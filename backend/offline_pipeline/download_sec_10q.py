#!/usr/bin/env python3
"""
Download SEC 10-Q filings for AAPL, AMZN, INTC, MSFT, NVDA
Convert HTML filings to PDF using headless Chrome (--print-to-pdf)
Naming format: "YYYY QX TICKER.pdf" (matching KG-RAG-datasets repo)

Run:
    python -m backend.offline_pipeline.download_sec_10q

Writes raw HTML/PDF output to data/raw/sec_10q/. Ingest reads from
data/kbs/<slug>/ instead — copy the PDFs you want over there manually.
"""

import logging
import os
import re
import sys
import shutil
import time
import zipfile
import subprocess
from pathlib import Path

import requests

from backend.logging_config import configure_logging

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent

# ============================================================
# CONFIGURATION
# ============================================================
# SEC EDGAR requires a real contact email in the User-Agent or it
# 403s/blocks requests. Set SEC_EDGAR_CONTACT_EMAIL before running.
CONTACT_EMAIL = os.environ.get('SEC_EDGAR_CONTACT_EMAIL')

HEADERS = {
    'User-Agent': f'Mozilla/5.0 (compatible; ResearchBot/1.0; {CONTACT_EMAIL})',
    'Accept': 'application/json'
}

COMPANIES = {
    'AAPL': '0000320193',
    'AMZN': '0001018724',
    'INTC': '0000050863',
    'MSFT': '0000789019',
    'NVDA': '0001045810'
}

# === CONFIGURATION ===
START_YEAR = 2025      # Change this
END_YEAR = 2026        # Optional: add this for a capped range
OUTPUT_DIR = str(ROOT / "data" / "raw" / "sec_10q")
RATE_LIMIT_SEC = 0.25  # SEC rate limit (be polite)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_calendar_quarter(date_str):
    """Return Q1/Q2/Q3/Q4 from a YYYY-MM-DD string."""
    month = int(date_str.split('-')[1])
    if month <= 3:
        return 'Q1'
    elif month <= 6:
        return 'Q2'
    elif month <= 9:
        return 'Q3'
    else:
        return 'Q4'


def get_all_10q_filings(cik, ticker):
    """
    Fetch all 10-Q filings for a company from SEC submissions API.
    Returns list of dicts with accession, filing_date, period_end, primary_doc, etc.
    """
    url = f'https://data.sec.gov/submissions/CIK{cik}.json'
    logger.info("  Fetching metadata for %s...", ticker)
    
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    
    filings = data.get('filings', {})
    recent = filings.get('recent', {})
    files_list = filings.get('files', [])
    
    all_10qs = []
    seen_accessions = set()

    # --- Process recent filings ---
    forms = recent.get('form', [])
    accession_numbers = recent.get('accessionNumber', [])
    filing_dates = recent.get('filingDate', [])
    report_dates = recent.get('reportDate', [])
    primary_docs = recent.get('primaryDocument', [])
    
    for i, form in enumerate(forms):
        if form == '10-Q':
            # Determine period end date
            if i < len(report_dates) and report_dates[i]:
                period_end = report_dates[i]
            elif i < len(primary_docs) and primary_docs[i]:
                match = re.search(r'(\d{4})(\d{2})(\d{2})', primary_docs[i])
                if match:
                    period_end = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                else:
                    period_end = filing_dates[i]
            else:
                period_end = filing_dates[i]
            
            year = int(period_end.split('-')[0])
            accession = accession_numbers[i].replace('-', '')
            if START_YEAR <= year <= END_YEAR and accession not in seen_accessions:
                seen_accessions.add(accession)
                all_10qs.append({
                    'accession': accession,
                    'filing_date': filing_dates[i],
                    'period_end': period_end,
                    'primary_doc': primary_docs[i] if i < len(primary_docs) else '',
                    'cik': cik,
                    'ticker': ticker,
                    'year': year,
                    'quarter': get_calendar_quarter(period_end)
                })
    
    # --- Process older files (archived submissions) ---
    for file_info in files_list:
        file_name = file_info.get('name', '')
        if not file_name:
            continue
            
        file_url = f"https://data.sec.gov/submissions/{file_name}"
        try:
            file_resp = requests.get(file_url, headers=HEADERS, timeout=30)
            file_resp.raise_for_status()
            file_data = file_resp.json()
            
            file_forms = file_data.get('form', [])
            file_accessions = file_data.get('accessionNumber', [])
            file_dates = file_data.get('filingDate', [])
            file_reports = file_data.get('reportDate', [])
            file_docs = file_data.get('primaryDocument', [])
            
            for i, fform in enumerate(file_forms):
                if fform == '10-Q':
                    if i < len(file_reports) and file_reports[i]:
                        period_end = file_reports[i]
                    elif i < len(file_docs) and file_docs[i]:
                        match = re.search(r'(\d{4})(\d{2})(\d{2})', file_docs[i])
                        if match:
                            period_end = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                        else:
                            period_end = file_dates[i]
                    else:
                        period_end = file_dates[i]
                    
                    year = int(period_end.split('-')[0])
                    accession = file_accessions[i].replace('-', '')
                    if START_YEAR <= year <= END_YEAR and accession not in seen_accessions:
                        seen_accessions.add(accession)
                        all_10qs.append({
                            'accession': accession,
                            'filing_date': file_dates[i],
                            'period_end': period_end,
                            'primary_doc': file_docs[i] if i < len(file_docs) else '',
                            'cik': cik,
                            'ticker': ticker,
                            'year': year,
                            'quarter': get_calendar_quarter(period_end)
                        })
            time.sleep(RATE_LIMIT_SEC)
        except Exception as e:
            logger.warning("    Could not fetch %s: %s", file_name, e)
    
    return all_10qs


def download_html(filing):
    """Download the HTML filing from SEC archives."""
    numeric_cik = str(int(filing['cik']))
    accession = filing['accession']
    filename = filing['primary_doc']
    ticker = filing['ticker']
    period_end = filing['period_end']
    
    url = f"https://www.sec.gov/Archives/edgar/data/{numeric_cik}/{accession}/{filename}"
    html_path = os.path.join(OUTPUT_DIR, f"{ticker}_{period_end}.htm")
    
    # Skip if already downloaded
    if os.path.exists(html_path) and os.path.getsize(html_path) > 1000:
        return html_path, True  # cached
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        with open(html_path, 'wb') as f:
            f.write(resp.content)
        return html_path, False
    except Exception as e:
        logger.error("    ERROR downloading %s %s: %s", ticker, period_end, e)
        return None, False


def _find_chrome_binary():
    """Locate an installed Chromium-family browser for headless print-to-pdf."""
    for name in ('google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser',
                 'microsoft-edge', 'brave-browser'):
        path = shutil.which(name)
        if path:
            return path
    for path in (
        '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        '/Applications/Chromium.app/Contents/MacOS/Chromium',
        '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
        '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
    ):
        if os.path.exists(path):
            return path
    return None


CHROME_BINARY = _find_chrome_binary()


def convert_to_pdf(html_path, filing):
    """Convert HTML to PDF using headless Chrome's --print-to-pdf."""
    ticker = filing['ticker']
    year = filing['year']
    quarter = filing['quarter']

    pdf_name = f"{year} {quarter} {ticker}.pdf"
    pdf_path = os.path.abspath(os.path.join(OUTPUT_DIR, pdf_name))

    # Skip if PDF already exists and is valid
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 10000:
        return pdf_path, True  # cached

    abs_html_uri = 'file://' + os.path.abspath(html_path)
    cmd = [
        CHROME_BINARY,
        '--headless=new',
        '--disable-gpu',
        '--no-sandbox',
        '--no-pdf-header-footer',
        '--virtual-time-budget=10000',
        f'--print-to-pdf={pdf_path}',
        abs_html_uri,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if result.returncode == 0 and os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
            return pdf_path, False
        else:
            logger.error(
                "    ERROR converting %s %s: PDF conversion failed: %s",
                ticker, filing['period_end'], result.stderr.strip()[:200],
            )
            return None, False
    except Exception as e:
        logger.error("    ERROR converting %s %s: %s", ticker, filing['period_end'], e)
        return None, False


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    configure_logging(plain=True)

    if not CONTACT_EMAIL:
        logger.error(
            "SEC_EDGAR_CONTACT_EMAIL not set. SEC EDGAR requires a real contact "
            "email in the User-Agent header or it will block requests. Set it, e.g.:\n"
            "  export SEC_EDGAR_CONTACT_EMAIL=you@example.com"
        )
        sys.exit(1)
    if not CHROME_BINARY:
        logger.error(
            "No Chromium-family browser found (Chrome/Chromium/Edge/Brave). "
            "Install one — it's used headless for HTML->PDF conversion, no "
            "extra PDF tool needed."
        )
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SEC 10-Q Downloader & PDF Converter")
    logger.info("Format: 'YYYY QX TICKER.pdf'")
    logger.info("=" * 60)

    # Step 1: Fetch all filing metadata
    all_filings = {}
    total_expected = 0

    for ticker, cik in COMPANIES.items():
        filings = get_all_10q_filings(cik, ticker)
        all_filings[ticker] = filings
        total_expected += len(filings)
        year_range = f"{min(f['year'] for f in filings)}-{max(f['year'] for f in filings)}" if filings else "N/A"
        logger.info("  %s: %d filings found (%s)", ticker, len(filings), year_range)
        time.sleep(RATE_LIMIT_SEC)

    logger.info("\nTotal filings to process: %d", total_expected)
    logger.info("-" * 60)

    # Step 2: Download HTML and convert to PDF
    converted = []
    failed = []

    for ticker in COMPANIES:
        filings = all_filings[ticker]
        logger.info("\nProcessing %s (%d filings)...", ticker, len(filings))

        for i, filing in enumerate(filings, 1):
            # Download HTML
            html_path, html_cached = download_html(filing)
            if not html_path:
                failed.append(filing)
                time.sleep(RATE_LIMIT_SEC)
                continue

            # Convert to PDF
            pdf_path, pdf_cached = convert_to_pdf(html_path, filing)
            if pdf_path:
                converted.append({
                    'ticker': ticker,
                    'pdf': pdf_path,
                    'cached': html_cached and pdf_cached,
                    'period_end': filing['period_end']
                })
                status = "cached" if (html_cached and pdf_cached) else "new"
                logger.info("  [%d/%d] %d %s %s (%s)", i, len(filings), filing['year'], filing['quarter'], ticker, status)
            else:
                failed.append(filing)

            time.sleep(RATE_LIMIT_SEC)

    # Step 3: Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Total converted: %d", len(converted))
    logger.info("Total failed: %d", len(failed))

    for ticker in COMPANIES:
        ticker_converted = sum(1 for c in converted if c['ticker'] == ticker)
        ticker_failed = sum(1 for f in failed if f['ticker'] == ticker)
        logger.info("  %s: %d OK, %d failed", ticker, ticker_converted, ticker_failed)

    # Step 4: Create ZIP file
    if converted:
        zip_path = os.path.join(OUTPUT_DIR, 'SEC_10Q_All_Companies_20Years.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for item in converted:
                arcname = os.path.basename(item['pdf'])
                zf.write(item['pdf'], arcname)

        zip_size = os.path.getsize(zip_path) / (1024 * 1024)
        logger.info("\nZIP created: %s", zip_path)
        logger.info("ZIP size: %.1f MB", zip_size)

    # Step 5: List failed if any
    if failed:
        logger.info("\nFailed filings:")
        for f in failed:
            logger.info("  %s %s (accession: %s)", f['ticker'], f['period_end'], f['accession'])

    logger.info("\nDone!")


if __name__ == '__main__':
    main()