"""Download arXiv papers by topic search, HTML-first with PDF fallback.

Ported from AIR-FastAgent's arxiv_downloader_tool/arxiv_downloader.py,
simplified: HTML is stripped to plain text (stdlib html.parser, no new
dependency) instead of kept raw, and the PDF fallback just saves the .pdf
directly instead of shelling out to `pdftotext` — backend/offline_pipeline/
loaders.py already parses .pdf natively, so no external binary is needed at
all. Output is always one of Shonku's directly-ingestible types (.txt/.pdf).

Writes to data/raw/arxiv/<topic>/ (scratch) — same pattern as
download_sec_10q.py. Copy what you want into data/kbs/<slug>/ and ingest.

Run:
    python -m backend.offline_pipeline.download_arxiv --topic "diffusion models" --count 10
"""

import argparse
import logging
import re
from html.parser import HTMLParser
from pathlib import Path

import arxiv
import requests

from backend.logging_config import configure_logging

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT / "data" / "raw" / "arxiv"

FULL_TEXT_MARKERS = (
    "<article", 'class="ltx_paper"', 'class="ltx_content"',
    'id="main-content"', '<section class="ltx_section"',
)
ABSTRACT_ONLY_MARKERS = ('<div id="abs">', 'class="abstract mathjax"', "no html for", "not yet available")


def _clean_filename(title: str) -> str:
    return re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")


class _TextExtractor(HTMLParser):
    """Strips HTML tags to plain text, dropping script/style content."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip > 0:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            text = data.strip()
            if text:
                self._parts.append(text)

    def text(self) -> str:
        return "\n".join(self._parts)


def _html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.text()


def _is_full_text_html(html: str) -> bool:
    lower = html.lower()
    if any(marker in lower for marker in ABSTRACT_ONLY_MARKERS):
        return False
    return any(marker in lower for marker in FULL_TEXT_MARKERS)


def _try_html(url: str) -> str | None:
    try:
        resp = requests.get(url, timeout=15)
    except requests.RequestException as exc:
        logger.warning("  HTML fetch failed for %s: %s", url, exc)
        return None
    if resp.status_code == 200 and _is_full_text_html(resp.text):
        return _html_to_text(resp.text)
    return None


def download_topic(topic: str, count: int, output_dir: Path | None = None) -> None:
    out_dir = output_dir or (OUTPUT_DIR / _clean_filename(topic))
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("searching arXiv for %r (%d papers)...", topic, count)
    client = arxiv.Client()
    search = arxiv.Search(query=topic, max_results=count, sort_by=arxiv.SortCriterion.Relevance)
    results = list(client.results(search))

    ok = failed = 0
    for i, result in enumerate(results, 1):
        paper_id = result.get_short_id().split("v")[0]
        stem = f"{paper_id}_{_clean_filename(result.title)}"
        logger.info("[%d/%d] %s (%s)", i, len(results), result.title, paper_id)

        text = _try_html(f"https://arxiv.org/html/{paper_id}")
        if text is None:
            text = _try_html(f"https://ar5iv.labs.arxiv.org/html/{paper_id}")

        if text is not None:
            (out_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
            logger.info("  saved (html -> text)")
            ok += 1
            continue

        try:
            result.download_pdf(dirpath=str(out_dir), filename=f"{stem}.pdf")
            logger.info("  saved (pdf, no full-text html available)")
            ok += 1
        except Exception as exc:
            logger.error("  FAILED: %s", exc)
            failed += 1

    logger.info("done: %d saved, %d failed -> %s", ok, failed, out_dir)


def main() -> None:
    configure_logging(plain=True)
    parser = argparse.ArgumentParser(description="Download arXiv papers by topic (HTML-first, PDF fallback).")
    parser.add_argument("--topic", required=True, help="Search topic/query.")
    parser.add_argument("--count", type=int, default=10, help="Number of papers to fetch.")
    parser.add_argument("--dir", type=str, default=None, help="Output dir (default: data/raw/arxiv/<topic>/).")
    args = parser.parse_args()
    download_topic(args.topic, args.count, Path(args.dir) if args.dir else None)


if __name__ == "__main__":
    main()
