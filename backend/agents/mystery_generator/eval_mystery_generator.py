"""Mystery Generator eval harness. Complements test_path_discipline.py (a
pass/fail regression guard for one specific past bug, file-path discipline)
with a scored quality eval across many samples — same LLM-as-judge approach
eval_story_developer.py uses, since this is generative writing with no
ground truth to check recall against.

Judge criteria deliberately mirror the fair-play-mystery-design skill
(skills/fair-play-mystery-design/SKILL.md) verbatim — the eval checks the
exact thing that skill is supposed to produce.

Run:
    python -m backend.agents.mystery_generator.eval_mystery_generator
"""

import json
import logging
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(ROOT / ".env")

from openai import OpenAI  # noqa: E402

from backend.agents.mystery_generator.agent import MYSTERIES_DIR, make_mystery_agent  # noqa: E402
from backend.config import get_router_config  # noqa: E402
from backend.logging_config import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

REPORT_PATH = ROOT / "data" / "raw" / "mystery_generator_eval_report.json"

SEED_THEMES = [
    "A jewel heist at a museum gala gone wrong.",
    "A poisoning at a family dinner reunion.",
    "A locked-room murder in a mountain cabin during a snowstorm.",
    "A disappearance on the last night of a cruise ship voyage.",
    "A sabotage at a Formula-style racing team's headquarters.",
    "A murder during a live theater performance.",
    "A death at a remote research station in Antarctica.",
    "A kidnapping at a high-society wedding.",
    "A fatal accident at a tech startup's product launch that wasn't an accident.",
    "A murder at a chess tournament between two rival grandmasters.",
]

EVAL_ID_OFFSET = 900_000_000  # synthetic conversation_ids, well clear of real DB ids

JUDGE_SYSTEM_PROMPT = """You are a strict fair-play mystery editor. Score a generated mystery case against fixed
criteria, comparing the public premise (what the player sees) against the private solution (the real answer).
Always respond with valid JSON only. No markdown, no explanation outside the JSON."""

JUDGE_USER_TEMPLATE = """Evaluate this mystery case against each criterion below. For each, answer true or false.

- exactly_one_culprit: the solution names exactly one culprit, not multiple or an ambiguous set.
- premise_no_culprit_leak: the public premise does NOT name or clearly imply the culprit or their motive.
- clues_shown_before_reveal: every key clue the solution depends on is already present (directly or as
  evidence pointing to it) in the public premise — nothing decisive is introduced only in the solution.
- red_herrings_plausible: any red herrings/suspects in the premise look genuinely suspicious on their own
  terms, not arbitrary padding.
- internally_consistent: method, motive, and timeline in the solution don't contradict anything established
  in the public premise.

Return a JSON object with exactly these 5 boolean keys.

PUBLIC PREMISE (what the player sees):
{premise}

PRIVATE SOLUTION (the real answer, never shown to the player):
{solution}"""


def _judge(client: OpenAI, model: str, premise: str, solution: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": JUDGE_USER_TEMPLATE.format(premise=premise, solution=solution)},
            ],
            temperature=0.0,
            max_tokens=300,
            timeout=30,
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception:
        logger.exception("judge call failed")
        return {}


CRITERIA = (
    "exactly_one_culprit",
    "premise_no_culprit_leak",
    "clues_shown_before_reveal",
    "red_herrings_plausible",
    "internally_consistent",
)


def run_eval(seed_themes: list[str] = SEED_THEMES) -> None:
    configure_logging(plain=True)
    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")
    judge_model = get_router_config().hard_model

    hits = {c: 0 for c in CRITERIA}
    overall_hits = 0
    file_discipline_failures = 0
    samples = []

    for i, theme in enumerate(seed_themes, 1):
        conversation_id = EVAL_ID_OFFSET + i
        case_dir = MYSTERIES_DIR / str(conversation_id)
        logger.info("[%d/%d] %r (conversation_id=%d)", i, len(seed_themes), theme, conversation_id)

        try:
            agent = make_mystery_agent(conversation_id)
            result = agent.invoke({"messages": [{"role": "user", "content": theme}]})
            premise = result["messages"][-1].content

            solution_path = case_dir / "solution.md"
            if not solution_path.exists() or not solution_path.read_text().strip():
                logger.warning("solution.md missing/empty for %r — file-path discipline failure, auto-fail", theme)
                file_discipline_failures += 1
                samples.append({"theme": theme, "scores": None, "all_pass": False, "file_discipline_ok": False})
                continue
            solution = solution_path.read_text()

            scores = _judge(client, judge_model, premise, solution)
            passed = {c: bool(scores.get(c, False)) for c in CRITERIA}
            for c in CRITERIA:
                if passed[c]:
                    hits[c] += 1
            all_pass = all(passed.values())
            if all_pass:
                overall_hits += 1
            samples.append({"theme": theme, "scores": passed, "all_pass": all_pass, "file_discipline_ok": True})
            logger.info("  -> %s", passed)
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    n = len(seed_themes)
    report = {
        "sample_size": n,
        "file_discipline_failures": file_discipline_failures,
        "criteria_pass_rate": {c: round(hits[c] / n, 4) for c in CRITERIA},
        "overall_pass_rate": round(overall_hits / n, 4),
        "samples": samples,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    logger.info("=== Mystery Generator eval (n=%d) ===", n)
    if file_discipline_failures:
        logger.warning("file-path discipline failures: %d/%d", file_discipline_failures, n)
    for c in CRITERIA:
        logger.info("%-24s %.1f%%", c, report["criteria_pass_rate"][c] * 100)
    logger.info("Overall pass rate: %.1f%%", report["overall_pass_rate"] * 100)
    logger.info("report written to %s", REPORT_PATH)


def main() -> None:
    run_eval()


if __name__ == "__main__":
    main()
