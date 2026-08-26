"""Heuristic router: picks easy/hard model from the question text, no LLM call."""

import logging
from typing import Literal

from backend.config import get_router_config

logger = logging.getLogger(__name__)

HARD_KEYWORDS = (
    "compare", "comparison", "versus", " vs ", "difference", "trend",
    "why", "analyze", "analysis", "explain", "growth", "correlation",
    "across", "relationship between",
)
HARD_WORD_COUNT = 25


def classify_intent(message: str) -> Literal["easy", "hard"]:
    text = message.lower()
    if len(text.split()) > HARD_WORD_COUNT:
        return "hard"
    if text.count("?") > 1:
        return "hard"
    if any(kw in text for kw in HARD_KEYWORDS):
        return "hard"
    return "easy"


def choose_model(message: str) -> str:
    config = get_router_config()
    classification = classify_intent(message)
    model = config.hard_model if classification == "hard" else config.easy_model
    logger.info("router classification=%s model=%s message=%r", classification, model, message)
    return model


# Labeled examples drawn from questions actually tested against this project's
# seeded KBs (hr-policies, product-docs, sec_10q). Self-check: python -m backend.agents.docqa.router
_EVAL_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("How many PTO days do employees get per year?", "easy"),
    ("What is the home office stipend amount?", "easy"),
    ("How many days of PTO carry over to next year?", "easy"),
    ("What is Project Shonku?", "easy"),
    ("What was Apple's Q1 2026 revenue?", "easy"),
    ("Which knowledge base has HR policies?", "easy"),
    ("Compare Apple and Amazon's Q1 2026 revenue growth and explain what drove the difference.", "hard"),
    ("What are the trends in Microsoft's revenue across the last four quarters?", "hard"),
    ("Why did Intel's margins change, and how does that compare to Nvidia's?", "hard"),
    ("Analyze the correlation between headcount growth and SG&A expenses across all five companies.", "hard"),
    ("What's the difference between the remote work policy and the PTO policy in terms of approval steps?", "hard"),
    ("Give me a full breakdown of quarterly revenue, margin trends, and headcount changes for every company in this knowledge base, and explain what's driving each one.", "hard"),
)


def _run_eval() -> None:
    correct = 0
    for message, expected in _EVAL_EXAMPLES:
        got = classify_intent(message)
        ok = got == expected
        correct += ok
        print(f"{'OK  ' if ok else 'MISS'} expected={expected:<5} got={got:<5} {message!r}")
    total = len(_EVAL_EXAMPLES)
    accuracy = correct / total
    print(f"\naccuracy: {correct}/{total} ({accuracy:.0%})")
    assert accuracy >= 0.8, f"router heuristic accuracy {accuracy:.0%} below 80% threshold"


if __name__ == "__main__":
    _run_eval()
