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
