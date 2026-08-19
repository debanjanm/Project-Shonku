"""Env-driven config. Router model names live here, not hardcoded in agent.py."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RouterConfig:
    easy_model: str
    hard_model: str


def get_router_config() -> RouterConfig:
    return RouterConfig(
        easy_model=os.environ.get("ROUTER_EASY_MODEL", "openai/gpt-5.4-nano"),
        hard_model=os.environ.get("ROUTER_HARD_MODEL", "openai/gpt-5.4-mini"),
    )
