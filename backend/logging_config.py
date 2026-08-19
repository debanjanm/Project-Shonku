"""Shared logging setup. Call once per entrypoint before doing anything else."""

import logging
import os


_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3")


def configure_logging(*, plain: bool = False) -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    fmt = "%(message)s" if plain else "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
