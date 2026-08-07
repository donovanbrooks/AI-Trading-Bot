"""Structured application logging with local rotation."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging() -> logging.Logger:
    """Configure a rotating file log once and return the app logger."""
    logger = logging.getLogger("trading_bot")
    if logger.handlers:
        return logger
    Path("logs").mkdir(exist_ok=True)
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler("logs/trading_bot.log", maxBytes=1_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    return logger
