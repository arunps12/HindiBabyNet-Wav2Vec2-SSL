"""Structured logging via rich."""

from __future__ import annotations

import logging
import sys

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logger with rich handler and return the package logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )
    logger = logging.getLogger("homewav2vec2")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


log = setup_logging()
