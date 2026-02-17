"""Structured logging with run_id support."""

import logging
import sys
import uuid
from typing import Optional


def get_logger(name: str, run_id: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Return a structured logger with optional run_id prefix."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        run_prefix = f"[run={run_id}] " if run_id else ""
        formatter = logging.Formatter(
            f"%(asctime)s %(levelname)-8s {run_prefix}%(name)s — %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def new_run_id() -> str:
    """Generate a short unique run identifier."""
    return str(uuid.uuid4())[:8]
