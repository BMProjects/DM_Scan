"""Lightweight logging configuration for microlens-defects."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def get_logger(name: str = "microlens_defects", level: Optional[int] = None) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name (default: 'microlens_defects')
        level: Optional logging level override

    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(level or logging.INFO)
    elif level is not None:
        logger.setLevel(level)

    return logger


def set_log_level(level: int) -> None:
    """Set log level for the default microlens_defects logger.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
    """
    get_logger().setLevel(level)
