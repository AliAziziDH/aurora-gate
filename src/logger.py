"""
Centralized logging module for AuroraGate project.
"""

import logging
from pathlib import Path

from src.config import EXPERIMENTS_DIR, LOG_FORMAT, LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """Return a configured console and experiment-file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False

    if not logger.handlers:
        c_handler = logging.StreamHandler()
        c_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(c_handler)

        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        f_handler = logging.FileHandler(EXPERIMENTS_DIR / f"{name}.log")
        f_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(f_handler)

    return logger


if __name__ == "__main__":
    get_logger("aurora_gate").info("AuroraGate logger initialized")
