"""
utils/logger.py

Centralized logging configuration for the project. Configures the root
logger once (console + optional file handler), so every module can just
call get_logger(__name__) and inherit consistent formatting.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

_ROOT_CONFIGURED = False

DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logger(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    fmt: str = DEFAULT_FORMAT,
) -> logging.Logger:
    """
    Configure the root logger with a console handler (and optionally a
    file handler). Safe to call multiple times — only configures once.

    Args:
        level: Logging level for the root logger
        log_file: Optional path to also write logs to a file
        fmt: Log message format string

    Returns:
        The configured root logging.Logger
    """
    global _ROOT_CONFIGURED

    root = logging.getLogger()

    if _ROOT_CONFIGURED:
        return root

    root.setLevel(level)
    formatter = logging.Formatter(fmt)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _ROOT_CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    """
    Get a module-level logger. Ensures the root logger is configured
    (with default settings) on first use, so modules don't need to call
    setup_logger() themselves unless they want custom settings (e.g. a
    log file) applied before any module-level loggers are created.

    Args:
        name: Usually __name__ of the calling module

    Returns:
        A logging.Logger that inherits the root's handlers/formatting
    """
    setup_logger()
    return logging.getLogger(name)