"""Console + structured logging helpers built on rich.

Kept intentionally boring: one console, one ``get_logger`` factory, and a couple
of helpers for printing tables/panels in CLI commands.
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()
_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    _CONFIGURED = True


def get_logger(name: str = "guardian") -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
