"""
AI Investor v2 — Centralized logging.

One idempotent setup so the API, scheduler, and launcher can all run in the same
process (bundled app) without clobbering each other's handlers. Writes to console
plus a daily rotating file in the per-user logs folder.
"""
import sys
from loguru import logger

from agent.paths import logs_dir

_configured = False


def setup_logging():
    global _configured
    if _configured:
        return
    logger.remove()
    # Console sink — only if there's a console (a bundled windowed app has no stdout).
    if sys.stdout is not None:
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # avoid cp1252 emoji crashes on Windows
        except Exception:
            pass
        logger.add(
            sys.stdout,
            format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
            level="INFO",
            colorize=True,
        )
    # File sink — UTF-8 so emojis/unicode in log lines never crash the handler.
    logger.add(
        str(logs_dir() / "agent_{time:YYYY-MM-DD}.log"),
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
        encoding="utf-8",
    )
    _configured = True
