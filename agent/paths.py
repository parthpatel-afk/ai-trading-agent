"""
Trend-Buyer v1 — Path resolution.

Everything is anchored to the PROJECT ROOT so the database and logs live visibly
next to the code (matches our reference-files workflow). No PyInstaller / per-user
AppData indirection — this runs from source.
"""
from pathlib import Path

APP_NAME = "TrendBuyer"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent  # …/AI Investor


def resource_dir() -> Path:
    return project_root()


def user_data_dir() -> Path:
    return project_root()


def logs_dir() -> Path:
    d = project_root() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def db_path() -> Path:
    return project_root() / "agent_data.db"


def dashboard_dir() -> Path:
    return project_root() / "dashboard" / "out"
