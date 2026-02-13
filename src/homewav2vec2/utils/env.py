"""Environment helpers — load .env safely, never print secrets."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_env(env_path: str | Path | None = None) -> None:
    """Load .env from *env_path* (default: project root)."""
    if env_path is None:
        # Walk up from this file to find .env
        env_path = Path(__file__).resolve().parents[3] / ".env"
    else:
        env_path = Path(env_path)
    if env_path.exists():
        load_dotenv(env_path, override=False)


def get_hf_token() -> str:
    """Return the HF_TOKEN from environment (never print it)."""
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN not set.  Copy .env.example → .env and fill in your token."
        )
    return token


def get_hf_repo_id() -> str:
    return os.environ.get("HF_REPO_ID", "arunps/wav2vec2-home-hindibabynet")


def get_hf_private() -> bool:
    return os.environ.get("HF_PRIVATE", "true").lower() == "true"
