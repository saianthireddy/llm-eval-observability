import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(os.getenv("TARGET_REPO", ROOT.parent / "rag-support-automation"))


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    # Tests must never reach a real model, whatever the developer's shell has set.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def target_repo() -> Path:
    if not (TARGET / "src" / "rag_support").is_dir():
        pytest.skip(f"rag-support-automation checkout not found at {TARGET} (set TARGET_REPO)")
    return TARGET
