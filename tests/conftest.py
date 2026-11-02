"""Shared pytest fixtures."""
import pytest


@pytest.fixture(autouse=True)
def _clean_version_bypass_env(monkeypatch):
    """Importing predict/config loads the local .env, which may set
    ALLOW_VERSION_MISMATCH=1 as a temporary dev bypass. Tests must never inherit
    it; tests that need the bypass set it explicitly with monkeypatch.setenv."""
    monkeypatch.delenv('ALLOW_VERSION_MISMATCH', raising=False)
