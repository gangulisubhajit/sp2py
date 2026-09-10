"""Shared fixtures.

The suite runs entirely offline: `mock_llm.py` is started as a real HTTP
server speaking both the OpenAI and Anthropic wire protocols, so the actual
SDK streaming and tool-call parsing is on the tested path rather than being
stubbed out.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def mock_api() -> str:
    """Start the mock model server; yield its base URL.

    It serves both the OpenAI and Anthropic wire protocols, so the same
    scripted trajectory can be replayed through either provider.
    """
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "tests", "mock_llm.py"), str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("mock server exited during startup")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)
    else:
        proc.kill()
        raise RuntimeError("mock server did not start in time")

    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    """Point the JSON stores at a temp dir so tests never touch real data/."""
    from src import feedback_notes, history_store

    monkeypatch.setattr(history_store, "HISTORY_PATH", str(tmp_path / "history.json"))
    monkeypatch.setattr(feedback_notes, "NOTES_PATH", str(tmp_path / "feedback_notes.json"))
    yield


@pytest.fixture
def oracle_sql() -> str:
    with open(os.path.join(ROOT, "sample_procedures", "sample_oracle.sql"), encoding="utf-8") as fh:
        return fh.read()
