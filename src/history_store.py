"""
Simple JSON-file-backed history store powering the Historical Audit
Dashboard and Agent Core Analytics workspaces. A real deployment would
likely swap this for SQLite/Postgres, but a flat JSON file keeps the
"run it in VS Code with zero setup" promise intact.
"""

from __future__ import annotations

import json
import os
import time
import uuid

HISTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "history.json")


def _ensure_store():
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    if not os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)


def load_history() -> list[dict]:
    _ensure_store()
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_record(record: dict) -> str:
    """Insert or update a history record (by id) and persist to disk."""
    _ensure_store()
    history = load_history()
    record_id = record.get("id") or str(uuid.uuid4())[:8]
    record["id"] = record_id
    record.setdefault("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))

    for i, existing in enumerate(history):
        if existing.get("id") == record_id:
            history[i] = record
            break
    else:
        history.append(record)

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return record_id


def get_record(record_id: str) -> dict | None:
    for r in load_history():
        if r.get("id") == record_id:
            return r
    return None


def compute_kpis(history: list[dict] | None = None) -> dict:
    history = history if history is not None else load_history()
    total = len(history)
    if total == 0:
        return {
            "total_processed": 0,
            "avg_duration_seconds": 0.0,
            "ast_success_rate": 0.0,
        }
    durations = [h.get("duration_seconds", 0) for h in history]
    successes = [1 for h in history if h.get("ast_valid")]
    return {
        "total_processed": total,
        "avg_duration_seconds": round(sum(durations) / total, 2),
        "ast_success_rate": round(100.0 * len(successes) / total, 1),
    }
