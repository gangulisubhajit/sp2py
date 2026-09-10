"""
Persisted "learned" corrections.

When a user flags a conversion as wrong and checks "apply this as a
general rule", the correction is stored here and folded into the system
prompt of *every future* conversion (see prompts.build_learned_notes_block).
This is what lets the agent's behaviour actually change over time instead
of only correcting itself within a single chat thread.
"""

from __future__ import annotations

import json
import os
import time
import uuid

NOTES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "feedback_notes.json")


def _ensure_store():
    os.makedirs(os.path.dirname(NOTES_PATH), exist_ok=True)
    if not os.path.exists(NOTES_PATH):
        with open(NOTES_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)


def load_notes() -> list[dict]:
    _ensure_store()
    with open(NOTES_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def add_note(note_text: str, dialect: str | None = None) -> dict:
    _ensure_store()
    notes = load_notes()
    record = {
        "id": str(uuid.uuid4())[:8],
        "note": note_text.strip(),
        "dialect": dialect,
        "added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    notes.append(record)
    with open(NOTES_PATH, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)
    return record


def delete_note(note_id: str) -> None:
    notes = [n for n in load_notes() if n.get("id") != note_id]
    with open(NOTES_PATH, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)


def notes_for_dialect(dialect: str | None) -> list[dict]:
    """Return general notes plus any notes scoped specifically to this dialect."""
    all_notes = load_notes()
    return [n for n in all_notes if not n.get("dialect") or n.get("dialect") == dialect]
