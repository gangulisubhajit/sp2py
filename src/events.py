"""
Events emitted while the agent works on one turn.

These live in their own module so both the agent loop and the
loop-owning providers can emit them without importing each other.
The UI renders whatever arrives, regardless of which backend produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TextDelta:
    """A fragment of the assistant's prose reply."""

    text: str


@dataclass
class ToolStarted:
    name: str
    arguments: dict


@dataclass
class ToolFinished:
    name: str
    arguments: dict
    result: dict
    ok: bool


@dataclass
class ArtifactUpdated:
    """A generated Python module was written or replaced."""

    artifact: Any


@dataclass
class TurnFinished:
    """The turn is over; `text` is the full reply for the transcript."""

    text: str
    #: Equivalent API cost in USD, when the backend reports it. On a
    #: Claude subscription this is an estimate of what the same work
    #: would have cost via the API, not an amount actually charged.
    cost_usd: float | None = None


AgentEvent = TextDelta | ToolStarted | ToolFinished | ArtifactUpdated | TurnFinished
