"""
Provider-neutral types shared by every backend.

The canonical transcript format is OpenAI-flavoured (role / content /
tool_calls / tool) because that is what the agent loop speaks. Each
provider translates to and from its own wire format.

Providers come in two shapes:

* **Chat providers** (OpenAI, Groq, Anthropic API) implement `iter_turn`
  and plug into the agent loop in `src/agent.py`.
* **Loop-owning providers** (the Claude Agent SDK) set `owns_loop = True`
  and drive the whole turn themselves, emitting the same agent events.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator


class ProviderError(RuntimeError):
    """Any backend failure, already phrased for an end user."""


@dataclass
class ToolCall:
    """One tool invocation requested by the model."""

    id: str
    name: str
    arguments: str  # raw JSON, parsed by the caller


@dataclass
class StreamResult:
    """Everything one streamed assistant turn produced."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    # Provider-native assistant content, kept so a transcript can be
    # replayed losslessly (Anthropic requires thinking blocks to be
    # echoed back unchanged on the same model).
    native: dict | None = None

    def as_assistant_message(self) -> dict:
        """Render this turn as a canonical `assistant` message."""
        message: dict[str, Any] = {"role": "assistant", "content": self.text or None}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        if self.native:
            message["_native"] = self.native
        return message


@dataclass
class ProviderSpec:
    """Everything the UI needs to offer and configure one backend."""

    key: str
    label: str
    blurb: str
    # How the backend is authenticated: "api_key" or "subscription".
    auth: str
    models: list[str]
    default_model: str
    # Where to get credentials, shown in the sidebar.
    credentials_url: str = ""
    key_label: str = "API key"
    key_env: str = ""
    supports_base_url: bool = False


class Provider(ABC):
    """Base class for a model backend."""

    #: True when the provider runs its own agent loop rather than
    #: returning one assistant turn at a time.
    owns_loop: bool = False

    def __init__(self, spec: ProviderSpec, model: str):
        self.spec = spec
        self.model = model

    @property
    def key(self) -> str:
        return self.spec.key

    @abstractmethod
    def iter_turn(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> Iterator[str]:
        """Stream one assistant turn.

        Yields text deltas as they arrive and *returns* a
        :class:`StreamResult` (read it via ``StopIteration.value``, or
        ``result = yield from provider.iter_turn(...)``).

        Raises :class:`ProviderError` on any backend failure.
        """
        raise NotImplementedError


def native_content(message: dict, provider_key: str):
    """Return a message's provider-native content, if it matches `provider_key`.

    A transcript can outlive a provider switch, so native content is
    tagged with the provider that produced it and ignored by everyone
    else -- that way replaying an Anthropic thinking block through Groq
    degrades to plain text instead of erroring.
    """
    native = message.get("_native")
    if isinstance(native, dict) and native.get("provider") == provider_key:
        return native.get("content")
    return None
