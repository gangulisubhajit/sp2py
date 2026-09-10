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
    #: Substrings of model ids that cannot accept *custom* tools, and so
    #: can't run this agent. Groq's "compound" systems are the case that
    #: matters: they're chat models with their own built-in tools, and
    #: they reject a `tools` array with a 400.
    no_custom_tools: tuple[str, ...] = ()


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

    def list_models(self) -> list[str]:
        """Chat-capable model ids this credential can actually use.

        Providers that expose a models endpoint override this. Hard-coded
        lists go stale as vendors retire models -- Groq in particular
        rotates them often -- so the UI prefers a live list when it can
        get one and falls back to :attr:`ProviderSpec.models` otherwise.

        Raises :class:`ProviderError` if the lookup fails.
        """
        return list(self.spec.models)


# Model ids that are real but useless here: speech, embeddings, images,
# moderation, safety classifiers. Matched as substrings, case-insensitive.
NON_CHAT_MODEL_MARKERS = (
    "whisper",
    "tts",
    "orpheus",
    "prompt-guard",
    "safeguard",
    "embed",
    "moderation",
    "dall-e",
    "davinci",
    "babbage",
    "image",
    "realtime",
    "audio",
    "transcribe",
    "rerank",
)


def is_chat_model(model_id: str) -> bool:
    """Whether a model id looks like something we can hold a conversation with."""
    lowered = (model_id or "").lower()
    return bool(lowered) and not any(m in lowered for m in NON_CHAT_MODEL_MARKERS)


def accepts_custom_tools(model_id: str, spec: ProviderSpec) -> bool:
    """Whether a model can be given this agent's tools.

    The agent is nothing but tool calls, so a model that rejects a
    `tools` array is unusable here however good it otherwise is.
    """
    lowered = (model_id or "").lower()
    return not any(m.lower() in lowered for m in spec.no_custom_tools)


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
