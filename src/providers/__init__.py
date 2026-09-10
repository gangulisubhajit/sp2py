"""
Model backends SP2PY can talk to.

Three providers, four ways in:

    openai       OpenAI            API key
    anthropic    Claude            Anthropic API key (billed separately)
    claude_code  Claude            your claude.ai subscription, via Claude Code
    groq         Groq              API key

`build()` turns a provider key plus credentials into a ready
:class:`~.base.Provider`. Everything above this layer works in terms of
the abstract provider and the events in `src/events.py`, so adding a
backend means adding a module here and one entry in `SPECS`.
"""

from __future__ import annotations

from .anthropic_api import ANTHROPIC_SPEC, AnthropicProvider
from .base import Provider, ProviderError, ProviderSpec, StreamResult, ToolCall
from .claude_code import CLAUDE_CODE_SPEC, ClaudeCodeProvider
from .claude_code import is_available as claude_code_available
from .openai_compat import GROQ_SPEC, OPENAI_SPEC, OpenAICompatProvider

__all__ = [
    "Provider",
    "ProviderError",
    "ProviderSpec",
    "StreamResult",
    "ToolCall",
    "SPECS",
    "PROVIDER_KEYS",
    "spec_for",
    "build",
    "claude_code_available",
]

SPECS: dict[str, ProviderSpec] = {
    spec.key: spec
    for spec in (OPENAI_SPEC, ANTHROPIC_SPEC, CLAUDE_CODE_SPEC, GROQ_SPEC)
}

#: Display order in the UI.
PROVIDER_KEYS = ["openai", "anthropic", "claude_code", "groq"]


def spec_for(key: str) -> ProviderSpec:
    try:
        return SPECS[key]
    except KeyError:
        raise ProviderError(f"Unknown provider '{key}'.") from None


def build(
    key: str,
    model: str = "",
    api_key: str = "",
    base_url: str = "",
) -> Provider:
    """Construct the provider identified by `key`.

    Raises :class:`ProviderError` with a user-readable message when the
    backend can't be used (missing key, CLI not installed, ...).
    """
    spec = spec_for(key)
    model = model or spec.default_model

    if key == "claude_code":
        return ClaudeCodeProvider(spec, model)
    if key == "anthropic":
        return AnthropicProvider(spec, model, api_key, base_url)
    return OpenAICompatProvider(spec, model, api_key, base_url or None)
