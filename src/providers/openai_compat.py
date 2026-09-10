"""
OpenAI and Groq.

Groq serves an OpenAI-compatible endpoint, so both run through the same
client and differ only in base URL, model list and a couple of parameter
quirks. The canonical transcript format is already OpenAI-shaped, so
this provider passes messages through untouched.
"""

from __future__ import annotations

import re
from typing import Any, Iterator

import openai
from openai import OpenAI

from .base import Provider, ProviderError, ProviderSpec, StreamResult, ToolCall

# Reasoning / o-series models reject `temperature` (and several other
# sampling params) with a 400. Detect them by name so the same code path
# works for gpt-4o and o4-mini alike.
_REASONING_MODEL_RE = re.compile(r"^(o\d|gpt-5)", re.IGNORECASE)

OPENAI_SPEC = ProviderSpec(
    key="openai",
    label="OpenAI",
    blurb="Pay-as-you-go API key from platform.openai.com.",
    auth="api_key",
    models=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "o4-mini"],
    default_model="gpt-4o-mini",
    credentials_url="https://platform.openai.com/api-keys",
    key_label="OpenAI API key",
    key_env="OPENAI_API_KEY",
    supports_base_url=True,
)

GROQ_SPEC = ProviderSpec(
    key="groq",
    label="Groq",
    blurb="Very fast open-weight models. Free-tier API key from console.groq.com.",
    auth="api_key",
    models=[
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "moonshotai/kimi-k2-instruct",
        "qwen/qwen3-32b",
    ],
    default_model="llama-3.3-70b-versatile",
    credentials_url="https://console.groq.com/keys",
    key_label="Groq API key",
    key_env="GROQ_API_KEY",
    supports_base_url=False,
)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def supports_temperature(model: str) -> bool:
    """Whether `model` accepts a `temperature` parameter."""
    return not _REASONING_MODEL_RE.match(model or "")


def friendly_error(exc: Exception, model: str = "", label: str = "The API") -> str:
    """Turn an OpenAI-SDK exception into something worth showing a user."""
    if isinstance(exc, openai.AuthenticationError):
        return (
            f"Your {label} key was rejected (401). Check the key in the sidebar "
            "— and that it has access to the selected model."
        )
    if isinstance(exc, openai.PermissionDeniedError):
        return f"This key isn't allowed to use `{model}` (403). Pick a different model."
    if isinstance(exc, openai.NotFoundError):
        return (
            f"The model `{model}` doesn't exist or isn't available to this key (404). "
            "Pick another model in the sidebar."
        )
    if isinstance(exc, openai.RateLimitError):
        return (
            f"{label} rate-limited the request (429). You may be out of quota or "
            "sending too fast — wait a moment and retry."
        )
    if isinstance(exc, openai.APIConnectionError):
        return (
            f"Couldn't reach {label}. Check your network connection, or the base "
            "URL if you're routing through a proxy."
        )
    if isinstance(exc, openai.BadRequestError):
        return f"{label} rejected the request (400): {getattr(exc, 'message', exc)}"
    if isinstance(exc, openai.APIStatusError):
        return f"{label} returned an error ({exc.status_code}): {getattr(exc, 'message', exc)}"
    return f"The model call failed: {exc}"


class OpenAICompatProvider(Provider):
    """Streaming chat + tool calling over an OpenAI-compatible endpoint."""

    def __init__(
        self,
        spec: ProviderSpec,
        model: str,
        api_key: str,
        base_url: str | None = None,
        temperature: float = 0.2,
    ):
        super().__init__(spec, model)
        if not api_key:
            raise ProviderError(f"No {spec.key_label} was provided.")
        self.temperature = temperature

        if spec.key == "groq":
            base_url = GROQ_BASE_URL

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def iter_turn(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> Iterator[str]:
        wire = [{"role": "system", "content": system}] + [
            _strip_private(m) for m in messages
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": wire,
            "stream": True,
        }
        if supports_temperature(self.model):
            kwargs["temperature"] = self.temperature
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        result = StreamResult()
        # index -> partial tool call, since fragments arrive interleaved.
        partial: dict[int, dict] = {}

        try:
            stream = self.client.chat.completions.create(**kwargs)
            for chunk in stream:
                if not getattr(chunk, "choices", None):
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    result.finish_reason = choice.finish_reason

                delta = choice.delta
                if delta is None:
                    continue

                piece = getattr(delta, "content", None)
                if piece:
                    result.text += piece
                    yield piece

                for call in getattr(delta, "tool_calls", None) or []:
                    slot = partial.setdefault(
                        call.index, {"id": "", "name": "", "arguments": ""}
                    )
                    if call.id:
                        slot["id"] = call.id
                    fn = getattr(call, "function", None)
                    if fn is not None:
                        if fn.name:
                            slot["name"] += fn.name
                        if fn.arguments:
                            slot["arguments"] += fn.arguments
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(friendly_error(exc, self.model, self.spec.label)) from exc

        result.tool_calls = [
            ToolCall(
                id=partial[i]["id"] or f"call_{i}",
                name=partial[i]["name"],
                arguments=partial[i]["arguments"],
            )
            for i in sorted(partial)
            if partial[i]["name"]
        ]
        return result


def _strip_private(message: dict) -> dict:
    """Drop the `_native` bookkeeping key before sending to the API."""
    return {k: v for k, v in message.items() if not k.startswith("_")}
