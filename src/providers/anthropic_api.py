"""
Anthropic Messages API (pay-as-you-go API key).

Translates the canonical OpenAI-shaped transcript to and from Anthropic's
content-block format:

    canonical                         Anthropic
    -------------------------------   ----------------------------------
    system (separate argument)     -> top-level `system`
    {"role":"user","content":str}  -> {"role":"user","content":[text]}
    assistant text + tool_calls    -> assistant content: text + tool_use
    {"role":"tool", ...}           -> user content: tool_result blocks

Consecutive `tool` messages are merged into a single user turn, because
Anthropic requires every tool_result for one assistant turn to arrive
together.

Thinking blocks must be echoed back unchanged on the same model, so the
assistant's native content list is stashed on the canonical message and
replayed verbatim when it is available.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

import anthropic

from .base import (
    Provider,
    ProviderError,
    ProviderSpec,
    StreamResult,
    ToolCall,
    is_chat_model,
    native_content,
)

ANTHROPIC_SPEC = ProviderSpec(
    key="anthropic",
    label="Claude (API key)",
    blurb=(
        "Anthropic API key from console.anthropic.com. Billed separately "
        "from a claude.ai Pro/Max subscription."
    ),
    auth="api_key",
    models=[
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-haiku-4-5",
        "claude-opus-4-8",
    ],
    default_model="claude-opus-5",
    credentials_url="https://console.anthropic.com/settings/keys",
    key_label="Anthropic API key",
    key_env="ANTHROPIC_API_KEY",
    supports_base_url=False,
)

MAX_TOKENS = 16000


def friendly_error(exc: Exception, model: str = "") -> str:
    """Turn an Anthropic-SDK exception into something worth showing a user."""
    if isinstance(exc, anthropic.AuthenticationError):
        return (
            "Your Anthropic API key was rejected (401). Note that a claude.ai "
            "Pro/Max subscription does not include API credits — you need a key "
            "from console.anthropic.com with billing set up."
        )
    if isinstance(exc, anthropic.PermissionDeniedError):
        return f"This API key isn't allowed to use `{model}` (403)."
    if isinstance(exc, anthropic.NotFoundError):
        return f"The model `{model}` doesn't exist or isn't available to this key (404)."
    if isinstance(exc, anthropic.RateLimitError):
        return (
            "Anthropic rate-limited the request (429). You may be out of credit "
            "or sending too fast — wait a moment and retry."
        )
    if isinstance(exc, anthropic.APIConnectionError):
        return "Couldn't reach the Anthropic API. Check your network connection."
    if isinstance(exc, anthropic.BadRequestError):
        return f"Anthropic rejected the request (400): {getattr(exc, 'message', exc)}"
    if isinstance(exc, anthropic.APIStatusError):
        return f"Anthropic returned an error ({exc.status_code}): {getattr(exc, 'message', exc)}"
    return f"The model call failed: {exc}"


def to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-style function schemas to Anthropic tool schemas."""
    converted = []
    for tool in tools or []:
        fn = tool.get("function", tool)
        converted.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return converted


def to_anthropic_messages(messages: list[dict]) -> list[dict]:
    """Convert the canonical transcript to Anthropic's message list."""
    out: list[dict] = []

    for message in messages:
        role = message.get("role")

        if role == "user":
            _append(out, "user", [{"type": "text", "text": message.get("content") or ""}])

        elif role == "assistant":
            replay = native_content(message, "anthropic")
            if replay:
                # Echo the model's own blocks back untouched -- required
                # for thinking blocks to stay valid.
                _append(out, "assistant", list(replay))
                continue

            blocks: list[dict] = []
            text = message.get("content")
            if text:
                blocks.append({"type": "text", "text": text})
            for call in message.get("tool_calls") or []:
                fn = call["function"]
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": fn["name"],
                        "input": _safe_json(fn.get("arguments")),
                    }
                )
            if blocks:
                _append(out, "assistant", blocks)

        elif role == "tool":
            _append(
                out,
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id", ""),
                        "content": message.get("content") or "",
                    }
                ],
            )

    return out


def _append(out: list[dict], role: str, blocks: list) -> None:
    """Add blocks, merging into the previous turn when the role repeats.

    Anthropic requires all tool_results for one assistant turn to sit in a
    single user message, and rejects an empty content list.
    """
    if not blocks:
        return
    if out and out[-1]["role"] == role:
        out[-1]["content"].extend(blocks)
    else:
        out.append({"role": role, "content": list(blocks)})


def _safe_json(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class AnthropicProvider(Provider):
    """Streaming chat + tool calling over the Anthropic Messages API."""

    def __init__(self, spec: ProviderSpec, model: str, api_key: str, base_url: str = ""):
        super().__init__(spec, model)
        if not api_key:
            raise ProviderError("No Anthropic API key was provided.")
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            # Not exposed in the UI; used by the offline test harness and
            # anyone fronting the API with a gateway.
            kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**kwargs)

    def list_models(self) -> list[str]:
        """Ask the API which Claude models this key can use."""
        try:
            page = self.client.models.list()
        except Exception as exc:
            raise ProviderError(friendly_error(exc, self.model)) from exc

        ids = [getattr(m, "id", "") for m in page]
        ids = [i for i in ids if is_chat_model(i)]
        if not ids:
            raise ProviderError("Anthropic returned no models for this key.")
        return ids

    def iter_turn(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> Iterator[str]:
        wire = to_anthropic_messages(messages)
        if not wire:
            raise ProviderError("There is nothing to send — the conversation is empty.")

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": wire,
        }
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)

        result = StreamResult()
        try:
            with self.client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    result.text += text
                    yield text
                final = stream.get_final_message()
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(friendly_error(exc, self.model)) from exc

        result.finish_reason = final.stop_reason

        # A safety decline arrives as a normal 200 with stop_reason "refusal".
        if final.stop_reason == "refusal":
            detail = getattr(final, "stop_details", None)
            category = getattr(detail, "category", None) or "unspecified"
            raise ProviderError(
                f"Claude declined this request (category: {category}). "
                "Rephrasing it, or switching provider, may help."
            )

        for block in final.content:
            if block.type == "tool_use":
                result.tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=json.dumps(block.input or {}),
                    )
                )

        # Keep the native blocks so thinking survives the round trip.
        result.native = {
            "provider": "anthropic",
            "content": [block.model_dump() for block in final.content],
        }
        return result
