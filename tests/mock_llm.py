"""Offline mock of the OpenAI and Anthropic streaming APIs, for testing SP2PY.

Serves both wire protocols from one scripted brain so the same agent
trajectory can be replayed through either provider:

    POST /v1/chat/completions   OpenAI-compatible SSE (also covers Groq)
    POST /v1/messages           Anthropic Messages API SSE

The scripted trajectory is: read_procedure -> write_python (deliberately
broken, to exercise the AST rejection path) -> write_python (valid) ->
prose answer. Asking a question instead answers without writing code;
saying "always ..." also saves a rule.

Run:  python mock_llm.py [port]
"""

import json
import sys
import time

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

BROKEN_CODE = "def process_order(session, customer_id:\n    return None\n"

GOOD_CODE = '''"""Service layer for the process_customer_order procedure."""

from dataclasses import dataclass


@dataclass
class OrderResult:
    """Outcome of creating an order.

    Attributes:
        order_id: Primary key of the newly created order.
        total_amount: Sum of all line items on the order.
    """

    order_id: int
    total_amount: float


def process_customer_order(session, customer_id: int) -> OrderResult:
    """Create an order for a customer from their staged items.

    Args:
        session: An open SQLAlchemy Session.
        customer_id: The customer placing the order.

    Returns:
        An OrderResult with the new order id and its total.

    Side effects:
        Inserts one row into `orders` and one per staged item into
        `order_items`, then commits. Rolls back on any failure.
    """
    # NOTE: mirrors the procedure's single commit at the end.
    with session.begin():
        return OrderResult(order_id=1, total_amount=0.0)
'''

EXPLANATION = (
    "This procedure inserts an order, iterates staged line items with a "
    "cursor, decrements stock, and commits once at the end. If stock is "
    "insufficient it raises an application error, which rolls the whole "
    "transaction back."
)

SUMMARY = (
    "Done — I wrote `process_customer_order.py`. It keeps the procedure's "
    "single commit at the end and rolls back on failure. Worth checking the "
    "stock-decrement ordering against your isolation level."
)


# --------------------------------------------------------------------------
# The scripted brain, shared by both protocols
# --------------------------------------------------------------------------


def decide(called, tool_errors, user_text):
    """Pick the next action from what has happened so far.

    Returns ("text", str) or ("tool", name, args, preamble).
    """
    lowered = (user_text or "").lower()

    if "read_procedure" not in called:
        return ("tool", "read_procedure", {}, "Let me read the procedure first.\n\n")

    if any(k in lowered for k in ("what does", "explain", "why did", "where are")):
        return ("text", EXPLANATION)

    if ("always" in lowered or "from now on" in lowered) and "save_rule" not in called:
        return (
            "tool",
            "save_rule",
            {
                "rule": "Always roll back the transaction when stock is insufficient.",
                "dialect": "Oracle PL/SQL",
            },
            "",
        )

    writes = [n for n in called if n == "write_python"]
    if not writes:
        # First write is deliberately broken: the app must reject it.
        return (
            "tool",
            "write_python",
            {
                "filename": "process_customer_order.py",
                "code": BROKEN_CODE,
                "summary": "First draft.",
                "source_procedure": "sample_oracle.sql",
            },
            "",
        )
    if len(writes) == 1 and tool_errors:
        return (
            "tool",
            "write_python",
            {
                "filename": "process_customer_order.py",
                "code": GOOD_CODE,
                "summary": "Service function preserving the single commit boundary.",
                "source_procedure": "sample_oracle.sql",
            },
            "",
        )

    return ("text", SUMMARY)


def _read_openai_history(messages):
    called, errors, user_text = [], False, ""
    for message in messages:
        for call in message.get("tool_calls") or []:
            called.append(call["function"]["name"])
        if message.get("role") == "tool":
            errors = errors or '"error"' in (message.get("content") or "")
        if message.get("role") == "user":
            user_text = message.get("content") or ""
    return called, errors, user_text


def _read_anthropic_history(messages):
    called, errors, user_text = [], False, ""
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            if message.get("role") == "user":
                user_text = content
            continue
        for block in content or []:
            btype = block.get("type")
            if btype == "tool_use":
                called.append(block.get("name"))
            elif btype == "tool_result":
                errors = errors or '"error"' in str(block.get("content") or "")
            elif btype == "text" and message.get("role") == "user":
                user_text = block.get("text") or ""
    return called, errors, user_text


# --------------------------------------------------------------------------
# OpenAI wire format
# --------------------------------------------------------------------------


def sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def frame(delta: dict, finish=None) -> dict:
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "mock-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def openai_text(text: str):
    yield sse(frame({"role": "assistant"}))
    for i in range(0, len(text), 24):
        yield sse(frame({"content": text[i : i + 24]}))
    yield sse(frame({}, "stop"))
    yield b"data: [DONE]\n\n"


def openai_tool(name: str, arguments: dict, preamble: str = ""):
    yield sse(frame({"role": "assistant"}))
    for i in range(0, len(preamble), 24):
        yield sse(frame({"content": preamble[i : i + 24]}))
    yield sse(
        frame(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": f"call_{name}",
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }
                ]
            }
        )
    )
    blob = json.dumps(arguments)
    for i in range(0, len(blob), 40):
        yield sse(
            frame({"tool_calls": [{"index": 0, "function": {"arguments": blob[i : i + 40]}}]})
        )
    yield sse(frame({}, "tool_calls"))
    yield b"data: [DONE]\n\n"


async def chat_completions(request: Request):
    body = await request.json()
    if not request.headers.get("authorization", "").startswith("Bearer "):
        return JSONResponse({"error": {"message": "missing key"}}, status_code=401)
    if body.get("model") == "trigger-500":
        return JSONResponse({"error": {"message": "boom"}}, status_code=500)

    action = decide(*_read_openai_history(body.get("messages", [])))
    if action[0] == "text":
        return StreamingResponse(openai_text(action[1]), media_type="text/event-stream")
    _, name, args, preamble = action
    return StreamingResponse(
        openai_tool(name, args, preamble), media_type="text/event-stream"
    )


# --------------------------------------------------------------------------
# Anthropic wire format
# --------------------------------------------------------------------------


def anthropic_sse(event: str, payload: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()


def _anthropic_open(model: str):
    yield anthropic_sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_mock",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        },
    )


def _anthropic_close(stop_reason: str):
    yield anthropic_sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 42},
        },
    )
    yield anthropic_sse("message_stop", {"type": "message_stop"})


def _anthropic_text_block(index: int, text: str):
    yield anthropic_sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "text", "text": ""},
        },
    )
    for i in range(0, len(text), 24):
        yield anthropic_sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": text[i : i + 24]},
            },
        )
    yield anthropic_sse(
        "content_block_stop", {"type": "content_block_stop", "index": index}
    )


def anthropic_text(model: str, text: str):
    yield from _anthropic_open(model)
    yield from _anthropic_text_block(0, text)
    yield from _anthropic_close("end_turn")


def anthropic_tool(model: str, name: str, arguments: dict, preamble: str = ""):
    yield from _anthropic_open(model)

    index = 0
    if preamble:
        yield from _anthropic_text_block(0, preamble)
        index = 1

    yield anthropic_sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "tool_use",
                "id": f"toolu_{name}",
                "name": name,
                "input": {},
            },
        },
    )
    blob = json.dumps(arguments)
    for i in range(0, len(blob), 40):
        yield anthropic_sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": blob[i : i + 40]},
            },
        )
    yield anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": index})
    yield from _anthropic_close("tool_use")


async def anthropic_messages(request: Request):
    body = await request.json()
    if not request.headers.get("x-api-key"):
        return JSONResponse(
            {"type": "error", "error": {"type": "authentication_error", "message": "missing key"}},
            status_code=401,
        )
    model = body.get("model", "mock-claude")
    if model == "trigger-500":
        return JSONResponse(
            {"type": "error", "error": {"type": "api_error", "message": "boom"}},
            status_code=500,
        )

    # Echo a compact summary of the request so tests can assert on the
    # translation. Kept as a flat delimited string rather than nested JSON,
    # because the SDK renders error bodies with Python repr and re-parsing
    # JSON out of that is needlessly fragile.
    if model == "echo-request":
        fields = [
            f"system_present={bool(body.get('system'))}",
            f"system_head={(body.get('system') or '')[:80]!r}",
            "roles=" + ",".join(m.get("role", "?") for m in body.get("messages", [])),
            "tool_fields=" + ",".join(sorted({k for t in body.get("tools", []) for k in t})),
            "tool_names=" + ",".join(t.get("name", "?") for t in body.get("tools", [])),
            "top_level_keys=" + ",".join(sorted(body)),
        ]
        return JSONResponse(
            {"type": "error", "error": {
                "type": "invalid_request_error", "message": "ECHO " + " | ".join(fields)}},
            status_code=400,
        )

    action = decide(*_read_anthropic_history(body.get("messages", [])))
    if action[0] == "text":
        return StreamingResponse(
            anthropic_text(model, action[1]), media_type="text/event-stream"
        )
    _, name, args, preamble = action
    return StreamingResponse(
        anthropic_tool(model, name, args, preamble), media_type="text/event-stream"
    )


# --------------------------------------------------------------------------
# Model listings
# --------------------------------------------------------------------------

# Deliberately mixed: chat models, a retired one, and non-chat models the
# app must filter out (speech, embeddings, safety classifiers).
MOCK_MODELS = [
    {"id": "mock-model", "active": True},
    {"id": "llama-3.3-70b-versatile", "active": True},
    {"id": "openai/gpt-oss-120b", "active": True},
    {"id": "qwen/qwen3-32b", "active": False},          # retired
    {"id": "whisper-large-v3", "active": True},         # speech
    {"id": "text-embedding-3-small", "active": True},   # embeddings
    {"id": "meta-llama/llama-prompt-guard-2-86m", "active": True},  # classifier
    {"id": "playai-tts", "active": True},               # speech
    # Chat models that reject a custom `tools` array (built-in tools only).
    {"id": "groq/compound", "active": True},
    {"id": "groq/compound-mini", "active": True},
]


async def list_models(request: Request):
    """Both SDKs GET /v1/models, so branch on which auth header arrived."""
    if request.headers.get("x-api-key"):
        return JSONResponse(
            {
                "data": [
                    {"type": "model", "id": "claude-opus-5", "display_name": "Claude Opus 5"},
                    {"type": "model", "id": "claude-sonnet-5", "display_name": "Claude Sonnet 5"},
                ],
                "has_more": False,
                "first_id": "claude-opus-5",
                "last_id": "claude-sonnet-5",
            }
        )

    if not request.headers.get("authorization", "").startswith("Bearer "):
        return JSONResponse({"error": {"message": "missing key"}}, status_code=401)

    return JSONResponse(
        {
            "object": "list",
            "data": [
                {"object": "model", "created": 1, "owned_by": "mock", **m}
                for m in MOCK_MODELS
            ],
        }
    )


app = Starlette(
    routes=[
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/v1/messages", anthropic_messages, methods=["POST"]),
        Route("/v1/models", list_models, methods=["GET"]),
    ]
)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8799
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
