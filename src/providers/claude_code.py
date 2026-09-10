"""
Claude via the Claude Agent SDK -- authenticated by your Claude subscription.

This is the one backend that needs no API key: the SDK drives the local
Claude Code CLI, which is already logged in to your claude.ai account, so
usage counts against your Pro/Max limits instead of API credits.

It is also the one backend that owns the agent loop. Rather than
returning one assistant turn for `src/agent.py` to drive, it runs Claude
Code's own harness and we expose the SP2PY tools to it as an in-process
MCP server. The same events come out either way, so the UI can't tell.

Two consequences worth knowing:

* The Claude Code CLI must be installed and logged in wherever the app
  runs, which effectively makes this a local, single-user option.
* Conversation state lives in the CLI's session, not in our transcript,
  so each turn resumes the previous `session_id`. That keeps context
  across Streamlit reruns without holding a connection open.
"""

from __future__ import annotations

import importlib.util
import json
import queue
import shutil
import tempfile
import threading
from typing import Any, Iterator

from ..events import ArtifactUpdated, TextDelta, ToolFinished, ToolStarted, TurnFinished
from .base import Provider, ProviderError, ProviderSpec

CLAUDE_CODE_SPEC = ProviderSpec(
    key="claude_code",
    label="Claude (subscription)",
    blurb=(
        "Runs through the Claude Code CLI signed in to your claude.ai "
        "account — no API key, no separate bill. Local use only."
    ),
    auth="subscription",
    models=["claude-opus-5", "claude-sonnet-5", "opus", "sonnet", "haiku"],
    default_model="claude-opus-5",
    credentials_url="https://claude.com/product/claude-code",
    key_label="",
    key_env="",
    supports_base_url=False,
)

MCP_SERVER_NAME = "sp2py"
MAX_TURNS = 24
# Sentinel pushed onto the bridge queue when the worker thread is done.
_DONE = object()


def is_available() -> tuple[bool, str]:
    """Whether this backend can run here, and why not if it can't."""
    if importlib.util.find_spec("claude_agent_sdk") is None:
        return False, (
            "The `claude-agent-sdk` package isn't installed. "
            "Run `pip install claude-agent-sdk`."
        )
    if shutil.which("claude") is None:
        return False, (
            "The Claude Code CLI isn't on your PATH. Install it from "
            "claude.com/product/claude-code and run `claude` once to sign in."
        )
    return True, ""


class ClaudeCodeProvider(Provider):
    """Loop-owning provider backed by the Claude Agent SDK."""

    owns_loop = True

    def __init__(self, spec: ProviderSpec, model: str):
        super().__init__(spec, model)
        ok, reason = is_available()
        if not ok:
            raise ProviderError(reason)
        # Resumed each turn so context survives Streamlit reruns.
        self.session_id: str | None = None

    # -- required by Provider, unused: this backend owns the loop --------

    def iter_turn(self, system, messages, tools):  # pragma: no cover
        raise NotImplementedError(
            "ClaudeCodeProvider owns the agent loop; call run_agent_turn()."
        )

    # -- the turn --------------------------------------------------------

    def run_agent_turn(self, host, user_message: str) -> Iterator[Any]:
        """Run one turn through Claude Code, yielding agent events.

        `host` is the AgentSession: it supplies the system prompt, the
        tool schemas and the dispatcher. Events are produced on a worker
        thread (the SDK is async) and handed to this generator through a
        queue, so the caller stays plain synchronous Streamlit code.
        """
        bridge: queue.Queue = queue.Queue()
        state: dict[str, Any] = {"error": None, "session_id": None, "cost": None}

        worker = threading.Thread(
            target=_run_async_turn,
            args=(self, host, user_message, bridge, state),
            daemon=True,
        )
        worker.start()

        segments: list[str] = []
        while True:
            event = bridge.get()
            if event is _DONE:
                break
            if isinstance(event, TextDelta):
                segments.append(event.text)
            yield event

        worker.join(timeout=10)

        if state["error"]:
            raise ProviderError(state["error"])

        if state["session_id"]:
            self.session_id = state["session_id"]

        yield TurnFinished("".join(segments).strip(), cost_usd=state["cost"])


def _run_async_turn(provider, host, user_message, bridge, state) -> None:
    """Thread entry point: run the async SDK turn and feed the queue."""
    import asyncio

    try:
        asyncio.run(_drive(provider, host, user_message, bridge, state))
    except Exception as exc:
        state["error"] = _friendly_error(exc)
    finally:
        bridge.put(_DONE)


async def _drive(provider, host, user_message, bridge, state) -> None:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        ResultMessage,
        StreamEvent,
        TextBlock,
    )

    server, tool_names = _build_tool_server(host, bridge)

    options = ClaudeAgentOptions(
        system_prompt=host.system_prompt(),
        mcp_servers={MCP_SERVER_NAME: server},
        allowed_tools=tool_names,
        # Deny anything not pre-approved: this agent gets the SP2PY tools
        # and nothing else -- no Bash, no Read/Write against the user's disk.
        permission_mode="dontAsk",
        model=provider.model,
        max_turns=MAX_TURNS,
        # Neutral working directory and no user settings, so a stray
        # file tool can't reach into the project.
        cwd=tempfile.gettempdir(),
        setting_sources=[],
        include_partial_messages=True,
        resume=provider.session_id,
    )

    # Text arrives twice: as partial deltas and again in the completed
    # AssistantMessage. Prefer the deltas and skip blocks already streamed.
    streamed_any = False

    async with ClaudeSDKClient(options=options) as client:
        await client.query(user_message)
        async for message in client.receive_response():
            if isinstance(message, StreamEvent):
                event = message.event or {}
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta") or {}
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        streamed_any = True
                        bridge.put(TextDelta(delta["text"]))
                continue

            if isinstance(message, AssistantMessage):
                if streamed_any:
                    continue
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text:
                        bridge.put(TextDelta(block.text))

            elif isinstance(message, ResultMessage):
                state["session_id"] = message.session_id
                state["cost"] = message.total_cost_usd
                if message.is_error:
                    state["error"] = (
                        message.result
                        or f"Claude Code ended with an error ({message.stop_reason})."
                    )


def _build_tool_server(host, bridge):
    """Expose the agent's tools to Claude Code as an in-process MCP server.

    Each wrapper emits the same ToolStarted / ToolFinished / ArtifactUpdated
    events the built-in loop does, so the UI's tool trace works identically.
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool

    wrapped = []
    names = []

    for schema in host.tool_schemas():
        fn = schema["function"]
        name = fn["name"]
        parameters = fn.get("parameters") or {"type": "object", "properties": {}}

        wrapped.append(
            tool(name, fn.get("description", ""), parameters)(
                _make_handler(host, bridge, name)
            )
        )
        names.append(f"mcp__{MCP_SERVER_NAME}__{name}")

    server = create_sdk_mcp_server(name=MCP_SERVER_NAME, version="1.0.0", tools=wrapped)
    return server, names


def _make_handler(host, bridge, name):
    """Build one async MCP handler bound to `name`."""

    async def handler(args: dict) -> dict:
        arguments = dict(args or {})
        bridge.put(ToolStarted(name, arguments))

        payload = host.dispatch(name, arguments)
        ok = not payload.get("error")

        artifact_name = payload.pop("_artifact", None)
        bridge.put(ToolFinished(name, arguments, payload, ok))
        if artifact_name:
            bridge.put(ArtifactUpdated(host.workspace.artifacts[artifact_name]))

        return {
            "content": [
                {"type": "text", "text": json.dumps(payload, default=str)[:20_000]}
            ],
            # Surfaces the rejection to Claude as a tool error, which is
            # what makes it retry a module that failed AST validation.
            "isError": not ok,
        }

    handler.__name__ = f"sp2py_{name}"
    return handler


def _friendly_error(exc: Exception) -> str:
    from claude_agent_sdk import CLIConnectionError, CLINotFoundError, ProcessError

    if isinstance(exc, CLINotFoundError):
        return (
            "Couldn't find the Claude Code CLI. Install it from "
            "claude.com/product/claude-code, then run `claude` once to sign in."
        )
    if isinstance(exc, CLIConnectionError):
        return (
            "Lost the connection to the Claude Code CLI. Check that `claude` "
            "runs in your terminal and that you're signed in."
        )
    if isinstance(exc, ProcessError):
        return (
            f"The Claude Code CLI exited unexpectedly: {exc}. If you're signed "
            "out, run `claude` in a terminal to log in again."
        )
    return f"The Claude Code run failed: {exc}"
