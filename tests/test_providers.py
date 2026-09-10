"""Cross-provider tests.

The mock server speaks both wire protocols, so the same scripted agent
trajectory is replayed through OpenAI and Anthropic and asserted to
behave identically. That is the property that matters: the UI and the
agent loop must not care which backend is selected.

The Claude Agent SDK provider talks to the real Claude Code CLI and your
subscription, so its live test is opt-in via SP2PY_LIVE_CLAUDE=1; its
wiring is covered offline with fakes.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from src import agent, events, providers, schema_utils, sql_utils
from src.providers import claude_code

CHAT_PROVIDERS = ["openai", "anthropic"]


@pytest.fixture
def session(oracle_sql):
    workspace = agent.Workspace(
        schema_text=schema_utils.schema_to_prompt_text(schema_utils.dummy_schema())
    )
    workspace.add_procedure(
        "sample_oracle.sql", oracle_sql, sql_utils.guess_dialect(oracle_sql)
    )
    return agent.AgentSession(workspace)


def make(provider_key, mock_api, model="mock-model"):
    # The OpenAI SDK wants a base URL that already includes /v1; the
    # Anthropic SDK appends /v1 itself, so hand it the bare root.
    base_url = mock_api.removesuffix("/v1") if provider_key == "anthropic" else mock_api
    return providers.build(provider_key, model, api_key="test-key", base_url=base_url)


def drive(session, provider, message):
    collected = {"text": "", "tools": [], "artifacts": [], "final": None}
    for event in session.run_turn(provider, message):
        if isinstance(event, events.TextDelta):
            collected["text"] += event.text
        elif isinstance(event, events.ToolFinished):
            collected["tools"].append((event.name, event.ok))
        elif isinstance(event, events.ArtifactUpdated):
            collected["artifacts"].append(event.artifact)
        elif isinstance(event, events.TurnFinished):
            collected["final"] = event.text
    return collected


# --------------------------------------------------------------------------
# Parity: identical behaviour across wire protocols
# --------------------------------------------------------------------------


@pytest.mark.parametrize("provider_key", CHAT_PROVIDERS)
def test_conversion_trajectory_is_identical(session, mock_api, provider_key):
    result = drive(session, make(provider_key, mock_api), "Convert this to Python")

    assert [name for name, _ in result["tools"]] == [
        "read_procedure", "write_python", "write_python"
    ]
    assert result["tools"][1][1] is False, "the broken draft must be rejected"
    assert result["tools"][2][1] is True

    artifact = session.workspace.artifacts["process_customer_order.py"]
    assert artifact.is_valid and artifact.repair_attempts == 1


@pytest.mark.parametrize("provider_key", CHAT_PROVIDERS)
def test_question_answered_without_code(session, mock_api, provider_key):
    result = drive(session, make(provider_key, mock_api), "What does this procedure do?")
    assert not session.workspace.artifacts
    assert "cursor" in result["final"]


@pytest.mark.parametrize("provider_key", CHAT_PROVIDERS)
def test_streaming_is_incremental(session, mock_api, provider_key):
    provider = make(provider_key, mock_api)
    deltas = [
        e.text
        for e in session.run_turn(provider, "What does this procedure do?")
        if isinstance(e, events.TextDelta)
    ]
    assert len(deltas) > 3, "text should arrive in fragments, not one lump"


@pytest.mark.parametrize("provider_key", CHAT_PROVIDERS)
def test_rule_saving_works_on_both(session, mock_api, provider_key):
    from src import feedback_notes

    drive(session, make(provider_key, mock_api), "Always roll back when stock is short")
    assert len(feedback_notes.load_notes()) == 1


@pytest.mark.parametrize("provider_key", CHAT_PROVIDERS)
def test_server_error_becomes_a_provider_error(session, mock_api, provider_key):
    provider = make(provider_key, mock_api, model="trigger-500")
    with pytest.raises(providers.ProviderError) as excinfo:
        list(session.run_turn(provider, "Convert this"))
    assert "500" in str(excinfo.value)


@pytest.mark.parametrize("provider_key", CHAT_PROVIDERS)
def test_conversation_recovers_after_an_error(session, mock_api, provider_key):
    with pytest.raises(providers.ProviderError):
        list(session.run_turn(make(provider_key, mock_api, "trigger-500"), "Convert"))

    drive(session, make(provider_key, mock_api), "Convert this to Python")
    assert session.workspace.artifacts


def test_a_transcript_survives_switching_provider(session, mock_api):
    """Anthropic-native blocks must not break a later OpenAI turn, or vice versa."""
    drive(session, make("anthropic", mock_api), "Convert this to Python")
    assert session.workspace.artifacts

    result = drive(session, make("openai", mock_api), "What does this procedure do?")
    assert result["final"]


# --------------------------------------------------------------------------
# Anthropic request shape (asserted against the wire)
# --------------------------------------------------------------------------


def echo_request(session, mock_api) -> str:
    """Send one turn to the mock's echo model; return its request summary."""
    provider = make("anthropic", mock_api, model="echo-request")
    with pytest.raises(providers.ProviderError) as excinfo:
        list(session.run_turn(provider, "Convert this"))

    message = str(excinfo.value)
    assert "ECHO " in message, f"no echo summary in: {message}"
    return message


def test_system_prompt_is_sent_top_level_not_as_a_message(session, mock_api):
    """Anthropic takes `system` as its own field, never a message role."""
    summary = echo_request(session, mock_api)

    assert "system_present=True" in summary
    assert "SP2PY" in summary
    assert "roles=user |" in summary, "the only message should be the user turn"
    assert "system" in summary.split("top_level_keys=")[1]


def test_tools_are_sent_in_anthropic_schema(session, mock_api):
    summary = echo_request(session, mock_api)

    assert "tool_fields=description,input_schema,name" in summary
    for schema in agent.TOOL_SCHEMAS:
        assert schema["function"]["name"] in summary


# --------------------------------------------------------------------------
# Live model listing (the fix for retired model ids)
# --------------------------------------------------------------------------


def test_openai_compatible_listing_filters_non_chat_models(mock_api):
    """Speech, embedding and classifier models must not reach the dropdown."""
    found = make("openai", mock_api).list_models()

    assert "mock-model" in found
    assert "llama-3.3-70b-versatile" in found
    for junk in (
        "whisper-large-v3",
        "text-embedding-3-small",
        "meta-llama/llama-prompt-guard-2-86m",
        "playai-tts",
    ):
        assert junk not in found, f"{junk} should have been filtered out"


def test_listing_drops_retired_models(mock_api):
    """Groq marks retired ids inactive; those are exactly the ones that 404."""
    found = make("openai", mock_api).list_models()
    assert "qwen/qwen3-32b" not in found


def test_listing_excludes_models_that_reject_custom_tools(mock_api, monkeypatch):
    """The agent is all tool calls, so a built-in-tools-only model is unusable.

    Groq's base URL is pinned to the real service, so the filter is
    exercised by lending Groq's exclusion rules to the mock-backed client.
    """
    provider = make("openai", mock_api)
    monkeypatch.setattr(
        provider, "spec", replace(provider.spec, no_custom_tools=("groq/compound",))
    )

    found = provider.list_models()
    assert "llama-3.3-70b-versatile" in found
    assert "groq/compound" not in found
    assert "groq/compound-mini" not in found


def test_listing_keeps_those_models_when_nothing_is_excluded(mock_api):
    """Sanity check that the previous test proves the filter, not the mock."""
    found = make("openai", mock_api).list_models()
    assert "groq/compound" in found


@pytest.mark.parametrize(
    "model,expected",
    [
        ("llama-3.3-70b-versatile", True),
        ("llama-3.1-8b-instant", True),
        ("openai/gpt-oss-120b", True),
        ("groq/compound", False),
        ("groq/compound-mini", False),
        ("GROQ/COMPOUND", False),  # matching must be case-insensitive
    ],
)
def test_custom_tool_capability_by_model(model, expected):
    assert providers.accepts_custom_tools(model, providers.SPECS["groq"]) is expected


def test_other_providers_block_nothing_by_default():
    for key in ("openai", "anthropic", "claude_code"):
        assert providers.SPECS[key].no_custom_tools == ()


def test_tool_support_400_names_working_models():
    """The raw Groq message is opaque; the app must say what to pick instead."""
    import openai

    httpx = getattr(openai._base_client, "httpx2", None) or openai._base_client.httpx
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {"error": {"message": "tool calling is not supported with this model"}}
    response = httpx.Response(400, request=request, json=body)
    exc = openai.BadRequestError(body["error"]["message"], response=response, body=body)

    from src.providers import openai_compat

    message = openai_compat.friendly_error(exc, "groq/compound", "Groq")
    assert "llama-3.3-70b-versatile" in message
    assert "compound" in message
    assert "400" not in message, "the raw status code isn't useful here"


def test_unrelated_400_is_passed_through_verbatim():
    import openai

    httpx = getattr(openai._base_client, "httpx2", None) or openai._base_client.httpx
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {"error": {"message": "max_tokens must be positive"}}
    response = httpx.Response(400, request=request, json=body)
    exc = openai.BadRequestError(body["error"]["message"], response=response, body=body)

    from src.providers import openai_compat

    message = openai_compat.friendly_error(exc, "llama-3.3-70b-versatile", "Groq")
    assert "max_tokens must be positive" in message
    assert "tool calling" not in message


def test_anthropic_listing_works(mock_api):
    found = make("anthropic", mock_api).list_models()
    assert "claude-opus-5" in found


def test_listing_failure_raises_provider_error(mock_api):
    provider = make("openai", mock_api)
    provider.client.api_key = ""  # force the 401 path
    with pytest.raises(providers.ProviderError):
        provider.list_models()


def test_bundled_groq_models_are_current_production_ids():
    """Guards against the stale-list bug: no known-retired ids in the fallback."""
    retired = {"moonshotai/kimi-k2-instruct", "qwen/qwen3-32b", "mixtral-8x7b-32768",
               "llama2-70b-4096", "gemma-7b-it"}
    assert not (set(providers.SPECS["groq"].models) & retired)


def test_model_not_found_error_points_at_the_refresh_button():
    import openai

    httpx = getattr(openai._base_client, "httpx2", None) or openai._base_client.httpx
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(404, request=request, json={"error": {"message": "no"}})
    exc = openai.NotFoundError("no", response=response, body=None)

    from src.providers import openai_compat

    message = openai_compat.friendly_error(exc, "qwen/qwen3-32b", "Groq")
    assert "Refresh models" in message
    assert "retires models" in message


def test_subscription_provider_falls_back_to_its_static_list():
    """The CLI picks the model; there is no endpoint to query."""
    available, _ = claude_code.is_available()
    if not available:
        pytest.skip("Claude Code CLI not available here")
    provider = providers.build("claude_code")
    assert provider.list_models() == providers.SPECS["claude_code"].models


# --------------------------------------------------------------------------
# Claude Agent SDK (subscription) -- wiring, offline
# --------------------------------------------------------------------------


def test_claude_code_reports_availability():
    available, reason = claude_code.is_available()
    assert isinstance(available, bool)
    if not available:
        assert reason, "an unavailable backend must explain why"


def test_claude_code_owns_the_loop():
    assert claude_code.ClaudeCodeProvider.owns_loop is True


def test_claude_code_missing_cli_is_a_friendly_error(monkeypatch):
    monkeypatch.setattr(claude_code.shutil, "which", lambda _: None)
    available, reason = claude_code.is_available()
    assert not available
    assert "Claude Code CLI" in reason
    with pytest.raises(providers.ProviderError):
        providers.build("claude_code")


def test_tool_server_exposes_every_agent_tool(session):
    """Each SP2PY tool must reach Claude Code under its mcp__ name."""
    pytest.importorskip("claude_agent_sdk")
    import queue

    _, names = claude_code._build_tool_server(session, queue.Queue())
    expected = {
        f"mcp__{claude_code.MCP_SERVER_NAME}__{s['function']['name']}"
        for s in agent.TOOL_SCHEMAS
    }
    assert set(names) == expected


def test_tool_handler_emits_events_and_flags_errors(session):
    """A rejected write must come back to Claude as an MCP tool error."""
    pytest.importorskip("claude_agent_sdk")
    import asyncio
    import json
    import queue

    bridge: queue.Queue = queue.Queue()
    handler = claude_code._make_handler(session, bridge, "write_python")

    response = asyncio.run(handler({"filename": "x.py", "code": "def f(:\n"}))
    assert response["isError"] is True
    assert "error" in json.loads(response["content"][0]["text"])

    emitted = []
    while not bridge.empty():
        emitted.append(bridge.get())
    assert isinstance(emitted[0], events.ToolStarted)
    assert isinstance(emitted[1], events.ToolFinished)
    assert emitted[1].ok is False


def test_tool_handler_emits_artifact_event_on_success(session):
    pytest.importorskip("claude_agent_sdk")
    import asyncio
    import queue

    bridge: queue.Queue = queue.Queue()
    handler = claude_code._make_handler(session, bridge, "write_python")

    response = asyncio.run(
        handler({"filename": "ok.py", "code": "def f():\n    '''d'''\n    return 1\n"})
    )
    assert response["isError"] is False

    emitted = []
    while not bridge.empty():
        emitted.append(bridge.get())
    assert any(isinstance(e, events.ArtifactUpdated) for e in emitted)
    assert "ok.py" in session.workspace.artifacts


# --------------------------------------------------------------------------
# Claude Agent SDK -- live, opt-in (uses your Claude subscription)
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    os.getenv("SP2PY_LIVE_CLAUDE") != "1",
    reason="set SP2PY_LIVE_CLAUDE=1 to run against the real Claude Code CLI",
)
def test_live_claude_code_conversion(session):
    provider = providers.build("claude_code", "claude-opus-5")
    result = drive(session, provider, "Convert the attached procedure to Python.")

    assert session.workspace.artifacts, "the agent should have written a module"
    artifact = next(iter(session.workspace.artifacts.values()))
    assert artifact.is_valid
    assert result["final"]
    assert provider.session_id, "a resumable session id should come back"
