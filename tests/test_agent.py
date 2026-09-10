"""Agent-loop tests, driven against the mock OpenAI server.

These exercise the real SDK streaming path: SSE parsing, tool-call
fragment reassembly, the AST rejection/retry cycle and error handling.
"""

from __future__ import annotations

import pytest

from src import agent, events, providers, schema_utils, sql_utils


@pytest.fixture
def session(oracle_sql):
    workspace = agent.Workspace(
        schema_text=schema_utils.schema_to_prompt_text(schema_utils.dummy_schema())
    )
    workspace.add_procedure(
        "sample_oracle.sql", oracle_sql, sql_utils.guess_dialect(oracle_sql)
    )
    return agent.AgentSession(workspace)


def drive(session, provider, message):
    """Run a turn, returning the emitted events grouped by type."""
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
# Workspace
# --------------------------------------------------------------------------


def test_single_procedure_resolves_without_a_name():
    ws = agent.Workspace()
    ws.add_procedure("only.sql", "SELECT 1", sql_utils.ORACLE)
    assert ws.resolve_procedure(None).name == "only.sql"


def test_ambiguous_lookup_without_a_name_returns_nothing():
    ws = agent.Workspace()
    ws.add_procedure("a.sql", "SELECT 1", sql_utils.ORACLE)
    ws.add_procedure("b.sql", "SELECT 2", sql_utils.ORACLE)
    assert ws.resolve_procedure(None) is None


def test_lookup_tolerates_case_and_extension():
    ws = agent.Workspace()
    ws.add_procedure("Process_Order.sql", "SELECT 1", sql_utils.ORACLE)
    assert ws.resolve_procedure("process_order").name == "Process_Order.sql"
    assert ws.resolve_procedure("process_order.py").name == "Process_Order.sql"


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


def test_write_python_rejects_unparseable_code(session):
    result = session._tool_write_python({"filename": "x.py", "code": "def f(:\n"})
    assert result["error"]
    assert "x.py" not in session.workspace.artifacts


def test_write_python_strips_stray_markdown_fences(session):
    result = session._tool_write_python(
        {"filename": "x.py", "code": "```python\ndef f():\n    '''d'''\n    return 1\n```"}
    )
    assert result["saved"]
    assert not session.workspace.artifacts["x.py"].code.startswith("```")


def test_write_python_appends_py_extension(session):
    session._tool_write_python({"filename": "orders", "code": "x = 1\n"})
    assert "orders.py" in session.workspace.artifacts


def test_repair_attempts_counted_separately_from_revisions(session):
    session._tool_write_python({"filename": "x.py", "code": "def f(:\n"})
    session._tool_write_python({"filename": "x.py", "code": "def f(:\n"})
    session._tool_write_python({"filename": "x.py", "code": "x = 1\n"})
    artifact = session.workspace.artifacts["x.py"]
    assert artifact.repair_attempts == 2
    assert artifact.revision == 1

    # A later user-requested rewrite bumps the revision, not the repairs.
    session._tool_write_python({"filename": "x.py", "code": "x = 2\n"})
    artifact = session.workspace.artifacts["x.py"]
    assert artifact.revision == 2 and artifact.repair_attempts == 0


def test_read_procedure_supports_line_ranges(session):
    result = session._tool_read_procedure({"start_line": 1, "end_line": 3})
    assert result["sql"].count("\n") == 2
    assert result["total_lines"] > 3


def test_read_procedure_rejects_out_of_range_start(session):
    assert session._tool_read_procedure({"start_line": 99999})["error"]


def test_read_unknown_procedure_lists_alternatives(session):
    result = session._tool_read_procedure({"name": "nope.sql"})
    assert result["error"] and "sample_oracle.sql" in result["available"]


def test_unknown_tool_is_reported_not_raised(session):
    assert session._dispatch("no_such_tool", {})["error"]


def test_tool_exceptions_are_caught(session, monkeypatch):
    def boom(_args):
        raise ValueError("kaboom")

    monkeypatch.setattr(session, "_tool_get_schema", boom)
    assert "kaboom" in session._dispatch("get_schema", {})["error"]


def test_malformed_tool_arguments_do_not_crash():
    assert agent._parse_arguments("{not json") == {"_raw": "{not json"}
    assert agent._parse_arguments("") == {}


# --------------------------------------------------------------------------
# Transcript integrity
# --------------------------------------------------------------------------


def test_dangling_tool_calls_are_backfilled(session):
    """An abandoned turn must not wedge the conversation with a 400."""
    session.messages = [
        {"role": "user", "content": "convert"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "read_procedure", "arguments": "{}"}}
            ],
        },
    ]
    healed = session._messages_for_api()
    tool_messages = [m for m in healed if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["call_1"]


def test_healing_is_idempotent(session):
    session.messages = [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "get_schema", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "{}"},
    ]
    before = len(session._messages_for_api())
    assert len(session._messages_for_api()) == before


# --------------------------------------------------------------------------
# End-to-end against the mock server
# --------------------------------------------------------------------------


def test_conversion_turn_reads_then_writes(session, mock_api):
    client = providers.build("openai", "mock-model", "sk-mock", mock_api)
    events = drive(session, client, "Convert this to Python")

    names = [name for name, _ in events["tools"]]
    assert names == ["read_procedure", "write_python", "write_python"]
    # The first write is deliberately broken and must be rejected.
    assert events["tools"][1] == ("write_python", False)
    assert events["tools"][2] == ("write_python", True)

    artifact = session.workspace.artifacts["process_customer_order.py"]
    assert artifact.is_valid and artifact.repair_attempts == 1
    assert events["final"]


def test_question_is_answered_without_generating_code(session, mock_api):
    client = providers.build("openai", "mock-model", "sk-mock", mock_api)
    events = drive(session, client, "What does this procedure do?")
    assert not session.workspace.artifacts
    assert "cursor" in events["final"]


def test_general_correction_saves_a_rule(session, mock_api):
    from src import feedback_notes

    client = providers.build("openai", "mock-model", "sk-mock", mock_api)
    drive(session, client, "Always roll back when stock is insufficient")
    notes = feedback_notes.load_notes()
    assert len(notes) == 1
    assert notes[0]["dialect"] == sql_utils.ORACLE


def test_saved_rules_reach_the_next_system_prompt(session, mock_api):
    from src import feedback_notes

    feedback_notes.add_note("Never use bare except.", dialect=None)
    assert "Never use bare except." in session.system_prompt()


def test_api_error_raises_a_friendly_conversion_error(session, mock_api):
    with pytest.raises(providers.ProviderError) as excinfo:
        list(session.run_turn(providers.build("openai", "trigger-500", "sk-mock", mock_api), "convert"))
    assert "500" in str(excinfo.value)


def test_conversation_recovers_after_an_api_error(session, mock_api):
    client = providers.build("openai", "mock-model", "sk-mock", mock_api)
    with pytest.raises(providers.ProviderError):
        list(session.run_turn(providers.build("openai", "trigger-500", "sk-mock", mock_api), "convert"))

    events = drive(session, client, "Convert this to Python")
    assert session.workspace.artifacts
    assert events["final"]


def test_streaming_yields_multiple_deltas(session, mock_api):
    """Text must arrive incrementally, not as one lump at the end."""
    client = providers.build("openai", "mock-model", "sk-mock", mock_api)
    deltas = [
        e.text
        for e in session.run_turn(client, "What does this procedure do?")
        if isinstance(e, events.TextDelta)
    ]
    assert len(deltas) > 3


def test_tool_iteration_cap_is_enforced(session, mock_api, monkeypatch):
    monkeypatch.setattr(agent, "MAX_TOOL_ITERATIONS", 2)
    client = providers.build("openai", "mock-model", "sk-mock", mock_api)
    events = drive(session, client, "Convert this to Python")
    assert "tool steps" in events["final"]


def test_empty_workspace_tells_the_model_to_ask_for_a_file():
    session = agent.AgentSession()
    result = session._tool_list_procedures({})
    assert result["procedures"] == []
    assert "hasn't uploaded" in result["hint"]
    assert "No stored procedure has been attached" in session.system_prompt()
