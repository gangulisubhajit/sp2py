"""Unit tests for the non-UI modules."""

from __future__ import annotations

import pandas as pd
import pytest

from src import agent, code_validator, feedback_notes, history_store, providers, schema_utils, sql_utils
from src.providers import anthropic_api, openai_compat


# --------------------------------------------------------------------------
# code_validator
# --------------------------------------------------------------------------


def test_valid_code_passes():
    result = code_validator.validate_python_ast("def f():\n    '''doc'''\n    return 1\n")
    assert result.is_valid
    assert not result.warnings


def test_syntax_error_is_reported_with_location():
    result = code_validator.validate_python_ast("def f(:\n    pass\n")
    assert not result.is_valid
    assert "line 1" in result.error_message


def test_empty_code_is_invalid():
    assert not code_validator.validate_python_ast("   ").is_valid


def test_missing_docstrings_warns():
    result = code_validator.validate_python_ast("def f():\n    return 1\n")
    assert result.is_valid
    assert any("docstring" in w for w in result.warnings)


def test_long_lines_warn():
    result = code_validator.validate_python_ast("x = '" + "a" * 200 + "'\n")
    assert result.is_valid
    assert any("99 characters" in w for w in result.warnings)


def test_extract_prefers_python_tagged_block():
    """A stray untagged block before the module must not be mistaken for it."""
    response = (
        "First, the SQL I'm replacing:\n\n"
        "```\nSELECT * FROM orders;\n```\n\n"
        "And the module:\n\n"
        "```python\ndef f():\n    return 1\n```\n"
    )
    explanation, code = code_validator.extract_python_code(response)
    assert code == "def f():\n    return 1"
    assert "SQL I'm replacing" in explanation


def test_extract_falls_back_to_untagged_block():
    _, code = code_validator.extract_python_code("here:\n```\nx = 1\n```\n")
    assert code == "x = 1"


def test_extract_with_no_block_yields_empty_code():
    explanation, code = code_validator.extract_python_code("I can't do that.")
    assert code == ""
    assert explanation == "I can't do that."


# --------------------------------------------------------------------------
# schema_utils
# --------------------------------------------------------------------------


def test_blank_nullable_does_not_fabricate_not_null():
    """A blank cell means 'unknown' -- inventing NOT NULL misleads the model."""
    df = pd.DataFrame(
        [("t", "c", "INT", "", "", "")], columns=schema_utils.EXPECTED_COLUMNS
    )
    assert "NOT NULL" not in schema_utils.schema_to_prompt_text(df)


def test_explicit_no_yields_not_null():
    df = pd.DataFrame(
        [("t", "c", "INT", "N", "Y", "")], columns=schema_utils.EXPECTED_COLUMNS
    )
    text = schema_utils.schema_to_prompt_text(df)
    assert "NOT NULL" in text and "PRIMARY KEY" in text


def test_nullable_yes_is_not_marked_not_null():
    df = pd.DataFrame(
        [("t", "c", "INT", "Y", "N", "")], columns=schema_utils.EXPECTED_COLUMNS
    )
    assert "NOT NULL" not in schema_utils.schema_to_prompt_text(df)


def test_missing_columns_are_filled_not_fatal():
    raw = pd.DataFrame([{"Table Name": "t", "Column Name": "c"}])
    normalised = schema_utils._normalize_columns(raw)
    assert list(normalised.columns) == schema_utils.EXPECTED_COLUMNS


def test_empty_schema_renders_empty():
    assert schema_utils.schema_to_prompt_text(pd.DataFrame()) == ""
    assert schema_utils.schema_to_prompt_text(None) == ""


def test_dummy_schema_is_renderable():
    text = schema_utils.schema_to_prompt_text(schema_utils.dummy_schema())
    assert "TABLE customers" in text and "REFERENCES orders.order_id" in text


def test_live_db_connection_fails_loudly():
    with pytest.raises(NotImplementedError):
        schema_utils.connect_to_database("dsn=whatever")


# --------------------------------------------------------------------------
# sql_utils
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("sample_oracle.sql", sql_utils.ORACLE),
        ("sample_postgres.sql", sql_utils.POSTGRES),
        ("sample_tsql.sql", sql_utils.TSQL),
    ],
)
def test_dialect_detection_on_samples(filename, expected):
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "sample_procedures", filename), encoding="utf-8") as fh:
        assert sql_utils.guess_dialect(fh.read()) == expected


def test_comments_do_not_sway_detection():
    sql = "-- ported from T-SQL, uses @@ROWCOUNT there\nSELECT sysdate FROM dual;"
    assert sql_utils.guess_dialect(sql) == sql_utils.ORACLE


def test_unrecognised_sql_falls_back_to_default():
    assert sql_utils.guess_dialect("SELECT 1;", default=sql_utils.TSQL) == sql_utils.TSQL
    assert sql_utils.guess_dialect("") == sql_utils.ORACLE


def test_procedure_names_extracted():
    assert sql_utils.procedure_names("CREATE PROCEDURE dbo.[Foo] AS BEGIN END") == ["dbo.Foo"]


# --------------------------------------------------------------------------
# upload policy
# --------------------------------------------------------------------------


def test_upload_accepts_a_normal_sql_file():
    sql, reason = sql_utils.check_upload("proc.sql", 20, b"SELECT 1;")
    assert reason is None and sql == "SELECT 1;"


def test_upload_rejects_wrong_extension():
    _, reason = sql_utils.check_upload("proc.exe", 20, b"SELECT 1;")
    assert "supported" in reason


def test_upload_rejects_oversized_file():
    _, reason = sql_utils.check_upload("proc.sql", 6 * 1024 * 1024, b"SELECT 1;", max_mb=5)
    assert "limit" in reason


def test_upload_rejects_empty_file():
    _, reason = sql_utils.check_upload("proc.sql", 3, b"   \n")
    assert "empty" in reason


def test_upload_survives_non_utf8_bytes():
    """Legacy exports are often cp1252; a stray byte must not lose the file."""
    sql, reason = sql_utils.check_upload("proc.sql", 30, b"SELECT '\xa9 2024' FROM dual;")
    assert reason is None and "FROM dual" in sql


# --------------------------------------------------------------------------
# providers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,expected",
    [("gpt-4o-mini", True), ("gpt-4.1", True), ("o4-mini", False), ("gpt-5", False)],
)
def test_temperature_support_by_model(model, expected):
    """Reasoning models 400 if sent `temperature`."""
    assert openai_compat.supports_temperature(model) is expected


def test_missing_key_is_rejected_before_any_request():
    with pytest.raises(providers.ProviderError):
        providers.build("openai", "gpt-4o-mini", api_key="")


def test_unknown_provider_is_rejected():
    with pytest.raises(providers.ProviderError):
        providers.build("not-a-provider")


def test_every_advertised_provider_has_a_spec():
    for key in providers.PROVIDER_KEYS:
        spec = providers.spec_for(key)
        assert spec.default_model in spec.models
        assert spec.auth in {"api_key", "subscription"}


def test_groq_is_pinned_to_its_openai_compatible_endpoint():
    provider = providers.build("groq", api_key="gsk-test")
    assert str(provider.client.base_url).rstrip("/") == openai_compat.GROQ_BASE_URL


def test_groq_ignores_a_stray_base_url_override():
    """Groq has one endpoint; an OpenAI proxy URL must not leak into it."""
    provider = providers.build("groq", api_key="gsk-test", base_url="http://proxy.local/v1")
    assert str(provider.client.base_url).rstrip("/") == openai_compat.GROQ_BASE_URL


def test_friendly_error_for_auth_failure():
    import openai

    # The SDK vendors its own httpx (httpx2 as of openai 3.x), so borrow
    # whichever module it is actually built against rather than pinning one.
    httpx = getattr(openai._base_client, "httpx2", None) or openai._base_client.httpx

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(401, request=request, json={"error": {"message": "bad key"}})
    exc = openai.AuthenticationError("bad key", response=response, body=None)
    message = openai_compat.friendly_error(exc, "gpt-4o-mini", "OpenAI")
    assert "rejected" in message and "401" in message


def test_friendly_error_falls_back_for_unknown_exceptions():
    message = openai_compat.friendly_error(RuntimeError("something odd"), "gpt-4o-mini")
    assert "something odd" in message


def test_anthropic_auth_error_explains_the_subscription_trap():
    """The Pro/Max-vs-API-credits confusion deserves an explicit answer."""
    import anthropic

    httpx = getattr(anthropic._base_client, "httpx2", None) or anthropic._base_client.httpx
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(401, request=request, json={"error": {"message": "bad key"}})
    exc = anthropic.AuthenticationError("bad key", response=response, body=None)

    message = anthropic_api.friendly_error(exc, "claude-opus-5")
    assert "subscription" in message.lower()
    assert "console.anthropic.com" in message


# --------------------------------------------------------------------------
# Anthropic message translation
# --------------------------------------------------------------------------


def test_tool_schemas_convert_to_anthropic_shape():
    converted = anthropic_api.to_anthropic_tools(agent.TOOL_SCHEMAS)
    assert len(converted) == len(agent.TOOL_SCHEMAS)
    for tool in converted:
        assert set(tool) == {"name", "description", "input_schema"}
        assert tool["input_schema"]["type"] == "object"


def test_tool_calls_and_results_translate():
    canonical = [
        {"role": "user", "content": "convert it"},
        {
            "role": "assistant",
            "content": "Reading first.",
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "read_procedure", "arguments": '{"name":"a.sql"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"sql":"SELECT 1"}'},
    ]
    wire = anthropic_api.to_anthropic_messages(canonical)

    assert [m["role"] for m in wire] == ["user", "assistant", "user"]
    blocks = wire[1]["content"]
    assert blocks[0]["type"] == "text"
    assert blocks[1] == {
        "type": "tool_use", "id": "c1", "name": "read_procedure",
        "input": {"name": "a.sql"},
    }
    assert wire[2]["content"][0]["type"] == "tool_result"
    assert wire[2]["content"][0]["tool_use_id"] == "c1"


def test_parallel_tool_results_merge_into_one_user_turn():
    """Anthropic requires all tool_results for a turn in a single message."""
    canonical = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "get_schema", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "list_rules", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "a", "content": "{}"},
        {"role": "tool", "tool_call_id": "b", "content": "{}"},
    ]
    wire = anthropic_api.to_anthropic_messages(canonical)
    assert [m["role"] for m in wire] == ["user", "assistant", "user"]
    assert len(wire[2]["content"]) == 2


def test_native_content_is_replayed_verbatim():
    """Thinking blocks must survive the round trip unchanged."""
    blocks = [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": "Done."},
    ]
    canonical = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Done.",
         "_native": {"provider": "anthropic", "content": blocks}},
    ]
    wire = anthropic_api.to_anthropic_messages(canonical)
    assert wire[1]["content"] == blocks


def test_native_content_from_another_provider_is_ignored():
    """A transcript can outlive a provider switch without erroring."""
    canonical = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Done.",
         "_native": {"provider": "openai", "content": [{"type": "bogus"}]}},
    ]
    wire = anthropic_api.to_anthropic_messages(canonical)
    assert wire[1]["content"] == [{"type": "text", "text": "Done."}]


def test_malformed_tool_arguments_translate_to_empty_input():
    canonical = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "x", "type": "function",
             "function": {"name": "get_schema", "arguments": "{not json"}},
        ]},
    ]
    wire = anthropic_api.to_anthropic_messages(canonical)
    assert wire[1]["content"][0]["input"] == {}


# --------------------------------------------------------------------------
# stores
# --------------------------------------------------------------------------


def test_history_roundtrip_and_kpis():
    history_store.save_record(
        {"filename": "a.sql", "dialect": "Oracle PL/SQL", "duration_seconds": 2.0, "ast_valid": True}
    )
    record_id = history_store.save_record(
        {"filename": "b.sql", "dialect": "Microsoft T-SQL", "duration_seconds": 4.0, "ast_valid": False}
    )
    kpis = history_store.compute_kpis()
    assert kpis["total_processed"] == 2
    assert kpis["avg_duration_seconds"] == 3.0
    assert kpis["ast_success_rate"] == 50.0
    assert history_store.get_record(record_id)["filename"] == "b.sql"


def test_saving_same_id_updates_rather_than_duplicates():
    history_store.save_record({"id": "fixed", "filename": "a.sql", "ast_valid": True})
    history_store.save_record({"id": "fixed", "filename": "a-renamed.sql", "ast_valid": True})
    history = history_store.load_history()
    assert len(history) == 1 and history[0]["filename"] == "a-renamed.sql"


def test_kpis_on_empty_history():
    assert history_store.compute_kpis([])["total_processed"] == 0


def test_notes_scoping():
    feedback_notes.add_note("global rule")
    feedback_notes.add_note("oracle rule", dialect="Oracle PL/SQL")
    oracle = [n["note"] for n in feedback_notes.notes_for_dialect("Oracle PL/SQL")]
    tsql = [n["note"] for n in feedback_notes.notes_for_dialect("Microsoft T-SQL")]
    assert oracle == ["global rule", "oracle rule"]
    assert tsql == ["global rule"]


def test_note_deletion():
    record = feedback_notes.add_note("temporary")
    feedback_notes.delete_note(record["id"])
    assert feedback_notes.load_notes() == []


def test_corrupt_store_degrades_to_empty(tmp_path, monkeypatch):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(history_store, "HISTORY_PATH", str(path))
    assert history_store.load_history() == []
