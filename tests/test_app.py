"""UI tests driving the real Streamlit app through Streamlit's AppTest.

Model traffic goes to the mock OpenAI server, so these run offline while
still exercising the genuine request/stream/tool-call path.
"""

from __future__ import annotations

import os

import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app.py")

CHAT, HISTORY, ANALYTICS = "💬 Chat", "📜 History", "📊 Analytics"


def assert_clean(at, label=""):
    assert not at.exception, f"{label}: {[str(e.value) for e in at.exception]}"


def configure(at, mock_api, provider="openai", model="mock-model"):
    """Point the app at the mock server for the given provider."""
    at.session_state["provider"] = provider
    at.session_state[f"key_{provider}"] = "sk-mock"
    at.session_state[f"model_{provider}"] = model
    at.session_state["base_url"] = mock_api
    at.run()
    return at


@pytest.fixture
def app(mock_api):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return configure(at, mock_api)


def load_sample(at):
    [b for b in at.sidebar.button if "Load sample" in b.label][0].click().run()
    return at


def send(at, text):
    """Queue a chat turn the way the composer does, then let the app run it."""
    at.session_state["pending"] = {"text": text, "files": []}
    at.run()
    return at


# --------------------------------------------------------------------------
# Boot & guard rails
# --------------------------------------------------------------------------


def test_app_boots_without_a_key():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert_clean(at, "boot")
    assert at.chat_input[0].disabled, "composer must be disabled without a key"


def test_composer_enabled_once_a_key_is_set(app):
    assert not app.chat_input[0].disabled


def test_empty_workspace_shows_the_hero(app):
    assert any("Attach a stored procedure" in m.value for m in app.markdown)


# --------------------------------------------------------------------------
# Attaching procedures
# --------------------------------------------------------------------------


def test_loading_a_sample_detects_its_dialect(app):
    load_sample(app)
    assert_clean(app, "load sample")
    procedures = app.session_state["session"].workspace.procedures
    assert len(procedures) == 1
    assert next(iter(procedures.values())).dialect in (
        "Oracle PL/SQL", "PostgreSQL PL/pgSQL", "Microsoft T-SQL"
    )


def test_attached_file_can_be_removed(app):
    load_sample(app)
    # The remove control is icon-only, so address it by key rather than label.
    remove = [b for b in app.sidebar.button if (b.key or "").startswith("rm_")]
    assert len(remove) == 1
    remove[0].click().run()
    assert_clean(app, "remove file")
    assert not app.session_state["session"].workspace.procedures


# --------------------------------------------------------------------------
# Conversation
# --------------------------------------------------------------------------


def test_conversion_turn_produces_a_validated_artifact(app):
    load_sample(app)
    send(app, "Convert this to Python")
    assert_clean(app, "conversion turn")

    artifacts = app.session_state["session"].workspace.artifacts
    assert len(artifacts) == 1
    artifact = next(iter(artifacts.values()))
    assert artifact.is_valid
    assert artifact.repair_attempts == 1, "the broken first draft should be rejected"

    record = app.session_state["transcript"][-1]
    assert record["role"] == "assistant" and record["error"] is None
    assert [t["name"] for t in record["tools"]] == [
        "read_procedure", "write_python", "write_python"
    ]


def test_artifact_panel_appears_with_a_download(app):
    load_sample(app)
    send(app, "Convert this to Python")
    assert any("Download" in b.label for b in app.download_button)


def test_conversion_is_written_to_history(app):
    from src import history_store

    load_sample(app)
    send(app, "Convert this to Python")
    history = history_store.load_history()
    assert len(history) == 1
    assert history[0]["ast_valid"] is True
    assert history[0]["self_repaired"] is True


def test_new_chat_clears_the_thread_but_keeps_files(app):
    load_sample(app)
    send(app, "Convert this to Python")
    assert app.session_state["transcript"]

    [b for b in app.sidebar.button if "New chat" in b.label][0].click().run()
    assert_clean(app, "new chat")
    assert app.session_state["transcript"] == []
    assert app.session_state["session"].workspace.procedures
    assert app.session_state["session"].messages == []


def test_history_ids_are_namespaced_per_session(app, mock_api):
    """A new session's revision 1 must not overwrite an older record."""
    from src import history_store

    load_sample(app)
    send(app, "Convert this to Python")
    first = history_store.load_history()
    assert len(first) == 1

    second = AppTest.from_file(APP, default_timeout=180)
    second.run()
    configure(second, mock_api)
    load_sample(second)
    send(second, "Convert this to Python")

    assert len(history_store.load_history()) == 2, "second run overwrote the first"


def test_a_question_does_not_generate_code(app):
    load_sample(app)
    send(app, "What does this procedure do?")
    assert_clean(app, "question turn")
    assert not app.session_state["session"].workspace.artifacts


def test_general_correction_is_saved_and_listed_in_the_sidebar(app):
    from src import feedback_notes

    load_sample(app)
    send(app, "Always roll back when stock is insufficient")
    assert_clean(app, "rule turn")
    assert len(feedback_notes.load_notes()) == 1

    app.run()
    assert_clean(app, "sidebar with rules")
    assert any("Learned rules (1)" in e.label for e in app.sidebar.expander)


# --------------------------------------------------------------------------
# Model list refresh
# --------------------------------------------------------------------------


def test_refresh_models_loads_the_live_list(app):
    refresh = [b for b in app.sidebar.button if "Refresh models" in b.label]
    assert refresh, "the refresh control should be present"
    assert not refresh[0].disabled, "it should be enabled once a key is set"

    refresh[0].click().run()
    assert_clean(app, "refresh models")

    live = app.session_state["live_models_openai"]
    assert "mock-model" in live
    assert "whisper-large-v3" not in live, "non-chat models must be filtered"


def test_refresh_replaces_a_retired_model_selection(app, mock_api):
    """A model that no longer exists should be swapped for one that does."""
    app.session_state["model_openai"] = "qwen/qwen3-32b"  # inactive in the mock
    app.run()

    [b for b in app.sidebar.button if "Refresh models" in b.label][0].click().run()
    assert_clean(app, "refresh after retirement")

    chosen = app.session_state["model_openai"]
    assert chosen != "qwen/qwen3-32b"
    assert chosen in app.session_state["live_models_openai"]


def test_refresh_is_disabled_without_a_key():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    refresh = [b for b in at.sidebar.button if "Refresh models" in b.label]
    assert refresh and refresh[0].disabled


# --------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------


def test_api_failure_is_shown_not_raised(app):
    load_sample(app)
    app.session_state["model_openai"] = "trigger-500"
    send(app, "Convert this")
    assert_clean(app, "API failure")  # no traceback reaches the user
    assert app.session_state["transcript"][-1]["error"]
    assert app.error, "the failure should be rendered as an st.error"


def test_conversation_continues_after_a_failure(app):
    load_sample(app)
    app.session_state["model_openai"] = "trigger-500"
    send(app, "Convert this")
    app.session_state["model_openai"] = "mock-model"
    send(app, "Convert this to Python")
    assert_clean(app, "recovery")
    assert app.session_state["session"].workspace.artifacts


def test_sending_without_a_key_prompts_for_one(mock_api):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.session_state["pending"] = {"text": "Convert this", "files": []}
    at.run()
    assert_clean(at, "no key")
    assert "API key" in at.session_state["transcript"][-1]["error"]


# --------------------------------------------------------------------------
# Schema panel
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expect_text",
    [
        ("Bundled dummy schema", True),
        ("Upload Excel (.xlsx)", False),   # nothing uploaded yet
        ("Paste DDL / notes", True),
    ],
)
def test_schema_sources_resolve(app, source, expect_text):
    app.session_state["schema_source"] = source
    app.session_state["pasted_schema"] = "CREATE TABLE t (a INT);"
    app.run()
    assert_clean(app, f"schema source {source}")
    text = app.session_state["session"].workspace.schema_text
    assert bool(text) is expect_text


def test_switching_schema_source_does_not_leave_a_stale_schema(app):
    app.run()
    assert app.session_state["session"].workspace.schema_text  # dummy schema
    app.session_state["schema_source"] = "Upload Excel (.xlsx)"
    app.run()
    assert app.session_state["session"].workspace.schema_text == ""


# --------------------------------------------------------------------------
# Other pages
# --------------------------------------------------------------------------


def test_history_page_renders_kpis(app):
    load_sample(app)
    send(app, "Convert this to Python")
    app.sidebar.radio[0].set_value(HISTORY).run()
    assert_clean(app, "history page")
    labels = [m.label for m in app.metric]
    assert "Scripts processed" in labels


def test_history_reload_returns_to_chat_with_the_record(app):
    load_sample(app)
    send(app, "Convert this to Python")
    app.sidebar.radio[0].set_value(HISTORY).run()
    [b for b in app.button if "Load into chat" in b.label][0].click().run()
    assert_clean(app, "history reload")

    assert app.session_state["page"] == CHAT
    workspace = app.session_state["session"].workspace
    assert workspace.procedures and workspace.artifacts


def test_analytics_page_renders(app):
    load_sample(app)
    send(app, "Convert this to Python")
    app.sidebar.radio[0].set_value(ANALYTICS).run()
    assert_clean(app, "analytics page")


def test_analytics_empty_state():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(ANALYTICS).run()
    assert_clean(at, "analytics empty")
    assert at.info


def test_history_filter_with_no_matches_is_handled(app):
    load_sample(app)
    send(app, "Convert this to Python")
    app.sidebar.radio[0].set_value(HISTORY).run()
    app.text_input[0].set_value("no-such-file-xyz").run()
    assert_clean(app, "empty filter result")
