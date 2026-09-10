"""
SP2PY -- AI-Powered Stored Procedure to Python Converter
=========================================================

A chat-first workspace: attach your stored procedures straight into the
chat box, talk to the agent about them, and watch the generated Python
appear (AST-validated) in the artifact panel beside the conversation.

Run with:
    streamlit run app.py

See README.md for full setup instructions.
"""

from __future__ import annotations

import os
import time
import uuid

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src import (
    agent,
    code_validator,
    events,
    feedback_notes,
    history_store,
    providers,
    schema_utils,
    sql_utils,
    ui,
)

load_dotenv()

# --------------------------------------------------------------------------
# Page config & constants
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="SP2PY - Stored Procedure to Python",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DIALECTS = sql_utils.DIALECTS
TARGET_FRAMEWORKS = ["FastAPI + SQLAlchemy", "Flask + PyODBC"]
ALLOWED_EXTENSIONS = ["sql", "prc", "txt"]
MAX_FILE_MB = 5

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(APP_DIR, "sample_procedures")
SAMPLE_SCHEMA_PATH = os.path.join(APP_DIR, "sample_schema", "sample_tables.xlsx")

SCHEMA_SOURCES = ["Bundled dummy schema", "Upload Excel (.xlsx)", "Paste DDL / notes"]

CHAT, HISTORY, ANALYTICS = "💬 Chat", "📜 History", "📊 Analytics"
PAGES = [CHAT, HISTORY, ANALYTICS]

STARTERS = [
    "Convert this to Python",
    "What does this procedure do?",
    "Where are the transaction boundaries?",
    "Add unit tests for the generated code",
]


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------


# Credentials and model choices live under one key per provider, so
# switching provider doesn't lose what you already typed. The API-key
# entries double as the text_input widget keys -- Streamlit forbids
# writing to a widget-keyed entry once the widget exists, so the widget
# owns the value and everything else reads it through these helpers.

def key_state(provider_key: str) -> str:
    return f"key_{provider_key}"


def model_state(provider_key: str) -> str:
    return f"model_{provider_key}"


def api_key_for(provider_key: str) -> str:
    return st.session_state.get(key_state(provider_key), "")


def model_for(provider_key: str) -> str:
    return st.session_state.get(
        model_state(provider_key), providers.SPECS[provider_key].default_model
    )


def _key_from_env(provider_key: str) -> str:
    """Preload a provider's API key from its conventional env var, if set."""
    env_name = providers.SPECS[provider_key].key_env
    return os.getenv(env_name, "") if env_name else ""


def _model_from_env(provider_key: str) -> str:
    """Starting model for a provider: OPENAI_MODEL is honoured for OpenAI."""
    spec = providers.SPECS[provider_key]
    if provider_key == "openai":
        return os.getenv("OPENAI_MODEL") or spec.default_model
    return spec.default_model


def init_state() -> None:
    for provider_key in providers.PROVIDER_KEYS:
        st.session_state.setdefault(key_state(provider_key), _key_from_env(provider_key))
        st.session_state.setdefault(model_state(provider_key), _model_from_env(provider_key))

    defaults = {
        "page": CHAT,
        "nav_radio": CHAT,      # the navigation widget's own state
        "nav_request": None,    # a page another view wants us to switch to
        "session": agent.AgentSession(),
        "transcript": [],       # display records; see _record_* helpers
        "pending": None,        # {"text": str, "files": [(name, sql)]}
        "provider": os.getenv("SP2PY_PROVIDER", providers.PROVIDER_KEYS[0]),
        "base_url": os.getenv("OPENAI_BASE_URL", ""),
        "schema_source": SCHEMA_SOURCES[0],
        "schema_df": None,
        "pasted_schema": "",
        "target_framework": TARGET_FRAMEWORKS[0],
        "turn_error": None,
        "last_turn_seconds": 0.0,
        # Namespaces history ids so a fresh session's "revision 1" doesn't
        # overwrite a previous session's record for the same filename.
        "run_id": uuid.uuid4().hex[:6],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def workspace() -> agent.Workspace:
    return st.session_state.session.workspace


def reset_conversation() -> None:
    """Start a fresh chat but keep the attached files and settings."""
    old = workspace()
    fresh = agent.AgentSession(
        agent.Workspace(
            procedures=dict(old.procedures),
            schema_text=old.schema_text,
            target_framework=old.target_framework,
        )
    )
    st.session_state.session = fresh
    st.session_state.transcript = []
    st.session_state.pending = None
    st.session_state.turn_error = None


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------


def render_sidebar() -> None:
    # Honour a navigation request from another view. Streamlit forbids
    # writing to a widget-keyed entry once that widget exists, so this has
    # to happen before the radio below is instantiated.
    requested = st.session_state.pop("nav_request", None)
    if requested in PAGES:
        st.session_state.nav_radio = requested

    with st.sidebar:
        ui.header("SP2PY", "Stored procedure → Python")

        st.session_state.page = st.radio(
            "Workspace",
            PAGES,
            key="nav_radio",
            label_visibility="collapsed",
        )

        if st.session_state.transcript:
            if st.button("✨ New chat", width="stretch",
                         help="Clear the conversation but keep the attached files"):
                reset_conversation()
                st.rerun()

        st.divider()
        _sidebar_model_settings()
        st.divider()
        _sidebar_conversion_settings()
        st.divider()
        _sidebar_schema()
        st.divider()
        _sidebar_rules()


def current_provider_key() -> str:
    return st.session_state.provider


def provider_ready() -> tuple[bool, str]:
    """Whether the selected provider can run, and why not if it can't."""
    key = current_provider_key()
    spec = providers.SPECS[key]
    if spec.auth == "subscription":
        return providers.claude_code_available()
    if not api_key_for(key):
        return False, f"Add your {spec.key_label} in the sidebar to start chatting."
    return True, ""


def live_models_state(provider_key: str) -> str:
    return f"live_models_{provider_key}"


def models_for(provider_key: str) -> list[str]:
    """Model options to offer: the live list once fetched, else the bundled one."""
    live = st.session_state.get(live_models_state(provider_key))
    if live:
        return live
    return list(providers.SPECS[provider_key].models)


def _render_model_refresh(provider_key: str, spec) -> None:
    """A button to replace the bundled model list with a live one.

    Vendors retire model ids between releases -- Groq especially -- so a
    hard-coded list eventually 404s. This asks the provider what the key
    can actually use.
    """
    if spec.auth == "subscription":
        return  # the CLI decides; there is no models endpoint to query

    live = st.session_state.get(live_models_state(provider_key))
    caption = f"{len(live)} models loaded from {spec.label}" if live else ""

    if st.button("🔄 Refresh models", width="stretch", key=f"refresh_models_{provider_key}",
                 disabled=not api_key_for(provider_key),
                 help="Ask the provider which models this key can use"):
        try:
            found = providers.build(
                provider_key,
                model_for(provider_key),
                api_key_for(provider_key),
                st.session_state.base_url if spec.supports_base_url else "",
            ).list_models()
        except providers.ProviderError as exc:
            st.error(str(exc), icon="🚨")
        else:
            st.session_state[live_models_state(provider_key)] = found
            # If the selected model no longer exists, move to one that does.
            if model_for(provider_key) not in found:
                fallback = spec.default_model if spec.default_model in found else found[0]
                st.session_state[model_state(provider_key)] = fallback
                st.toast(f"`{fallback}` selected — the previous model is gone.", icon="🔄")
            st.rerun()

    if caption:
        st.caption(caption)


def _sidebar_model_settings() -> None:
    ready, reason = provider_ready()
    label = "⚙️ Model & provider" + ("" if ready else "  ⚠️")

    with st.expander(label, expanded=not ready):
        picked = st.radio(
            "Provider",
            providers.PROVIDER_KEYS,
            index=providers.PROVIDER_KEYS.index(current_provider_key()),
            format_func=lambda k: providers.SPECS[k].label,
            key="provider",
            help="Each provider keeps its own key and model choice.",
        )
        spec = providers.SPECS[picked]
        st.caption(spec.blurb)

        if spec.auth == "subscription":
            available, why = providers.claude_code_available()
            if available:
                st.success("Signed in via the Claude Code CLI.", icon="✅")
            else:
                st.error(why, icon="🚫")
        else:
            # The widget owns this value; see the note by key_state().
            st.text_input(
                spec.key_label,
                type="password",
                key=key_state(picked),
                help="Kept only in this browser session's memory; never written to disk.",
            )
            if not api_key_for(picked):
                st.markdown(f"[Get a key →]({spec.credentials_url})")

        # Deliberately unkeyed: with a key, the widget's remembered value
        # would outrank `index`, so a model set anywhere else (env var,
        # provider switch, a live refresh) could never take effect.
        available = models_for(picked)
        current_model = model_for(picked)
        options = available + ["custom…"]
        index = available.index(current_model) if current_model in available else len(available)
        choice = st.selectbox("Model", options, index=index)
        if choice == "custom…":
            st.session_state[model_state(picked)] = st.text_input(
                "Custom model id", value=current_model
            )
        else:
            st.session_state[model_state(picked)] = choice

        # Catch a known-incompatible model here rather than letting the
        # first message fail with a 400.
        if not providers.accepts_custom_tools(model_for(picked), spec):
            st.error(
                f"`{model_for(picked)}` doesn't accept custom tool calling, "
                "which this agent needs. Pick another model.",
                icon="🚫",
            )

        _render_model_refresh(picked, spec)

        if spec.supports_base_url:
            st.session_state.base_url = st.text_input(
                "API base URL (optional)",
                value=st.session_state.base_url,
                help="Only needed for an OpenAI-compatible proxy or gateway.",
            )

        # Anthropic's two paths are easy to confuse; say it plainly.
        if picked == "anthropic":
            st.caption(
                "⚠️ A claude.ai Pro/Max subscription does **not** include API "
                "credits. For subscription usage pick *Claude (subscription)*."
            )

        ready, reason = provider_ready()
        if not ready and spec.auth != "subscription":
            st.warning(reason, icon="🔑")

    # Switching provider mid-conversation would replay a transcript the new
    # backend never produced, so the built provider is cached per key.
    if st.session_state.get("_active_provider") != picked:
        st.session_state._active_provider = picked
        st.session_state.pop("_provider_obj", None)


def _sidebar_conversion_settings() -> None:
    with st.expander("🎯 Target", expanded=False):
        st.session_state.target_framework = st.selectbox(
            "Python framework",
            TARGET_FRAMEWORKS,
            index=TARGET_FRAMEWORKS.index(st.session_state.target_framework),
            help="Shapes the style of the generated module.",
        )
        workspace().target_framework = st.session_state.target_framework

    procs = workspace().procedures
    with st.expander(f"📎 Attached files ({len(procs)})", expanded=bool(procs)):
        if not procs:
            st.caption("Attach a `.sql` file using the 📎 button in the chat box.")
        for name, proc in list(procs.items()):
            row = st.columns([5, 1], vertical_alignment="center")
            row[0].markdown(
                f"`{name}`  \n<span class='sp-panel-sub'>{proc.dialect} · "
                f"{proc.line_count} lines</span>",
                unsafe_allow_html=True,
            )
            if row[1].button(
                "", icon=":material/close:", key=f"rm_{name}",
                help=f"Remove {name} from the workspace",
            ):
                del procs[name]
                st.rerun()

        st.caption("Or load a bundled sample:")
        samples = sorted(f for f in os.listdir(SAMPLE_DIR) if f.endswith(".sql"))
        picked = st.selectbox("Sample", samples, label_visibility="collapsed")
        if st.button("Load sample", width="stretch"):
            with open(os.path.join(SAMPLE_DIR, picked), encoding="utf-8") as fh:
                attach_procedure(picked, fh.read())
            st.rerun()


def _sidebar_schema() -> None:
    with st.expander("📐 Schema", expanded=False):
        st.session_state.schema_source = st.radio(
            "Source",
            SCHEMA_SOURCES,
            index=SCHEMA_SOURCES.index(st.session_state.schema_source),
            label_visibility="collapsed",
        )
        source = st.session_state.schema_source

        if source == "Upload Excel (.xlsx)":
            st.caption(
                "Columns: table_name, column_name, data_type, nullable, "
                "primary_key, foreign_key."
            )
            try:
                with open(SAMPLE_SCHEMA_PATH, "rb") as fh:
                    st.download_button(
                        "Template",
                        data=fh.read(),
                        file_name="sample_tables.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        width="stretch",
                    )
            except OSError:
                st.caption("_(bundled template missing)_")

            uploaded = st.file_uploader("Table details", type=["xlsx"], key="schema_upload")
            if uploaded is not None:
                try:
                    st.session_state.schema_df = schema_utils.load_schema_from_excel(
                        uploaded.getvalue()
                    )
                except Exception as exc:
                    st.session_state.schema_df = None
                    st.error(f"Couldn't read that file: {exc}")

            df = st.session_state.schema_df
            if df is not None and not df.empty:
                st.caption(f"{len(df)} columns / {df['table_name'].nunique()} tables")
                st.dataframe(df, width="stretch", height=160)

        elif source == "Paste DDL / notes":
            st.session_state.pasted_schema = st.text_area(
                "CREATE TABLE statements or notes",
                value=st.session_state.pasted_schema,
                height=140,
                placeholder="CREATE TABLE orders (order_id NUMBER PRIMARY KEY, ...);",
            )

        else:
            st.caption("A generic e-commerce schema, so you can try the app immediately.")
            st.dataframe(schema_utils.dummy_schema(), width="stretch", height=160)

        # Recompute every run so switching source takes effect immediately,
        # rather than leaving a stale schema silently in the prompt.
        workspace().schema_text = current_schema_text()


def current_schema_text() -> str:
    """Resolve the schema text for whichever source is currently selected."""
    source = st.session_state.schema_source
    if source == "Upload Excel (.xlsx)":
        df = st.session_state.schema_df
        return schema_utils.schema_to_prompt_text(df) if df is not None else ""
    if source == "Paste DDL / notes":
        return st.session_state.pasted_schema.strip()
    return schema_utils.schema_to_prompt_text(schema_utils.dummy_schema())


def _sidebar_rules() -> None:
    notes = feedback_notes.load_notes()
    with st.expander(f"🧠 Learned rules ({len(notes)})", expanded=False):
        st.caption(
            "Corrections the agent saved as general rules. These are folded "
            "into every future conversion."
        )
        if not notes:
            st.caption("_None yet — tell the agent 'always ...' to create one._")
        for note in notes:
            row = st.columns([6, 1], vertical_alignment="center")
            with row[0]:
                ui.rule_card(note["note"], note.get("dialect") or "all dialects")
            if row[1].button("🗑", key=f"del_note_{note['id']}", help="Forget this rule"):
                feedback_notes.delete_note(note["id"])
                st.rerun()


# --------------------------------------------------------------------------
# Attaching procedures
# --------------------------------------------------------------------------


def attach_procedure(name: str, sql: str) -> agent.Procedure:
    """Add a procedure to the workspace, auto-detecting its dialect."""
    return workspace().add_procedure(name, sql, sql_utils.guess_dialect(sql))


def _ingest_uploads(files) -> list[str]:
    """Validate and attach chat-box attachments. Returns the accepted names."""
    accepted: list[str] = []
    for f in files or []:
        sql, reason = sql_utils.check_upload(
            f.name, f.size, f.getvalue(), max_mb=MAX_FILE_MB, allowed=tuple(ALLOWED_EXTENSIONS)
        )
        if reason:
            st.toast(f"Skipped {f.name}: {reason}.", icon="⚠️")
            continue
        attach_procedure(f.name, sql)
        accepted.append(f.name)
    return accepted


# --------------------------------------------------------------------------
# Chat page
# --------------------------------------------------------------------------


def _provider_chip_text() -> str:
    key = current_provider_key()
    spec = providers.SPECS[key]
    return f"{spec.label} · <b>{model_for(key)}</b>"


def page_chat() -> None:
    ws = workspace()
    ui.header("Conversion workspace", "Attach a stored procedure, then just talk to the agent.")

    valid = sum(1 for a in ws.artifacts.values() if a.is_valid)
    ui.chips(
        [
            (f"<b>{len(ws.procedures)}</b> procedure(s)", "ok" if ws.procedures else ""),
            (f"<b>{len(ws.artifacts)}</b> generated · {valid} AST-valid",
             "ok" if ws.artifacts and valid == len(ws.artifacts) else ("warn" if ws.artifacts else "")),
            (f"{_provider_chip_text()}", "ok" if provider_ready()[0] else "bad"),
            (f"→ <b>{st.session_state.target_framework}</b>", ""),
        ]
    )

    if ws.artifacts:
        chat_col, artifact_col = st.columns([1.1, 1], gap="large")
    else:
        chat_col, artifact_col = st.container(), None

    # Rendered before the turn runs so the (disabled) chat box stays visible
    # while the agent works. Streamlit pins it to the bottom of the viewport
    # regardless of where it is called.
    _render_composer()

    with chat_col:
        _render_transcript()
        _run_pending_turn()

    if artifact_col is not None:
        with artifact_col:
            _render_artifact_panel()


def _render_transcript() -> None:
    ws = workspace()
    if not st.session_state.transcript and st.session_state.pending is None:
        ui.hero(
            "🧬",
            "Attach a stored procedure to begin",
            "Use the 📎 button in the chat box below to attach one or more "
            ".sql files — or load a bundled sample from the sidebar — then ask "
            "for whatever you need: a conversion, an explanation, a fix, or a "
            "rule the agent should follow from now on.",
        )
        if ws.procedures:
            st.caption("Quick starts")
            cols = st.columns(len(STARTERS))
            for col, starter in zip(cols, STARTERS):
                if col.button(starter, key=f"starter_{starter}", width="stretch"):
                    st.session_state.pending = {"text": starter, "files": []}
                    st.rerun()

    for record in st.session_state.transcript:
        if record["role"] == "user":
            with st.chat_message("user", avatar="🧑‍💻"):
                _render_user_message(record["content"], record.get("files"))
        else:
            with st.chat_message("assistant", avatar="🧬"):
                _render_tool_trace(record.get("tools", []), record.get("seconds"))
                if record.get("content"):
                    st.markdown(record["content"])
                if record.get("error"):
                    st.error(record["error"], icon="🚨")


def _render_user_message(content: str, files: list[str] | None) -> None:
    """Render one user turn.

    The marker span is emitted separately from the message body so the
    body never needs `unsafe_allow_html` -- pasted SQL containing angle
    brackets stays inert text rather than becoming markup.
    """
    st.markdown('<span class="sp-user-marker"></span>', unsafe_allow_html=True)
    st.markdown(content)
    if files:
        st.markdown(ui.file_pills(files), unsafe_allow_html=True)


def _render_tool_trace(tools: list[dict], seconds: float | None) -> None:
    """Show what the agent actually did, collapsed by default."""
    if not tools:
        return
    label = _tool_trace_label(tools, seconds)
    with st.expander(label, expanded=False):
        for step in tools:
            icon = "✅" if step["ok"] else "⚠️"
            st.markdown(f"{icon} **`{step['name']}`** {_summarise_args(step)}")
            detail = step.get("detail")
            if detail:
                st.caption(detail)


def _tool_trace_label(tools: list[dict], seconds: float | None) -> str:
    names = []
    for step in tools:
        if step["name"] not in names:
            names.append(step["name"])
    suffix = f" · {seconds:.1f}s" if seconds else ""
    return f"🛠 {len(tools)} step(s): {', '.join(names)}{suffix}"


def _summarise_args(step: dict) -> str:
    """One-line, human-readable description of a tool call."""
    name, args = step["name"], step.get("args", {})
    if name == "read_procedure":
        return f"— read `{args.get('name') or 'the attached procedure'}`"
    if name == "write_python":
        return f"— wrote `{args.get('filename', '?')}`"
    if name == "save_rule":
        return f"— saved rule: _{args.get('rule', '')}_"
    if name == "get_schema":
        return "— read the supplied schema"
    if name == "list_procedures":
        return "— listed attached files"
    if name == "list_rules":
        return "— reviewed learned rules"
    return ""


def _render_composer() -> None:
    """The chat box. Must stay top-level so Streamlit pins it to the bottom."""
    ready, reason = provider_ready()
    disabled = not ready
    placeholder = (
        reason or "Configure a provider in the sidebar to start…"
        if disabled
        else "Attach a .sql file with 📎, or ask about the ones already attached…"
    )
    submitted = st.chat_input(
        placeholder,
        accept_file="multiple",
        file_type=ALLOWED_EXTENSIONS,
        disabled=disabled or st.session_state.pending is not None,
        key="composer",
    )
    if not submitted:
        return

    text = (submitted.text or "").strip()
    names = _ingest_uploads(submitted.files)

    if not text and not names:
        return
    if not text:
        # Attaching a file with no question is a clear enough intent.
        listed = ", ".join(f"`{n}`" for n in names)
        text = f"I've attached {listed}. Convert it to Python."

    st.session_state.pending = {"text": text, "files": names}
    st.rerun()


def _run_pending_turn() -> None:
    """Execute the queued user turn, streaming the agent's work as it goes."""
    pending = st.session_state.pending
    if not pending:
        return

    with st.chat_message("user", avatar="🧑‍💻"):
        _render_user_message(pending["text"], pending["files"])

    st.session_state.transcript.append(
        {"role": "user", "content": pending["text"], "files": pending["files"]}
    )

    ready, reason = provider_ready()
    provider = _provider_or_none() if ready else None
    if provider is None:
        st.session_state.pending = None
        st.session_state.transcript.append(
            {"role": "assistant", "content": "", "tools": [],
             "error": reason or "That provider isn't configured yet — check the sidebar."}
        )
        st.rerun()
        return

    record: dict = {"role": "assistant", "content": "", "tools": [], "error": None}
    started = time.time()

    with st.chat_message("assistant", avatar="🧬"):
        status = st.status("Working…", expanded=True)
        text_slot = st.empty()
        accumulated = ""
        touched: list[str] = []
        cost: float | None = None

        try:
            for event in st.session_state.session.run_turn(provider, pending["text"]):
                if isinstance(event, events.TextDelta):
                    accumulated += event.text
                    text_slot.markdown(accumulated)
                elif isinstance(event, events.ToolStarted):
                    status.update(label=_running_label(event.name), state="running")
                elif isinstance(event, events.ToolFinished):
                    step = {
                        "name": event.name,
                        "args": event.arguments,
                        "ok": event.ok,
                        "detail": _tool_detail(event),
                    }
                    record["tools"].append(step)
                    with status:
                        icon = "✅" if event.ok else "⚠️"
                        st.markdown(f"{icon} `{event.name}` {_summarise_args(step)}")
                elif isinstance(event, events.ArtifactUpdated):
                    touched.append(event.artifact.filename)
                    _record_history(event.artifact, time.time() - started)
                elif isinstance(event, events.TurnFinished):
                    accumulated = event.text or accumulated
                    cost = event.cost_usd
                    text_slot.markdown(accumulated)
        except providers.ProviderError as exc:
            record["error"] = str(exc)
        except Exception as exc:  # never leave the user staring at a traceback
            record["error"] = f"Something went wrong while running the agent: {exc}"

        elapsed = time.time() - started
        status.update(
            label=_finished_label(record, elapsed, cost),
            state="error" if record["error"] else "complete",
            expanded=False,
        )
        if record["error"]:
            st.error(record["error"], icon="🚨")

    record["content"] = accumulated
    record["seconds"] = elapsed
    record["cost_usd"] = cost
    record["provider"] = providers.SPECS[current_provider_key()].label
    st.session_state.transcript.append(record)
    st.session_state.last_turn_seconds = elapsed
    st.session_state.pending = None

    if touched:
        st.toast(f"Updated {', '.join(touched)}", icon="🐍")
    st.rerun()


def _running_label(tool_name: str) -> str:
    return {
        "read_procedure": "Reading the procedure…",
        "get_schema": "Checking the schema…",
        "write_python": "Writing Python…",
        "save_rule": "Saving the rule…",
        "list_procedures": "Looking at attached files…",
        "list_rules": "Reviewing learned rules…",
    }.get(tool_name, f"Running {tool_name}…")


def _tool_detail(event) -> str:
    result = event.result or {}
    if result.get("error"):
        return f"{result['error']} {result.get('detail', '')}".strip()
    if event.name == "write_python":
        warnings = result.get("warnings") or []
        rev = result.get("revision", 1)
        bits = [f"revision {rev}", "AST valid"]
        if warnings:
            bits.append(f"{len(warnings)} style warning(s)")
        return " · ".join(bits)
    if event.name == "read_procedure":
        return f"lines {result.get('start_line')}–{result.get('end_line')} of {result.get('total_lines')}"
    return ""


def _finished_label(record: dict, elapsed: float, cost: float | None = None) -> str:
    if record["error"]:
        return "Failed"
    parts = [f"Done in {elapsed:.1f}s"]
    if record["tools"]:
        parts.append(f"{len(record['tools'])} tool step(s)")
    if cost:
        # On a subscription this is the equivalent API cost, not a charge.
        parts.append(f"~${cost:.3f}")
    return " · ".join(parts)


def _provider_or_none():
    """Build (and cache) the selected provider, or None if it can't run.

    The instance is cached because the Claude Code backend carries a
    resumable session id across turns; rebuilding it each rerun would
    silently drop the conversation's context.
    """
    key = current_provider_key()
    model = model_for(key)
    api_key = api_key_for(key)
    base_url = st.session_state.base_url if providers.SPECS[key].supports_base_url else ""

    signature = (key, model, api_key, base_url)
    if st.session_state.get("_provider_sig") == signature:
        cached = st.session_state.get("_provider_obj")
        if cached is not None:
            return cached

    try:
        provider = providers.build(key, model, api_key, base_url)
    except providers.ProviderError as exc:
        st.error(str(exc), icon="🚨")
        return None

    st.session_state._provider_sig = signature
    st.session_state._provider_obj = provider
    return provider


def _record_history(artifact: agent.Artifact, seconds: float) -> None:
    """Log a generated artifact to the audit history."""
    ws = workspace()
    proc = ws.resolve_procedure(artifact.source_procedure)
    history_store.save_record(
        {
            "id": f"{st.session_state.run_id}-{artifact.filename}#{artifact.revision}",
            "filename": proc.name if proc else artifact.filename,
            "dialect": proc.dialect if proc else "unknown",
            "target_framework": ws.target_framework,
            "duration_seconds": round(seconds, 2),
            "ast_valid": artifact.is_valid,
            "self_repaired": artifact.repair_attempts > 0,
            "sql_code": proc.sql if proc else "",
            "python_code": artifact.code,
            "revised_after_feedback": artifact.revision > 1,
        }
    )


# --------------------------------------------------------------------------
# Artifact panel
# --------------------------------------------------------------------------


def _render_artifact_panel() -> None:
    ws = workspace()
    artifacts = list(ws.artifacts.values())
    ui.panel_head("Generated Python", f"{len(artifacts)} file(s)")

    tabs = st.tabs([a.filename for a in artifacts])
    for tab, artifact in zip(tabs, artifacts):
        with tab:
            if artifact.is_valid:
                warn = artifact.validation.warnings or []
                st.success(
                    f"Passed AST validation · revision {artifact.revision}"
                    + (f" · {len(warn)} style note(s)" if warn else ""),
                    icon="✅",
                )
                for message in warn:
                    st.caption(f"⚠️ {message}")
            else:
                st.error("Did not pass AST validation.", icon="❌")

            if artifact.summary:
                st.caption(artifact.summary)

            view = st.segmented_control(
                "View",
                ["Python", "Source SQL"],
                default="Python",
                key=f"view_{artifact.filename}",
                label_visibility="collapsed",
            )
            proc = ws.resolve_procedure(artifact.source_procedure)
            if view == "Source SQL" and proc is not None:
                st.code(proc.sql, language="sql", line_numbers=True, height=460)
            else:
                st.code(artifact.code, language="python", line_numbers=True, height=460)

            st.download_button(
                "⬇️ Download",
                data=artifact.code,
                file_name=artifact.filename,
                mime="text/x-python",
                key=f"dl_{artifact.filename}",
                width="stretch",
            )


# --------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------

HISTORY_COLUMNS = [
    "id", "filename", "dialect", "target_framework",
    "created_at", "duration_seconds", "ast_valid",
]


def page_history() -> None:
    ui.header("Historical audit", "Every conversion this project has produced.", icon="📜")
    history = history_store.load_history()
    kpis = history_store.compute_kpis(history)

    cols = st.columns(3)
    cols[0].metric("Scripts processed", kpis["total_processed"])
    cols[1].metric("Avg. conversion time", f"{kpis['avg_duration_seconds']}s")
    cols[2].metric("AST success rate", f"{kpis['ast_success_rate']}%")

    st.divider()
    if not history:
        st.info("No conversions yet — run one from the Chat workspace.", icon="💬")
        return

    df = pd.DataFrame(history)
    # Tolerate records written by older versions / hand edits.
    for column in HISTORY_COLUMNS:
        if column not in df.columns:
            df[column] = None
    display = df[HISTORY_COLUMNS].sort_values("created_at", ascending=False)

    search = st.text_input("🔍 Filter by filename or dialect", placeholder="orders, Oracle, …")
    if search:
        mask = display["filename"].astype(str).str.contains(search, case=False, na=False) | display[
            "dialect"
        ].astype(str).str.contains(search, case=False, na=False)
        display = display[mask]

    st.dataframe(
        display,
        width="stretch",
        height=320,
        hide_index=True,
        column_config={
            "id": st.column_config.TextColumn("Run"),
            "filename": st.column_config.TextColumn("Source file"),
            "dialect": st.column_config.TextColumn("Dialect"),
            "target_framework": st.column_config.TextColumn("Target"),
            "created_at": st.column_config.TextColumn("When"),
            "duration_seconds": st.column_config.NumberColumn("Duration", format="%.2f s"),
            "ast_valid": st.column_config.CheckboxColumn("AST ✓"),
        },
    )
    if display.empty:
        st.caption("No records match that filter.")
        return

    st.divider()
    st.subheader("Reload into the chat workspace")
    options = {
        f"{row['filename']} — {row['created_at']} ({row['id']})": row["id"]
        for _, row in display.iterrows()
    }
    pick = st.selectbox("Record", list(options))
    if st.button("↩️ Load into chat", type="primary"):
        _reload_history_record(options[pick])


def _reload_history_record(record_id: str) -> None:
    record = history_store.get_record(record_id)
    if not record:
        st.error("That record could not be found — it may have been deleted.", icon="🚨")
        return

    ws = workspace()
    sql = record.get("sql_code") or ""
    if sql.strip():
        attach_procedure(record.get("filename", "reloaded.sql"), sql)

    code = record.get("python_code") or ""
    if code.strip():
        filename = os.path.splitext(record.get("filename", "reloaded"))[0] + ".py"
        ws.artifacts[filename] = agent.Artifact(
            filename=filename,
            code=code,
            summary="Reloaded from the audit history.",
            source_procedure=record.get("filename"),
            validation=code_validator.validate_python_ast(code),
        )

    st.session_state.transcript.append(
        {
            "role": "assistant",
            "content": f"Reloaded **{record.get('filename')}** from history "
                       f"({record.get('created_at')}). Ask me to change anything.",
            "tools": [],
        }
    )
    st.session_state.nav_request = CHAT
    st.rerun()


# --------------------------------------------------------------------------
# Analytics
# --------------------------------------------------------------------------


def page_analytics() -> None:
    ui.header("Agent analytics", "How the conversion agent is performing.", icon="📊")
    history = history_store.load_history()
    if not history:
        st.info("No data yet — analytics populate as you run conversions.", icon="📊")
        return

    import plotly.express as px

    df = pd.DataFrame(history)
    for column in ("dialect", "ast_valid", "duration_seconds", "created_at"):
        if column not in df.columns:
            df[column] = None
    df["ast_valid"] = df["ast_valid"].fillna(False).astype(bool)
    df["duration_seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce").fillna(0)

    accent = "#6d5efc"
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("**Conversions by dialect**")
        counts = df.groupby("dialect", dropna=False).size().reset_index(name="count")
        fig = px.bar(counts, x="dialect", y="count", color_discrete_sequence=[accent])
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=300, xaxis_title=None)
        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("**AST success rate by dialect**")
        success = df.groupby("dialect", dropna=False)["ast_valid"].mean().reset_index()
        success["ast_valid"] = (success["ast_valid"] * 100).round(1)
        fig2 = px.bar(
            success, x="dialect", y="ast_valid",
            labels={"ast_valid": "Success rate (%)"},
            color_discrete_sequence=["#22d3ee"],
        )
        fig2.update_layout(
            margin=dict(l=0, r=0, t=10, b=0), height=300,
            yaxis_range=[0, 100], xaxis_title=None,
        )
        st.plotly_chart(fig2, width="stretch")

    st.markdown("**Conversion duration over time**")
    ordered = df.sort_values("created_at")
    fig3 = px.line(
        ordered, x="created_at", y="duration_seconds", markers=True,
        color_discrete_sequence=[accent],
    )
    fig3.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=280, xaxis_title=None)
    st.plotly_chart(fig3, width="stretch")

    revised = sum(1 for h in history if h.get("revised_after_feedback"))
    st.caption(
        f"{revised} of {len(history)} conversions were revised at least once "
        "after feedback."
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    init_state()
    ui.inject_styles()
    render_sidebar()

    page = st.session_state.page
    if page == HISTORY:
        page_history()
    elif page == ANALYTICS:
        page_analytics()
    else:
        page_chat()


if __name__ == "__main__":
    main()
