# SP2PY — Stored Procedure → Python Converter

An AI agent that converts legacy stored procedures (Oracle PL/SQL,
PostgreSQL PL/pgSQL, Microsoft T-SQL) into clean, modular Python backend
code — through a chat interface. Attach a `.sql` file to the chat box,
say what you want, and the agent reads it, writes the module, and shows
it beside the conversation.

## What you get

- **Chat-first workspace** — attach one or more `.sql` / `.prc` / `.txt`
  files with the 📎 button in the chat box (or load a bundled sample),
  then just talk. The source dialect is detected automatically from the
  SQL itself.
- **Four backends, one interface** — OpenAI, Claude (by API key *or* your
  claude.ai subscription) and Groq. Switch in the sidebar; the agent, the
  tools and the UI behave identically on all of them.
- **A real agent, not a fixed pipeline** — the model chooses what to do
  with tools: read the procedure, read your schema, write a module, or
  save a durable rule. Ask "what does this do?" and you get an
  explanation, not a pointless regeneration. Ask for a fix and only the
  code changes.
- **AST validation as a hard gate** — generated code is run through
  Python's `ast.parse` *before it is accepted*. Invalid code is rejected
  and the syntax error is handed straight back to the model, which
  retries. You never see a module that doesn't parse.
- **Live artifact panel** — the generated Python appears next to the
  chat with its validation status, revision count, a source-SQL toggle
  and a download button.
- **Rules that stick** — tell the agent "always roll back when stock is
  insufficient" and it saves that as a project rule, folded into the
  system prompt of every future conversion. Manage them in the sidebar.
- **Historical audit dashboard** — KPI cards (scripts processed, average
  conversion time, AST success rate) plus a searchable log. Reload any
  past run back into the chat.
- **Agent analytics** — conversion volume, success rate by dialect, and
  duration trends.

## Providers

Pick one in the sidebar under **⚙️ Model & provider**. Each keeps its own
key and model choice, so you can switch mid-project to compare.

| Provider | Auth | Notes |
|---|---|---|
| **OpenAI** | API key | `gpt-4o`, `gpt-4.1`, `o4-mini`, … Supports a custom base URL for proxies/Azure gateways. |
| **Claude (subscription)** | Your claude.ai login | **No API key and no separate bill** — runs through the Claude Code CLI you're already signed in to. Local, single-user only. |
| **Claude (API key)** | Anthropic API key | `claude-opus-5`, `claude-sonnet-5`, … Pay-as-you-go. |
| **Groq** | API key | Very fast open-weight models (Llama 3.x, gpt-oss). Has a free tier. |

**If a model 404s**, click **🔄 Refresh models** in the sidebar. It asks the
provider which models your key can actually use and rewrites the dropdown
— vendors retire model ids between releases (Groq especially), so the
bundled list is only a fallback. Non-chat models (speech, embeddings,
safety classifiers) are filtered out automatically, and if your current
selection has been retired the app moves you to one that still exists.

> ⚠️ **A claude.ai Pro/Max subscription does *not* include Anthropic API
> credits.** They are billed separately. If you have a subscription and
> want to use it, pick **Claude (subscription)** — not *Claude (API key)*.

**Claude (subscription)** additionally needs the
[Claude Code CLI](https://claude.com/product/claude-code) installed and
signed in (run `claude` once in a terminal). The app checks for this and
tells you if it's missing. Because the agent's tools are exposed to it as
an in-process MCP server with `permission_mode="dontAsk"`, it gets the
SP2PY tools and nothing else — no shell, no access to your filesystem.

## Requirements

- Python 3.10+
- Credentials for at least one provider (see the table above)

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) preload credentials so you don't paste them every run
cp .env.example .env
# then edit .env — it lists the env var for each provider

# 4. Run the app
streamlit run app.py
```

Streamlit opens the app at `http://localhost:8501`. If you skipped step 3,
paste your API key into the sidebar — it's kept only in that session's
memory and is never written to disk. (The *Claude (subscription)*
provider needs no key at all.)

## Running from VS Code

1. Open this folder in VS Code (`File > Open Folder...`).
2. Open a terminal (`` Ctrl+` ``) and run the setup commands above.
3. Run `streamlit run app.py`; VS Code shows a clickable `localhost` link.

## Using it

1. Click 📎 in the chat box and attach a stored procedure (or open
   **📎 Attached files** in the sidebar and load a bundled sample).
2. Ask for what you need:
   - *"Convert this to Python"*
   - *"What does this procedure do?"*
   - *"The transaction should roll back if stock is insufficient"*
   - *"From now on, always use explicit savepoints"* — saves a rule
   - *"Add unit tests for the generated code"*
3. The generated module appears in the right-hand panel. Expand the
   **🛠 steps** line under any reply to see exactly which tools ran.

Pick the target framework (FastAPI + SQLAlchemy, or Flask + PyODBC) under
**🎯 Target** in the sidebar.

## Providing your own table/schema details

Open **📐 Schema** in the sidebar. Three options:

- **Bundled dummy schema** (default) — a small generic e-commerce schema
  (`customers`, `orders`, `order_items`, `products`) so the app is usable
  immediately.
- **Upload Excel (.xlsx)** — use the "Template" button to download the
  expected layout:

  | table_name | column_name | data_type | nullable | primary_key | foreign_key |
  |---|---|---|---|---|---|

  Blank `nullable` / `primary_key` cells mean "unknown" — the app will not
  claim a constraint you didn't supply.
- **Paste DDL / notes** — paste `CREATE TABLE` statements or free-form
  column notes.

## Project layout

```
app.py                      Streamlit UI (entry point)
src/
  agent.py                   Tool-calling agent: tools, workspace, turn loop
  events.py                  Events the agent emits; the UI renders these
  providers/
    base.py                  Provider interface + canonical transcript types
    openai_compat.py         OpenAI and Groq (one OpenAI-compatible client)
    anthropic_api.py         Anthropic Messages API + message translation
    claude_code.py           Claude Agent SDK (subscription); owns its own loop
  prompts.py                 System prompt construction per dialect/framework
  code_validator.py          AST-based validation of generated code
  schema_utils.py            Excel schema parsing + dummy schema + (future) DB connect stub
  sql_utils.py               Dialect detection, upload vetting
  feedback_notes.py          Persisted "learned" rules from user feedback
  history_store.py           JSON-file-backed conversion history + KPIs
  ui.py                      Stylesheet and small presentation helpers
tests/
  mock_llm.py                Offline server speaking both the OpenAI and
                             Anthropic wire protocols
  test_units.py              Validator, schema, dialect, store, provider units
  test_agent.py              Agent tool loop + streaming, against the mock server
  test_providers.py          Cross-provider parity + Claude Agent SDK wiring
  test_app.py                Full UI tests via Streamlit's AppTest
sample_procedures/           Example Oracle / PostgreSQL / T-SQL procedures
sample_schema/               Example Excel schema template
.streamlit/config.toml       Theme and server settings
data/                        Created at runtime: history.json, feedback_notes.json
```

## Tests

The suite runs fully offline: `tests/mock_llm.py` is started as a real
HTTP server speaking **both** the OpenAI and Anthropic wire protocols, so
the genuine SDK streaming and tool-call paths are exercised without a key
or network access. The same scripted trajectory is replayed through both
providers and asserted to behave identically.

```bash
pip install -r requirements-dev.txt
pytest -q
```

One test is opt-in because it spends real subscription usage — it drives
the actual Claude Code CLI end to end:

```bash
SP2PY_LIVE_CLAUDE=1 pytest tests/test_providers.py -k live
```

## Notes & limitations

- History and learned rules are stored as local JSON files under `data/`
  — fine for individual/local use; swap `history_store.py` and
  `feedback_notes.py` for a real database for multi-user deployment.
- **Live database connection is a future capability.** Supply table
  details via Excel upload, pasted DDL, or the bundled dummy schema.
  `src/schema_utils.connect_to_database()` is a stubbed placeholder.
- Reasoning models (`o4-mini`, `gpt-5`) are handled — the app omits
  `temperature`, which those models reject.
- Switching provider mid-conversation is supported; provider-native
  content (such as Claude thinking blocks) is tagged and ignored by
  backends that didn't produce it.
- Generated code is a strong first draft, not a guaranteed drop-in
  replacement. Always review it — especially transaction boundaries and
  error handling — before running it against a production database.
