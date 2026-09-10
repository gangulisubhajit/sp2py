"""
The conversion agent.

Instead of a single fixed "convert this procedure" call, the chat UI runs a
tool-calling loop: the model decides whether the user is asking for a
conversion, a question about the SQL, a targeted fix, or a durable rule
change -- and calls the matching tool.

Tools are deliberately small and side-effect-explicit:

  list_procedures   what the user has uploaded
  read_procedure    the SQL source (optionally a line range, for big files)
  get_schema        table/column details the user supplied
  write_python      emit/replace a generated module -- AST-validated, and the
                    error is handed straight back so the model can retry
  save_rule         persist a correction so it applies to future conversions
  list_rules        what has already been learned

`AgentSession` owns the transcript and the workspace; `run_turn` drives one
user turn to completion, yielding events the UI renders as it goes.

The loop is provider-agnostic: OpenAI, Anthropic and Groq each return one
assistant turn at a time and this module drives the tool cycle, while the
Claude Agent SDK runs its own harness and is handed the same tools. Either
way the UI sees the same stream of events from `src/events.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator

from . import code_validator, feedback_notes, prompts
from .events import (
    AgentEvent,
    ArtifactUpdated,
    TextDelta,
    ToolFinished,
    ToolStarted,
    TurnFinished,
)

MAX_TOOL_ITERATIONS = 8
# Keep a single read_procedure result from blowing the context window.
MAX_SOURCE_CHARS = 60_000


# --------------------------------------------------------------------------
# Workspace: the files and artifacts this conversation is working on
# --------------------------------------------------------------------------


@dataclass
class Procedure:
    """One uploaded/pasted stored procedure."""

    name: str
    sql: str
    dialect: str

    @property
    def line_count(self) -> int:
        return self.sql.count("\n") + 1


@dataclass
class Artifact:
    """One generated Python module."""

    filename: str
    code: str
    summary: str = ""
    source_procedure: str | None = None
    validation: Any = None
    revision: int = 1
    # How many drafts the model had rejected by ast.parse before this one
    # was accepted -- i.e. genuine self-repair, as opposed to `revision`,
    # which counts user-requested rewrites.
    repair_attempts: int = 0

    @property
    def is_valid(self) -> bool:
        return bool(self.validation and self.validation.is_valid)


@dataclass
class Workspace:
    """Everything the agent's tools can see or touch for one conversation."""

    procedures: dict[str, Procedure] = field(default_factory=dict)
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    schema_text: str = ""
    target_framework: str = "FastAPI + SQLAlchemy"
    # filename -> drafts rejected by ast.parse since the last accepted write
    rejected_writes: dict[str, int] = field(default_factory=dict)

    def add_procedure(self, name: str, sql: str, dialect: str) -> Procedure:
        proc = Procedure(name=name, sql=sql, dialect=dialect)
        self.procedures[name] = proc
        return proc

    def resolve_procedure(self, name: str | None) -> Procedure | None:
        """Look a procedure up leniently -- exact, then case/extension-insensitive."""
        if not self.procedures:
            return None
        if not name:
            # A single uploaded file is unambiguous; don't make the model guess.
            if len(self.procedures) == 1:
                return next(iter(self.procedures.values()))
            return None
        if name in self.procedures:
            return self.procedures[name]
        target = name.strip().lower()
        for key, proc in self.procedures.items():
            low = key.lower()
            if low == target or low.rsplit(".", 1)[0] == target.rsplit(".", 1)[0]:
                return proc
        for key, proc in self.procedures.items():
            if target in key.lower():
                return proc
        return None


# --------------------------------------------------------------------------
# Tool schemas
# --------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_procedures",
            "description": (
                "List the stored procedures the user has added to this "
                "conversation, with their dialect and size. Call this first if "
                "you are unsure which files are available."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_procedure",
            "description": (
                "Read the SQL source of one stored procedure. Omit "
                "'name' when only one procedure has been uploaded. Use "
                "start_line/end_line to page through a very large file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Filename of the procedure, e.g. 'sample_oracle.sql'.",
                    },
                    "start_line": {"type": "integer", "description": "1-based first line."},
                    "end_line": {"type": "integer", "description": "1-based last line, inclusive."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": (
                "Get the table/column details the user supplied (Excel upload, "
                "pasted DDL, or the bundled dummy schema). Call this before "
                "mapping SQL types to Python/ORM types."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_python",
            "description": (
                "Write (or overwrite) a generated Python module. The code is "
                "parsed with ast.parse before being accepted: if it fails, the "
                "syntax error is returned to you and nothing is saved -- fix it "
                "and call this tool again. Always pass the COMPLETE module, "
                "never a diff or a fragment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Target filename, e.g. 'process_order.py'.",
                    },
                    "code": {
                        "type": "string",
                        "description": "The complete Python module source, with no markdown fences.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "One or two sentences on what this module does and any notable translation decisions.",
                    },
                    "source_procedure": {
                        "type": "string",
                        "description": "Filename of the stored procedure this was converted from, if any.",
                    },
                },
                "required": ["filename", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_rule",
            "description": (
                "Persist a correction as a durable rule applied to every FUTURE "
                "conversion in this project. Only call this when the user asks "
                "for the correction to stick generally ('always...', 'from now "
                "on...', 'as a rule...'), not for a one-off fix to this file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rule": {
                        "type": "string",
                        "description": "The rule, phrased imperatively, e.g. 'Always roll back the transaction when stock is insufficient.'",
                    },
                    "dialect": {
                        "type": "string",
                        "description": "Restrict the rule to one source dialect, or omit for all dialects.",
                        "enum": ["Oracle PL/SQL", "PostgreSQL PL/pgSQL", "Microsoft T-SQL"],
                    },
                },
                "required": ["rule"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_rules",
            "description": "List the durable rules already learned from earlier user feedback.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------


class AgentSession:
    """Owns the message transcript and executes the tool loop for one chat."""

    def __init__(self, workspace: Workspace | None = None):
        self.workspace = workspace or Workspace()
        self.messages: list[dict] = []

    # -- prompt ------------------------------------------------------------

    def system_prompt(self) -> str:
        dialects = sorted({p.dialect for p in self.workspace.procedures.values()})
        prompt = prompts.build_agent_system_prompt(
            dialects=dialects,
            target_framework=self.workspace.target_framework,
            procedure_names=list(self.workspace.procedures),
            has_schema=bool(self.workspace.schema_text.strip()),
        )
        learned = prompts.build_learned_notes_block(
            _relevant_notes(dialects)
        )
        if learned:
            prompt += "\n" + learned
        return prompt

    def tool_schemas(self) -> list[dict]:
        """The canonical (OpenAI-shaped) tool schemas, for any provider."""
        return TOOL_SCHEMAS

    def dispatch(self, name: str, args: dict) -> dict:
        """Public tool entry point, used by loop-owning providers."""
        return self._dispatch(name, args)

    def _messages_for_api(self) -> list[dict]:
        # Healed each turn so an abandoned turn can't wedge the conversation.
        # The system prompt is passed separately by the provider.
        self._heal_transcript()
        return list(self.messages)

    def _heal_transcript(self) -> None:
        """Guarantee every assistant tool_call has a matching `tool` reply.

        A turn can be abandoned part-way -- the UI reran, the browser tab
        closed, an exception unwound the generator -- leaving an assistant
        message whose tool_calls were never answered. The API rejects that
        with a 400, permanently wedging the conversation, so backfill a
        stub reply for anything left dangling.
        """
        answered = {
            m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"
        }
        healed: list[dict] = []
        for message in self.messages:
            healed.append(message)
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                if call["id"] in answered:
                    continue
                healed.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(
                            {"error": "This tool call was interrupted and never ran."}
                        ),
                    }
                )
                answered.add(call["id"])
        self.messages = healed

    # -- turn --------------------------------------------------------------

    def run_turn(self, provider, user_message: str) -> Iterator[AgentEvent]:
        """Drive one user turn to completion, yielding events as they happen.

        Providers that own their own agent loop (the Claude Agent SDK) are
        handed the turn directly; everything else runs the tool cycle here.
        """
        if getattr(provider, "owns_loop", False):
            yield from provider.run_agent_turn(self, user_message)
            return
        yield from self._run_tool_loop(provider, user_message)

    def _run_tool_loop(self, provider, user_message: str) -> Iterator[AgentEvent]:
        """Loop provider -> tool calls -> provider until the model is done."""
        self.messages.append({"role": "user", "content": user_message})

        # Text can arrive in several segments across tool-calling rounds
        # ("Let me read that file..." then the real answer); the UI shows
        # them as one message, so the transcript stores them joined.
        segments: list[str] = []
        for _ in range(MAX_TOOL_ITERATIONS):
            # Re-yield each token as it arrives so the UI can render live.
            stream = provider.iter_turn(
                self.system_prompt(),
                self._messages_for_api(),
                TOOL_SCHEMAS,
            )
            while True:
                try:
                    piece = next(stream)
                except StopIteration as stop:
                    result = stop.value
                    break
                yield TextDelta(piece)

            if result.text.strip():
                segments.append(result.text.strip())
            self.messages.append(result.as_assistant_message())

            if not result.tool_calls:
                break

            for call in result.tool_calls:
                args = _parse_arguments(call.arguments)
                yield ToolStarted(call.name, args)
                payload = self._dispatch(call.name, args)
                ok = not payload.get("error")
                yield ToolFinished(call.name, args, payload, ok)

                artifact_name = payload.pop("_artifact", None)
                if artifact_name:
                    yield ArtifactUpdated(self.workspace.artifacts[artifact_name])

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(payload, default=str)[:20_000],
                    }
                )
        else:
            overrun = (
                f"I used all {MAX_TOOL_ITERATIONS} tool steps without settling on "
                "an answer. Try narrowing the request, or ask me to convert one "
                "procedure at a time."
            )
            segments.append(overrun)
            self.messages.append({"role": "assistant", "content": overrun})

        yield TurnFinished("\n\n".join(segments))

    # -- tools -------------------------------------------------------------

    def _dispatch(self, name: str, args: dict) -> dict:
        handler = {
            "list_procedures": self._tool_list_procedures,
            "read_procedure": self._tool_read_procedure,
            "get_schema": self._tool_get_schema,
            "write_python": self._tool_write_python,
            "save_rule": self._tool_save_rule,
            "list_rules": self._tool_list_rules,
        }.get(name)
        if handler is None:
            return {"error": f"Unknown tool '{name}'."}
        try:
            return handler(args)
        except Exception as exc:  # a tool bug must not kill the conversation
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _tool_list_procedures(self, args: dict) -> dict:
        if not self.workspace.procedures:
            return {
                "procedures": [],
                "hint": "The user hasn't uploaded any stored procedure yet. Ask them to attach one.",
            }
        return {
            "procedures": [
                {"name": p.name, "dialect": p.dialect, "lines": p.line_count, "characters": len(p.sql)}
                for p in self.workspace.procedures.values()
            ]
        }

    def _tool_read_procedure(self, args: dict) -> dict:
        proc = self.workspace.resolve_procedure(args.get("name"))
        if proc is None:
            return {
                "error": "No such procedure.",
                "available": list(self.workspace.procedures),
            }

        lines = proc.sql.splitlines()
        start = max(1, int(args.get("start_line") or 1))
        end = int(args.get("end_line") or len(lines))
        end = min(end, len(lines))
        if start > len(lines):
            return {"error": f"start_line {start} is past the end of the file ({len(lines)} lines)."}

        body = "\n".join(lines[start - 1 : end])
        truncated = len(body) > MAX_SOURCE_CHARS
        if truncated:
            body = body[:MAX_SOURCE_CHARS]

        return {
            "name": proc.name,
            "dialect": proc.dialect,
            "start_line": start,
            "end_line": end,
            "total_lines": len(lines),
            "truncated": truncated,
            "sql": body,
        }

    def _tool_get_schema(self, args: dict) -> dict:
        text = self.workspace.schema_text.strip()
        if not text:
            return {
                "schema": "",
                "hint": (
                    "No schema was supplied. Infer types from how each column is "
                    "used and comment every assumption you make."
                ),
            }
        return {"schema": text}

    def _tool_write_python(self, args: dict) -> dict:
        filename = (args.get("filename") or "generated.py").strip()
        if not filename.endswith(".py"):
            filename += ".py"

        code = args.get("code") or ""
        # Models occasionally wrap the code in fences despite instructions.
        if code.lstrip().startswith("```"):
            _, stripped = code_validator.extract_python_code(code)
            code = stripped or code

        validation = code_validator.validate_python_ast(code)
        if not validation.is_valid:
            # Reject the write: the model gets the error and retries.
            rejected = self.workspace.rejected_writes.get(filename, 0) + 1
            self.workspace.rejected_writes[filename] = rejected
            return {
                "error": "The code failed AST validation and was NOT saved.",
                "detail": validation.error_message,
                "attempt": rejected,
                "instruction": "Fix the syntax error and call write_python again with the complete module.",
            }

        existing = self.workspace.artifacts.get(filename)
        artifact = Artifact(
            filename=filename,
            code=code,
            summary=(args.get("summary") or "").strip(),
            source_procedure=args.get("source_procedure"),
            validation=validation,
            revision=(existing.revision + 1) if existing else 1,
            repair_attempts=self.workspace.rejected_writes.pop(filename, 0),
        )
        self.workspace.artifacts[filename] = artifact

        return {
            "saved": True,
            "filename": filename,
            "revision": artifact.revision,
            "ast_valid": True,
            "warnings": validation.warnings or [],
            "_artifact": filename,
        }

    def _tool_save_rule(self, args: dict) -> dict:
        rule = (args.get("rule") or "").strip()
        if not rule:
            return {"error": "rule text was empty."}
        dialect = args.get("dialect") or None
        record = feedback_notes.add_note(rule, dialect=dialect)
        return {
            "saved": True,
            "id": record["id"],
            "rule": record["note"],
            "scope": dialect or "all dialects",
        }

    def _tool_list_rules(self, args: dict) -> dict:
        notes = feedback_notes.load_notes()
        return {
            "rules": [
                {"id": n["id"], "rule": n["note"], "scope": n.get("dialect") or "all dialects"}
                for n in notes
            ]
        }


def _relevant_notes(dialects: list[str]) -> list[dict]:
    """Notes that are global, or scoped to any dialect currently in play."""
    notes = feedback_notes.load_notes()
    if not dialects:
        return notes
    return [n for n in notes if not n.get("dialect") or n.get("dialect") in dialects]


def _parse_arguments(raw: str) -> dict:
    """Parse a tool-call argument blob, tolerating an empty or malformed one."""
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return parsed if isinstance(parsed, dict) else {"_value": parsed}
