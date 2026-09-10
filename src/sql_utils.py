"""
Small helpers for working with the uploaded SQL itself.

Currently: guessing which dialect a procedure is written in, so the chat
UI can pre-select it instead of making the user classify every upload.
"""

from __future__ import annotations

import re

ORACLE = "Oracle PL/SQL"
POSTGRES = "PostgreSQL PL/pgSQL"
TSQL = "Microsoft T-SQL"

DIALECTS = [ORACLE, POSTGRES, TSQL]

# (regex, dialect, weight). Weights favour markers that are unambiguous:
# `$$ ... LANGUAGE plpgsql` can only be Postgres, whereas `BEGIN` is noise.
_SIGNALS: list[tuple[str, str, int]] = [
    (r"\blanguage\s+plpgsql\b", POSTGRES, 6),
    (r"\$\$", POSTGRES, 4),
    (r"\breturns\s+(table|setof|trigger)\b", POSTGRES, 3),
    (r"\braise\s+(notice|exception)\b", POSTGRES, 3),
    (r"\bcreate\s+or\s+replace\s+function\b", POSTGRES, 2),
    (r"::\w+", POSTGRES, 1),
    (r"\bcreate\s+or\s+replace\s+(procedure|package)\b", ORACLE, 4),
    (r"\bvarchar2\b|\bnumber\s*\(|\bnvarchar2\b", ORACLE, 3),
    (r"%rowtype\b|%type\b", ORACLE, 4),
    (r"\bdbms_output\b|\braise_application_error\b", ORACLE, 5),
    (r"\bbulk\s+collect\b|\bforall\b", ORACLE, 4),
    (r"\bpragma\s+autonomous_transaction\b", ORACLE, 5),
    (r"\bexception\s+when\b", ORACLE, 2),
    (r"\bfrom\s+dual\b", ORACLE, 4),
    (r"\bcreate\s+(or\s+alter\s+)?proc(edure)?\b", TSQL, 2),
    (r"@@\w+", TSQL, 5),
    (r"\bnvarchar\s*\(|\bnvarchar\b", TSQL, 2),
    (r"\bbegin\s+try\b|\bend\s+catch\b", TSQL, 5),
    (r"\bset\s+nocount\b|\bset\s+xact_abort\b", TSQL, 5),
    (r"#\w+\s|\bsp_executesql\b", TSQL, 4),
    (r"\bdeclare\s+@\w+", TSQL, 4),
    (r"\bas\s+begin\b", TSQL, 1),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), d, w) for p, d, w in _SIGNALS]

_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


ALLOWED_EXTENSIONS = ("sql", "prc", "txt")


def check_upload(
    name: str,
    size: int,
    data: bytes,
    max_mb: int = 5,
    allowed: tuple[str, ...] = ALLOWED_EXTENSIONS,
) -> tuple[str | None, str | None]:
    """Vet one uploaded file. Returns (decoded_sql, rejection_reason).

    Exactly one of the two is None. Kept free of Streamlit so the upload
    policy can be tested directly.
    """
    import os

    extension = os.path.splitext(name)[1].lower().lstrip(".")
    if extension not in allowed:
        pretty = ", ".join(f".{e}" for e in allowed)
        return None, f"only {pretty} files are supported"
    if size > max_mb * 1024 * 1024:
        return None, f"over the {max_mb}MB limit"

    # Legacy procedure exports are often cp1252/latin-1; replace rather than
    # fail so a stray byte can't block an otherwise fine 2000-line file.
    sql = data.decode("utf-8", errors="replace")
    if not sql.strip():
        return None, "the file is empty"
    return sql, None


def guess_dialect(sql: str, default: str = ORACLE) -> str:
    """Best-effort guess of the SQL dialect from distinctive syntax markers.

    Comments are stripped first so prose like "-- ported from T-SQL" in an
    Oracle file doesn't swing the result. Returns `default` when nothing
    distinctive is found.
    """
    if not sql or not sql.strip():
        return default

    body = _BLOCK_COMMENT_RE.sub(" ", sql)
    body = _LINE_COMMENT_RE.sub(" ", body)

    scores = dict.fromkeys(DIALECTS, 0)
    for pattern, dialect, weight in _COMPILED:
        if pattern.search(body):
            scores[dialect] += weight

    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else default


def procedure_names(sql: str) -> list[str]:
    """Extract declared procedure/function names, for display purposes."""
    pattern = re.compile(
        r"\bcreate\s+(?:or\s+replace\s+|or\s+alter\s+)?"
        r"(?:procedure|proc|function)\s+([\w.\[\]\"]+)",
        re.IGNORECASE,
    )
    seen: list[str] = []
    for match in pattern.finditer(sql or ""):
        # Quoting can appear per identifier part -- dbo.[Foo], "s"."f" --
        # so strip the delimiters everywhere, not just at the ends.
        name = re.sub(r'["\[\]]', "", match.group(1))
        if name and name not in seen:
            seen.append(name)
    return seen
