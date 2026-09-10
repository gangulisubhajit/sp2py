"""
Lightweight validation for AI-generated Python code.

Implements the "AST Syntax Verification Sandbox" step from the PRD: we
never trust the model's output blindly -- we parse it with Python's own
`ast` module before showing it to the user or writing it to disk.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass


@dataclass
class ValidationResult:
    is_valid: bool
    error_message: str | None = None
    warnings: list[str] | None = None


_PY_BLOCK_RE = re.compile(r"```(?:python|py)[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_BLOCK_RE = re.compile(r"```[ \t]*\r?\n(.*?)```", re.DOTALL)


def extract_python_code(model_response: str) -> tuple[str, str]:
    """Split a model response into (explanation_text, python_code).

    Prefers an explicitly ```python-tagged block; only falls back to an
    untagged ``` block if there is no tagged one, so a stray shell or SQL
    snippet earlier in the reply can't be mistaken for the module. If no
    block is found at all, the whole response is treated as explanation
    and the code comes back empty (which fails validation, triggering the
    self-repair pass).
    """
    match = _PY_BLOCK_RE.search(model_response) or _ANY_BLOCK_RE.search(model_response)
    if not match:
        return model_response.strip(), ""
    code = match.group(1).strip()
    explanation = model_response[: match.start()].strip()
    return explanation, code


def validate_python_ast(code: str) -> ValidationResult:
    """Run the generated code through ast.parse and a few light PEP 8 checks."""
    if not code.strip():
        return ValidationResult(is_valid=False, error_message="No Python code was found in the response.")

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return ValidationResult(
            is_valid=False,
            error_message=f"SyntaxError: {exc.msg} (line {exc.lineno}, col {exc.offset})",
        )

    warnings: list[str] = []

    has_docstringed_def = False
    total_defs = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            total_defs += 1
            if ast.get_docstring(node):
                has_docstringed_def = True

    if total_defs and not has_docstringed_def:
        warnings.append("None of the generated functions/classes have docstrings.")

    long_lines = [i + 1 for i, line in enumerate(code.splitlines()) if len(line) > 99]
    if long_lines:
        warnings.append(
            f"{len(long_lines)} line(s) exceed 99 characters (PEP 8 soft limit), e.g. line {long_lines[0]}."
        )

    return ValidationResult(is_valid=True, warnings=warnings or None)
