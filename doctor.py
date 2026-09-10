"""
Preflight check for SP2PY.

Run it when something isn't working and you want to know whether the
problem is your environment, your credentials, or the app itself:

    python doctor.py

Exits non-zero if nothing is usable, so it also works in CI.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OK, WARN, BAD = "[ ok ]", "[warn]", "[FAIL]"


def line(status: str, label: str, detail: str = "") -> None:
    # Deliberately ASCII-only: the Windows console defaults to cp1252 and
    # would mangle (or crash on) anything fancier -- a diagnostic tool must
    # not be the thing that fails.
    print(f"  {status} {label}" + (f"  {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}")


def check_python() -> bool:
    section("Python")
    version = sys.version_info
    good = version >= (3, 10)
    line(OK if good else BAD, f"{sys.version.split()[0]}", "" if good else "3.10+ required")
    line(OK, "interpreter", sys.executable)
    in_venv = sys.prefix != sys.base_prefix
    line(OK if in_venv else WARN, "virtualenv", "active" if in_venv else "not active")
    return good


def check_packages() -> bool:
    section("Packages")
    required = ["streamlit", "pandas", "openpyxl", "plotly", "python-dotenv", "openai", "anthropic"]
    optional = ["claude-agent-sdk", "pytest"]
    ok = True

    for name in required:
        try:
            line(OK, name, md.version(name))
        except md.PackageNotFoundError:
            line(BAD, name, "missing — pip install -r requirements.txt")
            ok = False

    for name in optional:
        try:
            line(OK, name, md.version(name) + "  (optional)")
        except md.PackageNotFoundError:
            line(WARN, name, "not installed (optional)")
    return ok


def check_providers() -> bool:
    section("Providers")
    try:
        from src import providers
    except Exception as exc:
        line(BAD, "import src.providers", str(exc))
        return False

    any_ready = False
    for key in providers.PROVIDER_KEYS:
        spec = providers.SPECS[key]
        if spec.auth == "subscription":
            ready, why = providers.claude_code_available()
            detail = "signed in via Claude Code CLI" if ready else why
        else:
            ready = bool(os.getenv(spec.key_env))
            detail = f"{spec.key_env} set" if ready else f"set {spec.key_env}, or paste the key in the sidebar"
        any_ready = any_ready or ready
        line(OK if ready else WARN, f"{spec.label:<24}", detail)

    if not any_ready:
        line(WARN, "no provider configured", "the app will run but the chat box stays disabled")
    return True


def check_cli() -> None:
    section("Claude Code CLI (only needed for the subscription provider)")
    path = shutil.which("claude")
    if path:
        line(OK, "claude", path)
    else:
        line(WARN, "claude", "not on PATH — install from claude.com/product/claude-code")


def check_data() -> bool:
    section("App data")
    try:
        from src import feedback_notes, history_store
    except Exception as exc:
        line(BAD, "import stores", str(exc))
        return False

    try:
        history = history_store.load_history()
        line(OK, "history.json", f"{len(history)} record(s)")
    except Exception as exc:
        line(BAD, "history.json", str(exc))
        return False

    try:
        notes = feedback_notes.load_notes()
        line(OK, "feedback_notes.json", f"{len(notes)} learned rule(s)")
    except Exception as exc:
        line(BAD, "feedback_notes.json", str(exc))
        return False
    return True


def check_samples() -> bool:
    section("Bundled samples")
    try:
        from src import sql_utils
    except Exception as exc:
        line(BAD, "import src.sql_utils", str(exc))
        return False

    root = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(root, "sample_procedures")
    if not os.path.isdir(sample_dir):
        line(BAD, "sample_procedures/", "missing")
        return False

    for name in sorted(os.listdir(sample_dir)):
        if not name.endswith(".sql"):
            continue
        with open(os.path.join(sample_dir, name), encoding="utf-8") as fh:
            sql = fh.read()
        line(OK, name, f"detected as {sql_utils.guess_dialect(sql)}")
    return True


def main() -> int:
    print("SP2PY doctor")
    results = [
        check_python(),
        check_packages(),
        check_providers(),
        check_data(),
        check_samples(),
    ]
    check_cli()

    print()
    if all(results):
        print("All checks passed. Start the app with:  streamlit run app.py")
        return 0
    print("Some checks failed — see [FAIL] lines above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
