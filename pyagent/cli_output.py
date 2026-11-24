"""Shared output contract for the ``pyagent`` command line interface.

Every command produces a :class:`Result`. With ``--json`` the CLI prints a
single JSON envelope on stdout::

    {"ok": true, "command": "profiles.list", "data": {...}}
    {"ok": false, "command": "profiles.list", "error": {"code": "...", "message": "..."}}

``ok`` is ``true`` if and only if the process exit code is ``0``. On failure
the envelope either carries an ``error`` object (unexpected failure, produced
by :func:`emit_error`) or a ``data`` payload with details (an expected
negative result such as a failed ``agents validate`` or a failed extension
install). Scripts should key off the exit code and ``ok``, not off the shape
of ``data``.

Without ``--json`` the command's human renderer writes to stdout; errors are
written to stderr. Exit codes are stable:

- ``0`` success
- ``1`` runtime failure (agent error, I/O, upstream API, missing resource)
- ``2`` usage/validation error (bad arguments, invalid definition, bad profile)
- ``130`` interrupted (Ctrl-C)

Handlers never print directly; they return a :class:`Result` (or raise
:class:`CliError`). ``main()`` renders the result and maps exceptions to exit
codes, so every command behaves the same way.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from tabulate import tabulate

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2


class CliError(Exception):
    """An expected, user-facing failure.

    ``code`` is a stable snake_case identifier for scripts; ``message`` is
    human-readable and may change between releases.
    """

    def __init__(self, message: str, *, code: str = "error", exit_code: int = EXIT_ERROR) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.exit_code = exit_code


@dataclass
class Result:
    """Outcome of a command, before rendering."""

    command: str
    data: Any
    human: Callable[[], str] | None = None
    exit_code: int = EXIT_OK
    already_emitted: bool = False

    def render(self) -> str:
        if self.human is not None:
            return self.human()
        return default_human(self.data)


def default_human(data: Any) -> str:
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    return json.dumps(data, indent=2, default=str, ensure_ascii=False)


def emit(result: Result, *, json_output: bool) -> None:
    if result.already_emitted:
        return
    if json_output:
        payload = {
            "ok": result.exit_code == EXIT_OK,
            "command": result.command,
            "data": result.data,
        }
        print(json.dumps(payload, indent=2, default=str, ensure_ascii=False))
        return
    text = result.render()
    if text:
        print(text)


def emit_error(command: str, exc: BaseException, *, json_output: bool) -> int:
    if isinstance(exc, CliError):
        code, message, exit_code = exc.code, exc.message, exc.exit_code
    elif _is_usage_error(exc):
        code, message, exit_code = "usage_error", str(exc), EXIT_USAGE
    elif isinstance(exc, OSError):
        code, message, exit_code = "io_error", str(exc), EXIT_ERROR
    else:
        code, message, exit_code = "internal_error", str(exc), EXIT_ERROR
    if json_output:
        payload = {
            "ok": False,
            "command": command,
            "error": {"code": code, "message": message},
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        sys.stderr.write(f"{message}\n")
    return exit_code


def _is_usage_error(exc: BaseException) -> bool:
    """Errors caused by bad user input rather than a runtime failure.

    ``SessionError`` (invalid checkpoint/session data) is a ``ValueError``
    subclass, so a single check covers both.
    """
    return isinstance(exc, ValueError)


def note(message: str) -> None:
    """Write a diagnostic line to stderr (never stdout)."""
    sys.stderr.write(message + "\n")


def render_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    rows = list(rows)
    if not rows:
        return ""
    return tabulate(rows, headers=list(headers))


def render_detail(title: str | None, fields: Sequence[tuple[str, Any]]) -> str:
    width = max((len(name) for name, _ in fields), default=0)
    lines = [f"{name.ljust(width)}  {value}" for name, value in fields]
    if title:
        return "\n".join([title, *lines])
    return "\n".join(lines)
