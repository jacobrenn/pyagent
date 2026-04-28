from __future__ import annotations

import re
from pathlib import Path

from .templates import render_tool_template
from .user_runtime import (
    ensure_user_subdir,
    resolve_user_dir,
    user_tools_dir,
)


_VALID_TOOL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


class ScaffoldError(ValueError):
    """Raised for predictable scaffolding failures (bad name, conflict, etc.)."""


def _normalize_tool_name(raw: str) -> str:
    name = raw.strip()
    if name.endswith(".py"):
        name = name[: -len(".py")]
    if not _VALID_TOOL_NAME.fullmatch(name):
        raise ScaffoldError(
            "Tool name must start with a letter or underscore and contain only "
            "letters, digits, underscores, or hyphens."
        )
    return name


def create_user_tool(
    name: str,
    *,
    user_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a starter tool script to ``~/.pyagent/tools/<name>.py``.

    Refuses to overwrite an existing file unless ``overwrite=True``.
    Returns the absolute path to the new script.
    """
    tool_name = _normalize_tool_name(name)
    resolved = (
        Path(user_dir).expanduser().resolve() if user_dir is not None else resolve_user_dir()
    )
    tools = ensure_user_subdir(user_tools_dir(resolved))
    target = tools / f"{tool_name}.py"

    if target.exists() and not overwrite:
        raise ScaffoldError(
            f"Cannot create tool: `{target}` already exists. Pass overwrite=True to replace it."
        )

    target.write_text(render_tool_template(tool_name), encoding="utf-8")
    try:
        target.chmod(0o755)
    except OSError:
        # Permission errors on chmod are non-fatal (e.g. on some FS types
        # where permissions are not honored). Execution still works via
        # `uv run <script>`.
        pass
    return target
