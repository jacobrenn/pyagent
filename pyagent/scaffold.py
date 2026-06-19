from __future__ import annotations

import re
from pathlib import Path

from .templates import render_tool_template
from .user_runtime import (
    ensure_user_subdir,
    resolve_user_dir,
    user_extensions_dir,
    user_tools_dir,
)


_VALID_TOOL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


EXTENSION_TEMPLATE = '''\
"""PyAgent extension: {name}.

Extensions subscribe to lifecycle events via ``register(bus, name)`` and may
inject skills into the system prompt via ``ctx.add_skill``. See
``extensions_prd.md`` for the event catalog and the mutation model.

Skills are plain Markdown under ``~/.pyagent/skills/extensions/<key>.md``.
Calling ``ctx.add_skill("<key>")`` from ``input``/``before_agent_start``/
``turn_start`` injects the skill THIS turn; calling it from ``turn_end`` (or a
later event) applies it NEXT turn. The injection is auto-expunged after the
turn, so re-declare each turn to keep it.
"""
from __future__ import annotations


def register(bus, name):
    @bus.on("tool_call")
    def on_tool_call(payload, ctx):
        # Example safeguard: block destructive bash. Return a dict to mutate
        # the payload (veto keys like ``blocked`` short-circuit), or None to
        # pass through.
        if payload["name"] == "bash" and "rm -rf" in payload["input"].get("command", ""):
            return {"blocked": True, "reason": f"{name} blocks destructive commands"}
        return None

    @bus.on("turn_end")
    def on_turn_end(payload, ctx):
        # Inject a skill next turn once the conversation grows. Declaring at
        # turn_end applies it next turn (one-turn lag); the skill text is read
        # from ~/.pyagent/skills/extensions/<key>.md.
        if payload.get("message_count", 0) > 20:
            ctx.add_skill("{name}")
'''


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
        Path(user_dir).expanduser().resolve(
        ) if user_dir is not None else resolve_user_dir()
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


def create_user_extension(
    name: str,
    *,
    user_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a starter extension to ``~/.pyagent/extensions/<name>.py``.

    Refuses to overwrite an existing file unless ``overwrite=True``. Returns
    the absolute path to the new script.
    """
    ext_name = _normalize_tool_name(name)
    resolved = (
        Path(user_dir).expanduser().resolve()
        if user_dir is not None else resolve_user_dir()
    )
    exts = ensure_user_subdir(user_extensions_dir(resolved))
    target = exts / f"{ext_name}.py"

    if target.exists() and not overwrite:
        raise ScaffoldError(
            f"Cannot create extension: `{target}` already exists. Pass overwrite=True to replace it."
        )

    target.write_text(EXTENSION_TEMPLATE.replace("{name}", ext_name), encoding="utf-8")
    return target
