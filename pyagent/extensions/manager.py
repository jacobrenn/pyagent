"""The ``/extension`` command surface: list, reload, new, load, unload.

Plain functions taking ``(agent, args)`` and returning output text, wired into
the UI's slash-command dispatcher. No manifests — ``load``/``unload`` operate
on the in-memory bus; ``new`` scaffolds a starter file on disk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..scaffold import ScaffoldError, create_user_extension
from ..user_runtime import resolve_user_dir, user_extensions_dir
from .loader import _discover, load_all, load_one, unload_one


def _ext_dir(agent: Any) -> Path:
    return user_extensions_dir(resolve_user_dir(agent.config.user_dir))


def handle_extension_command(agent: Any, args: list[str]) -> str:
    """Dispatch ``/extension`` subcommands. Returns output text."""
    if not args or args[0].lower() in {"list", "ls"}:
        return _cmd_list(agent)
    sub, rest = args[0].lower(), args[1:]
    if sub == "reload":
        return _cmd_reload(agent)
    if sub == "new":
        if not rest:
            return "Usage: `/extension new <name>`"
        return _cmd_new(agent, rest[0])
    if sub == "load":
        if not rest:
            return "Usage: `/extension load <name>`"
        return _cmd_load(agent, rest[0])
    if sub == "unload":
        if not rest:
            return "Usage: `/extension unload <name>`"
        return _cmd_unload(agent, rest[0])
    return "Usage: `/extension [list|reload|new <name>|load <name>|unload <name>]`"


def _cmd_list(agent: Any) -> str:
    on_disk = _discover(_ext_dir(agent))
    loaded = set(agent.bus.loaded_extensions())
    if not on_disk and not loaded:
        return "No extensions found in `~/.pyagent/extensions/`."
    lines = ["Extensions:"]
    for name in on_disk:
        state = "loaded" if name in loaded else "not loaded"
        lines.append(f"- {name} [{state}]")
    for name in sorted(loaded - set(on_disk)):
        lines.append(f"- {name} [loaded (not on disk)]")
    return "\n".join(lines)


def _cmd_reload(agent: Any) -> str:
    agent.bus.clear()
    loaded, failed = load_all(agent.bus, _ext_dir(agent), agent._ext_log)
    agent._rebuild_external_tools()
    parts = [f"Reloaded: {', '.join(loaded) or '<none>'}."]
    if failed:
        parts.append("Failed: " + ", ".join(f"{n} ({e})" for n, e in failed))
    return "\n".join(parts)


def _cmd_new(agent: Any, name: str) -> str:
    try:
        path = create_user_extension(name, user_dir=agent.config.user_dir)
    except ScaffoldError as exc:
        return str(exc)
    ext_dir = _ext_dir(agent)
    return (
        f"Created extension at `{path}`.\n"
        f"A starter tool was also placed at `{ext_dir / name / 'tools' / f'{name}.py'}`.\n"
        f"Edit them, then `/extension load {name}`."
    )


def _cmd_load(agent: Any, name: str) -> str:
    try:
        load_one(agent.bus, name, _ext_dir(agent), agent._ext_log)
    except Exception as exc:
        return f"Load failed: {exc}"
    agent._rebuild_external_tools()
    return f"Loaded `{name}`."


def _cmd_unload(agent: Any, name: str) -> str:
    if name not in agent.bus.loaded_extensions():
        return f"`{name}` is not loaded."
    unload_one(agent.bus, name)
    agent._rebuild_external_tools()
    return f"Unloaded `{name}`."
