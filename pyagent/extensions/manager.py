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


def _disabled_ext_dir(agent: Any) -> Path:
    return _ext_dir(agent) / "disabled"


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
    if sub == "enable":
        if not rest:
            return "Usage: `/extension enable <name>`"
        return _cmd_enable(agent, rest[0])
    if sub == "disable":
        if not rest:
            return "Usage: `/extension disable <name>`"
        return _cmd_disable(agent, rest[0])
    if sub == "remove":
        if not rest:
            return "Usage: `/extension remove <name>`"
        return _cmd_remove(agent, rest[0])
    return "Usage: `/extension [list|reload|new <name>|load <name>|unload <name>|enable <name>|disable <name>|remove <name>]`"


def _cmd_list(agent: Any) -> str:
    on_disk = _discover(_ext_dir(agent))
    disabled = _discover(_disabled_ext_dir(agent))
    loaded = set(agent.bus.loaded_extensions())
    if not on_disk and not loaded and not disabled:
        return "No extensions found in `~/.pyagent/extensions/`."
    lines = ["Extensions:"]
    for name in on_disk:
        state = "loaded" if name in loaded else "not loaded"
        lines.append(f"- {name} [{state}]")
    for name in disabled:
        lines.append(f"- {name} [disabled]")
    for name in sorted(loaded - set(on_disk) - set(disabled)):
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
    ext_dir = _ext_dir(agent)
    disabled_dir = _disabled_ext_dir(agent)

    # If the extension was disabled (moved to disabled/), move it back first
    if disabled_dir.is_dir():
        for entry in disabled_dir.iterdir():
            if entry.name == name or (entry.is_file() and entry.stem == name):
                entry.rename(ext_dir / entry.name)
                break

    try:
        load_one(agent.bus, name, ext_dir, agent._ext_log)
    except Exception as exc:
        return f"Load failed: {exc}"
    agent._rebuild_external_tools()
    return f"Loaded `{name}`."


def _cmd_unload(agent: Any, name: str) -> str:
    if name not in agent.bus.loaded_extensions():
        return f"`{name}` is not loaded."
    # Persist: move to disabled/ and remove from the bus (so it stays
    # unloaded across restarts until `/extension load` restores it).
    _cmd_disable(agent, name)
    return f"Unloaded and disabled `{name}`. Use `/extension load {name}` to re-enable."

def _cmd_enable(agent: Any, name: str) -> str:
    disabled_dir = _disabled_ext_dir(agent)
    ext_dir = _ext_dir(agent)
    
    # Search in disabled dir
    found = False
    # Extensions can be dirs or .py files
    if disabled_dir.is_dir():
        for entry in disabled_dir.iterdir():
            if entry.name == name or (entry.is_file() and entry.stem == name):
                # Move back to main ext dir
                dest = ext_dir / entry.name
                entry.rename(dest)
                found = True
                break
            
    if not found:
        return f"Extension `{name}` not found in disabled directory."
    
    return f"Enabled extension `{name}`. Run `/extension load {name}` to load it now."

def _cmd_disable(agent: Any, name: str) -> str:
    ext_dir = _ext_dir(agent)
    disabled_dir = _disabled_ext_dir(agent)
    disabled_dir.mkdir(exist_ok=True)
    
    # Find the extension on disk
    found_entry = None
    for entry in ext_dir.iterdir():
        if entry.name == name or (entry.is_file() and entry.stem == name):
            found_entry = entry
            break
            
    if found_entry is None:
        return f"Extension `{name}` not found in extensions directory."
    
    # Unload from bus if loaded
    if name in agent.bus.loaded_extensions():
        unload_one(agent.bus, name)
        agent._rebuild_external_tools()

    dest = disabled_dir / found_entry.name
    found_entry.rename(dest)
    return f"Disabled extension `{name}`. It will not be loaded on startup."

def _cmd_remove(agent: Any, name: str) -> str:
    ext_dir = _ext_dir(agent)
    disabled_dir = _disabled_ext_dir(agent)
    
    # Try to find in active or disabled
    found_entry = None
    for d in [ext_dir, disabled_dir]:
        if not d.is_dir(): continue
        for entry in d.iterdir():
            if entry.name == name or (entry.is_file() and entry.stem == name):
                found_entry = entry
                break
        if found_entry: break
        
    if found_entry is None:
        return f"Extension `{name}` not found."
    
    # Unload if loaded
    if name in agent.bus.loaded_extensions():
        unload_one(agent.bus, name)
        agent._rebuild_external_tools()
        
    import shutil
    if found_entry.is_dir():
        shutil.rmtree(found_entry)
    else:
        found_entry.unlink()
        
    return f"Removed extension `{name}`."
