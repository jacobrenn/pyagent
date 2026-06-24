"""Discover and load extensions from ``~/.pyagent/extensions/``.

A plain directory scan + ``importlib``.
``load_one`` imports a module and calls its ``register(bus, name)``;
``unload_one`` removes its handlers by name. A failing extension is logged
and skipped — it never blocks startup.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Sequence

from ..user_runtime import resolve_user_dir, user_extensions_dir, user_skills_dir
from .bus import EventBus

MAX_EXTENSION_SKILLS_CHARS = 15_000


def extension_tools_dir(ext_dir: Path, name: str) -> Path:
    """The ``tools/`` subdir of a colocated package extension."""
    return ext_dir / name / "tools"


def extension_skills_dir(ext_dir: Path, name: str) -> Path:
    """The ``skills/`` subdir of a colocated package extension."""
    return ext_dir / name / "skills"


def loaded_extension_tool_dirs(
    ext_dir: Path, loaded_names: Sequence[str]
) -> list[Path]:
    """Tool dirs of *loaded* package extensions that actually have a tools dir.

    This is the gate that keeps an unloaded extension's tools undiscoverable:
    only loaded extensions appear here, so discovery never scans their
    ``tools/`` subdir until they are loaded.
    """
    dirs: list[Path] = []
    for name in loaded_names:
        tools_dir = extension_tools_dir(ext_dir, name)
        if tools_dir.is_dir():
            dirs.append(tools_dir)
    return dirs


def _discover(ext_dir: Path) -> list[str]:
    """On-disk extension names (packages with ``__init__.py`` or single ``.py``)."""
    if not ext_dir.is_dir():
        return []
    names: list[str] = []
    for entry in sorted(ext_dir.iterdir()):
        name = entry.name
        if name.startswith(".") or name == "__pycache__":
            continue
        if entry.is_dir() and (entry / "__init__.py").exists():
            names.append(name)
        elif entry.is_file() and entry.suffix == ".py":
            names.append(entry.stem)
    return names


def _import_ext(name: str, ext_dir: Path) -> Any:
    dir_str = str(ext_dir)
    if dir_str not in sys.path:
        sys.path.insert(0, dir_str)
    sys.modules.pop(name, None)
    importlib.invalidate_caches()

    pkg = ext_dir / name / "__init__.py"
    if pkg.exists():
        spec = importlib.util.spec_from_file_location(
            f"pyagent_ext_{name}", pkg,
            submodule_search_locations=[str(ext_dir / name)],
        )
    else:
        spec = importlib.util.spec_from_file_location(
            f"pyagent_ext_{name}", ext_dir / f"{name}.py"
        )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build spec for extension {name!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_one(bus: EventBus, name: str, ext_dir: Path, log: Any) -> None:
    """Import ``name`` and wire its handlers. Idempotent (clears prior handlers)."""
    bus.off_extension(name)
    module = _import_ext(name, ext_dir)
    register = getattr(module, "register", None)
    if not callable(register):
        raise AttributeError(
            f"extension {name!r} must define register(bus, name)"
        )
    register(bus.scoped(name), name)


def unload_one(bus: EventBus, name: str) -> None:
    bus.off_extension(name)


def load_all(bus: EventBus, ext_dir: Path, log: Any) -> tuple[list[str], list[tuple[str, str]]]:
    """Scan ``ext_dir`` and load every discovered extension.

    Returns ``(loaded, failed)`` where ``failed`` is ``[(name, error)]``.
    """
    loaded: list[str] = []
    failed: list[tuple[str, str]] = []
    for name in _discover(ext_dir):
        try:
            load_one(bus, name, ext_dir, log)
            loaded.append(name)
        except Exception as exc:
            failed.append((name, str(exc)))
            _log_error(log, f"extension {name} failed to load", {
                       "error": str(exc)})
    return loaded, failed


def _log_error(log: Any, body: str, attrs: dict[str, Any]) -> None:
    err = getattr(log, "error", None)
    if callable(err):
        err(body, attrs)


def collect_skill_text(agent: Any) -> str:
    """Build the system-prompt suffix for skills active THIS turn.

    Reads ``agent._this_skills`` (composite ``"ext/key"`` entries declared this
    turn at ``input``/``before_agent_start``/``turn_start``, or rotated in from
    last turn's ``turn_end`` declarations) and appends each matching skill
    Markdown file. Skills live colocated with their extension under
    ``~/.pyagent/extensions/<ext>/skills/<key>.md``; the legacy
    ``~/.pyagent/skills/extensions/<key>.md`` location is a fallback so
    existing files keep working until moved.

    A total char budget guards against bloat; missing/unreadable files are
    skipped (fault isolation). Called from ``Agent._refresh_system_message``
    every turn/iteration.
    """
    keys = getattr(agent, "_this_skills", None)
    if not keys:
        return ""
    user_dir = resolve_user_dir(agent.config.user_dir)
    ext_dir = user_extensions_dir(user_dir)
    legacy_skills_dir = user_skills_dir(user_dir) / "extensions"
    chunks: list[str] = []
    used = 0
    for entry in sorted(keys):
        ext, key = _split_skill_entry(entry)
        if not ext or not key:
            continue
        path = extension_skills_dir(ext_dir, ext) / f"{key}.md"
        if not path.is_file():
            path = legacy_skills_dir / f"{key}.md"
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not text:
            continue
        if used + len(text) > MAX_EXTENSION_SKILLS_CHARS:
            remaining = MAX_EXTENSION_SKILLS_CHARS - used
            if remaining <= 0:
                break
            text = text[:remaining] + \
                f"\n\n[truncated {len(text) - remaining} characters]"
        chunk = f"# Extension skill: {key}\n\n{text}"
        chunks.append(chunk)
        used += len(chunk)
    return "\n\n---\n\n".join(chunks)


def _split_skill_entry(entry: Any) -> tuple[str, str]:
    """Split a composite ``"ext/key"`` skill entry into ``(ext, key)``.

    Bare keys (no slash) are tolerated and resolve to an empty extension, which
    means they will be skipped under the colocated model. This keeps the agent
    loop robust if an extension ever declares a skill before its name is known.
    """
    text = str(entry)
    if "/" in text:
        ext, key = text.split("/", 1)
        return ext.strip(), key.strip()
    return "", text.strip()
