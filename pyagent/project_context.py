from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .user_runtime import (
    resolve_user_dir,
    user_agents_file,
    user_skills_dir,
)

MAX_PROJECT_CONTEXT_FILE_CHARS = 12_000
MAX_PROJECT_CONTEXT_TOTAL_CHARS = 30_000
MAX_USER_GLOBAL_CONTEXT_TOTAL_CHARS = 15_000

GLOBAL_SCOPE = "global"
PROJECT_SCOPE = "project"


@dataclass(frozen=True, slots=True)
class ContextSource:
    """A single instruction file loaded into the system prompt."""

    scope: str
    path: Path
    label: str

    def display_path(self) -> str:
        return self.label


def discover_project_instruction_files(cwd: str | Path) -> list[Path]:
    base = Path(cwd).resolve()
    candidates: list[Path] = []

    agents_file = base / "AGENTS.md"
    if agents_file.is_file():
        candidates.append(agents_file)

    for pattern in ("*.skill", "skills/**/*.md", "skills/**/*.skill"):
        for path in sorted(base.glob(pattern)):
            if path.is_file():
                candidates.append(path)

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(resolved)

    return unique_candidates


def discover_user_global_instruction_files(user_dir: str | Path | None = None) -> list[Path]:
    base = (
        Path(user_dir).expanduser().resolve()
        if user_dir is not None
        else resolve_user_dir()
    )
    candidates: list[Path] = []

    agents_file = user_agents_file(base)
    if agents_file.is_file():
        candidates.append(agents_file)

    skills_root = user_skills_dir(base)
    if skills_root.is_dir():
        for pattern in ("*.md", "*.skill", "**/*.md", "**/*.skill"):
            for path in sorted(skills_root.glob(pattern)):
                if path.is_file():
                    candidates.append(path)

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _truncate_content(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[truncated {omitted} characters]", True


def _label_for(path: Path, scope: str, base: Path) -> str:
    try:
        relative = path.relative_to(base).as_posix()
    except ValueError:
        relative = path.name
    if scope == GLOBAL_SCOPE:
        return f"~/.pyagent/{relative}"
    return relative


def _render_section(
    *,
    files: Iterable[Path],
    scope: str,
    base: Path,
    budget: int,
    section_heading: str,
    section_intro: str,
) -> tuple[str, list[ContextSource], int]:
    files = list(files)
    if not files:
        return "", [], 0

    total_chars = 0
    sources: list[ContextSource] = []
    body: list[str] = [f"# {section_heading}", section_intro]

    truncated_section = False
    for path in files:
        label = _label_for(path, scope, base)
        try:
            raw_text = path.read_text(encoding="utf-8")
        except OSError as exc:
            body.append(f"## {label}\n\n[unreadable: {exc}]")
            sources.append(ContextSource(scope=scope, path=path, label=label))
            continue

        remaining = budget - total_chars
        if remaining <= 0:
            truncated_section = True
            break

        per_file_limit = min(MAX_PROJECT_CONTEXT_FILE_CHARS, remaining)
        content, truncated = _truncate_content(
            raw_text.strip(), per_file_limit
        )
        body.append(
            f"## {label}\n\n{content if content else '[empty file]'}"
        )
        total_chars += len(content)
        sources.append(ContextSource(scope=scope, path=path, label=label))

        if truncated and per_file_limit == remaining:
            truncated_section = True
            break

    if truncated_section:
        body.append(
            f"\nAdditional {scope} instruction files were omitted because the {scope} "
            "context budget was reached."
        )

    return "\n\n".join(body), sources, total_chars


def load_full_context(
    cwd: str | Path,
    *,
    user_dir: str | Path | None = None,
) -> tuple[str, list[ContextSource]]:
    """Load both user-global and project-local instructions.

    Returns the composed system-prompt body and a list of
    :class:`ContextSource` describing each loaded file (with scope).
    Global context is rendered before project context so project files
    layer on top.
    """
    project_base = Path(cwd).resolve()
    resolved_user_dir = (
        Path(user_dir).expanduser().resolve()
        if user_dir is not None
        else resolve_user_dir()
    )

    sections: list[str] = []
    sources: list[ContextSource] = []

    global_files = discover_user_global_instruction_files(resolved_user_dir)
    global_text, global_sources, _ = _render_section(
        files=global_files,
        scope=GLOBAL_SCOPE,
        base=resolved_user_dir,
        budget=MAX_USER_GLOBAL_CONTEXT_TOTAL_CHARS,
        section_heading="User-global instructions",
        section_intro=(
            "User-global instructions live under `~/.pyagent/` and apply to every "
            "PyAgent session for this user. Project-specific instructions, when "
            "present, layer on top and may override these defaults."
        ),
    )
    if global_text:
        sections.append(global_text)
        sources.extend(global_sources)

    project_files = discover_project_instruction_files(project_base)
    project_text, project_sources, _ = _render_section(
        files=project_files,
        scope=PROJECT_SCOPE,
        base=project_base,
        budget=MAX_PROJECT_CONTEXT_TOTAL_CHARS,
        section_heading="Project-specific instructions",
        section_intro=(
            "Project-specific instructions are loaded from local instruction files in the current working "
            "directory. Follow them in addition to the base system prompt and any user-global instructions."
        ),
    )
    if project_text:
        sections.append(project_text)
        sources.extend(project_sources)

    if not sections:
        return "", []

    return "\n\n".join(sections), sources


def load_project_context(
    cwd: str | Path,
    *,
    user_dir: str | Path | None = None,
) -> tuple[str, list[str]]:
    """Compatibility shim that returns ``(text, [labelled paths])``.

    The labels include a ``~/.pyagent/`` prefix for user-global sources
    and a project-relative path for project-local sources, so the UI can
    distinguish them at a glance. ``user_dir`` is mostly intended for
    tests; production callers can rely on the ``PYAGENT_USER_DIR`` env
    var resolved by :func:`pyagent.user_runtime.resolve_user_dir`.
    """
    text, sources = load_full_context(cwd, user_dir=user_dir)
    return text, [source.label for source in sources]
