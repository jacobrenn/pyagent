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
MAX_SKILL_PREVIEW_CHARS = 180

GLOBAL_SCOPE = "global"
PROJECT_SCOPE = "project"
USER_SKILL_SCOPE = "user"
PROJECT_SKILL_SCOPE = "project"


@dataclass(frozen=True, slots=True)
class ContextSource:
    """A single instruction file loaded into the system prompt."""

    scope: str
    path: Path
    label: str

    def display_path(self) -> str:
        return self.label


@dataclass(frozen=True, slots=True)
class LoadedSkill:
    path: Path
    label: str

    def __str__(self):
        return self.label


@dataclass(frozen=True, slots=True)
class AvailableSkill:
    """A skill file available for explicit or tool-driven loading."""

    scope: str
    path: Path
    label: str
    title: str
    preview: str
    error: str | None = None

    @property
    def id(self) -> str:
        return f"{self.scope}:{self.label}"

    def __str__(self) -> str:
        return self.id


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def discover_project_instruction_files(cwd: str | Path) -> list[Path]:
    """Return project files that load into the system prompt by default.

    Project-local skills are intentionally excluded. They are available through
    the skill catalog and can be loaded explicitly by the user or by the model's
    skill tools.
    """
    base = Path(cwd).resolve()
    agents_file = base / "AGENTS.md"
    return [agents_file.resolve()] if agents_file.is_file() else []


def discover_project_skill_files(cwd: str | Path) -> list[Path]:
    """Return project-local skill files available for explicit loading."""
    base = Path(cwd).resolve()
    candidates: list[Path] = []
    for pattern in ("*.skill", "skills/**/*.md", "skills/**/*.skill"):
        for path in sorted(base.glob(pattern)):
            if path.is_file():
                candidates.append(path)
    return _unique_paths(candidates)


def discover_user_global_instruction_files(user_dir: str | Path | None = None) -> list[Path]:
    base = (
        Path(user_dir).expanduser().resolve()
        if user_dir is not None
        else resolve_user_dir()
    )
    agents_file = user_agents_file(base)
    return [agents_file.resolve()] if agents_file.is_file() else []


def discover_user_skill_files(user_dir: str | Path | None = None) -> list[Path]:
    base = (
        Path(user_dir).expanduser().resolve()
        if user_dir is not None
        else resolve_user_dir()
    )
    skills_root = user_skills_dir(base)
    if not skills_root.is_dir():
        return []

    candidates: list[Path] = []
    for pattern in ("*.md", "*.skill", "**/*.md", "**/*.skill"):
        for path in sorted(skills_root.glob(pattern)):
            if not path.is_file():
                continue
            try:
                rel_parts = path.relative_to(skills_root).parts
            except ValueError:
                rel_parts = ()
            if rel_parts[:1] == ("extensions",):
                continue
            candidates.append(path)

    return _unique_paths(candidates)


def list_user_skills(user_dir: str | Path | None = None) -> list[LoadedSkill]:
    base = (
        Path(user_dir).expanduser().resolve()
        if user_dir is not None
        else resolve_user_dir()
    )
    skills_root = user_skills_dir(base)
    skills: list[LoadedSkill] = []
    for path in discover_user_skill_files(base):
        try:
            relative = path.relative_to(skills_root).as_posix()
        except ValueError:
            relative = path.name
        skills.append(LoadedSkill(path=path, label=relative))
    return skills


def resolve_user_skill(skill_name: str, user_dir: str | Path | None = None) -> LoadedSkill | None:
    wanted = skill_name.strip().strip("/")
    if wanted.startswith(f"{USER_SKILL_SCOPE}:"):
        wanted = wanted.split(":", 1)[1]
    if not wanted or Path(wanted).is_absolute():
        return None
    normalized = Path(wanted).as_posix()
    for skill in list_user_skills(user_dir):
        if skill.label == normalized:
            return skill
    return None


def _relative_label(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.name


def _skill_title_and_preview(path: Path) -> tuple[str, str, str | None]:
    title = path.name
    preview = ""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return title, preview, str(exc)
    except UnicodeDecodeError as exc:
        return title, preview, str(exc)

    first_body_line = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            if heading and title == path.name:
                title = heading
            continue
        if not first_body_line:
            first_body_line = line
            break

    if first_body_line:
        preview = first_body_line
    else:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                preview = line
                break

    if len(preview) > MAX_SKILL_PREVIEW_CHARS:
        preview = f"{preview[:MAX_SKILL_PREVIEW_CHARS]}..."
    return title, preview, None


def list_project_skills(cwd: str | Path) -> list[AvailableSkill]:
    base = Path(cwd).resolve()
    skills: list[AvailableSkill] = []
    for path in discover_project_skill_files(base):
        label = _relative_label(path, base)
        title, preview, error = _skill_title_and_preview(path)
        skills.append(
            AvailableSkill(
                scope=PROJECT_SKILL_SCOPE,
                path=path,
                label=label,
                title=title,
                preview=preview,
                error=error,
            )
        )
    return skills


def list_available_skills(
    cwd: str | Path,
    *,
    user_dir: str | Path | None = None,
) -> list[AvailableSkill]:
    """List user and project skills available for explicit/tool loading."""
    resolved_user_dir = (
        Path(user_dir).expanduser().resolve()
        if user_dir is not None
        else resolve_user_dir()
    )
    skills_root = user_skills_dir(resolved_user_dir)
    available: list[AvailableSkill] = []

    for loaded in list_user_skills(resolved_user_dir):
        title, preview, error = _skill_title_and_preview(loaded.path)
        available.append(
            AvailableSkill(
                scope=USER_SKILL_SCOPE,
                path=loaded.path,
                label=loaded.label,
                title=title,
                preview=preview,
                error=error,
            )
        )

    available.extend(list_project_skills(cwd))
    available.sort(key=lambda skill: (skill.scope, skill.label))
    return available


def resolve_available_skill(
    skill_name: str,
    cwd: str | Path,
    *,
    user_dir: str | Path | None = None,
) -> AvailableSkill | None:
    """Resolve a scoped or backward-compatible unscoped skill reference."""
    wanted = skill_name.strip().strip("/")
    if not wanted:
        return None

    scoped_scope: str | None = None
    scoped_label = wanted
    if ":" in wanted:
        prefix, rest = wanted.split(":", 1)
        if prefix in {USER_SKILL_SCOPE, PROJECT_SKILL_SCOPE}:
            scoped_scope = prefix
            scoped_label = rest.strip().strip("/")

    if not scoped_label or Path(scoped_label).is_absolute():
        return None
    normalized = Path(scoped_label).as_posix()

    available = list_available_skills(cwd, user_dir=user_dir)
    if scoped_scope is not None:
        for skill in available:
            if skill.scope == scoped_scope and skill.label == normalized:
                return skill
        return None

    # Backward compatibility: unscoped names resolve to user skills first, then
    # to a unique project skill label.
    for skill in available:
        if skill.scope == USER_SKILL_SCOPE and skill.label == normalized:
            return skill

    project_matches = [
        skill
        for skill in available
        if skill.scope == PROJECT_SKILL_SCOPE and skill.label == normalized
    ]
    if len(project_matches) == 1:
        return project_matches[0]
    return None


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
        except UnicodeDecodeError as exc:
            body.append(f"## {label}\n\n[unreadable: {exc}]")
            sources.append(ContextSource(scope=scope, path=path, label=label))
            continue

        remaining = budget - total_chars
        if remaining <= 0:
            truncated_section = True
            break

        per_file_limit = min(MAX_PROJECT_CONTEXT_FILE_CHARS, remaining)
        content, truncated = _truncate_content(raw_text.strip(), per_file_limit)
        body.append(f"## {label}\n\n{content if content else '[empty file]'}")
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


def _split_loaded_skill_paths(
    skill_refs: Iterable[str | Path],
    *,
    cwd: Path,
    user_dir: Path,
) -> tuple[list[Path], list[Path]]:
    user_skill_paths: list[Path] = []
    project_skill_paths: list[Path] = []

    for skill_ref in skill_refs:
        if isinstance(skill_ref, Path):
            resolved = skill_ref.expanduser().resolve()
            if not resolved.is_file():
                continue
            try:
                resolved.relative_to(cwd)
            except ValueError:
                user_skill_paths.append(resolved)
            else:
                project_skill_paths.append(resolved)
            continue

        skill = resolve_available_skill(str(skill_ref), cwd, user_dir=user_dir)
        if skill is None:
            continue
        if skill.scope == USER_SKILL_SCOPE:
            user_skill_paths.append(skill.path)
        else:
            project_skill_paths.append(skill.path)

    return _unique_paths(user_skill_paths), _unique_paths(project_skill_paths)


def load_full_context(
    cwd: str | Path,
    *,
    user_dir: str | Path | None = None,
    loaded_user_skills: Iterable[str | Path] | None = None,
) -> tuple[str, list[ContextSource]]:
    """Load always-on instructions plus explicitly selected skills.

    Startup context includes ``~/.pyagent/AGENTS.md`` and project ``AGENTS.md``
    by default. User and project skills are available through the skill catalog,
    but load into the system prompt only when explicitly listed in
    ``loaded_user_skills`` (kept as the parameter name for compatibility).
    """
    project_base = Path(cwd).resolve()
    resolved_user_dir = (
        Path(user_dir).expanduser().resolve()
        if user_dir is not None
        else resolve_user_dir()
    )

    sections: list[str] = []
    sources: list[ContextSource] = []

    loaded_user_skill_paths, loaded_project_skill_paths = _split_loaded_skill_paths(
        loaded_user_skills or [], cwd=project_base, user_dir=resolved_user_dir
    )

    global_files = _unique_paths(
        [*discover_user_global_instruction_files(resolved_user_dir), *loaded_user_skill_paths]
    )
    global_text, global_sources, _ = _render_section(
        files=global_files,
        scope=GLOBAL_SCOPE,
        base=resolved_user_dir,
        budget=MAX_USER_GLOBAL_CONTEXT_TOTAL_CHARS,
        section_heading="User-global instructions",
        section_intro=(
            "User-global instructions live under `~/.pyagent/` and apply to "
            "PyAgent sessions for this user. `~/.pyagent/AGENTS.md` loads by "
            "default; user skills under `~/.pyagent/skills/` load into the "
            "system prompt only when explicitly enabled. Project-specific "
            "instructions, when present, layer on top and may override these "
            "defaults."
        ),
    )
    if global_text:
        sections.append(global_text)
        sources.extend(global_sources)

    project_files = _unique_paths(
        [*discover_project_instruction_files(project_base), *loaded_project_skill_paths]
    )
    project_text, project_sources, _ = _render_section(
        files=project_files,
        scope=PROJECT_SCOPE,
        base=project_base,
        budget=MAX_PROJECT_CONTEXT_TOTAL_CHARS,
        section_heading="Project-specific instructions",
        section_intro=(
            "Project-specific instructions are loaded from `AGENTS.md` and any "
            "project skills explicitly loaded into the current session. Other "
            "project skills are discoverable through the `list_skills` tool and "
            "can be loaded on demand with `load_skills`."
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
    loaded_user_skills: Iterable[str | Path] | None = None,
) -> tuple[str, list[str]]:
    """Compatibility shim that returns ``(text, [labelled paths])``."""
    text, sources = load_full_context(
        cwd,
        user_dir=user_dir,
        loaded_user_skills=loaded_user_skills,
    )
    return text, [source.label for source in sources]
