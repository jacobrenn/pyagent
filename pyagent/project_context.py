from __future__ import annotations

from pathlib import Path

MAX_PROJECT_CONTEXT_FILE_CHARS = 12_000
MAX_PROJECT_CONTEXT_TOTAL_CHARS = 30_000


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


def _truncate_content(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[truncated {omitted} characters]", True


def load_project_context(cwd: str | Path) -> tuple[str, list[str]]:
    base = Path(cwd).resolve()
    files = discover_project_instruction_files(base)
    if not files:
        return "", []

    sections: list[str] = [
        "Project-specific instructions are loaded from local instruction files in the current working directory. "
        "Follow them in addition to the base system prompt."
    ]
    total_chars = 0
    loaded_paths: list[str] = []

    for path in files:
        rel_path = path.relative_to(base).as_posix()
        try:
            raw_text = path.read_text(encoding="utf-8")
        except Exception as exc:
            sections.append(f"## {rel_path}\n\n[unreadable: {exc}]")
            loaded_paths.append(rel_path)
            continue

        remaining = MAX_PROJECT_CONTEXT_TOTAL_CHARS - total_chars
        if remaining <= 0:
            sections.append(
                "\nAdditional project instruction files were omitted because the total context budget was reached.")
            break

        per_file_limit = min(MAX_PROJECT_CONTEXT_FILE_CHARS, remaining)
        content, truncated = _truncate_content(
            raw_text.strip(), per_file_limit)
        sections.append(
            f"## {rel_path}\n\n{content if content else '[empty file]'}")
        total_chars += len(content)
        loaded_paths.append(rel_path)

        if truncated and per_file_limit == remaining:
            sections.append(
                "\nAdditional project instruction files were omitted because the total context budget was reached.")
            break

    return "\n\n".join(sections), loaded_paths
