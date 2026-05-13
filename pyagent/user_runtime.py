from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
from pathlib import Path


DEFAULT_USER_DIR = "~/.pyagent"
USER_DIR_ENV_VAR = "PYAGENT_USER_DIR"
TOOL_RUNNER_ENV_VAR = "PYAGENT_TOOL_RUNNER"
DEFAULT_TOOL_RUNNER = "uv"


def resolve_user_dir(override: str | os.PathLike[str] | None = None) -> Path:
    """Return the absolute path to the user-global ``~/.pyagent/`` directory.

    Resolution order:
    1. The ``override`` argument if provided.
    2. ``PYAGENT_USER_DIR`` env var if set and non-empty.
    3. ``~/.pyagent`` (matching the default ``models.json`` location).

    The path is expanded but not created. Callers that need a guaranteed
    directory should use :func:`ensure_user_subdir`.
    """
    raw = (
        str(override)
        if override is not None
        else os.environ.get(USER_DIR_ENV_VAR, "").strip() or DEFAULT_USER_DIR
    )
    return Path(os.path.expanduser(raw)).resolve()


def user_skills_dir(user_dir: Path | None = None) -> Path:
    return (user_dir or resolve_user_dir()) / "skills"


def user_tools_dir(user_dir: Path | None = None) -> Path:
    return (user_dir or resolve_user_dir()) / "tools"


def user_agents_file(user_dir: Path | None = None) -> Path:
    return (user_dir or resolve_user_dir()) / "AGENTS.md"


def user_tools_cache_dir(user_dir: Path | None = None) -> Path:
    return user_tools_dir(user_dir) / ".cache"


def user_tools_disabled_dir(user_dir: Path | None = None) -> Path:
    return user_tools_dir(user_dir) / "disabled"


def user_log_dir(user_dir: Path | None = None) -> Path:
    return (user_dir or resolve_user_dir()) / "logs"


def ensure_user_subdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True, slots=True)
class RunnerStatus:
    name: str
    available: bool
    executable: str | None
    message: str | None = None


def resolve_tool_runner(override: str | None = None) -> str:
    """Return the runner command name (currently ``uv`` only).

    The plan locks the contract to ``uv`` for clarity. We still read the env
    var so a future runner can be added without touching call sites.
    """
    raw = (
        override
        if override is not None
        else os.environ.get(TOOL_RUNNER_ENV_VAR, "").strip() or DEFAULT_TOOL_RUNNER
    )
    normalized = raw.strip().lower()
    if normalized != "uv":
        return DEFAULT_TOOL_RUNNER
    return DEFAULT_TOOL_RUNNER


def check_runner_available(runner: str | None = None) -> RunnerStatus:
    name = runner or resolve_tool_runner()
    executable = shutil.which(name)
    if executable:
        return RunnerStatus(name=name, available=True, executable=executable)
    return RunnerStatus(
        name=name,
        available=False,
        executable=None,
        message=(
            f"`{name}` was not found on PATH. External tools in "
            f"~/.pyagent/tools/ are disabled until `{name}` is installed. "
            "See https://docs.astral.sh/uv/ for installation instructions."
        ),
    )
