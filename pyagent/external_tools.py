from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

from .tools import MAX_TOOL_OUTPUT_CHARS, ToolSpec, _truncate
from .user_runtime import (
    RunnerStatus,
    check_runner_available,
    ensure_user_subdir,
    resolve_tool_runner,
    resolve_user_dir,
    user_tools_cache_dir,
    user_tools_dir,
    user_tools_disabled_dir,
)


CACHE_FILE_NAME = "manifests.json"
CACHE_VERSION = 1
DEFAULT_DESCRIBE_TIMEOUT_SECONDS = 10.0
DEFAULT_INVOKE_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class ExternalToolManifest:
    name: str
    description: str
    parameters: dict[str, Any]
    version: str
    script_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "version": self.version,
            "script_path": self.script_path,
        }

    @classmethod
    def from_describe_payload(cls, payload: Any, script_path: Path) -> "ExternalToolManifest":
        if not isinstance(payload, dict):
            raise ValueError("describe output must be a JSON object")

        name = str(payload.get("name") or "").strip()
        if not name:
            raise ValueError("describe output is missing a non-empty 'name'")

        description = str(payload.get("description") or "").strip()
        if not description:
            raise ValueError(
                "describe output is missing a non-empty 'description'")

        parameters = payload.get("parameters") or {
            "type": "object", "properties": {}}
        if not isinstance(parameters, dict):
            raise ValueError(
                "describe output 'parameters' must be a JSON object")

        version = str(payload.get("version") or "1").strip() or "1"

        return cls(
            name=name,
            description=description,
            parameters=parameters,
            version=version,
            script_path=str(script_path),
        )


@dataclass(slots=True)
class ExternalToolEntry:
    script_path: Path
    manifest: ExternalToolManifest | None = None
    error: str | None = None
    disabled: bool = False
    collision: bool = False

    @property
    def is_loaded(self) -> bool:
        return self.manifest is not None and not self.disabled and self.error is None


@dataclass(slots=True)
class DiscoveryResult:
    user_dir: Path
    tools_dir: Path
    runner: str
    runner_available: bool
    runner_message: str | None
    loaded: list[ExternalToolEntry] = field(default_factory=list)
    disabled: list[ExternalToolEntry] = field(default_factory=list)
    broken: list[ExternalToolEntry] = field(default_factory=list)

    def all_entries(self) -> list[ExternalToolEntry]:
        return [*self.loaded, *self.broken, *self.disabled]


def _file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns}-{stat.st_size}"


def _cache_key(path: Path, runner: str, runner_command: Sequence[str]) -> str:
    canonical = f"{path.resolve()}|{_file_fingerprint(path)}|{runner}|{'|'.join(runner_command)}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.is_file():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return {}
    return entries


def _save_cache(cache_path: Path, entries: dict[str, dict[str, Any]]) -> None:
    try:
        ensure_user_subdir(cache_path.parent)
        cache_path.write_text(
            json.dumps({"version": CACHE_VERSION,
                       "entries": entries}, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        # A read-only home or a permissions error must not break startup.
        return


def default_runner_command(runner: str | None = None) -> list[str]:
    """Return the prefix tokens used to invoke a script with the runner.

    For ``uv`` we use ``uv run <script>``. The contract is intentionally
    locked to ``uv`` for the moment but the prefix is a list so a future
    runner can be substituted via ``runner_command`` overrides.
    """
    name = runner or resolve_tool_runner()
    if name == "uv":
        return ["uv", "run"]
    return [name]


def _build_command(
    runner_command: Sequence[str],
    script_path: Path,
    subcommand: str,
    extra: Sequence[str] = (),
) -> list[str]:
    return [*runner_command, str(script_path), subcommand, *extra]


def _run_subprocess(
    command: Sequence[str],
    timeout: float,
    *,
    cwd: str | os.PathLike[str] | None = None,
) -> tuple[int, str, str, bool]:
    """Run a subprocess with a wall-clock timeout.

    Returns ``(returncode, stdout, stderr, timed_out)``. On timeout the
    process is killed and ``timed_out`` is True.
    """
    try:
        proc = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            cwd=str(cwd) if cwd is not None else None,
            text=True,
        )
    except FileNotFoundError as exc:
        return -1, "", f"runner not found: {exc}", False
    except OSError as exc:
        return -1, "", f"could not start runner: {exc}", False

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            stdout, stderr = proc.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        return proc.returncode if proc.returncode is not None else -1, stdout, stderr, True

    return proc.returncode, stdout, stderr, False


def _candidate_scripts(tools_dir: Path) -> list[Path]:
    if not tools_dir.is_dir():
        return []
    candidates: list[Path] = []
    for entry in sorted(tools_dir.iterdir()):
        name = entry.name
        if name.startswith(".") or name == "__pycache__":
            continue
        if not entry.is_file():
            continue
        if entry.suffix != ".py":
            continue
        candidates.append(entry)
    return candidates


def _disabled_scripts(tools_dir: Path) -> list[Path]:
    disabled_dir = tools_dir / "disabled"
    if not disabled_dir.is_dir():
        return []
    scripts: list[Path] = []
    for entry in sorted(disabled_dir.iterdir()):
        if entry.suffix != ".py" or not entry.is_file():
            continue
        scripts.append(entry)
    return scripts


def _describe_script(
    script_path: Path,
    *,
    runner_command: Sequence[str],
    timeout: float,
) -> tuple[ExternalToolManifest | None, str | None]:
    command = _build_command(runner_command, script_path, "describe")
    returncode, stdout, stderr, timed_out = _run_subprocess(
        command, timeout=timeout)

    if timed_out:
        return None, f"`describe` timed out after {timeout:.0f}s"
    if returncode != 0:
        message = stderr.strip() or stdout.strip() or f"exit code {returncode}"
        return None, f"`describe` failed (exit {returncode}): {message}"

    text = stdout.strip()
    if not text:
        return None, "`describe` produced no JSON on stdout"

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        preview = text[:200]
        return None, f"`describe` did not emit valid JSON: {exc}: {preview!r}"

    try:
        return ExternalToolManifest.from_describe_payload(payload, script_path), None
    except ValueError as exc:
        return None, f"`describe` payload invalid: {exc}"


def discover_external_tools(
    *,
    user_dir: Path | str | None = None,
    runner: str | None = None,
    runner_command: Sequence[str] | None = None,
    runner_status: RunnerStatus | None = None,
    describe_timeout: float = DEFAULT_DESCRIBE_TIMEOUT_SECONDS,
    cache_enabled: bool = True,
) -> DiscoveryResult:
    """Scan ``~/.pyagent/tools/`` and return manifests for each script.

    The runner check is gentle: when the runner is missing we still walk
    the directory so users can see their tools listed (with a clear error
    explaining how to install the runner). We just skip ``describe``.

    The describe output is cached on disk keyed by
    ``(script path, mtime, size, runner)`` so repeated TUI startups do
    not re-run the (potentially slow) ``uv run`` for unchanged scripts.
    """
    resolved_user_dir = (
        Path(user_dir).expanduser().resolve(
        ) if user_dir is not None else resolve_user_dir()
    )
    tools_dir = user_tools_dir(resolved_user_dir)
    runner_name = runner or resolve_tool_runner()
    command_prefix = list(
        runner_command) if runner_command is not None else default_runner_command(runner_name)
    status = runner_status or check_runner_available(runner_name)
    cache_path = user_tools_cache_dir(resolved_user_dir) / CACHE_FILE_NAME

    result = DiscoveryResult(
        user_dir=resolved_user_dir,
        tools_dir=tools_dir,
        runner=runner_name,
        runner_available=status.available,
        runner_message=status.message,
    )

    if not tools_dir.is_dir():
        return result

    cache = _load_cache(cache_path) if cache_enabled else {}
    fresh_cache: dict[str, dict[str, Any]] = {}

    for script_path in _candidate_scripts(tools_dir):
        if not status.available:
            result.broken.append(
                ExternalToolEntry(
                    script_path=script_path,
                    error=status.message or f"runner `{runner_name}` is not available",
                )
            )
            continue

        cache_key = _cache_key(script_path, runner_name, command_prefix)
        cached = cache.get(cache_key) if cache_enabled else None
        manifest: ExternalToolManifest | None = None
        error: str | None = None

        if cached:
            try:
                manifest = ExternalToolManifest.from_describe_payload(
                    {
                        "name": cached.get("name"),
                        "description": cached.get("description"),
                        "parameters": cached.get("parameters"),
                        "version": cached.get("version"),
                    },
                    script_path,
                )
            except ValueError:
                manifest = None

        if manifest is None:
            manifest, error = _describe_script(
                script_path,
                runner_command=command_prefix,
                timeout=describe_timeout,
            )

        if manifest is None:
            result.broken.append(
                ExternalToolEntry(script_path=script_path,
                                  error=error or "unknown describe failure")
            )
            continue

        fresh_cache[cache_key] = manifest.to_dict()
        result.loaded.append(ExternalToolEntry(
            script_path=script_path, manifest=manifest))

    for script_path in _disabled_scripts(tools_dir):
        result.disabled.append(
            ExternalToolEntry(script_path=script_path, disabled=True)
        )

    if cache_enabled and (fresh_cache or cache):
        _save_cache(cache_path, fresh_cache)

    return result


class ExternalToolHandler:
    """Callable wrapper that invokes an external tool script.

    Bound into a :class:`pyagent.tools.ToolSpec` and called with the
    tool arguments the model produced. The arguments are written to a
    tempfile to avoid quoting issues and command-line length limits.
    """

    def __init__(
        self,
        script_path: Path,
        *,
        runner_command: Sequence[str],
        invoke_timeout: float = DEFAULT_INVOKE_TIMEOUT_SECONDS,
        max_output_chars: int = MAX_TOOL_OUTPUT_CHARS,
    ):
        self.script_path = Path(script_path)
        self.runner_command = list(runner_command)
        self.invoke_timeout = invoke_timeout
        self.max_output_chars = max_output_chars

    def __call__(self, **arguments: Any) -> str:
        try:
            payload = json.dumps(arguments, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as exc:
            return f"Error: could not serialize tool arguments to JSON: {exc}"

        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".json", prefix="pyagent-tool-args-", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(payload)
                args_path = tmp.name
        except OSError as exc:
            return f"Error: could not write tool arguments tempfile: {exc}"

        try:
            command = _build_command(
                self.runner_command,
                self.script_path,
                "invoke",
                ["--args-file", args_path],
            )
            returncode, stdout, stderr, timed_out = _run_subprocess(
                command, timeout=self.invoke_timeout
            )
        finally:
            try:
                os.unlink(args_path)
            except OSError:
                pass

        if timed_out:
            return (
                f"Error: tool exceeded its {self.invoke_timeout:.0f}s timeout. "
                "Increase `PYAGENT_USER_TOOL_TIMEOUT` or simplify the request."
            )
        if returncode != 0:
            stderr_text = stderr.strip()
            stdout_text = stdout.strip()
            preview = stderr_text or stdout_text or f"exit code {returncode}"
            return _truncate(
                f"Error: tool failed (exit {returncode}): {preview}",
                max_chars=self.max_output_chars,
            )

        return _truncate(stdout if stdout else "<empty>", max_chars=self.max_output_chars)


def build_external_tool_specs(
    discovery: DiscoveryResult,
    *,
    invoke_timeout: float = DEFAULT_INVOKE_TIMEOUT_SECONDS,
    runner_command: Sequence[str] | None = None,
) -> list[ToolSpec]:
    """Convert successfully-discovered manifests into :class:`ToolSpec`s."""
    command_prefix = list(
        runner_command) if runner_command is not None else default_runner_command(discovery.runner)
    specs: list[ToolSpec] = []
    for entry in discovery.loaded:
        manifest = entry.manifest
        if manifest is None:
            continue
        handler = ExternalToolHandler(
            entry.script_path,
            runner_command=command_prefix,
            invoke_timeout=invoke_timeout,
        )
        specs.append(
            ToolSpec(
                name=manifest.name,
                description=manifest.description,
                parameters=manifest.parameters,
                handler=handler,
            )
        )
    return specs


def move_tool_script(
    name: str,
    *,
    user_dir: Path | str | None = None,
    enable: bool,
) -> tuple[Path | None, str | None]:
    """Move ``<name>.py`` into or out of ``tools/disabled/``.

    Returns ``(new_path, error)``. ``error`` is non-None on failure.
    The ``name`` argument may include or omit the ``.py`` suffix.
    """
    resolved_user_dir = (
        Path(user_dir).expanduser().resolve(
        ) if user_dir is not None else resolve_user_dir()
    )
    tools = user_tools_dir(resolved_user_dir)
    disabled_dir = user_tools_disabled_dir(resolved_user_dir)

    file_name = name if name.endswith(".py") else f"{name}.py"
    enabled_path = tools / file_name
    disabled_path = disabled_dir / file_name

    if enable:
        if not disabled_path.is_file():
            return None, f"No disabled tool named `{file_name}` was found at `{disabled_path}`."
        if enabled_path.exists():
            return None, f"Cannot enable: `{enabled_path}` already exists."
        ensure_user_subdir(tools)
        shutil.move(str(disabled_path), str(enabled_path))
        return enabled_path, None

    if not enabled_path.is_file():
        return None, f"No enabled tool named `{file_name}` was found at `{enabled_path}`."
    if disabled_path.exists():
        return None, f"Cannot disable: `{disabled_path}` already exists."
    ensure_user_subdir(disabled_dir)
    shutil.move(str(enabled_path), str(disabled_path))
    return disabled_path, None


def find_tool_script(
    name: str,
    *,
    user_dir: Path | str | None = None,
) -> Path | None:
    resolved_user_dir = (
        Path(user_dir).expanduser().resolve(
        ) if user_dir is not None else resolve_user_dir()
    )
    tools = user_tools_dir(resolved_user_dir)
    file_name = name if name.endswith(".py") else f"{name}.py"
    for candidate in (tools / file_name, tools / "disabled" / file_name):
        if candidate.is_file():
            return candidate
    return None
