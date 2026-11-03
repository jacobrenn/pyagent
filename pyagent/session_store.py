"""Read-only discovery and validation of saved run checkpoints.

Logs are data, never executable replay instructions. Parent links describe
ancestry only; a checkpoint does not depend on its ancestors being available.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from datetime import datetime
import json
import os
from pathlib import Path
import re
import stat
from typing import Any
from uuid import UUID

from .session_logger import LOG_SCHEMA_VERSION
from .user_runtime import resolve_user_dir, user_log_dir

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_RECORD_BYTES = 16 * 1024 * 1024
MAX_MESSAGES = 4096
MAX_CONTENT_CHARS = 8 * 1024 * 1024
MAX_LOG_FILES = 10_000
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 100_000
_RUN_NAME = re.compile(r"run-.+-([0-9a-fA-F-]{36})\.jsonl\Z")


class SessionError(ValueError):
    """A saved run cannot be inspected or resumed safely."""


class SessionNotFoundError(SessionError):
    """No saved run has the requested identity."""


def _identity(value: Any) -> str:
    if not isinstance(value, str):
        raise SessionError("Run IDs must be UUID strings.")
    try:
        return str(UUID(value))
    except ValueError as exc:
        raise SessionError("Run IDs must be UUID strings.") from exc


def _reject_constant(value: str) -> None:
    raise SessionError(f"Invalid JSON constant: {value}")


def _check_json_limits(value: Any) -> None:
    # Iterators keep auxiliary memory proportional to depth, not array length.
    stack = [(iter([value]), 0)]
    nodes = 0
    while stack:
        items, depth = stack[-1]
        try:
            item = next(items)
        except StopIteration:
            stack.pop()
            continue
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise SessionError("Log record exceeds the JSON node limit.")
        if isinstance(item, (dict, list)):
            if depth >= MAX_JSON_DEPTH:
                raise SessionError(
                    "Log record exceeds the JSON nesting limit.")
            children = item.values() if isinstance(item, dict) else item
            stack.append((iter(children), depth + 1))


@dataclass
class SavedRun:
    reference: str
    run_id: str | None = None
    parent_run_id: str | None = None
    timestamp: str = ""
    prompt: str = ""
    outcome: str = "incomplete"
    resumable: bool = False
    reason: str = "Missing run_end; the run is incomplete or still active."
    metadata: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    conversation_id: str | None = None
    lineage_warning: str | None = None


def validate_messages(messages: Any, *, provider: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate complete tool blocks, normalizing only native Ollama missing IDs."""
    if not isinstance(messages, list) or not messages or len(messages) > MAX_MESSAGES:
        raise SessionError(
            f"Checkpoint must contain 1..{MAX_MESSAGES} messages.")
    history = copy.deepcopy(messages)
    warnings: list[str] = []
    for index, message in enumerate(history):
        if not isinstance(message, dict):
            raise SessionError("Checkpoint messages must be objects.")
        role = message.get("role")
        if not isinstance(role, str) or role not in {"system", "user", "assistant", "tool"}:
            raise SessionError("Unsupported checkpoint message role.")
        content = message.get("content")
        if not isinstance(content, str) or len(content) > MAX_CONTENT_CHARS:
            raise SessionError(
                "Checkpoint message content is invalid or too large.")
        if role == "system" and index != 0:
            raise SessionError(
                "System messages are only allowed at the start.")
        if role != "assistant" and message.get("tool_calls"):
            raise SessionError(
                "Only assistant messages may contain tool calls.")

    index = 0
    while index < len(history):
        message = history[index]
        if message["role"] == "tool":
            raise SessionError("Checkpoint contains an orphan tool result.")
        calls = message.get("tool_calls", [])
        if not isinstance(calls, list):
            raise SessionError("Assistant tool_calls must be a list.")
        if not calls:
            index += 1
            continue
        end = index + 1
        while end < len(history) and history[end]["role"] == "tool":
            end += 1
        outputs = history[index + 1:end]
        if len(outputs) != len(calls):
            raise SessionError(
                "Checkpoint contains an unfinished tool-call block.")
        call_ids: dict[str, str] = {}
        for position, call in enumerate(calls):
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                raise SessionError("Malformed checkpoint tool call.")
            function = call["function"]
            name = function.get("name")
            if not isinstance(name, str) or not name:
                raise SessionError("Tool calls require a name.")
            arguments = function.get("arguments", {})
            if not isinstance(arguments, (dict, str)):
                raise SessionError(
                    "Tool arguments must be an object or JSON string.")
            call_id = call.get("id")
            if not call_id:
                output = outputs[position]
                if provider != "ollama" or (output.get("name") or output.get("tool_name")) != name:
                    raise SessionError(
                        "Cannot unambiguously pair a tool call without an ID.")
                call_id = output.get(
                    "tool_call_id") or f"resume_{index}_{position}"
                call["id"] = output["tool_call_id"] = call_id
                warnings.append(
                    "Normalized native Ollama tool-call IDs in restored history.")
            if not isinstance(call_id, str) or not call_id or call_id in call_ids:
                raise SessionError("Invalid or duplicate tool-call ID.")
            call_ids[call_id] = name
        seen: set[str] = set()
        for output in outputs:
            call_id = output.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in call_ids or call_id in seen:
                raise SessionError("Tool results do not match their calls.")
            name = output.get("name") or output.get("tool_name")
            if name is not None and name != call_ids[call_id]:
                raise SessionError("Tool result name does not match its call.")
            seen.add(call_id)
        index = end
    return history, list(dict.fromkeys(warnings))


class SessionStore:
    def __init__(self, user_dir: str | Path | None = None) -> None:
        self.user_dir = resolve_user_dir(user_dir)
        self.root = user_log_dir(self.user_dir).resolve()

    def _paths(self) -> list[Path]:
        paths: list[Path] = []
        for path in self.root.glob("*.jsonl"):
            if path.name.startswith(("run-", "session-")):
                paths.append(path)
                if len(paths) > MAX_LOG_FILES:
                    raise SessionError(
                        f"Too many run files (limit {MAX_LOG_FILES}); archive older logs.")
        return sorted(paths)

    def _resolve(self, reference: str) -> Path:
        # Filenames are accepted for inspecting legacy/corrupt entries in lists.
        if reference.endswith(".jsonl"):
            if Path(reference).name != reference or "/" in reference or "\\" in reference:
                raise SessionError(
                    "Use a run UUID or a log basename, not a path.")
            if not reference.startswith(("run-", "session-")):
                raise SessionError("Not a saved run filename.")
            return self.root / reference
        identity = _identity(reference)
        matches = [p for p in self._paths(
        ) if p.name.lower().endswith(f"-{identity}.jsonl")]
        if not matches:
            raise SessionNotFoundError(f"No saved run for {identity}.")
        if len(matches) != 1:
            raise SessionError(
                f"Expected one saved run for {identity}; found {len(matches)}.")
        return matches[0]

    def inspect(self, reference: str) -> SavedRun:
        path = self._resolve(reference)
        try:
            if path.is_symlink() or path.resolve().parent != self.root:
                raise SessionError(
                    "Log symlinks and paths outside the log directory are not allowed.")
            fd = os.open(path, os.O_RDONLY | getattr(
                os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
            with os.fdopen(fd, "rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
                    raise SessionError(
                        "Log must be a regular file no larger than 64 MiB.")
                if path.name.startswith("session-"):
                    return SavedRun(path.name, outcome="legacy", reason="Legacy session logs require migration; not supported for resume.")
                result = self._read(stream, path)
                after = os.fstat(stream.fileno())
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise SessionError(
                        "Log changed while being read; retry after the run finishes.")
                return result
        except (OSError, UnicodeError, RecursionError, ValueError) as exc:
            if isinstance(exc, SessionError):
                raise
            if isinstance(exc, FileNotFoundError):
                raise SessionNotFoundError(
                    "Saved run no longer exists.") from exc
            raise SessionError(f"Cannot read {path.name}: {exc}") from exc

    def _read(self, stream: Any, path: Path) -> SavedRun:
        match = _RUN_NAME.fullmatch(path.name)
        if match is None:
            raise SessionError("Invalid run filename.")
        identity = _identity(match[1])
        saved = SavedRun(identity, run_id=identity)
        start = None
        terminal = None
        sequence = 0
        total = 0
        while True:
            line = stream.readline(MAX_RECORD_BYTES + 1)
            if not line:
                break
            total += len(line)
            if len(line) > MAX_RECORD_BYTES or total > MAX_FILE_BYTES:
                raise SessionError("Log exceeds record/file size limits.")
            if not line.endswith(b"\n"):
                raise SessionError("Log has a truncated final record.")
            record = json.loads(line, parse_constant=_reject_constant)
            _check_json_limits(record)
            sequence += 1
            if not isinstance(record, dict):
                raise SessionError("Log records must be objects.")
            if type(record.get("schema_version")) is not int or record["schema_version"] != LOG_SCHEMA_VERSION:
                raise SessionError("Unsupported log schema version.")
            if record.get("run_id") != identity:
                raise SessionError("Mixed or mismatched run IDs in log.")
            if type(record.get("sequence")) is not int or record["sequence"] != sequence:
                raise SessionError("Invalid log record sequence.")
            kind = record.get("type")
            attrs = record.get("attributes", {})
            if not isinstance(kind, str) or not kind or not isinstance(attrs, dict):
                raise SessionError("Malformed log record.")
            timestamp = record.get("timestamp")
            if not isinstance(timestamp, str) or datetime.fromisoformat(timestamp).tzinfo is None:
                raise SessionError("Log timestamp must include a timezone.")
            if terminal is not None:
                raise SessionError("Unexpected records after run_end.")
            if kind == "run_start":
                if sequence != 1 or start is not None:
                    raise SessionError("run_start must be the first record.")
                start = attrs
                metadata = attrs.get("metadata")
                if not isinstance(metadata, dict) or not isinstance(attrs.get("prompt"), str):
                    raise SessionError("Invalid run_start metadata or prompt.")
                saved.metadata = metadata
                saved.prompt = attrs["prompt"]
                saved.timestamp = timestamp
                parent = attrs.get("parent_run_id")
                saved.parent_run_id = _identity(
                    parent) if parent is not None else None
                if saved.parent_run_id == identity:
                    raise SessionError("A run cannot be its own parent.")
            elif kind == "run_end":
                terminal = attrs
                saved.timestamp = timestamp
        if terminal is None:
            return saved
        saved.outcome = terminal.get("outcome", "invalid")
        if not isinstance(saved.outcome, str) or saved.outcome not in {"completed", "failed", "cancelled"}:
            raise SessionError("Invalid run outcome.")
        if type(terminal.get("started")) is not bool:
            raise SessionError(
                "run_end must specify whether execution started.")
        if not terminal["started"]:
            saved.reason = "Run never started; no conversation checkpoint to resume."
        elif saved.outcome != "completed":
            saved.reason = f"{saved.outcome.capitalize()} runs are inspection-only in this release."
        elif start is None:
            raise SessionError("Completed checkpoint is missing run_start.")
        else:
            saved.messages, saved.warnings = validate_messages(
                terminal.get("messages"), provider=saved.metadata.get("provider", ""))
            saved.resumable = True
            saved.reason = "Completed checkpoint."
        return saved

    def load(self, reference: str) -> SavedRun:
        saved = self.inspect(reference)
        if not saved.resumable:
            raise SessionError(saved.reason)
        # Resolve again by identity to reject duplicate copies with the same ID.
        self._resolve(saved.run_id)
        return saved

    def list(self) -> list[SavedRun]:
        runs = []
        for path in self._paths():
            try:
                saved = self.inspect(path.name)
                # Keep listings bounded even when individual histories are large.
                metadata = {key: str(saved.metadata[key])[:1000]
                            for key in ("profile", "model", "workspace", "agent", "agent_revision")
                            if key in saved.metadata}
                runs.append(replace(saved, messages=[],
                            prompt=saved.prompt[:160], metadata=metadata))
            except SessionError as exc:
                runs.append(
                    SavedRun(path.name, outcome="invalid", reason=str(exc)))
        by_id = {run.run_id: run for run in runs if run.run_id}
        counts: dict[str, int] = {}
        for run in runs:
            counts[run.reference] = counts.get(run.reference, 0) + 1
        for run in runs:
            if counts[run.reference] > 1:
                run.resumable = False
                run.reason = "Duplicate saved run ID; remove duplicate copies before resuming."
            current = run
            seen: set[str] = set()
            while current.run_id:
                if current.run_id in seen or len(seen) >= 256:
                    run.lineage_warning = "Cyclic or excessively deep parent chain."
                    break
                seen.add(current.run_id)
                run.conversation_id = current.run_id
                if not current.parent_run_id:
                    break
                if current.parent_run_id not in by_id or counts.get(current.parent_run_id, 0) > 1:
                    run.conversation_id = current.parent_run_id
                    run.lineage_warning = "Conversation parent is unavailable or ambiguous."
                    break
                current = by_id[current.parent_run_id]
        return sorted(runs, key=lambda item: (item.timestamp, item.reference), reverse=True)

    def lineage(self, reference: str) -> tuple[list[str], str | None]:
        """Return explicit ancestors, newest first, and any missing/cyclic parent."""
        saved = self.inspect(reference)
        chain: list[str] = []
        while saved.run_id:
            if saved.run_id in chain or len(chain) >= 256:
                return chain, "Cyclic or excessively deep parent chain."
            chain.append(saved.run_id)
            if saved.parent_run_id is None:
                return chain, None
            try:
                saved = self.inspect(saved.parent_run_id)
            except SessionError:
                return chain, f"Parent {saved.parent_run_id} is unavailable."
        return chain, None


def format_sessions(runs: list[SavedRun], limit: int = 20) -> str:
    """Compact plain-text listing shared by the CLI and TUI."""
    if not runs:
        return "No saved runs. Enable PYAGENT_LOGGING to save future prompts."
    lines = [
        "Saved checkpoints (newest conversations first; grouped by explicit parent links):"]
    groups: dict[str, list[SavedRun]] = {}
    for run in runs[:limit]:
        groups.setdefault(
            run.conversation_id or run.run_id or run.reference, []).append(run)
    for root, group in groups.items():
        lines.append(f"Conversation {root}")
        for run in group:
            metadata = run.metadata
            runtime = str(metadata.get("agent") or metadata.get(
                "profile") or "unknown runtime")[:200]
            if metadata.get("agent_revision"):
                runtime += f" revision {str(metadata['agent_revision'])[:20]}"
            model = str(metadata.get("model", "unknown model"))[:200]
            workspace = str(metadata.get("workspace", ""))[:500]
            preview = " ".join(run.prompt[:1000].split())[:100]
            state = "resumable" if run.resumable else run.reason
            lines.append(
                f"{run.reference} | {run.timestamp} | {run.outcome} | {state}")
            lines.append(f"  {runtime} / {model} | {workspace}")
            lines.append(f"  parent: {run.parent_run_id or '-'} | {preview}")
            if run.lineage_warning:
                lines.append(f"  {run.lineage_warning}")
    if len(runs) > limit:
        lines.append(f"Showing {limit} of {len(runs)} runs.")
    return "\n".join("".join(c if c.isprintable() else " " for c in line) for line in lines)
