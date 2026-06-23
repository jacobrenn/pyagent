from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SEVERITY_DEBUG = "DEBUG"
_SEVERITY_INFO = "INFO"
_SEVERITY_WARN = "WARN"
_SEVERITY_ERROR = "ERROR"


def _iso_timestamp() -> str:
    """Current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _log_dir_path(log_dir: str | os.PathLike[str] | None) -> Path:
    """Resolve the logs directory, creating it if needed."""
    base = (
        Path(os.path.expanduser(str(log_dir)))
        if log_dir
        else Path.home() / ".pyagent" / "logs"
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


class SessionLogger:
    """
    Writes OpenTelemetry-compatible JSON Lines (JSONL) logs to ~/.pyagent/logs/.
    Each log line is a JSON object with OTel log record fields::

    {
      "timeUnixNano": "<ISO 8601 timestamp>",
      "severityNumber": <int>,
      "severityText": "<string>",
      "body": "<message text>",
      "attributes": { ... }
    }

    Each call creates a new log file named with the session start timestamp.
    """

    def __init__(self, log_dir: str | os.PathLike[str] | None = None) -> None:
        self._log_dir = _log_dir_path(log_dir)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        self._log_path = self._log_dir / f"session-{timestamp}.jsonl"
        self._file = open(self._log_path, "a", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._log_path

    def _write_entry(
        self,
        severity_text: str,
        body: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "timeUnixNano": _iso_timestamp(),
            "severityText": severity_text,
            "body": body,
        }
        if attributes:
            entry["attributes"] = attributes
        self._file.write(json.dumps(
            entry, ensure_ascii=False, default=str) + "\n")
        self._file.flush()

    def info(self, body: str, attributes: dict[str, Any] | None = None) -> None:
        self._write_entry(_SEVERITY_INFO, body, attributes)

    def debug(self, body: str, attributes: dict[str, Any] | None = None) -> None:
        self._write_entry(_SEVERITY_DEBUG, body, attributes)

    def warn(self, body: str, attributes: dict[str, Any] | None = None) -> None:
        self._write_entry(_SEVERITY_WARN, body, attributes)

    def error(self, body: str, attributes: dict[str, Any] | None = None) -> None:
        self._write_entry(_SEVERITY_ERROR, body, attributes)

    def log_event(
        self,
        event_name: str,
        extension: str | None,
        payload: dict[str, Any],
        result: Any = None,
    ) -> None:
        """Record an extension bus event."""
        body = f"Extension event: {event_name}"
        attrs: dict[str, Any] = {
            "event_type": "extension_event",
            "event": event_name,
            "payload": dict(payload),
        }
        if extension:
            body += f" ({extension})"
            attrs["extension"] = extension
        if result is not None:
            attrs["result"] = result
        self._write_entry(_SEVERITY_DEBUG, body, attrs)

    def log_extension_skills(
        self, keys: list[str], text_chars: int
    ) -> None:
        """Record extension skills that were injected into the system prompt."""
        if not keys:
            return
        self._write_entry(
            _SEVERITY_INFO,
            f"Loaded extension skills: {', '.join(keys)}",
            {
                "event_type": "extension_skills_loaded",
                "keys": list(keys),
                "text_chars": text_chars,
            },
        )

    def log_skill_load(self, skill_id: str) -> None:
        """Record a user/project skill explicitly loaded into context."""
        self._write_entry(
            _SEVERITY_INFO,
            f"Skill loaded: {skill_id}",
            {"event_type": "skill_load", "skill_id": skill_id},
        )

    def log_skill_unload(self, skill_id: str) -> None:
        """Record a user/project skill explicitly unloaded from context."""
        self._write_entry(
            _SEVERITY_INFO,
            f"Skill unloaded: {skill_id}",
            {"event_type": "skill_unload", "skill_id": skill_id},
        )

    @classmethod
    def from_config(cls, config: Any) -> SessionLogger:
        """Create a SessionLogger from a PyAgent config object."""
        from .user_runtime import user_log_dir

        log_dir = user_log_dir(Path(config.user_dir))
        instance = cls(log_dir=log_dir)
        return instance

    @property
    def jsonl_path(self) -> Path:
        return self._log_path

    def log_entry(self, entry: dict[str, Any]) -> None:
        self._write_entry(
            _SEVERITY_INFO,
            entry.get("body", entry.get("content", "")),
            {k: v for k, v in entry.items() if k not in ("body", "content")},
        )

    def log_turn(
        self,
        turn_number: int,
        input_messages: list[dict[str, Any]],
        output_messages: list[dict[str, Any]],
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Log a complete turn: the full input message history and the model's response(s).

        *input_messages* is the list of messages sent to the model (including the
        system prompt and any prior tool results).  *output_messages* is the list
        of new messages produced by the model (assistant content, tool_calls,
        tool results).

        For compactness the JSONL body contains a summary; the full payloads are
        stored in the ``attributes`` dict so downstream consumers can replay or
        inspect the conversation state.
        """
        tool_names = [
            tc.get("function", {}).get("name", "<unknown>")
            for m in output_messages
            for tc in m.get("tool_calls", [])
        ]
        tool_summary = f" — {', '.join(tool_names)}" if tool_names else ""
        body = (
            f"Turn {turn_number}: "
            f"{len(output_messages)} response(s), "
            f"{len(tool_names)} tool call(s){tool_summary}"
        )
        attrs = dict(attributes or {})
        attrs.update(
            {
                "event_type": "turn",
                "turn_number": turn_number,
                "input_message_count": len(input_messages),
                "output_message_count": len(output_messages),
                "input_messages": input_messages,
                "output_messages": output_messages,
            }
        )
        self._write_entry(_SEVERITY_INFO, body, attrs)

    def log_session_start(self, attributes: dict[str, Any] | None = None) -> None:
        """Log the start of a session with environment/profile metadata."""
        attrs = dict(attributes or {})
        attrs["event_type"] = "session_start"
        self._write_entry(
            _SEVERITY_INFO,
            "Session started",
            attrs,
        )

    def log_session_end(
        self, turn_count: int, attributes: dict[str, Any] | None = None
    ) -> None:
        """Log the end of a session."""
        attrs = dict(attributes or {})
        attrs.update({"event_type": "session_end", "turn_count": turn_count})
        self._write_entry(
            _SEVERITY_INFO,
            f"Session ended after {turn_count} turn(s)",
            attrs,
        )

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
