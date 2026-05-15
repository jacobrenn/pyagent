from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# OTel severity numbers (subset we use)
_SEVERITY_INFO = 9
_SEVERITY_WARN = 13
_SEVERITY_ERROR = 17
_SEVERITY_DEBUG = 5


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
        severity_number: int,
        severity_text: str,
        body: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "timeUnixNano": _iso_timestamp(),
            "severityNumber": severity_number,
            "severityText": severity_text,
            "body": body,
        }
        if attributes:
            entry["attributes"] = attributes
        self._file.write(json.dumps(
            entry, ensure_ascii=False, default=str) + "\n")
        self._file.flush()

    def info(self, body: str, attributes: dict[str, Any] | None = None) -> None:
        self._write_entry(_SEVERITY_INFO, "INFO", body, attributes)

    def debug(self, body: str, attributes: dict[str, Any] | None = None) -> None:
        self._write_entry(_SEVERITY_DEBUG, "DEBUG", body, attributes)

    def warn(self, body: str, attributes: dict[str, Any] | None = None) -> None:
        self._write_entry(_SEVERITY_WARN, "WARN", body, attributes)

    def error(self, body: str, attributes: dict[str, Any] | None = None) -> None:
        self._write_entry(_SEVERITY_ERROR, "ERROR", body, attributes)

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
            "INFO",
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
        body = (
            f"Turn {turn_number}: "
            f"{len(output_messages)} response(s), "
            f"{sum(len(m.get('tool_calls', [])) for m in output_messages if m.get('tool_calls'))} tool call(s)"
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
        self._write_entry(_SEVERITY_INFO, "INFO", body, attrs)

    def log_session_start(self, attributes: dict[str, Any] | None = None) -> None:
        """Log the start of a session with environment/profile metadata."""
        attrs = dict(attributes or {})
        attrs["event_type"] = "session_start"
        self._write_entry(
            _SEVERITY_INFO,
            "INFO",
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
            "INFO",
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
