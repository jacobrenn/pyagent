"""Small per-run recorder shared by all entry points (no transport ownership)."""
from __future__ import annotations

import sys
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from .session_logger import SessionLogger
from .user_runtime import resolve_user_dir, user_log_dir


def safe_endpoint(value: str) -> str:
    """Keep endpoint routing metadata, never URL credentials/query/fragment."""
    try:
        parts = urlsplit(value)
        host = parts.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        if parts.port is not None:
            host += f":{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    except ValueError:
        return "<invalid endpoint>"


def runtime_metadata(agent: Any) -> dict[str, Any]:
    """Allowlist configuration metadata; never serialize credentials/config wholesale."""
    profile = agent.current_profile()
    return {
        "source": agent.run_source,
        "profile": profile.name,
        "provider": profile.provider,
        "api_mode": profile.resolved_api_mode(),
        "model": agent.model_override or profile.model,
        "base_url": safe_endpoint(profile.base_url),
        "workspace": str(agent.workspace),
        "context_files": list(agent.project_context_files),
        "extensions": agent.bus.loaded_extensions(),
        "tools": [item["function"]["name"] for item in agent.tools],
        "settings": {
            key: getattr(agent.config, key) for key in (
                "max_iterations", "max_history_messages",
                "max_previous_tool_result_chars", "tools_enabled",
                "builtin_tools_enabled", "user_tools_enabled", "bash_enabled",
                "bash_readonly_mode",
            )
        },
        **{key: agent.run_metadata[key] for key in ("agent", "agent_revision", "skills")
           if key in agent.run_metadata},
    }


class RunLog:
    def __init__(self, agent: Any) -> None:
        self.run_id = str(uuid4())
        self.logger: SessionLogger | None = None
        self.started = False
        self.finished = False
        self.outcome = "cancelled"
        self.response = ""
        self.error: str | None = None
        self.partial: list[str] = []
        self._warn: Callable[[str], None] = agent._logging_warning
        if agent.logging_enabled:
            try:
                self.logger = SessionLogger(
                    user_log_dir(resolve_user_dir(agent.config.user_dir)),
                    run_id=self.run_id,
                    on_error=self._warn,
                )
                # Exclusive creation can change the ID after a filename collision.
                self.run_id = self.logger.run_id
            except Exception as exc:
                self._warn(f"Could not start run logging: {exc}")

    def record(self, record_type: str, **data: Any) -> None:
        if self.logger is not None:
            self.logger.record(record_type, **data)

    def start(self, agent: Any, prompt: str) -> None:
        self.started = True
        if self.logger is None:
            return
        try:
            self.record(
                "run_start", prompt=prompt, parent_run_id=agent.parent_run_id,
                initial_messages=agent.messages, metadata=runtime_metadata(
                    agent),
            )
        except Exception as exc:
            self.logger._failed(exc)

    def message(self, message: dict[str, Any]) -> None:
        self.record("message", message=message)
        if message.get("role") == "assistant":
            self.partial.clear()

    def event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "content_delta":
            # Preserve partial output on failure without writing a line per token.
            if self.logger is not None:
                self.partial.append(str(event.get("delta", "")))
        elif kind == "assistant_done":
            self.outcome = "completed"
            self.response = str(event.get("content", ""))
            self.partial.clear()
        elif kind == "error":
            self.outcome = "failed"
            self.error = str(event.get("message") or "Agent run failed")
            self.record("error", message=self.error)
        elif kind in {"tool_call", "tool_result"}:
            self.record(kind, event=event)

    def finish(self, agent: Any) -> None:
        if self.finished:
            return
        self.finished = True
        try:
            if self.partial:
                self.record("partial_output", content="".join(self.partial))
            self.record(
                "run_end", outcome=self.outcome, response=self.response,
                error=self.error, messages=agent.messages,
                started=self.started,
            )
        finally:
            if self.logger is not None:
                self.logger.close()


def prepare_run_id(agent: Any, *, source: str) -> str:
    """Use the agent identity, retaining support for lightweight stream adapters."""
    prepare = getattr(agent, "prepare_run", None)
    run_id = prepare(source=source) if callable(prepare) else None
    if not isinstance(run_id, str):
        run_id = str(uuid4())
    return run_id


def report_warning(message: str) -> None:
    try:
        print(message, file=sys.stderr)
    except Exception:
        pass
