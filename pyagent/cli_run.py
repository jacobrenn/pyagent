"""Shared agent-run execution for the CLI entry points.

``--prompt``, ``agents run`` and ``resume --prompt`` all funnel through
:func:`run_and_report` so they agree on output shape, error handling, and exit
codes. ``run_and_report`` never closes the agent; callers own the lifecycle.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from .cli_output import EXIT_ERROR, EXIT_OK, Result, emit, note


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def build_summary(
    agent: Any,
    response: str,
    *,
    with_messages: bool,
    warnings: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        profile = agent.current_profile()
        profile_name = profile.name
        provider = profile.provider
        api_mode = profile.resolved_api_mode()
        model = getattr(agent, "model_override", None) or profile.model
    except Exception:
        profile_name = provider = api_mode = model = None
    context_files_raw = getattr(agent, "project_context_files", None)
    context_files = [str(item) for item in context_files_raw] if isinstance(
        context_files_raw, (list, tuple)) else []
    data: dict[str, Any] = {
        "run_id": _str_or_none(getattr(agent, "run_id", None)),
        "parent_run_id": _str_or_none(getattr(agent, "parent_run_id", None)),
        "log_saved": bool(getattr(agent, "log_saved", False)),
        "warnings": list(warnings),
        "response": response,
        "profile": profile_name,
        "provider": provider,
        "api_mode": api_mode,
        "model": model,
        "context_files": context_files,
    }
    if with_messages:
        messages = getattr(agent, "messages", None)
        data["messages"] = messages if isinstance(messages, list) else []
    if extra:
        data.update(extra)
    return data


def _emit_stream_line(command: str, event: dict[str, Any]) -> None:
    payload = {"ok": True, "command": command, "event": event}
    print(json.dumps(payload, default=str, ensure_ascii=False))


def _emit_stream_error(command: str, code: str, message: str) -> None:
    payload = {
        "ok": False,
        "command": command,
        "error": {"code": code, "message": message},
    }
    print(json.dumps(payload, ensure_ascii=False))


def _emit_json_error(command: str, code: str, message: str) -> None:
    payload = {
        "ok": False,
        "command": command,
        "error": {"code": code, "message": message},
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def run_and_report(
    agent: Any,
    prompt: str,
    *,
    command: str,
    json_output: bool = False,
    stream: bool = False,
    with_messages: bool = False,
    quiet: bool = False,
    source: str = "cli",
    warnings: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run ``agent`` on ``prompt`` and render the result.

    Returns ``(exit_code, summary)``. ``summary`` always contains at least
    ``response``; on failure it also carries ``error``. The agent is not
    closed here; callers own its lifecycle.
    """
    warnings = list(warnings or [])
    agent.run_source = source
    if warnings and not json_output and not quiet:
        for warning in warnings:
            note(warning)
    try:
        agent.load_extensions()
    except Exception as exc:
        message = f"could not load extensions: {exc}"
        if not quiet:
            if json_output:
                _emit_json_error(command, "extension_load_failed", message)
            else:
                print(f"Error: {message}", file=sys.stderr)
        return EXIT_ERROR, {"response": "", "error": message, "warnings": warnings}

    final_text = ""
    error_message: str | None = None
    streamed_any = False

    try:
        for event in agent.run(prompt):
            event_type = event.get("type")
            if event_type == "content_delta":
                delta = str(event.get("delta", ""))
                if stream:
                    if json_output:
                        if not quiet:
                            _emit_stream_line(
                                command, {"type": "content_delta", "delta": delta})
                    elif not quiet:
                        sys.stdout.write(delta)
                        sys.stdout.flush()
                    streamed_any = True
            elif event_type == "assistant_done":
                final_text = str(event.get("content", ""))
                if stream and json_output and not quiet:
                    _emit_stream_line(
                        command, {"type": "assistant_done", "content": final_text})
            elif event_type == "error":
                error_message = str(event.get("message") or "Agent run failed")
                if stream and json_output and not quiet:
                    _emit_stream_line(
                        command, {"type": "error", "message": error_message})
            elif stream and json_output and not quiet and event_type in {
                "assistant_start",
                "tool_call",
                "tool_result",
                "compaction",
            }:
                _emit_stream_line(command, event)
    except Exception as exc:  # agent blew up mid-run
        error_message = str(exc)

    if error_message is not None:
        summary = build_summary(
            agent, "", with_messages=with_messages, warnings=warnings, extra=extra
        )
        summary["error"] = error_message
        if not quiet:
            if json_output:
                if stream:
                    _emit_stream_error(command, "agent_error", error_message)
                else:
                    _emit_json_error(command, "agent_error", error_message)
            else:
                print(f"Error: {error_message}", file=sys.stderr)
        return EXIT_ERROR, summary

    summary = build_summary(
        agent, final_text, with_messages=with_messages, warnings=warnings, extra=extra
    )

    if quiet:
        return EXIT_OK, summary

    if json_output:
        if stream:
            _emit_stream_line(command, {"type": "done", "data": summary})
        else:
            emit(Result(command=command, data=summary), json_output=True)
        return EXIT_OK, summary

    if stream:
        if not streamed_any and final_text:
            sys.stdout.write(final_text)
        sys.stdout.flush()
    else:
        sys.stdout.write(final_text)
        sys.stdout.flush()
    return EXIT_OK, summary
