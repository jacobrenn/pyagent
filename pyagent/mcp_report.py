"""Shared MCP reporting helpers used by the HTTP API and the CLI.

Everything here returns plain dicts/lists so callers can wrap them in
pydantic models (API) or render them directly (CLI). No FastAPI imports.
"""

from __future__ import annotations

from typing import Any

from .mcp_config import redacted_server_dict


def mcp_specs_for(config: Any) -> tuple[list[Any], str | None]:
    """Return ``(mcp_specs, error)`` for registry construction."""
    if not getattr(config, "mcp_enabled", True):
        return [], None
    try:
        from .mcp_runtime import get_mcp_manager

        discovery = get_mcp_manager().ensure_tools(
            user_dir=config.user_dir,
            connect_timeout=config.mcp_connect_timeout,
        )
        if discovery.error:
            return [], discovery.error
        return discovery.specs, None
    except Exception as exc:  # never break tool listing because of MCP
        return [], str(exc)


def mcp_status_map(config: Any) -> tuple[dict[str, Any], str | None]:
    """Return ``(statuses_by_name, error)`` using the pooled connections."""
    if not getattr(config, "mcp_enabled", True):
        return {}, None
    try:
        from .mcp_runtime import get_mcp_manager

        discovery = get_mcp_manager().ensure_tools(
            user_dir=config.user_dir,
            connect_timeout=config.mcp_connect_timeout,
        )
        return {s.name: s for s in discovery.servers}, discovery.error
    except Exception as exc:  # pragma: no cover - defensive
        return {}, str(exc)


def mcp_entry_dict(server: Any, status: Any | None) -> dict[str, Any]:
    """Redacted, JSON-serializable view of one configured MCP server."""
    redacted = redacted_server_dict(server)
    entry: dict[str, Any] = {
        "name": server.name,
        "transport": server.transport,
        "enabled": server.enabled,
        "command": redacted.get("command"),
        "args": list(redacted.get("args") or []),
        "env": dict(redacted.get("env") or {}),
        "cwd": redacted.get("cwd"),
        "url": redacted.get("url"),
        "headers": dict(redacted.get("headers") or {}),
        "missing_env": list(redacted.get("missing_env") or []),
        "status": "unknown" if server.enabled else "disabled",
        "tool_count": 0,
        "tools": [],
        "error": None,
    }
    if status is not None:
        entry.update(
            status=status.status,
            tool_count=status.tool_count,
            tools=list(status.tool_names),
            error=status.error,
        )
    return entry


def mcp_list_payload(config: Any) -> dict[str, Any]:
    """Mirror of the ``GET /mcp`` response body as a plain dict."""
    from .mcp_config import default_mcp_config_path, load_mcp_config, McpConfigError

    enabled = bool(getattr(config, "mcp_enabled", True))
    try:
        parsed = load_mcp_config(user_dir=config.user_dir)
    except McpConfigError as exc:
        return {
            "enabled": enabled,
            "config_path": str(default_mcp_config_path(config.user_dir)),
            "servers": [],
            "error": str(exc),
        }
    statuses, status_error = mcp_status_map(config)
    return {
        "enabled": enabled,
        "config_path": str(parsed.path),
        "servers": [
            mcp_entry_dict(server, statuses.get(server.name))
            for server in parsed.servers
        ],
        "error": status_error,
    }
