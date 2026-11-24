"""Shared tool-registry report used by the HTTP API and the CLI.

Returns a plain dict mirroring the ``GET /tools`` response body. No FastAPI
imports so the CLI can use it without the ``api`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import AppConfig
from .external_tools import (
    build_external_tool_specs,
    default_runner_command,
    discover_external_tools,
)
from .mcp_report import mcp_specs_for
from .resources import TOOL_KIND, list_resources
from .tools import (
    BUILTIN_ORIGIN,
    EXTERNAL_ORIGIN,
    MCP_ORIGIN,
    create_default_tool_registry,
)
from .user_runtime import resolve_user_dir, user_extensions_dir


def _tool_info(registry: Any, definitions: dict[str, dict[str, Any]], name: str, origin: str) -> dict[str, Any]:
    definition = definitions.get(name, {})
    return {
        "name": name,
        "origin": origin,
        "source": registry.source(name),
        "description": definition.get("description"),
        "parameters": definition.get("parameters"),
    }


def tools_report(config: AppConfig | None = None) -> dict[str, Any]:
    config = config or AppConfig.from_env()
    discovery = None
    discovery_error: str | None = None
    if config.user_tools_enabled:
        try:
            from .extensions.loader import _discover, loaded_extension_tool_dirs

            resolved_user_dir = resolve_user_dir(config.user_dir)
            extension_dir = user_extensions_dir(resolved_user_dir)
            extension_tool_dirs = loaded_extension_tool_dirs(
                extension_dir, _discover(extension_dir)
            )
            discovery = discover_external_tools(
                user_dir=config.user_dir,
                runner=config.tool_runner,
                describe_timeout=config.user_tool_describe_timeout,
                extra_tool_dirs=extension_tool_dirs,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            discovery_error = str(exc)

    external_specs = (
        build_external_tool_specs(
            discovery,
            invoke_timeout=config.user_tool_timeout,
            runner_command=default_runner_command(discovery.runner),
        )
        if discovery is not None
        else None
    )
    mcp_specs, mcp_error = mcp_specs_for(config)
    registry = create_default_tool_registry(
        config, external_specs=external_specs, mcp_specs=mcp_specs)
    definitions = {
        item["function"]["name"]: item["function"]
        for item in registry.definitions()
        if isinstance(item.get("function"), dict)
    }

    runner = {
        "name": discovery.runner if discovery is not None else config.tool_runner,
        "available": discovery.runner_available if discovery is not None else None,
        "message": discovery.runner_message if discovery is not None else None,
    }

    builtin = [
        _tool_info(registry, definitions, name, BUILTIN_ORIGIN)
        for name in sorted(registry.names_by_origin(BUILTIN_ORIGIN))
    ]
    external = [
        _tool_info(registry, definitions, name, EXTERNAL_ORIGIN)
        for name in sorted(registry.names_by_origin(EXTERNAL_ORIGIN))
    ]
    mcp_tools = [
        _tool_info(registry, definitions, name, MCP_ORIGIN)
        for name in sorted(registry.names_by_origin(MCP_ORIGIN))
    ]
    files = [
        {
            "label": resource.label,
            "path": str(resource.path),
            "disabled": "disabled" in Path(resource.label).parts[:-1],
        }
        for resource in list_resources(TOOL_KIND, user_dir=config.user_dir)
    ]

    broken: list[dict[str, Any]] = []
    disabled: list[dict[str, Any]] = []
    if discovery is not None:
        broken = [
            {"script_path": str(entry.script_path), "error": entry.error}
            for entry in discovery.broken
        ]
        disabled = [
            {"script_path": str(entry.script_path)}
            for entry in discovery.disabled
        ]

    return {
        "tools_enabled": config.tools_enabled,
        "builtin_tools_enabled": config.builtin_tools_enabled,
        "user_tools_enabled": config.user_tools_enabled,
        "user_dir": str(resolve_user_dir(config.user_dir)),
        "runner": runner,
        "builtin": builtin,
        "external": external,
        "files": files,
        "broken": broken,
        "disabled": disabled,
        "collisions": [
            {
                "name": collision.name,
                "external_path": collision.external_path,
                "loser_origin": getattr(collision, "loser_origin", None),
            }
            for collision in registry.collisions()
        ],
        "discovery_error": discovery_error,
        "mcp": mcp_tools,
        "mcp_error": mcp_error,
    }
