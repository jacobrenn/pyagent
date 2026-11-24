"""Configuration for MCP (Model Context Protocol) servers.

MCP servers are configured in a single user-global file, ``~/.pyagent/mcp.json``
(or ``$PYAGENT_USER_DIR/mcp.json``). The format mirrors the de-facto standard
used by other MCP hosts so config snippets from server documentation work
out of the box::

    {
      "version": 1,
      "mcpServers": {
        "github": {
          "command": "docker",
          "args": ["run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
                   "ghcr.io/github/github-mcp-server"],
          "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_PERSONAL_ACCESS_TOKEN}"}
        },
        "docs": {
          "url": "https://docs.example.com/mcp",
          "headers": {"Authorization": "Bearer ${DOCS_TOKEN}"}
        }
      }
    }

Two transports are supported:

- ``stdio``: a local subprocess (``command``/``args``/``env``/``cwd``).
- ``http`` (Streamable HTTP): a remote endpoint (``url``/``headers``).

``${VAR}`` references in ``command``, ``args``, ``env`` values, ``cwd``,
``url`` and ``headers`` values are expanded from the process environment at
load time. Referencing an unset variable marks the server as misconfigured;
the variable name is reported, never a value. Secrets should live in the
environment, not in the file.

This module is deliberately free of any ``mcp`` SDK imports so that config
handling works even if the optional SDK is unavailable or broken.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from .user_runtime import resolve_user_dir, user_mcp_config_path

CONFIG_VERSION = 1
CONFIG_FILE_NAME = "mcp.json"
MCP_TOOL_PREFIX = "mcp__"
SERVER_NAME_RE = re.compile(r"^[a-z0-9_-]{1,32}$")
MAX_TOOL_NAME_CHARS = 64
_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

TRANSPORT_STDIO = "stdio"
TRANSPORT_HTTP = "http"
_HTTP_ALIASES = {"http", "streamable-http", "streamable_http", "streamablehttp"}
_STDIO_ALIASES = {"stdio", "stdio_server"}

SECRET_MASK = "********"


class McpConfigError(ValueError):
    """Raised when the MCP config file is invalid or cannot be parsed."""


# ---------------------------------------------------------------------------
# Server model
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class McpServerConfig:
    """A single configured MCP server (unexpanded, as stored on disk)."""

    name: str
    transport: str  # "stdio" | "http"
    enabled: bool = True
    # stdio
    command: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: str | None = None
    # http
    url: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    # Unknown keys preserved verbatim on round-trip.
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class McpConfig:
    path: Path
    servers: tuple[McpServerConfig, ...]

    def server_names(self) -> list[str]:
        return [server.name for server in self.servers]

    def get(self, name: str) -> McpServerConfig | None:
        for server in self.servers:
            if server.name == name:
                return server
        return None


@dataclass(frozen=True, slots=True)
class ResolvedServer:
    """A server config with ``${VAR}`` references expanded."""

    config: McpServerConfig
    transport: str
    command: str | None
    args: tuple[str, ...]
    env: Mapping[str, str]
    cwd: str | None
    url: str | None
    headers: Mapping[str, str]
    fingerprint: str
    missing_env: tuple[str, ...]


# ---------------------------------------------------------------------------
# Tool name helpers
# ---------------------------------------------------------------------------

def _sanitize_segment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", value)
    return cleaned.strip("_") or "x"


def qualify_tool_name(server: str, tool: str) -> str:
    """Return the model-facing tool name for ``tool`` on ``server``.

    Guarantees: matches ``[A-Za-z0-9_-]{1,64}``, starts with ``mcp__`` and is
    deterministic. Collisions after sanitisation are disambiguated with a
    short hash of the original pair.
    """
    safe_server = _sanitize_segment(server)[:24]
    safe_tool = _sanitize_segment(tool)
    base = f"{MCP_TOOL_PREFIX}{safe_server}__{safe_tool}"
    if safe_server == server and safe_tool == tool and len(base) <= MAX_TOOL_NAME_CHARS:
        return base
    digest = hashlib.sha256(f"{server}\x00{tool}".encode("utf-8")).hexdigest()[:8]
    tool_budget = MAX_TOOL_NAME_CHARS - len(MCP_TOOL_PREFIX) - len(safe_server) - 2 - 1 - len(digest)
    safe_tool = safe_tool[: max(1, tool_budget)]
    return f"{MCP_TOOL_PREFIX}{safe_server}__{safe_tool}-{digest}"


def is_mcp_tool_name(name: str) -> bool:
    return name.startswith(MCP_TOOL_PREFIX)


# ---------------------------------------------------------------------------
# Variable expansion
# ---------------------------------------------------------------------------

def _expand_string(value: str, environ: Mapping[str, str], missing: list[str]) -> str:
    def _replace(match: re.Match[str]) -> str:
        var = match.group(1)
        resolved = environ.get(var)
        if resolved is None:
            if var not in missing:
                missing.append(var)
            return ""
        return resolved

    return _VAR_RE.sub(_replace, value)


def _expand_mapping(
    values: Mapping[str, str],
    environ: Mapping[str, str],
    missing: list[str],
    *,
    expand_keys: bool = False,
) -> dict[str, str]:
    expanded: dict[str, str] = {}
    for key, value in values.items():
        out_key = _expand_string(key, environ, missing) if expand_keys else key
        expanded[out_key] = _expand_string(value, environ, missing)
    return expanded


def resolve_server(
    server: McpServerConfig,
    environ: Mapping[str, str] | None = None,
) -> ResolvedServer:
    """Expand ``${VAR}`` references and compute a stable fingerprint."""
    env_source = os.environ if environ is None else environ
    missing: list[str] = []

    command = _expand_string(server.command, env_source, missing) if server.command else None
    args = tuple(_expand_string(arg, env_source, missing) for arg in server.args)
    env = _expand_mapping(server.env, env_source, missing)
    cwd = _expand_string(server.cwd, env_source, missing) if server.cwd else None
    url = _expand_string(server.url, env_source, missing) if server.url else None
    headers = _expand_mapping(server.headers, env_source, missing)

    fingerprint_src = json.dumps(
        {
            "transport": server.transport,
            "command": command,
            "args": list(args),
            "env": env,
            "cwd": cwd,
            "url": url,
            "headers": headers,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(fingerprint_src.encode("utf-8")).hexdigest()

    return ResolvedServer(
        config=server,
        transport=server.transport,
        command=command,
        args=args,
        env=env,
        cwd=cwd,
        url=url,
        headers=headers,
        fingerprint=fingerprint,
        missing_env=tuple(missing),
    )


# ---------------------------------------------------------------------------
# Parsing / validation
# ---------------------------------------------------------------------------

def validate_server_name(name: str) -> str:
    candidate = (name or "").strip()
    if not SERVER_NAME_RE.match(candidate):
        raise McpConfigError(
            "Invalid server name "
            f"{name!r}: use 1-32 characters from [a-z0-9_-]."
        )
    return candidate


def _expect_str(value: Any, field: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise McpConfigError(f"Server {name!r}: {field} must be a non-empty string.")
    return value


def _expect_str_list(value: Any, field: str, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise McpConfigError(f"Server {name!r}: {field} must be a list of strings.")
    out: list[str] = []
    for item in value:
        if isinstance(item, (str, int, float, bool)):
            out.append(str(item))
        else:
            raise McpConfigError(f"Server {name!r}: {field} entries must be strings.")
    return tuple(out)


def _expect_str_map(value: Any, field: str, name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise McpConfigError(f"Server {name!r}: {field} must be an object of string values.")
    out: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise McpConfigError(f"Server {name!r}: {field} keys must be strings.")
        if not isinstance(item, str):
            raise McpConfigError(f"Server {name!r}: {field}[{key!r}] must be a string.")
        out[key] = item
    return out


def _server_from_dict(name: str, data: Any) -> McpServerConfig:
    if not isinstance(data, dict):
        raise McpConfigError(f"Server {name!r} must be a JSON object.")

    enabled = data.get("enabled", True)
    if not isinstance(enabled, bool):
        raise McpConfigError(f"Server {name!r}: 'enabled' must be a boolean.")

    command = data.get("command")
    url = data.get("url")
    transport_raw = data.get("transport")

    if transport_raw is not None:
        if not isinstance(transport_raw, str):
            raise McpConfigError(f"Server {name!r}: 'transport' must be a string.")
        normalized = transport_raw.strip().lower()
        if normalized in _STDIO_ALIASES:
            transport_raw = TRANSPORT_STDIO
        elif normalized in _HTTP_ALIASES:
            transport_raw = TRANSPORT_HTTP
        else:
            raise McpConfigError(
                f"Server {name!r}: unsupported transport {transport_raw!r}. "
                "Supported transports: 'stdio' and 'http' (Streamable HTTP)."
            )

    if command is not None and not isinstance(command, str):
        raise McpConfigError(f"Server {name!r}: 'command' must be a string.")
    if url is not None and not isinstance(url, str):
        raise McpConfigError(f"Server {name!r}: 'url' must be a string.")

    has_command = bool(command and command.strip())
    has_url = bool(url and url.strip())

    if transport_raw == TRANSPORT_STDIO:
        transport = TRANSPORT_STDIO
        if has_url:
            raise McpConfigError(f"Server {name!r}: http-only field 'url' is not allowed for stdio transport.")
    elif transport_raw == TRANSPORT_HTTP:
        transport = TRANSPORT_HTTP
        if has_command:
            raise McpConfigError(f"Server {name!r}: stdio field 'command' is not allowed for http transport.")
    elif has_command and has_url:
        raise McpConfigError(f"Server {name!r}: set either 'command' (stdio) or 'url' (http), not both.")
    elif has_command:
        transport = TRANSPORT_STDIO
    elif has_url:
        transport = TRANSPORT_HTTP
    else:
        raise McpConfigError(
            f"Server {name!r}: one of 'command' (stdio) or 'url' (http) is required."
        )

    if transport == TRANSPORT_STDIO and not has_command:
        raise McpConfigError(f"Server {name!r}: stdio transport requires 'command'.")
    if transport == TRANSPORT_HTTP:
        assert url is not None
        parsed = urlsplit(url.strip())
        if parsed.scheme not in {"http", "https"}:
            raise McpConfigError(
                f"Server {name!r}: 'url' must start with http:// or https://."
            )

    known = {
        "transport", "enabled", "command", "args", "env", "cwd", "url", "headers",
    }
    extra = {key: value for key, value in data.items() if key not in known}

    cwd = data.get("cwd")
    if cwd is not None and not isinstance(cwd, str):
        raise McpConfigError(f"Server {name!r}: 'cwd' must be a string.")

    return McpServerConfig(
        name=name,
        transport=transport,
        enabled=enabled,
        command=command.strip() if isinstance(command, str) else None,
        args=_expect_str_list(data.get("args"), "args", name),
        env=_expect_str_map(data.get("env"), "env", name),
        cwd=cwd if isinstance(cwd, str) and cwd.strip() else None,
        url=url.strip() if isinstance(url, str) else None,
        headers=_expect_str_map(data.get("headers"), "headers", name),
        extra=extra,
    )


def parse_servers(payload: Mapping[str, Any]) -> dict[str, McpServerConfig]:
    raw = payload.get("mcpServers")
    if raw is None:
        raw = payload.get("servers")  # tolerated alias
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise McpConfigError("'mcpServers' must be an object mapping names to server configs.")
    servers: dict[str, McpServerConfig] = {}
    for raw_name, raw_config in raw.items():
        name = validate_server_name(str(raw_name))
        # Preserve ordering; reject duplicates after name normalization.
        if name in servers:
            raise McpConfigError(f"Duplicate server name {name!r} in mcp.json.")
        servers[name] = _server_from_dict(name, raw_config)
    return servers


# ---------------------------------------------------------------------------
# Loading / saving
# ---------------------------------------------------------------------------

def default_mcp_config_path(user_dir: str | os.PathLike[str] | None = None) -> Path:
    if user_dir is not None:
        return Path(user_dir).expanduser().resolve() / CONFIG_FILE_NAME
    return user_mcp_config_path(resolve_user_dir())


def load_mcp_config(
    path: str | os.PathLike[str] | None = None,
    *,
    user_dir: str | os.PathLike[str] | None = None,
) -> McpConfig:
    """Load the config; a missing file yields an empty config.

    Raises :class:`McpConfigError` when the file exists but is invalid.
    """
    resolved = Path(path).expanduser() if path is not None else default_mcp_config_path(user_dir)
    if not resolved.is_file():
        return McpConfig(path=resolved, servers=())
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise McpConfigError(f"{resolved}: invalid JSON: {exc}") from exc
    except OSError as exc:
        raise McpConfigError(f"{resolved}: could not read file: {exc}") from exc
    if not isinstance(payload, dict):
        raise McpConfigError(f"{resolved}: top-level value must be a JSON object.")
    version = payload.get("version", CONFIG_VERSION)
    if not isinstance(version, int) or version > CONFIG_VERSION:
        raise McpConfigError(
            f"{resolved}: unsupported config version {version!r} "
            f"(this PyAgent understands up to version {CONFIG_VERSION})."
        )
    servers = parse_servers(payload)
    return McpConfig(path=resolved, servers=tuple(servers.values()))


def server_to_dict(server: McpServerConfig) -> dict[str, Any]:
    """Serialize a server config to its storage representation."""
    return _server_to_dict(server)


def server_from_dict(name: str, data: Mapping[str, Any]) -> McpServerConfig:
    """Validate a raw mapping into a server config (raises McpConfigError)."""
    return _server_from_dict(name, dict(data))


def _server_to_dict(server: McpServerConfig) -> dict[str, Any]:
    data: dict[str, Any] = dict(server.extra)
    data["enabled"] = server.enabled
    if server.transport == TRANSPORT_STDIO:
        data["command"] = server.command
        if server.args:
            data["args"] = list(server.args)
        if server.env:
            data["env"] = dict(server.env)
        if server.cwd:
            data["cwd"] = server.cwd
    else:
        data["url"] = server.url
        if server.headers:
            data["headers"] = dict(server.headers)
    # Explicit transport only when it cannot be inferred unambiguously.
    if server.transport not in (TRANSPORT_STDIO, TRANSPORT_HTTP):
        data["transport"] = server.transport
    return data


def save_mcp_config(
    servers: Mapping[str, McpServerConfig],
    path: str | os.PathLike[str] | None = None,
    *,
    user_dir: str | os.PathLike[str] | None = None,
) -> Path:
    resolved = Path(path).expanduser() if path is not None else default_mcp_config_path(user_dir)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CONFIG_VERSION,
        "mcpServers": {
            name: _server_to_dict(server) for name, server in servers.items()
        },
    }
    tmp = resolved.with_name(f".{resolved.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, resolved)
    return resolved


def _load_raw_servers(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, dict):
        return {}
    raw = payload.get("mcpServers", payload.get("servers"))
    if not isinstance(raw, dict):
        return {}
    return {str(name): dict(value) for name, value in raw.items() if isinstance(value, dict)}


def _validate_server_dicts(raw: Mapping[str, Any]) -> dict[str, McpServerConfig]:
    servers: dict[str, McpServerConfig] = {}
    for name, data in raw.items():
        validated_name = validate_server_name(str(name))
        servers[validated_name] = _server_from_dict(validated_name, data)
    return servers


def upsert_server(
    server: McpServerConfig,
    *,
    path: str | os.PathLike[str] | None = None,
    user_dir: str | os.PathLike[str] | None = None,
    replace: bool = False,
) -> McpServerConfig:
    """Insert or update ``server``; preserves unknown keys of other entries."""
    resolved = Path(path).expanduser() if path is not None else default_mcp_config_path(user_dir)
    raw = _load_raw_servers(resolved)
    if server.name in raw and not replace:
        raise McpConfigError(f"Server {server.name!r} already exists in {resolved}.")
    base = dict(raw.get(server.name, {}))
    base.update(_server_to_dict(server))
    raw[server.name] = base
    parsed = _validate_server_dicts(raw)
    save_mcp_config(parsed, resolved)
    return parsed[server.name]


def remove_server(
    name: str,
    *,
    path: str | os.PathLike[str] | None = None,
    user_dir: str | os.PathLike[str] | None = None,
) -> bool:
    resolved = Path(path).expanduser() if path is not None else default_mcp_config_path(user_dir)
    raw = _load_raw_servers(resolved)
    key = name.strip()
    if key not in raw:
        return False
    del raw[key]
    parsed = _validate_server_dicts(raw)
    save_mcp_config(parsed, resolved)
    return True


def set_server_enabled(
    name: str,
    enabled: bool,
    *,
    path: str | os.PathLike[str] | None = None,
    user_dir: str | os.PathLike[str] | None = None,
) -> McpServerConfig | None:
    resolved = Path(path).expanduser() if path is not None else default_mcp_config_path(user_dir)
    raw = _load_raw_servers(resolved)
    key = name.strip()
    if key not in raw:
        return None
    raw[key] = {**raw[key], "enabled": bool(enabled)}
    parsed = _validate_server_dicts(raw)
    save_mcp_config(parsed, resolved)
    return parsed[key]


# ---------------------------------------------------------------------------
# Redaction for display / API responses
# ---------------------------------------------------------------------------

def _mask_url(url: str) -> str:
    try:
        parts = urlsplit(url)
    except ValueError:
        return SECRET_MASK
    netloc = parts.netloc
    if "@" in netloc:
        userinfo, host = netloc.rsplit("@", 1)
        user = userinfo.split(":", 1)[0]
        netloc = f"{user}:{SECRET_MASK}@{host}"
    query = "<redacted>" if parts.query else ""
    fragment = "<redacted>" if parts.fragment else ""
    return urlunsplit((parts.scheme, netloc, parts.path, query, fragment))


def redacted_server_dict(server: McpServerConfig) -> dict[str, Any]:
    """Config suitable for display: secrets replaced by a fixed mask."""
    data: dict[str, Any] = {
        "name": server.name,
        "transport": server.transport,
        "enabled": server.enabled,
    }
    if server.transport == TRANSPORT_STDIO:
        data["command"] = server.command
        data["args"] = list(server.args)
        data["cwd"] = server.cwd
        data["env"] = {key: SECRET_MASK for key in server.env}
    else:
        data["url"] = _mask_url(server.url or "")
        data["headers"] = {key: SECRET_MASK for key in server.headers}
    missing = resolve_server(server).missing_env
    if missing:
        data["missing_env"] = list(dict.fromkeys(missing))
    return data
