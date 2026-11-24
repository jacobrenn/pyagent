"""Persistent, pooled connections to MCP (Model Context Protocol) servers.

The MCP SDK is asyncio-based while PyAgent's tool handlers are synchronous,
so this module runs a single background event loop in a dedicated thread and
bridges calls into it with :func:`asyncio.run_coroutine_threadsafe`.

Sessions are pooled process-wide and keyed by a fingerprint of the *effective*
server configuration (after ``${VAR}`` expansion). Agent instances are cheap
and numerous; connecting per agent would respawn stdio subprocesses on every
API request, so they share sessions. ``reload()`` drops pooled sessions for
configs that changed or vanished.

Nothing in this module raises for ordinary failures (unreachable servers,
bad configs): problems are reported per server so the rest of the registry
keeps working. Failures from network/spawn operations are cached briefly so
a dead server does not stall every tool listing.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, field
import atexit
import json
import subprocess
import threading
import time
from typing import Any

from .mcp_config import (
    ResolvedServer,
    McpConfigError,
    default_mcp_config_path,
    load_mcp_config,
    qualify_tool_name,
    resolve_server,
)
from .tools import MAX_TOOL_OUTPUT_CHARS, ToolSpec, _truncate
from .user_runtime import resolve_user_dir, user_log_dir

DEFAULT_CONNECT_TIMEOUT = 15.0
DEFAULT_CALL_TIMEOUT = 60.0
FAILURE_CACHE_TTL = 30.0
MAX_LIST_TOOLS_PAGES = 16
MAX_TOOLS_PER_SERVER = 512
CLOSE_TIMEOUT = 10.0

SOURCE_PREFIX = "mcp:"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class McpServerStatus:
    name: str
    transport: str
    enabled: bool
    status: str  # "ok" | "error" | "disabled"
    tool_count: int = 0
    error: str | None = None
    tool_names: list[str] = field(default_factory=list)


@dataclass
class McpDiscoveryResult:
    path: str
    servers: list[McpServerStatus] = field(default_factory=list)
    specs: list[ToolSpec] = field(default_factory=list)
    error: str | None = None

    @property
    def tools(self) -> list[ToolSpec]:
        return self.specs

    @property
    def ok(self) -> bool:
        return self.error is None

    def server_status(self, name: str) -> McpServerStatus | None:
        for server in self.servers:
            if server.name == name:
                return server
        return None

    def counts(self) -> tuple[int, int, int]:
        """Return ``(ok, failed, disabled)`` counts."""
        ok = sum(1 for s in self.servers if s.status == "ok")
        failed = sum(1 for s in self.servers if s.status == "error")
        disabled = sum(1 for s in self.servers if s.status == "disabled")
        return ok, failed, disabled


# ---------------------------------------------------------------------------
# Tool handler bridging sync -> async
# ---------------------------------------------------------------------------

class McpToolHandler:
    """Sync callable registered with the tool registry."""

    def __init__(self, manager: "McpManager", fingerprint: str, server: str, tool: str):
        self._manager = manager
        self.fingerprint = fingerprint
        self.server = server
        self.tool = tool
        # Surfaced as the registry's ``source`` for listings.
        self.script_path = f"{SOURCE_PREFIX}{server}"

    def __call__(self, **arguments: Any) -> str:
        args = arguments if isinstance(arguments, dict) else {}
        return self._manager.call_tool(self.fingerprint, self.tool, args)


# ---------------------------------------------------------------------------
# Internal session bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class _SessionEntry:
    name: str
    fingerprint: str
    server: ResolvedServer
    stack: AsyncExitStack
    session: Any
    raw_tools: list[Any]


def _render_content(blocks: Any) -> str:
    parts: list[str] = []
    for block in blocks or []:
        kind = getattr(block, "type", None)
        if kind == "text":
            parts.append(getattr(block, "text", "") or "")
        elif kind == "image":
            parts.append(
                f"[image omitted: {getattr(block, 'mime_type', 'image')}]")
        elif kind == "audio":
            parts.append(
                f"[audio omitted: {getattr(block, 'mime_type', 'audio')}]")
        elif kind == "resource":
            resource = getattr(block, "resource", None)
            text = getattr(resource, "text", None)
            if isinstance(text, str):
                parts.append(text)
            else:
                parts.append(
                    f"[resource omitted: {getattr(resource, 'uri', 'unknown URI')}]")
        elif kind == "resource_link":
            name = getattr(block, "name", "")
            uri = getattr(block, "uri", "")
            parts.append(f"[resource link: {name} — {uri}]")
        else:
            parts.append(f"[{kind or 'unknown'} content omitted]")
    return "\n".join(part for part in parts if part)


def render_tool_result(result: Any) -> str:
    """Convert an MCP ``CallToolResult`` into the string the agent loop expects."""
    is_error = bool(getattr(result, "is_error", False))
    content = getattr(result, "content", None)
    if content is None and not hasattr(result, "content"):
        # e.g. InputRequiredResult: interactive input is not supported.
        return "Error: MCP tool requested interactive input, which PyAgent does not support."
    text = _render_content(content)
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        try:
            is_redundant_result = (
                isinstance(structured, dict)
                and set(structured) == {"result"}
                and str(structured["result"]) == text.strip()
            )
        except Exception:
            is_redundant_result = False
        if not is_redundant_result:
            try:
                structured_text = json.dumps(
                    structured, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                structured_text = str(structured)
            text = f"{text}\n\n{structured_text}".strip(
            ) if text else structured_text
    if not text:
        text = "<empty result>"
    if is_error:
        text = f"Error: {text}"
    return _truncate(text, MAX_TOOL_OUTPUT_CHARS)


def _json_schema(value: Any) -> dict[str, Any]:
    fallback = {"type": "object", "properties": {}}
    if not isinstance(value, dict):
        return fallback
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return fallback


class McpManager:
    """Owns the background loop and the pooled server sessions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._sessions: dict[str, _SessionEntry] = {}
        self._failures: dict[str, tuple[float, str]] = {}
        self._log_dir = None

    def _set_log_dir(self, user_dir: str | None) -> None:
        try:
            log_dir = user_log_dir(resolve_user_dir(user_dir))
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_dir = log_dir
        except OSError:
            self._log_dir = None

    def _stdio_errlog(self, res: ResolvedServer):
        """Open a per-server stderr log; fall back to DEVNULL.

        MCP servers may be noisy on stderr; routing it to a file keeps the
        TUI and API logs clean. The handle stays open for the life of the
        session and is closed when the session stack closes.
        """
        if self._log_dir is not None:
            try:
                safe = "".join(
                    ch if ch.isalnum() or ch in "-_" else "_" for ch in res.config.name
                )
                path = self._log_dir / f"mcp-{safe}-{res.fingerprint[:8]}.log"
                return open(path, "a", encoding="utf-8")
            except OSError:
                pass
        return subprocess.DEVNULL

    # -- loop management ---------------------------------------------------

    def _ensure_loop_locked(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None:
            return self._loop
        ready = threading.Event()

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            ready.set()
            loop.run_forever()
            with suppress(Exception):
                loop.run_until_complete(loop.shutdown_asyncgens())

        self._thread = threading.Thread(
            target=_run, name="pyagent-mcp-loop", daemon=True
        )
        self._thread.start()
        ready.wait(timeout=5)
        if self._loop is None:
            raise RuntimeError("Failed to start the MCP event loop thread.")
        return self._loop

    def _submit(self, coro, timeout: float) -> Any:
        loop = self._ensure_loop_locked()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout)

    # -- discovery -----------------------------------------------------------

    def ensure_tools(
        self,
        *,
        user_dir: str | None = None,
        path: str | None = None,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    ) -> McpDiscoveryResult:
        """Return specs for all enabled servers, connecting as needed."""
        with self._lock:
            self._set_log_dir(user_dir)
            try:
                config = load_mcp_config(path, user_dir=user_dir)
            except McpConfigError as exc:
                return McpDiscoveryResult(
                    path=str(path or default_mcp_config_path(user_dir)),
                    error=str(exc),
                )

            enabled = [s for s in config.servers if s.enabled]
            disabled = [s for s in config.servers if not s.enabled]
            resolved: list[ResolvedServer] = []
            pre_errors: dict[str, str] = {}
            for server in enabled:
                res = resolve_server(server)
                if res.missing_env:
                    pre_errors[server.name] = (
                        "Missing environment variable(s): "
                        + ", ".join(res.missing_env)
                    )
                else:
                    resolved.append(res)

            # Drop sessions whose fingerprint is no longer configured.
            self._prune_locked({res.fingerprint for res in resolved})

            # Attach cached failures that are still fresh.
            now = time.monotonic()
            pending: list[ResolvedServer] = []
            cached_failure: dict[str, str] = {}
            for res in resolved:
                if res.fingerprint in self._sessions:
                    continue
                failure = self._failures.get(res.fingerprint)
                if failure and failure[0] > now:
                    cached_failure[res.config.name] = failure[1]
                    continue
                pending.append(res)

            if pending:
                self._connect_pending(pending, connect_timeout)

            servers: list[McpServerStatus] = []
            specs: list[ToolSpec] = []
            for server in config.servers:
                if not server.enabled:
                    servers.append(
                        McpServerStatus(
                            name=server.name,
                            transport=server.transport,
                            enabled=False,
                            status="disabled",
                        )
                    )
                    continue
                if server.name in pre_errors:
                    servers.append(
                        McpServerStatus(
                            name=server.name,
                            transport=server.transport,
                            enabled=True,
                            status="error",
                            error=pre_errors[server.name],
                        )
                    )
                    continue
                res = next(r for r in resolved if r.config.name == server.name)
                entry = self._sessions.get(res.fingerprint)
                if entry is None:
                    message = cached_failure.get(server.name) or self._failures.get(
                        res.fingerprint, (0.0, "Failed to start MCP server.")
                    )[1]
                    servers.append(
                        McpServerStatus(
                            name=server.name,
                            transport=server.transport,
                            enabled=True,
                            status="error",
                            error=message,
                        )
                    )
                    continue
                server_specs = _specs_for_entry(self, entry)
                specs.extend(server_specs)
                servers.append(
                    McpServerStatus(
                        name=server.name,
                        transport=server.transport,
                        enabled=True,
                        status="ok",
                        tool_count=len(server_specs),
                        tool_names=[spec.name for spec in server_specs],
                    )
                )

            return McpDiscoveryResult(path=str(config.path), servers=servers, specs=specs)

    def _connect_pending(self, pending: list[ResolvedServer], timeout: float) -> None:
        async def _gather() -> list[tuple[ResolvedServer, _SessionEntry | None, str | None]]:
            async def _one(res: ResolvedServer):
                try:
                    entry = await asyncio.wait_for(
                        self._connect_one(res), timeout=timeout
                    )
                    return res, entry, None
                except Exception as exc:  # noqa: BLE001 - surfaced to the user
                    return res, None, _format_connect_error(exc)

            return await asyncio.gather(*[_one(res) for res in pending])

        total_timeout = max(timeout * 2, timeout + 5.0)
        loop = self._ensure_loop_locked()
        future = asyncio.run_coroutine_threadsafe(_gather(), loop)
        try:
            results = future.result(total_timeout)
        except Exception as exc:  # timed out or loop died
            future.cancel()
            message = f"Timed out connecting to MCP server(s): {exc}"
            for res in pending:
                self._failures[res.fingerprint] = (
                    time.monotonic() + FAILURE_CACHE_TTL,
                    message,
                )
            return

        for res, entry, error in results:
            if entry is not None:
                self._sessions[res.fingerprint] = entry
                self._failures.pop(res.fingerprint, None)
            else:
                self._failures[res.fingerprint] = (
                    time.monotonic() + FAILURE_CACHE_TTL,
                    error or "Failed to start MCP server.",
                )

    async def _connect_one(self, res: ResolvedServer) -> _SessionEntry:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import get_default_environment, stdio_client
        except ImportError as exc:
            raise RuntimeError(
                "The 'mcp' package is required for MCP support. "
                "Reinstall pyagent-harness (mcp is a core dependency)."
            ) from exc

        stack = AsyncExitStack()
        try:
            if res.transport == "stdio":
                env = {**get_default_environment(), **dict(res.env)}
                params = StdioServerParameters(
                    command=res.command or "",
                    args=list(res.args),
                    env=env,
                    cwd=res.cwd,
                )
                errlog = self._stdio_errlog(res)
                if errlog is not subprocess.DEVNULL:
                    stack.push_async_callback(_close_quietly, errlog)
                read, write = await stack.enter_async_context(
                    stdio_client(params, errlog=errlog)
                )
            else:
                read, write = await self._enter_http_transport(stack, res)

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            raw_tools = await self._list_all_tools(session)
            return _SessionEntry(
                name=res.config.name,
                fingerprint=res.fingerprint,
                server=res,
                stack=stack,
                session=session,
                raw_tools=raw_tools,
            )
        except BaseException:
            with suppress(Exception):
                await stack.aclose()
            raise

    async def _enter_http_transport(self, stack: AsyncExitStack, res: ResolvedServer):
        try:
            from mcp.client.streamable_http import (  # type: ignore[import-not-found]
                create_mcp_http_client,
                streamable_http_client,
            )
        except ImportError:  # very old SDK layouts
            from mcp.client.streamable_http import streamablehttp_client  # type: ignore

            read, write = await stack.enter_async_context(
                streamablehttp_client(
                    res.url or "", headers=dict(res.headers) or None)
            )
            return read, write

        http_client = create_mcp_http_client(
            headers=dict(res.headers) or None,
        )
        stack.push_async_callback(http_client.aclose)
        return await stack.enter_async_context(
            streamable_http_client(res.url or "", http_client=http_client)
        )

    async def _list_all_tools(self, session: Any) -> list[Any]:
        from mcp.types import PaginatedRequestParams

        tools: list[Any] = []
        cursor: str | None = None
        for _ in range(MAX_LIST_TOOLS_PAGES):
            params = PaginatedRequestParams(cursor=cursor) if cursor else None
            result = await session.list_tools(params=params)
            tools.extend(getattr(result, "tools", []) or [])
            cursor = getattr(result, "next_cursor", None)
            if not cursor or len(tools) >= MAX_TOOLS_PER_SERVER:
                break
        return tools[:MAX_TOOLS_PER_SERVER]

    def _prune_locked(self, wanted_fingerprints: set[str]) -> None:
        stale = [fp for fp in self._sessions if fp not in wanted_fingerprints]
        if not stale:
            for key in list(self._failures):
                if key not in wanted_fingerprints:
                    del self._failures[key]
            return
        entries = [self._sessions.pop(fp) for fp in stale]
        for fp in stale:
            self._failures.pop(fp, None)
        with suppress(Exception):
            self._submit(self._aclose_entries(entries), CLOSE_TIMEOUT)

    @staticmethod
    async def _aclose_entries(entries: list[_SessionEntry]) -> None:
        for entry in entries:
            with suppress(Exception):
                await entry.stack.aclose()

    # -- tool calls ----------------------------------------------------------

    def call_tool(
        self,
        fingerprint: str,
        tool: str,
        arguments: dict[str, Any],
        *,
        timeout: float = DEFAULT_CALL_TIMEOUT,
    ) -> str:
        with self._lock:
            entry = self._sessions.get(fingerprint)
        if entry is None:
            return (
                "Error: MCP server session is not available (config changed or "
                "server failed). Run `/mcp reload` or `pyagent mcp list` to check server status."
            )

        async def _call() -> Any:
            return await asyncio.wait_for(
                entry.session.call_tool(tool, arguments or {}),
                timeout=timeout,
            )

        try:
            result = self._submit(_call(), timeout + 5.0)
        except TimeoutError:
            return f"Error: MCP tool '{tool}' timed out after {timeout:.0f}s."
        except Exception as exc:  # noqa: BLE001 - surface as tool output
            message = str(exc).strip() or exc.__class__.__name__
            return _truncate(f"Error: MCP tool '{tool}' failed: {message}", MAX_TOOL_OUTPUT_CHARS)
        return render_tool_result(result)

    def test_server(
        self,
        name: str,
        *,
        user_dir: str | None = None,
        path: str | None = None,
        timeout: float = DEFAULT_CONNECT_TIMEOUT,
    ) -> dict[str, Any]:
        """Connect to one server right now and report status and tools."""
        try:
            config = load_mcp_config(path, user_dir=user_dir)
        except McpConfigError as exc:
            return {"server": name, "ok": False, "error": str(exc)}
        server = config.get(name)
        if server is None:
            return {"server": name, "ok": False, "error": f"No MCP server named {name!r} is configured."}
        if not server.enabled:
            return {"server": name, "ok": False, "error": f"Server {name!r} is disabled."}
        self._set_log_dir(user_dir)
        res = resolve_server(server)
        if res.missing_env:
            return {
                "server": name,
                "ok": False,
                "error": "Missing environment variable(s): " + ", ".join(res.missing_env),
            }
        started = time.monotonic()
        with self._lock:
            entry = self._sessions.get(res.fingerprint)
            if entry is None:
                if res.fingerprint in self._failures:
                    # Bypass a stale negative cache entry for an explicit test.
                    del self._failures[res.fingerprint]
                try:
                    self._connect_pending([res], timeout)
                except Exception as exc:  # noqa: BLE001
                    return {"server": name, "ok": False, "error": str(exc)}
                entry = self._sessions.get(res.fingerprint)
        latency_ms = round((time.monotonic() - started) * 1000)
        if entry is None:
            _, message = self._failures.get(
                res.fingerprint, (0.0, "Connection failed."))
            return {"server": name, "ok": False, "error": message}
        return {
            "server": name,
            "ok": True,
            "transport": server.transport,
            "latency_ms": latency_ms,
            "tool_count": len(entry.raw_tools),
            "tools": [getattr(tool, "name", "?") for tool in entry.raw_tools],
        }

    # -- teardown ------------------------------------------------------------

    def reload(
        self,
        *,
        user_dir: str | None = None,
        path: str | None = None,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    ) -> McpDiscoveryResult:
        with self._lock:
            self._set_log_dir(user_dir)
            entries = list(self._sessions.values())
            self._sessions.clear()
            self._failures.clear()
            if entries:
                with suppress(Exception):
                    self._submit(self._aclose_entries(entries), CLOSE_TIMEOUT)
        return self.ensure_tools(
            user_dir=user_dir, path=path, connect_timeout=connect_timeout
        )

    def shutdown(self) -> None:
        with self._lock:
            entries = list(self._sessions.values())
            self._sessions.clear()
            self._failures.clear()
            loop = self._loop
            self._loop = None
        if loop is not None and loop.is_running():
            if entries:
                with suppress(Exception):
                    future = asyncio.run_coroutine_threadsafe(
                        self._aclose_entries(entries), loop
                    )
                    future.result(CLOSE_TIMEOUT)
            loop.call_soon_threadsafe(loop.stop)
            thread = self._thread
            if thread is not None:
                thread.join(timeout=5)
        if loop is not None:
            with suppress(Exception):
                loop.close()
        self._thread = None


def _close_quietly(stream) -> None:
    with suppress(Exception):
        stream.close()


def _specs_for_entry(manager: McpManager, entry: _SessionEntry) -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    seen: set[str] = set()
    for tool in entry.raw_tools:
        raw_name = str(getattr(tool, "name", "") or "").strip()
        if not raw_name:
            continue
        qualified = qualify_tool_name(entry.name, raw_name)
        if qualified in seen:
            # Extremely unlikely (hash-suffixed), but never emit duplicates.
            continue
        seen.add(qualified)
        description = (getattr(tool, "description", None) or raw_name).strip()
        schema = _json_schema(getattr(tool, "input_schema", None))
        specs.append(
            ToolSpec(
                name=qualified,
                description=f"[MCP: {entry.name}] {description}",
                parameters=schema,
                handler=McpToolHandler(
                    manager, entry.fingerprint, entry.name, raw_name),
            )
        )
    return specs


def _format_connect_error(exc: BaseException) -> str:
    message = str(exc).strip()
    name = exc.__class__.__name__
    if isinstance(exc, TimeoutError) or "TimeoutError" in name:
        return f"Connection timed out ({message})" if message else "Connection timed out"
    if not message:
        return f"Failed to start MCP server ({name})"
    # Collapse multi-line subprocess output to keep status lines readable.
    first_line = message.splitlines()[0]
    return f"{first_line} ({name})" if name not in first_line else first_line


_MANAGER: McpManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_mcp_manager() -> McpManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = McpManager()
            atexit.register(_shutdown_at_exit)
        return _MANAGER


def shutdown_mcp_manager() -> None:
    """Close all pooled sessions and stop the background loop.

    Intended for tests and orderly embedding; normal interpreter shutdown is
    handled by an atexit hook.
    """
    global _MANAGER
    with _MANAGER_LOCK:
        manager = _MANAGER
        _MANAGER = None
    if manager is not None:
        with suppress(Exception):
            manager.shutdown()


def _shutdown_at_exit() -> None:
    shutdown_mcp_manager()
