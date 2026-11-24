"""Tests for MCP (Model Context Protocol) support."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from pyagent import mcp_config
from pyagent.mcp_config import (
    McpConfigError,
    McpServerConfig,
    load_mcp_config,
    qualify_tool_name,
    redacted_server_dict,
    resolve_server,
    save_mcp_config,
    server_from_dict,
    set_server_enabled,
    upsert_server,
    remove_server,
    validate_server_name,
)
from pyagent.mcp_runtime import (
    get_mcp_manager,
    render_tool_result,
    shutdown_mcp_manager,
)
from pyagent.tools import MCP_ORIGIN, create_default_tool_registry
from pyagent.tools import ToolSpec

try:
    import mcp  # noqa: F401
    MCP_AVAILABLE = True
except ImportError:  # pragma: no cover
    MCP_AVAILABLE = False

FAKE_SERVER = '''
from mcp.server.mcpserver import MCPServer

server = MCPServer(name="fake", version="0.1.0")

@server.tool()
def echo(text: str) -> str:
    """Echo the input text back."""
    return f"echo: {text}"

@server.tool()
def add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)

@server.tool()
def fail() -> str:
    """Always fails."""
    raise RuntimeError("boom")

if __name__ == "__main__":
    server.run("stdio")
'''


def write_fake_server(directory: Path) -> Path:
    path = directory / "fake_mcp_server.py"
    path.write_text(FAKE_SERVER)
    return path


def write_config(user_dir: Path, servers: dict) -> Path:
    user_dir.mkdir(parents=True, exist_ok=True)
    path = user_dir / "mcp.json"
    path.write_text(json.dumps({"version": 1, "mcpServers": servers}))
    return path


class McpConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.user_dir = Path(self.tmp.name)

    def test_missing_file_yields_empty_config(self):
        config = load_mcp_config(user_dir=str(self.user_dir))
        self.assertEqual(config.servers, ())

    def test_round_trip_stdio_and_http(self):
        path = self.user_dir / "mcp.json"
        servers = {
            "fake": McpServerConfig(
                name="fake", transport="stdio", command="python3",
                args=("server.py",), env={"TOKEN": "abc"}, cwd="/tmp",
            ),
            "web": McpServerConfig(
                name="web", transport="http", url="https://example.com/mcp",
                headers={"Authorization": "Bearer xyz"},
            ),
        }
        save_mcp_config(servers, path)
        self.assertEqual((path.stat().st_mode & 0o777), 0o600)
        loaded = load_mcp_config(path)
        self.assertEqual([s.name for s in loaded.servers], ["fake", "web"])
        fake = loaded.get("fake")
        self.assertEqual(fake.args, ("server.py",))
        self.assertEqual(fake.env, {"TOKEN": "abc"})
        self.assertEqual(loaded.get("web").headers, {"Authorization": "Bearer xyz"})

    def test_unknown_server_keys_preserved(self):
        path = self.user_dir / "mcp.json"
        path.write_text(json.dumps({
            "version": 1,
            "mcpServers": {"fake": {"command": "x", "futureField": {"a": 1}}},
        }))
        config = load_mcp_config(path)
        self.assertEqual(config.get("fake").extra, {"futureField": {"a": 1}})
        upsert_server(config.get("fake"), path=path, replace=True)
        raw = json.loads(path.read_text())
        self.assertEqual(raw["mcpServers"]["fake"]["futureField"], {"a": 1})

    def test_invalid_json_raises(self):
        path = self.user_dir / "mcp.json"
        path.write_text("{not json")
        with self.assertRaises(McpConfigError):
            load_mcp_config(path)

    def test_version_too_new_rejected(self):
        path = self.user_dir / "mcp.json"
        path.write_text(json.dumps({"version": 999, "mcpServers": {}}))
        with self.assertRaises(McpConfigError):
            load_mcp_config(path)

    def test_conflicting_transport_rejected(self):
        with self.assertRaises(McpConfigError):
            server_from_dict("x", {"command": "run", "url": "https://a.example.com/mcp"})

    def test_missing_transport_rejected(self):
        with self.assertRaises(McpConfigError):
            server_from_dict("x", {})

    def test_http_requires_http_scheme(self):
        with self.assertRaises(McpConfigError):
            server_from_dict("x", {"url": "ftp://example.com/mcp"})

    def test_invalid_server_name_rejected(self):
        for bad in ("", "Has Space", "ümlaut", "a" * 33, "dot.name"):
            with self.assertRaises(McpConfigError, msg=bad):
                validate_server_name(bad)
        self.assertEqual(validate_server_name("ok-name_1"), "ok-name_1")

    def test_env_expansion_and_missing(self):
        server = McpServerConfig(
            name="s", transport="stdio", command="run",
            args=("--token", "${MCP_TEST_TOKEN}"),
            env={"KEY": "${MCP_TEST_KEY}"},
        )
        resolved = resolve_server(server, environ={"MCP_TEST_TOKEN": "abc"})
        self.assertEqual(resolved.args, ("--token", "abc"))
        self.assertEqual(resolved.missing_env, ("MCP_TEST_KEY",))
        self.assertEqual(resolved.env["KEY"], "")

    def test_fingerprint_changes_with_config(self):
        base = McpServerConfig(name="s", transport="stdio", command="a")
        other = McpServerConfig(name="s", transport="stdio", command="b")
        self.assertNotEqual(
            resolve_server(base, environ={}).fingerprint,
            resolve_server(other, environ={}).fingerprint,
        )

    def test_redaction_masks_secrets(self):
        server = McpServerConfig(
            name="s", transport="http",
            url="https://user:pass@example.com/mcp?token=sekret",
            headers={"Authorization": "Bearer abc"},
        )
        redacted = redacted_server_dict(server)
        self.assertNotIn("pass", redacted["url"])
        self.assertNotIn("sekret", redacted["url"])
        self.assertEqual(redacted["headers"]["Authorization"], "********")

    def test_upsert_remove_enable(self):
        path = self.user_dir / "mcp.json"
        server = McpServerConfig(name="s1", transport="stdio", command="true")
        upsert_server(server, path=path)
        self.assertEqual(len(load_mcp_config(path).servers), 1)
        with self.assertRaises(McpConfigError):
            upsert_server(server, path=path)  # no replace
        updated = set_server_enabled("s1", False, path=path)
        self.assertFalse(updated.enabled)
        self.assertTrue(remove_server("s1", path=path))
        self.assertFalse(remove_server("s1", path=path))

    def test_upsert_preserves_unknown_keys_of_entry(self):
        path = self.user_dir / "mcp.json"
        path.write_text(json.dumps({
            "version": 1,
            "mcpServers": {"s1": {"command": "old", "futureField": 42}},
        }))
        server = McpServerConfig(name="s1", transport="stdio", command="new")
        upsert_server(server, path=path, replace=True)
        raw = json.loads(path.read_text())["mcpServers"]["s1"]
        self.assertEqual(raw["command"], "new")
        self.assertEqual(raw["futureField"], 42)


class QualifyToolNameTests(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(qualify_tool_name("fake", "echo"), "mcp__fake__echo")

    def test_special_characters_sanitized(self):
        name = qualify_tool_name("my server", "tool/name.with.dots")
        self.assertRegex(name, r"^mcp__my_server__tool_name_with_dots-[0-9a-f]{8}$")

    def test_long_names_truncated_to_64(self):
        name = qualify_tool_name("s" * 24, "t" * 100)
        self.assertLessEqual(len(name), 64)
        self.assertTrue(name.startswith("mcp__"))

    def test_uniqueness_after_sanitization(self):
        a = qualify_tool_name("a b", "x")
        b = qualify_tool_name("a_b", "x")
        self.assertNotEqual(a, b)


class _McpTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.user_dir = Path(self.tmp.name) / "user"
        self.user_dir.mkdir(parents=True)
        self.addCleanup(shutdown_mcp_manager)

    def write_fake(self, name="fake"):
        server_path = write_fake_server(Path(self.tmp.name))
        return server_path

    def make_config(self, servers: dict) -> None:
        write_config(self.user_dir, servers)

    def fake_server_entry(self):
        script = write_fake_server(Path(self.tmp.name))
        return {"command": sys.executable, "args": [str(script)]}


@unittest.skipUnless(MCP_AVAILABLE, "mcp package not installed")
class McpRuntimeTests(_McpTestCase):
    def test_discovery_and_call(self):
        self.make_config({"fake": self.fake_server_entry()})
        result = get_mcp_manager().ensure_tools(user_dir=str(self.user_dir), connect_timeout=10)
        self.assertEqual([s.status for s in result.servers], ["ok"])
        names = [spec.name for spec in result.specs]
        self.assertEqual(
            sorted(names), ["mcp__fake__add", "mcp__fake__echo", "mcp__fake__fail"])
        echo = next(s for s in result.specs if s.name == "mcp__fake__echo")
        self.assertEqual(echo.handler(text="hi").strip().splitlines()[0], "echo: hi")

    def test_tool_error_is_reported_not_raised(self):
        self.make_config({"fake": self.fake_server_entry()})
        result = get_mcp_manager().ensure_tools(user_dir=str(self.user_dir), connect_timeout=10)
        fail = next(s for s in result.specs if s.name == "mcp__fake__fail")
        self.assertTrue(fail.handler().startswith("Error:"))

    def test_broken_server_does_not_break_discovery(self):
        self.make_config({
            "ghost": {"command": "definitely-not-a-real-binary-xyz"},
        })
        result = get_mcp_manager().ensure_tools(user_dir=str(self.user_dir), connect_timeout=5)
        self.assertEqual(result.servers[0].status, "error")
        self.assertEqual(result.specs, [])

    def test_missing_env_var_reported(self):
        entry = self.fake_server_entry()
        entry["env"] = {"TOKEN": "${PYAGENT_TEST_MISSING_VAR_XYZ}"}
        self.make_config({"fake": entry})
        result = get_mcp_manager().ensure_tools(user_dir=str(self.user_dir), connect_timeout=5)
        self.assertEqual(result.servers[0].status, "error")
        self.assertIn("PYAGENT_TEST_MISSING_VAR_XYZ", result.servers[0].error)

    def test_disabled_server_not_connected(self):
        entry = self.fake_server_entry()
        entry["enabled"] = False
        self.make_config({"fake": entry})
        result = get_mcp_manager().ensure_tools(user_dir=str(self.user_dir), connect_timeout=5)
        self.assertEqual(result.servers[0].status, "disabled")
        self.assertEqual(result.specs, [])

    def test_sessions_are_pooled(self):
        self.make_config({"fake": self.fake_server_entry()})
        manager = get_mcp_manager()
        manager.ensure_tools(user_dir=str(self.user_dir), connect_timeout=10)
        session_count = len(manager._sessions)
        manager.ensure_tools(user_dir=str(self.user_dir), connect_timeout=10)
        self.assertEqual(len(manager._sessions), session_count)

    def test_config_change_reconnects(self):
        self.make_config({"fake": self.fake_server_entry()})
        manager = get_mcp_manager()
        first = manager.ensure_tools(user_dir=str(self.user_dir), connect_timeout=10)
        first_fp = next(iter(manager._sessions))
        self.make_config({"fake": {**self.fake_server_entry(), "env": {"EXTRA": "1"}}})
        second = manager.ensure_tools(user_dir=str(self.user_dir), connect_timeout=10)
        second_fp = next(iter(manager._sessions))
        self.assertNotEqual(first_fp, second_fp)
        self.assertEqual(second.servers[0].status, "ok")

    def test_reload(self):
        self.make_config({"fake": self.fake_server_entry()})
        manager = get_mcp_manager()
        result = manager.reload(user_dir=str(self.user_dir), connect_timeout=10)
        self.assertEqual(result.servers[0].status, "ok")

    def test_test_server_reports_tools(self):
        self.make_config({"fake": self.fake_server_entry()})
        result = get_mcp_manager().test_server("fake", user_dir=str(self.user_dir), timeout=10)
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool_count"], 3)

    def test_registry_collision_prefers_builtin_and_external(self):
        spec = ToolSpec(
            name="list_files",
            description="mcp imposter",
            parameters={"type": "object", "properties": {}},
            handler=lambda **_: "mcp",
        )
        from pyagent.tools import MCP_ORIGIN, BUILTIN_ORIGIN, EXTERNAL_ORIGIN

        registry = create_default_tool_registry(mcp_specs=[spec])
        self.assertEqual(registry.origin("list_files"), BUILTIN_ORIGIN)
        self.assertIn("mcp", [c.loser_origin for c in registry.collisions()])
        registry = create_default_tool_registry(
            external_specs=[ToolSpec(
                name="mcp__fake__echo", description="x",
                parameters={"type": "object", "properties": {}},
                handler=lambda **_: "external",
            )],
            mcp_specs=[ToolSpec(
                name="mcp__fake__echo", description="y",
                parameters={"type": "object", "properties": {}},
                handler=lambda **_: "mcp",
            )],
        )
        self.assertEqual(registry.origin("mcp__fake__echo"), EXTERNAL_ORIGIN)

    def test_restricted_supports_globs(self):
        spec = ToolSpec(
            name="mcp__fake__echo", description="x",
            parameters={"type": "object", "properties": {}},
            handler=lambda **_: "mcp",
        )
        registry = create_default_tool_registry(mcp_specs=[spec])
        names = registry.restricted(["mcp__fake__*"]).names()
        self.assertEqual(names, ["mcp__fake__echo"])

    def test_render_tool_result_variants(self):
        class Block:
            def __init__(self, type_, **kw):
                self.type = type_
                self.__dict__.update(kw)

        class Result:
            def __init__(self, content, is_error=False, structured_content=None):
                self.content = content
                self.is_error = is_error
                self.structured_content = structured_content

        text = render_tool_result(Result([Block("text", text="hello")]))
        self.assertEqual(text, "hello")

        err = render_tool_result(Result([Block("text", text="nope")], is_error=True))
        self.assertTrue(err.startswith("Error:"))

        img = render_tool_result(Result([Block("image", mime_type="image/png")]))
        self.assertIn("image omitted", img)

        redundant = render_tool_result(
            Result([Block("text", text="echo hi")], structured_content={"result": "echo hi"}))
        self.assertEqual(redundant, "echo hi")

        structured = render_tool_result(
            Result([Block("text", text="done")], structured_content={"count": 3}))
        self.assertIn('"count": 3', structured)


class McpAgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.user_dir = Path(self.tmp.name) / "user"
        self.user_dir.mkdir(parents=True)
        self.profiles = Path(self.tmp.name) / "profiles.json"
        self.profiles.write_text(json.dumps({
            "default_profile": "test",
            "profiles": {
                "test": {"provider": "ollama", "model": "test-model",
                         "base_url": "http://localhost:11434"},
            },
        }))
        self.addCleanup(shutdown_mcp_manager)

    def _config(self, **overrides):
        from pyagent.config import AppConfig
        params = dict(
            user_dir=str(self.user_dir),
            model_profiles_path=str(self.profiles),
            system_prompt_path=str(Path(self.tmp.name) / "system.txt"),
            mcp_connect_timeout=10.0,
        )
        params.update(overrides)
        return AppConfig(**params)

    @unittest.skipUnless(MCP_AVAILABLE, "mcp package not installed")
    def test_agent_registers_mcp_tools(self):
        server = write_fake_server(Path(self.tmp.name))
        write_config(self.user_dir, {"fake": {"command": sys.executable, "args": [str(server)]}})
        from pyagent.agent import Agent

        agent = Agent(config=self._config(), workspace=self.tmp.name)
        try:
            names = agent.tool_registry.names()
            self.assertIn("mcp__fake__echo", names)
            self.assertEqual(agent.tool_registry.origin("mcp__fake__echo"), MCP_ORIGIN)
            out = agent.tool_registry.execute("mcp__fake__echo", {"text": "hi"})
            self.assertIn("echo: hi", out)
        finally:
            agent.close()

    def test_agent_mcp_disabled_registers_nothing(self):
        server = write_fake_server(Path(self.tmp.name))
        write_config(self.user_dir, {"fake": {"command": sys.executable, "args": [str(server)]}})
        from pyagent.agent import Agent

        agent = Agent(config=self._config(mcp_enabled=False), workspace=self.tmp.name)
        try:
            self.assertFalse(any(n.startswith("mcp__") for n in agent.tool_registry.names()))
            self.assertIsNone(agent.mcp_discovery)
        finally:
            agent.close()


class McpApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.user_dir = Path(self.tmp.name) / "user"
        self.user_dir.mkdir(parents=True)
        self.env = mock.patch.dict(os.environ, {"PYAGENT_USER_DIR": str(self.user_dir)})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(shutdown_mcp_manager)
        from fastapi.testclient import TestClient
        from pyagent.api import create_app
        self.client = TestClient(create_app())
        self.addCleanup(self.client.close)

    def test_crud_round_trip(self):
        server = write_fake_server(Path(self.tmp.name))
        response = self.client.post("/mcp", json={
            "name": "fake", "command": sys.executable, "args": [str(server)],
            "env": {"SECRET": "supersecret"},
        })
        self.assertEqual(response.status_code, 201, response.text)
        body = response.json()
        self.assertEqual(body["env"], {"SECRET": "********"})

        response = self.client.get("/mcp")
        self.assertEqual(response.status_code, 200)
        servers = response.json()["servers"]
        self.assertEqual(servers[0]["name"], "fake")
        self.assertEqual(servers[0]["env"], {"SECRET": "********"})
        self.assertEqual(servers[0]["status"], "ok")
        self.assertEqual(servers[0]["tool_count"], 3)

        response = self.client.post("/mcp/fake/test")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])

        response = self.client.post("/mcp/fake/disable")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["enabled"])

        tools = self.client.get("/tools").json()
        self.assertEqual(tools["mcp"], [])

        response = self.client.post("/mcp/fake/enable")
        self.assertTrue(response.json()["enabled"])

        response = self.client.put("/mcp/fake", json={"env": {"OTHER": "x"}})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["env"], {"OTHER": "********"})

        response = self.client.delete("/mcp/fake")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get("/mcp").json()["servers"], [])

    def test_create_conflict_and_validation(self):
        write_config(self.user_dir, {"dup": {"command": "true"}})
        response = self.client.post("/mcp", json={"name": "dup", "command": "true"})
        self.assertEqual(response.status_code, 409)
        response = self.client.post("/mcp", json={"name": "bad name", "command": "true"})
        self.assertEqual(response.status_code, 400)
        response = self.client.post(
            "/mcp", json={"name": "both", "command": "true", "url": "https://x.example.com/mcp"})
        self.assertEqual(response.status_code, 400)
        response = self.client.get("/mcp/missing")
        self.assertEqual(response.status_code, 404)

    @unittest.skipUnless(MCP_AVAILABLE, "mcp package not installed")
    def test_tools_endpoint_lists_mcp_tools(self):
        server = write_fake_server(Path(self.tmp.name))
        write_config(self.user_dir, {"fake": {"command": sys.executable, "args": [str(server)]}})
        response = self.client.get("/tools")
        self.assertEqual(response.status_code, 200)
        names = [tool["name"] for tool in response.json()["mcp"]]
        self.assertIn("mcp__fake__echo", names)


class McpCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.user_dir = str(Path(self.tmp.name) / "user")
        self.env = mock.patch.dict(os.environ, {"PYAGENT_USER_DIR": self.user_dir})
        self.env.start()
        self.addCleanup(self.env.stop)

    def _run(self, argv):
        from pyagent.main import main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                main(argv)
            except SystemExit as exc:
                return exc.code or 0, buf.getvalue()
        return 0, buf.getvalue()

    def test_add_list_remove_flow(self):
        server = write_fake_server(Path(self.tmp.name))
        code, out = self._run(
            ["mcp", "add", "fake", "--command", sys.executable, "--arg", str(server)])
        self.assertEqual(code, 0, out)
        code, out = self._run(["mcp", "list", "--no-connect"])
        self.assertEqual(code, 0)
        self.assertIn("fake", out)
        code, out = self._run(["mcp", "disable", "fake"])
        self.assertEqual(code, 0)
        code, out = self._run(["mcp", "remove", "fake"])
        self.assertEqual(code, 0)
        config = load_mcp_config(user_dir=self.user_dir)
        self.assertEqual(config.servers, ())

    def test_add_rejects_bad_name(self):
        code, out = self._run(["mcp", "add", "Bad Name", "--command", "true"])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(MCP_AVAILABLE, "mcp package not installed")
class McpHttpTransportTests(_McpTestCase):
    def test_streamable_http_round_trip(self):
        try:
            import uvicorn
            from mcp.server.mcpserver import MCPServer
        except ImportError:
            self.skipTest("uvicorn or mcp server extras not available")

        import socket
        import threading
        import time

        server = MCPServer(name="httpfake", version="0.1.0")

        @server.tool()
        def greet(name: str) -> str:
            """Greet someone over HTTP."""
            return f"hello {name}"

        app = server.streamable_http_app()
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        instance = uvicorn.Server(config)
        thread = threading.Thread(target=instance.run, daemon=True)
        thread.start()
        deadline = time.time() + 15
        while not instance.started and time.time() < deadline:
            time.sleep(0.05)
        try:
            self.make_config({"web": {"url": f"http://127.0.0.1:{port}/mcp"}})
            result = get_mcp_manager().ensure_tools(
                user_dir=str(self.user_dir), connect_timeout=10)
            self.assertEqual(result.servers[0].status, "ok")
            greet = next(s for s in result.specs if s.name == "mcp__web__greet")
            self.assertEqual(greet.handler(name="agent").strip().splitlines()[0], "hello agent")
        finally:
            instance.should_exit = True
            thread.join(timeout=5)


@unittest.skipUnless(MCP_AVAILABLE, "mcp package not installed")
class McpAgentDefinitionValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.user_dir = self.root / "user"
        self.user_dir.mkdir(parents=True)
        self.profiles = self.root / "profiles.json"
        self.profiles.write_text(json.dumps({
            "default_profile": "test",
            "profiles": {
                "test": {"provider": "ollama", "model": "test-model",
                         "base_url": "http://localhost:11434"},
            },
        }))
        self.addCleanup(shutdown_mcp_manager)

    def _definition(self, tools):
        from pyagent.agent_definitions import AgentDefinition

        return AgentDefinition(
            schema_version=1,
            name="test-agent",
            revision=1,
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
            tools=tuple(tools) if tools is not None else None,
            workspace=str(self.root),
        )

    def _validate(self, tools):
        from pyagent.agent_definitions import validate_agent_definition
        from pyagent.config import AppConfig

        config = AppConfig(
            user_dir=str(self.user_dir),
            model_profiles_path=str(self.profiles),
            system_prompt_path=str(self.root / "system.txt"),
            mcp_connect_timeout=10.0,
        )
        return validate_agent_definition(
            self._definition(tools), config=config, cwd=self.root)

    def test_wildcard_expands_to_concrete_tools(self):
        server = write_fake_server(self.root)
        write_config(self.user_dir, {"fake": {"command": sys.executable, "args": [str(server)]}})
        validation = self._validate(["mcp__fake__*"])
        self.assertTrue(validation.valid, validation.errors)
        self.assertEqual(
            sorted(validation.resolved["tools"]),
            ["mcp__fake__add", "mcp__fake__echo", "mcp__fake__fail"],
        )

    def test_unknown_mcp_tool_is_an_error_when_discovery_works(self):
        server = write_fake_server(self.root)
        write_config(self.user_dir, {"fake": {"command": sys.executable, "args": [str(server)]}})
        validation = self._validate(["mcp__fake__nonexistent"])
        self.assertFalse(validation.valid)
        self.assertIn("mcp__fake__nonexistent", " ".join(validation.errors))

    def test_mcp_names_warn_not_error_when_discovery_fails(self):
        from unittest import mock

        with mock.patch(
            "pyagent.mcp_runtime.get_mcp_manager",
            side_effect=RuntimeError("boom"),
        ):
            validation = self._validate(["mcp__anything__tool"])
        self.assertTrue(
            any("Could not discover MCP tools" in w for w in validation.warnings),
            validation.warnings,
        )
        self.assertNotIn(
            "Unknown or disabled tool: mcp__anything__tool",
            validation.errors,
        )


@unittest.skipUnless(MCP_AVAILABLE, "mcp package not installed")
class McpAgentDefinitionBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.user_dir = self.root / "user"
        self.user_dir.mkdir(parents=True)
        self.profiles = self.root / "profiles.json"
        self.profiles.write_text(json.dumps({
            "default_profile": "test",
            "profiles": {
                "test": {"provider": "ollama", "model": "test-model",
                         "base_url": "http://localhost:11434"},
            },
        }))
        self.addCleanup(shutdown_mcp_manager)

    def test_wildcard_allowlist_builds_restricted_agent(self):
        from pyagent.agent_definitions import (
            AgentDefinitionStore,
            build_agent_from_definition,
        )
        from pyagent.config import AppConfig

        server = write_fake_server(self.root)
        write_config(self.user_dir, {"fake": {"command": sys.executable, "args": [str(server)]}})
        store = AgentDefinitionStore(self.user_dir)
        store.create({
            "name": "mcp-bot",
            "tools": ["mcp__fake__*"],
            "workspace": str(self.root),
        })
        agent, _definition, validation = build_agent_from_definition(
            "mcp-bot",
            user_dir=self.user_dir,
            config=AppConfig(
                user_dir=str(self.user_dir),
                model_profiles_path=str(self.profiles),
                system_prompt_path=str(self.root / "system.txt"),
                mcp_connect_timeout=10.0,
            ),
        )
        try:
            self.assertTrue(validation.valid, validation.errors)
            names = agent.tool_registry.names()
            self.assertEqual(
                sorted(names), ["mcp__fake__add", "mcp__fake__echo", "mcp__fake__fail"])
            self.assertNotIn("bash", names)
        finally:
            agent.close()
