from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from pyagent.agent import Agent
from pyagent.api import create_app
from pyagent.client import PyAgentClient
from pyagent.config import AppConfig
from pyagent.main import run_agent_definition, run_single_shot
from pyagent.run_logging import safe_endpoint
from pyagent.session_logger import SessionLogger
from pyagent.streaming import stream_agent_sse
from pyagent.tools import ToolRegistry, ToolSpec
from pyagent.ui import PyAgentApp


class FakeClient:
    def __init__(self, streams=None):
        self.streams = iter(streams or [[{"content": "Done"}]])
        self.closed = False

    def chat_stream(self, messages, tools=None):
        yield from next(self.streams)

    def close(self):
        self.closed = True


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sse_events(body):
    return [json.loads(line[6:]) for line in body.splitlines()
            if line.startswith("data: ")]


class RunLoggingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.profiles = self.root / "profiles.json"
        self.profiles.write_text(json.dumps({
            "default_profile": "test",
            "profiles": {"test": {
                "provider": "ollama", "model": "test-model",
                "base_url": "http://localhost:11434",
            }},
        }))
        self.env = mock.patch.dict(os.environ, {
            "PYAGENT_LOGGING": "on",
            "PYAGENT_USER_DIR": str(self.root / "user"),
            "PYAGENT_USER_TOOLS_ENABLED": "off",
            "PYAGENT_MODEL_PROFILES_PATH": str(self.profiles),
            "PYAGENT_SYSTEM_PROMPT_PATH": str(self.root / "system.txt"),
        }, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.clients = []

        def build(profile, **kwargs):
            client = FakeClient()
            client.profile = profile
            self.clients.append(client)
            return client

        self.patch_client = mock.patch(
            "pyagent.agent.build_chat_client", side_effect=build)
        self.patch_client.start()
        self.addCleanup(self.patch_client.stop)

    def agent(self, streams=None, **config_changes):
        config = AppConfig.from_env()
        for key, value in config_changes.items():
            setattr(config, key, value)
        agent = Agent(config=config, tool_registry=ToolRegistry(
            []), workspace=self.root)
        if streams is not None:
            agent.client.streams = iter(streams)
        self.addCleanup(agent.close)
        return agent

    def logs(self):
        return list((self.root / "user" / "logs").glob("*.jsonl"))

    def test_env_boolean_values_and_default(self):
        for value in ("on", "1", "true", "yes", " ON ", "TrUe"):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"PYAGENT_LOGGING": value}):
                self.assertTrue(AppConfig.from_env().logging_enabled)
        for value in ("off", "0", "false", "no", "", "typo", " OFF "):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"PYAGENT_LOGGING": value}):
                self.assertFalse(AppConfig.from_env().logging_enabled)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(AppConfig.from_env().logging_enabled)
        self.assertFalse(AppConfig().logging_enabled)

    def test_disabled_run_has_identity_but_does_not_create_directory(self):
        agent = self.agent(logging_enabled=False)
        list(agent.run("Hi"))
        UUID(agent.run_id)
        self.assertIsNone(agent.last_log_path)
        self.assertFalse((self.root / "user" / "logs").exists())

    def test_complete_run_has_versioned_checkpoint_and_unique_sequence(self):
        agent = self.agent()
        list(agent.run("Hi"))
        log = records(agent.last_log_path)
        self.assertEqual(log[0]["type"], "run_start")
        self.assertEqual(log[0]["attributes"]["prompt"], "Hi")
        self.assertEqual(log[-1]["type"], "run_end")
        self.assertEqual(log[-1]["attributes"]["outcome"], "completed")
        self.assertEqual(log[-1]["attributes"]["messages"], agent.messages)
        self.assertEqual(log[-1]["attributes"]["response"], "Done")
        self.assertEqual({r["run_id"] for r in log}, {agent.run_id})
        self.assertEqual({r["schema_version"] for r in log}, {1})
        self.assertEqual([r["sequence"]
                         for r in log], list(range(1, len(log) + 1)))
        self.assertIn(agent.run_id, agent.last_log_path.name)
        if os.name == "posix":
            self.assertEqual(agent.last_log_path.stat().st_mode & 0o777, 0o600)
        self.assertIsNone(agent._run_log)
        self.assertIsNone(agent._active_run_generator)

    def test_multiple_turns_link_parent_and_reset_starts_new_conversation(self):
        agent = self.agent(
            [[{"content": "one"}], [{"content": "two"}], [{"content": "three"}]])
        list(agent.run("first"))
        first_id = agent.run_id
        first_path = agent.last_log_path
        first_bytes = first_path.read_bytes()
        list(agent.run("second"))
        log = records(agent.last_log_path)
        self.assertNotEqual(first_id, agent.run_id)
        self.assertEqual(log[0]["attributes"]["parent_run_id"], first_id)
        self.assertTrue(
            any(m.get("content") == "one" for m in log[0]["attributes"]["initial_messages"]))
        self.assertEqual(first_path.read_bytes(), first_bytes)
        agent.reset()
        list(agent.run("third"))
        self.assertIsNone(records(agent.last_log_path)[
                          0]["attributes"]["parent_run_id"])
        self.assertEqual(len(self.logs()), 3)

    def test_messages_survive_history_trimming_and_tool_ids_are_preserved(self):
        def call(n): return {"tool_calls": [
            {"id": n, "function": {"name": "echo", "arguments": {}}}]}
        agent = self.agent([[call("call-1")], [call("call-2")],
                           [{"content": "done"}]], max_history_messages=2)
        agent.tool_registry = ToolRegistry([ToolSpec(
            name="echo", description="echo", parameters={"type": "object", "properties": {}},
            handler=lambda: "full tool output",
        )])
        agent.tools = agent.tool_registry.definitions()
        list(agent.run("use tools"))
        log = records(agent.last_log_path)
        messages = [r["attributes"]["message"]
                    for r in log if r["type"] == "message"]
        self.assertEqual([m["tool_call_id"] for m in messages if m["role"] == "tool"], [
                         "call-1", "call-2"])
        self.assertEqual([m["tool_calls"][0]["id"] for m in messages if m.get(
            "tool_calls")], ["call-1", "call-2"])
        self.assertEqual(messages[0], {"role": "user", "content": "use tools"})
        self.assertGreater(len(messages), len(agent.messages))
        self.assertEqual(log[-1]["attributes"]["messages"], agent.messages)
        self.assertEqual(
            len([r for r in log if r["type"] == "model_request"]), 3)

    def test_extension_fault_is_logged_and_does_not_fail_run(self):
        agent = self.agent()

        def broken(payload, ctx):
            raise RuntimeError("extension fault")
        agent.bus.on("tool_unused", broken, extension="audit")
        agent.bus.on("input", broken, extension="audit")
        list(agent.run("hi"))
        text = agent.last_log_path.read_text()
        self.assertIn("extension handler failed", text)
        self.assertIn("traceback", text)
        self.assertEqual(records(agent.last_log_path)
                         [-1]["attributes"]["outcome"], "completed")

    def test_error_and_exception_preserve_partial_output(self):
        def broken():
            yield {"content": "partial"}
            raise RuntimeError("exploded")
        for stream in ([{"content": "partial"}, {"error": "boom"}], broken()):
            with self.subTest(stream=type(stream)):
                agent = self.agent([stream])
                try:
                    list(agent.run("hi"))
                except RuntimeError:
                    pass
                log = records(agent.last_log_path)
                self.assertEqual(log[-1]["attributes"]["outcome"], "failed")
                self.assertEqual([r["attributes"]["content"]
                                 for r in log if r["type"] == "partial_output"], ["partial"])
                self.assertIsNotNone(log[-1]["attributes"]["error"])

    def test_early_terminal_consumer_close_is_completed_and_closes_logger(self):
        agent = self.agent()
        ended = []
        agent.bus.on("turn_end", lambda payload, ctx: ended.append(True))
        generator = agent.run("hi")
        for event in generator:
            if event["type"] == "assistant_done":
                break
        agent.close()
        self.assertEqual(ended, [True])
        self.assertEqual(records(agent.last_log_path)
                         [-1]["attributes"]["outcome"], "completed")
        self.assertTrue(agent.client.closed)
        agent.close()
        self.assertEqual(
            sum(r["type"] == "run_end" for r in records(agent.last_log_path)), 1)

    def test_cancelled_and_unstarted_runs_close_cleanly(self):
        for iterate in (True, False):
            with self.subTest(iterate=iterate):
                agent = self.agent()
                generator = agent.run("hi")
                if iterate:
                    next(generator)
                agent.close()
                self.assertEqual(records(agent.last_log_path)
                                 [-1]["attributes"]["outcome"], "cancelled")
                self.assertIsNone(agent._run_log)
                self.assertIsNone(agent._active_run_generator)

    def test_open_failure_does_not_change_response_or_stdout(self):
        agent = self.agent()
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch("pyagent.run_logging.SessionLogger", side_effect=PermissionError("denied")), contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
            events = list(agent.run("hi"))
        self.assertEqual(events[-1]["content"], "Done")
        self.assertIn("Could not start run logging", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")
        self.assertIsNotNone(agent.logging_error)
        self.assertIsNone(agent.last_log_path)

    def test_write_failure_warns_once_and_leaves_incomplete_log(self):
        agent = self.agent()
        agent.prepare_run()
        logger = agent._run_log.logger
        real_file = logger._file
        failed_file = mock.Mock(wraps=real_file)
        failed_file.write.side_effect = OSError("disk full")
        logger._file = failed_file
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            events = list(agent.run("hi"))
        self.assertEqual(events[-1]["content"], "Done")
        self.assertEqual(stderr.getvalue().count("disk full"), 1)
        self.assertTrue(logger.failed)
        self.assertTrue(real_file.closed)
        self.assertFalse(
            any(r["type"] == "run_end" for r in records(agent.last_log_path)))

    def test_exclusive_creation_handles_same_timestamp_and_forced_collision(self):
        identity = str(uuid4())
        with mock.patch("pyagent.session_logger.datetime") as clock:
            clock.now.return_value.strftime.return_value = "fixed"
            first = SessionLogger(self.root, run_id=identity)
            first.info("original")
            first.close()
            before = first.path.read_bytes()
            second = SessionLogger(self.root, run_id=identity)
            second.close()
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertNotEqual(first.path, second.path)
        self.assertEqual(first.path.read_bytes(), before)

    def test_parallel_runs_have_distinct_files_and_no_mixed_records(self):
        agents = [self.agent() for _ in range(12)]

        def execute(item):
            index, agent = item
            list(agent.run(f"prompt-{index}"))
            return agent
        with ThreadPoolExecutor(max_workers=6) as pool:
            completed = list(pool.map(execute, enumerate(agents)))
        self.assertEqual(len({a.run_id for a in completed}), 12)
        self.assertEqual(len(self.logs()), 12)
        for index, agent in enumerate(completed):
            log = records(agent.last_log_path)
            self.assertEqual({r["run_id"] for r in log}, {agent.run_id})
            self.assertEqual(log[0]["attributes"]["prompt"], f"prompt-{index}")

    def test_metadata_excludes_credentials_and_sanitizes_endpoint(self):
        agent = self.agent()
        agent.client.profile = replace(
            agent.current_profile(),
            base_url="https://user:url-secret@example.com/v1?api_key=query-secret#fragment-secret",
            api_key="inline-secret",
            headers={"Authorization": "Bearer header-secret"},
        )
        agent.run_metadata["api_key"] = "metadata-secret"
        list(agent.run("hi"))
        metadata = records(agent.last_log_path)[0]["attributes"]["metadata"]
        self.assertEqual(metadata["base_url"], "https://example.com/v1")
        text = agent.last_log_path.read_text()
        for secret in ("url-secret", "query-secret", "fragment-secret", "inline-secret", "header-secret", "metadata-secret"):
            self.assertNotIn(secret, text)
        self.assertEqual(safe_endpoint(
            "http://[::1]:8000/v1?token=secret"), "http://[::1]:8000/v1")

    def test_cli_and_named_cli_record_source_and_definition(self):
        from pyagent.agent_definitions import AgentDefinitionStore
        AgentDefinitionStore().create({"name": "helper"})
        self.assertEqual(run_single_shot(prompt="one"), "Done")
        self.assertEqual(run_agent_definition(
            name="helper", prompt="two", cwd=str(self.root)), "Done")
        logs = [records(p) for p in self.logs()]
        self.assertEqual(len(logs), 2)
        self.assertEqual({log[0]["attributes"]["metadata"]
                         ["source"] for log in logs}, {"cli"})
        named = next(
            log for log in logs if log[0]["attributes"]["prompt"] == "two")
        self.assertEqual(named[0]["attributes"]["metadata"]["agent"], "helper")
        self.assertEqual(named[0]["attributes"]
                         ["metadata"]["agent_revision"], 1)
        self.assertTrue(
            all(log[-1]["attributes"]["outcome"] == "completed" for log in logs))

    def test_all_api_routes_share_response_and_file_identity(self):
        from pyagent.agent_definitions import AgentDefinitionStore
        AgentDefinitionStore().create({"name": "helper"})
        client = TestClient(create_app())
        ids = []
        for route in ("/run", "/run/stream", "/agents/helper/run", "/agents/helper/run/stream"):
            with self.subTest(route=route):
                response = client.post(
                    route, json={"message": "hello", "cwd": str(self.root)})
                self.assertEqual(response.status_code, 200, response.text)
                if route.endswith("stream"):
                    events = sse_events(response.text)
                    run_id = events[0]["run_id"]
                    self.assertEqual({e["run_id"] for e in events}, {run_id})
                    self.assertEqual([e["type"] for e in events if e["type"] in {
                                     "done", "error"}], ["done"])
                    self.assertFalse(any(e["type"] == "debug" for e in events))
                else:
                    run_id = response.json()["run_id"]
                UUID(run_id)
                ids.append(run_id)
                log_path, = [p for p in self.logs() if run_id in p.name]
                log = records(log_path)
                self.assertEqual({r["run_id"] for r in log}, {run_id})
                self.assertEqual(log[0]["attributes"]
                                 ["metadata"]["source"], "api")
                self.assertEqual(log[-1]["attributes"]["outcome"], "completed")
                self.assertNotIn(str(log_path), response.text)
        self.assertEqual(len(set(ids)), 4)

    def test_api_disabled_and_invalid_requests_do_not_create_logs(self):
        client = TestClient(create_app())
        for route in ("/run", "/run/stream"):
            response = client.post(
                route, json={"message": "hi", "profile": "missing"})
            self.assertEqual(response.status_code, 400)
        self.assertEqual(self.logs(), [])
        with mock.patch.dict(os.environ, {"PYAGENT_LOGGING": "off"}):
            for route in ("/run", "/run/stream"):
                self.assertEqual(client.post(
                    route, json={"message": "hi"}).status_code, 200)
        self.assertEqual(self.logs(), [])

    def test_api_error_outcomes_and_debug_capture_are_independent(self):
        client = TestClient(create_app())
        for route in ("/run", "/run/stream?include_debug=true"):
            agent = self.agent([[{"error": "boom"}]])
            with mock.patch("pyagent.main.build_agent_for_request", return_value=agent):
                response = client.post(route, json={"message": "hi"})
            if "stream" in route:
                events = sse_events(response.text)
                self.assertEqual(events[-1]["type"], "error")
                self.assertEqual(sum(e["type"] == "error" for e in events), 1)
                self.assertTrue(any(e["type"] == "debug" for e in events))
            else:
                self.assertEqual(response.status_code, 502)
            self.assertEqual(records(agent.last_log_path)
                             [-1]["attributes"]["outcome"], "failed")

    def test_disconnect_during_blocked_model_waits_for_execution_owner(self):
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()

        def chunks():
            entered.set()
            release.wait(5)
            yield {"content": "partial"}
            yield {"content": "more"}
        agent = self.agent([chunks()])
        original_close = agent.close

        def close(*args, **kwargs):
            try:
                original_close(*args, **kwargs)
            finally:
                closed.set()
        agent.close = close

        async def exercise():
            stream = stream_agent_sse(
                agent, "hi", start_data={}, completion_data=lambda text: {"response": text})
            start = sse_events(await anext(stream))[0]
            self.assertEqual(start["run_id"], agent.run_id)
            self.assertTrue(await asyncio.to_thread(entered.wait, 3))
            await stream.aclose()  # cancellation even immediately after the start event
            self.assertFalse(closed.is_set())
            release.set()
            self.assertTrue(await asyncio.to_thread(closed.wait, 3))
        try:
            asyncio.run(exercise())
        finally:
            release.set()
        log = records(agent.last_log_path)
        self.assertEqual(log[-1]["attributes"]["outcome"], "cancelled")
        self.assertEqual(sum(r["type"] == "run_end" for r in log), 1)
        self.assertTrue(agent.client.closed)

    def test_disconnect_during_tool_execution_retains_completed_tool_result(self):
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()
        agent = self.agent([[{"tool_calls": [{
            "id": "side-effect", "function": {"name": "slow", "arguments": {}},
        }]}]])

        def slow():
            entered.set()
            release.wait(5)
            return "side effect completed"
        agent.tool_registry = ToolRegistry([ToolSpec(
            name="slow", description="slow tool", parameters={"type": "object", "properties": {}}, handler=slow,
        )])
        agent.tools = agent.tool_registry.definitions()
        original_close = agent.close

        def close(*args, **kwargs):
            try:
                original_close(*args, **kwargs)
            finally:
                closed.set()
        agent.close = close

        async def exercise():
            stream = stream_agent_sse(
                agent, "hi", start_data={}, completion_data=lambda text: {"response": text})
            await anext(stream)
            self.assertTrue(await asyncio.to_thread(entered.wait, 3))
            await stream.aclose()
            self.assertFalse(closed.is_set())
            release.set()
            self.assertTrue(await asyncio.to_thread(closed.wait, 3))
        try:
            asyncio.run(exercise())
        finally:
            release.set()
        log = records(agent.last_log_path)
        self.assertEqual(log[-1]["attributes"]["outcome"], "cancelled")
        messages = [r["attributes"]["message"]
                    for r in log if r["type"] == "message"]
        tool = next(m for m in messages if m["role"] == "tool")
        self.assertEqual(tool["tool_call_id"], "side-effect")
        self.assertEqual(tool["content"], "side effect completed")
        self.assertIn(tool, log[-1]["attributes"]["messages"])

    def test_tui_initial_setting_and_runtime_overrides(self):
        agent = self.agent(
            [[{"content": "one"}], [{"content": "two"}], [{"content": "three"}]])
        app = PyAgentApp(agent=agent)
        app._add_system_note = mock.Mock()
        app._set_status = mock.Mock()
        self.assertTrue(app.logging_enabled)
        self.assertEqual(agent.run_source, "tui")
        app._handle_logging_command(["off"])
        list(agent.run("one"))
        self.assertEqual(self.logs(), [])
        for _ in range(2):
            app._handle_logging_command(["on"])
        list(agent.run("two"))
        self.assertEqual(len(self.logs()), 1)
        app._handle_logging_command([])
        self.assertIn(str(agent.last_log_path),
                      app._add_system_note.call_args.args[0])
        app.is_processing = True
        app._handle_logging_command(["off"])
        self.assertTrue(agent.logging_enabled)
        app.is_processing = False
        app._handle_logging_command(["off"])
        list(agent.run("three"))
        self.assertEqual(len(self.logs()), 1)

    def test_headless_tui_runs_without_duplicate_transcript_logging(self):
        agent = self.agent()
        app = PyAgentApp(agent=agent)

        async def exercise():
            async with app.run_test() as pilot:
                await pilot.pause()
                with mock.patch.object(app, "_add_system_note", wraps=app._add_system_note) as note:
                    await app.process_message("hello")
                    await pilot.pause()
                    note.assert_not_called()
                    app._handle_logging_command([])
                    self.assertIn(str(agent.last_log_path),
                                  note.call_args.args[0])
                self.assertIn("Log: on", app._status_summary())
        asyncio.run(exercise())
        log = records(agent.last_log_path)
        self.assertEqual(sum(r["type"] == "run_start" for r in log), 1)
        self.assertEqual(sum(r["type"] == "run_end" for r in log), 1)
        self.assertEqual([r["attributes"]["message"]["role"]
                         for r in log if r["type"] == "message"], ["user", "assistant"])

    def test_parallel_api_streams_have_independent_checkpoints(self):
        barrier = threading.Barrier(4)
        agents = [self.agent() for _ in range(4)]

        def chunks(index):
            barrier.wait(timeout=5)
            yield {"content": f"answer-{index}"}
        for index, agent in enumerate(agents):
            agent.client.streams = iter([chunks(index)])

        def build(**kwargs):
            return agents[int(kwargs["model"])]

        def request(index):
            with TestClient(create_app()) as client:
                return client.post("/run/stream", json={"message": f"prompt-{index}", "model": str(index)})
        with mock.patch("pyagent.main.build_agent_for_request", side_effect=build), ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(request, range(4)))
        for index, response in enumerate(responses):
            events = sse_events(response.text)
            self.assertEqual(events[-1]["response"], f"answer-{index}")
            self.assertEqual(events[0]["run_id"], agents[index].run_id)
            log = records(agents[index].last_log_path)
            self.assertEqual(log[0]["attributes"]["prompt"], f"prompt-{index}")
            self.assertEqual(log[-1]["attributes"]
                             ["response"], f"answer-{index}")
        self.assertEqual(len(self.logs()), 4)

    def test_api_logging_failure_does_not_add_terminal_events(self):
        with mock.patch("pyagent.run_logging.SessionLogger", side_effect=OSError("unwritable")), contextlib.redirect_stderr(io.StringIO()):
            client = TestClient(create_app())
            response = client.post("/run", json={"message": "hello"})
            stream = client.post("/run/stream", json={"message": "hello"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["response"], "Done")
        events = sse_events(stream.text)
        self.assertEqual([e["type"] for e in events if e["type"] in {
                         "done", "error"}], ["done"])
        self.assertEqual(events[-1]["response"], "Done")
        self.assertEqual(self.logs(), [])

    def test_tui_failure_is_visible_and_later_run_can_retry(self):
        agent = self.agent([[{"content": "one"}], [{"content": "two"}]])
        app = PyAgentApp(agent=agent)

        async def exercise():
            async with app.run_test() as pilot:
                with mock.patch("pyagent.run_logging.SessionLogger", side_effect=PermissionError("denied")), contextlib.redirect_stderr(io.StringIO()), mock.patch.object(app, "_add_system_note", wraps=app._add_system_note) as note:
                    await app.process_message("first")
                    note.assert_called_once_with(agent.logging_error)
                self.assertIn("Log: error", app._status_summary())
                self.assertIsNone(agent.last_log_path)
                await app.process_message("second")
                await pilot.pause()
                self.assertIn("Log: on", app._status_summary())
        asyncio.run(exercise())
        self.assertEqual(len(self.logs()), 1)
        self.assertEqual(records(agent.last_log_path)
                         [-1]["attributes"]["response"], "two")

    def test_original_prompt_and_effective_context_are_distinct_records(self):
        agent = self.agent()
        agent.bus.on("input", lambda payload, ctx: {
                     "action": "transform", "text": "transformed"})
        agent.bus.on("before_agent_start", lambda payload,
                     ctx: {"system_prompt": "custom system"})
        list(agent.run("original"))
        log = records(agent.last_log_path)
        self.assertEqual(log[0]["attributes"]["prompt"], "original")
        request = next(r for r in log if r["type"] == "model_request")
        self.assertEqual(request["attributes"]["messages"]
                         [0]["content"], "custom system")
        self.assertEqual(request["attributes"]["messages"]
                         [-1]["content"], "transformed")
        self.assertFalse(any(m.get(
            "content") == "transformed" for m in log[0]["attributes"]["initial_messages"]))

    def test_client_run_ids_are_optional_for_older_servers(self):
        client = PyAgentClient()
        base = {"response": "ok", "profile": "p", "provider": "ollama",
                "model": "m", "messages": [], "agent": "helper", "revision": 1}
        for identity in (None, str(uuid4())):
            payload = dict(base)
            if identity:
                payload["run_id"] = identity
            with mock.patch.object(client, "_request_json", return_value=payload):
                self.assertEqual(client.run("hi").run_id, identity)
                self.assertEqual(client.run_agent(
                    "helper", "hi").run_id, identity)


if __name__ == "__main__":
    unittest.main()
