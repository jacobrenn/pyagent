from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextlib
import io
import os
import threading
import unittest
from unittest import mock
from uuid import uuid4

from fastapi.testclient import TestClient

from pyagent.agent_definitions import AgentDefinitionStore, build_agent_from_definition
from pyagent.api import ChatRequest, _prepare_request_agent, create_app
from pyagent.client import PyAgentClient, PyAgentClientError
from pyagent.config import AppConfig
from pyagent.session_logger import SessionLogger
from pyagent.streaming import stream_agent_sse
from test_run_logging import FakeClient, records, sse_events
import test_sessions as fixtures


class SessionApiTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SessionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        enabled = mock.patch.dict(os.environ, {"PYAGENT_SESSION_API": "on"})
        enabled.start()
        self.addCleanup(enabled.stop)
        self.client = TestClient(create_app())
        self.addCleanup(self.client.close)
        self.store = self.fixture.store

    def source(self, named=False):
        if not named:
            return self.fixture.saved()
        definitions = AgentDefinitionStore()
        definitions.create(
            {"name": "reviewer", "workspace": str(self.fixture.root), "tools": []})
        agent, _, _ = build_agent_from_definition("reviewer")
        self.addCleanup(agent.close)
        list(agent.run("original prompt"))
        definitions.update("reviewer", {"model": "new-model"})
        return agent

    def test_opt_in_defaults_off_and_gates_contents_and_execution(self):
        source = self.source()
        for value in ("off", "", "garbage"):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"PYAGENT_SESSION_API": value}):
                self.assertFalse(AppConfig.from_env().session_api_enabled)
                capability = self.client.get("/sessions/capabilities")
                self.assertEqual(capability.json(), {
                                 "enabled": False, "resume_supported": True})
                for path in ("/sessions", f"/sessions/{source.run_id}"):
                    self.assertEqual(self.client.get(path).status_code, 403)
                for path in ("/run", "/run/stream", "/agents/reviewer/run", "/agents/reviewer/run/stream"):
                    response = self.client.post(
                        path, json={"message": "new", "resume_from_run_id": source.run_id})
                    self.assertEqual(response.status_code, 403, response.text)
        self.assertEqual(len(self.store.list()), 1)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(AppConfig.from_env().session_api_enabled)
        for value in (" ON ", "1", "TrUe", "yes"):
            with mock.patch.dict(os.environ, {"PYAGENT_SESSION_API": value}):
                self.assertTrue(AppConfig.from_env().session_api_enabled)
        with mock.patch.dict(os.environ, {"PYAGENT_SESSION_API": "off"}):
            self.assertEqual(self.client.post(
                "/run", json={"message": "ordinary"}).status_code, 200)

    def test_read_only_listing_detail_pagination_and_metadata_allowlist(self):
        source = self.source()
        self.fixture.rewrite(source, lambda log: log[0]["attributes"]["metadata"].update(
            api_key="inline-secret", headers={"Authorization": "header-secret"}))
        child = self.fixture.restore(source.run_id).agent
        list(child.run("child"))
        legacy = self.store.root / "session-legacy.jsonl"
        legacy.write_text("{}\n")
        before = source.last_log_path.read_bytes()
        with mock.patch.object(FakeClient, "chat_stream", side_effect=AssertionError("must not run")):
            listing = self.client.get("/sessions")
            detail = self.client.get(f"/sessions/{source.run_id}")
        self.assertEqual(listing.status_code, 200, listing.text)
        self.assertEqual(detail.status_code, 200, detail.text)
        self.assertEqual(listing.headers["cache-control"], "no-store")
        self.assertEqual(detail.headers["cache-control"], "no-store")
        self.assertEqual(listing.json()["total"], 2)
        self.assertEqual(listing.json()["unavailable_count"], 1)
        items = {item["run_id"]: item for item in listing.json()["runs"]}
        self.assertEqual(items[child.run_id]["parent_run_id"], source.run_id)
        self.assertEqual(items[child.run_id]["conversation_id"], source.run_id)
        self.assertNotIn("messages", items[source.run_id])
        self.assertEqual(detail.json()["messages"], source.messages)
        for secret in ("inline-secret", "header-secret", str(self.store.root), source.last_log_path.name):
            self.assertNotIn(secret, listing.text + detail.text)
        page = self.client.get("/sessions?offset=1&limit=1").json()
        self.assertEqual(page["runs"], listing.json()["runs"][1:2])
        for query in ("limit=0", "limit=101", "offset=-1"):
            self.assertEqual(self.client.get(
                "/sessions?" + query).status_code, 422)
        self.assertEqual(source.last_log_path.read_bytes(), before)
        self.assertEqual(len(self.store.list()), 3)

    def test_duplicate_id_listing_and_oversized_runtime_metadata(self):
        source = self.source()
        self.fixture.rewrite(source, lambda log: log[0]["attributes"]["metadata"].update(
            agent_revision="9" * 5000))
        detail = self.client.get(f"/sessions/{source.run_id}")
        self.assertEqual(detail.status_code, 200)
        self.assertIsNone(detail.json()["agent_revision"])
        duplicate = self.store.root / f"run-copy-{source.run_id}.jsonl"
        duplicate.write_bytes(source.last_log_path.read_bytes())
        listing = self.client.get("/sessions").json()
        self.assertEqual(listing["total"], 1)
        self.assertEqual(listing["unavailable_count"], 1)
        self.assertFalse(listing["runs"][0]["resumable"])
        self.assertEqual(self.client.get(
            f"/sessions/{source.run_id}").status_code, 409)

    def test_all_four_routes_resume_with_fresh_identity_and_pinned_revision(self):
        for named in (False, True):
            source = self.source(named)
            before = source.last_log_path.read_bytes()
            base = "/agents/reviewer/run" if named else "/run"
            for suffix in ("", "/stream"):
                response = self.client.post(base + suffix, json={
                    "message": "continue", "resume_from_run_id": source.run_id,
                })
                self.assertEqual(response.status_code, 200, response.text)
                if suffix:
                    events = sse_events(response.text)
                    data = events[-1]
                    self.assertEqual(data["type"], "done")
                    self.assertEqual(
                        sum(e["type"] in {"done", "error"} for e in events), 1)
                    self.assertEqual(events[0]["parent_run_id"], source.run_id)
                    self.assertTrue(events[0]["warnings"])
                else:
                    data = response.json()
                self.assertNotEqual(data["run_id"], source.run_id)
                self.assertEqual(data["parent_run_id"], source.run_id)
                self.assertTrue(data["log_saved"])
                self.assertTrue(data["warnings"])
                users = [m["content"]
                         for m in data["messages"] if m["role"] == "user"]
                self.assertEqual(users, ["original prompt", "continue"])
                if named:
                    self.assertEqual(data["agent"], "reviewer")
                    self.assertEqual(data["revision"], 1)
                    self.assertEqual(data["model"], "test-model")
                checkpoint = self.store.load(data["run_id"])
                self.assertEqual(checkpoint.parent_run_id, source.run_id)
                self.assertEqual(checkpoint.metadata["source"], "api")
                self.assertTrue(self.fixture.clients[-1].closed)
            self.assertEqual(source.last_log_path.read_bytes(), before)

    def test_conflicting_fields_fail_before_execution_including_empty_values(self):
        source = self.source()
        for suffix in ("", "/stream"):
            for field, value in (("messages", []), ("skills", []), ("messages", [{"role": "user", "content": "extra"}])):
                response = self.client.post("/run" + suffix, json={
                    "message": "new", "resume_from_run_id": source.run_id, field: value,
                })
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(
                    response.headers["content-type"], "application/json")
        self.assertEqual(len(self.store.list()), 1)

    def test_missing_invalid_unstarted_and_unsafe_ids_rejected(self):
        source = self.source()
        missing = str(uuid4())
        self.assertEqual(self.client.get(
            f"/sessions/{missing}").status_code, 404)
        for path in ("/run", "/run/stream"):
            self.assertEqual(self.client.post(
                path, json={"message": "new", "resume_from_run_id": missing}).status_code, 404)
            for identity in ("../secret", source.last_log_path.name, str(source.last_log_path)):
                self.assertEqual(self.client.post(
                    path, json={"message": "new", "resume_from_run_id": identity}).status_code, 422)
        unstarted = self.fixture.agent()
        unstarted.run("no execution").close()
        self.assertFalse(self.client.get(
            f"/sessions/{unstarted.run_id}").json()["resumable"])
        self.assertEqual(self.client.post(
            "/run/stream", json={"message": "new", "resume_from_run_id": unstarted.run_id}).status_code, 409)
        self.fixture.rewrite(
            source, lambda log: log[0].update(schema_version=99))
        response = self.client.get(f"/sessions/{source.run_id}")
        self.assertEqual(response.status_code, 409)
        self.assertNotIn(str(self.store.root), response.text)
        self.assertEqual(len(self.store.list()), 2)

    def test_named_route_identity_revision_and_workspace_conflicts(self):
        named = self.source(True)
        ad_hoc = self.source()
        cases = [
            ("/run", named.run_id, {}),
            ("/agents/other/run", named.run_id, {}),
            ("/agents/reviewer/run", ad_hoc.run_id, {}),
            ("/agents/reviewer/run", named.run_id, {"revision": 2}),
        ]
        for base, identity, extra in cases:
            for suffix in ("", "/stream"):
                response = self.client.post(
                    base + suffix, json={"message": "new", "resume_from_run_id": identity, **extra})
                self.assertEqual(response.status_code, 409, response.text)
        response = self.client.post("/agents/reviewer/run/stream", json={
            "message": "new", "resume_from_run_id": named.run_id, "cwd": str(self.fixture.root),
        })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(self.store.list()), 2)

    def test_logging_off_still_restores_but_reports_no_saved_continuation(self):
        source = self.source()
        with mock.patch.dict(os.environ, {"PYAGENT_LOGGING": "off"}):
            for path in ("/run", "/run/stream"):
                response = self.client.post(
                    path, json={"message": "continue", "resume_from_run_id": source.run_id})
                data = sse_events(
                    response.text)[-1] if path.endswith("stream") else response.json()
                self.assertFalse(data["log_saved"])
                self.assertEqual(data["parent_run_id"], source.run_id)
                self.assertIn("Logging is off", "\n".join(data["warnings"]))
                self.assertEqual(self.client.get(
                    f"/sessions/{data['run_id']}").status_code, 404)
        self.assertEqual(len(self.store.list()), 1)

    def test_terminal_log_write_failure_is_reported_after_producer_cleanup(self):
        source = self.source()
        original = SessionLogger._write_entry

        def fail_end(logger, severity, body, attributes=None):
            if body == "run_end":
                logger._failed(OSError("disk full"))
            else:
                original(logger, severity, body, attributes)
        with mock.patch.object(SessionLogger, "_write_entry", fail_end), contextlib.redirect_stderr(io.StringIO()):
            for path in ("/run", "/run/stream"):
                response = self.client.post(
                    path, json={"message": "continue", "resume_from_run_id": source.run_id})
                self.assertEqual(response.status_code, 200)
                data = sse_events(
                    response.text)[-1] if path.endswith("stream") else response.json()
                self.assertFalse(data["log_saved"])
                self.assertEqual(data["response"], "Done")
                self.assertFalse(self.store.inspect(data["run_id"]).resumable)

    def test_ordinary_history_requests_do_not_infer_parentage(self):
        source = self.source()
        response = self.client.post(
            "/run", json={"message": "continue", "messages": source.messages})
        data = response.json()
        self.assertIsNone(data["parent_run_id"])
        self.assertTrue(data["log_saved"])
        self.assertIsNone(self.store.load(data["run_id"]).parent_run_id)

    def test_concurrent_resumes_branch_without_modifying_the_source(self):
        source = self.source()
        before = source.last_log_path.read_bytes()

        def submit(index):
            with TestClient(create_app()) as client:
                response = client.post("/run/stream", json={
                    "message": f"branch {index}", "resume_from_run_id": source.run_id,
                })
                self.assertEqual(response.status_code, 200)
                return sse_events(response.text)[-1]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(submit, range(4)))
        self.assertEqual(len({result["run_id"] for result in results}), 4)
        for result in results:
            self.assertEqual(result["parent_run_id"], source.run_id)
            self.assertTrue(result["log_saved"])
            self.assertEqual(self.store.load(
                result["run_id"]).parent_run_id, source.run_id)
        self.assertEqual(source.last_log_path.read_bytes(), before)

    def test_resumed_stream_disconnect_retains_producer_owned_cleanup(self):
        source = self.source()
        before = source.last_log_path.read_bytes()
        agent, extra = _prepare_request_agent(ChatRequest(
            message="continue", resume_from_run_id=source.run_id))
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()

        def chunks():
            entered.set()
            release.wait(5)
            yield {"content": "partial"}
        agent.client.streams = iter([chunks()])
        original_close = agent.close

        def close(*args, **kwargs):
            try:
                original_close(*args, **kwargs)
            finally:
                closed.set()
        agent.close = close

        async def exercise():
            stream = stream_agent_sse(
                agent, "continue", start_data=extra, completion_data=lambda text: {"response": text})
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
        self.assertEqual(log[0]["attributes"]["parent_run_id"], source.run_id)
        self.assertEqual(log[-1]["attributes"]["outcome"], "cancelled")
        self.assertFalse(agent.log_saved)
        self.assertEqual(source.last_log_path.read_bytes(), before)

    def test_python_client_resume_payloads_capabilities_and_optional_response_fields(self):
        client = PyAgentClient()
        identity = str(uuid4())
        response = {"run_id": "child", "parent_run_id": identity, "log_saved": True,
                    "warnings": ["notice"], "response": "ok", "profile": "p", "provider": "ollama",
                    "model": "m", "messages": [], "agent": "reviewer", "revision": 1}

        def request(method, path, payload=None):
            return {"enabled": True, "resume_supported": True} if path.endswith("capabilities") else response
        with mock.patch.object(client, "_request_json", side_effect=request) as send:
            for invoke in (lambda: client.run("new", resume_from_run_id=identity),
                           lambda: client.run_agent("reviewer", "new", resume_from_run_id=identity)):
                result = invoke()
                self.assertTrue(result.log_saved)
                self.assertEqual(result.parent_run_id, identity)
                self.assertEqual(result.warnings, ["notice"])
                payload = send.call_args.args[2]
                self.assertEqual(payload["resume_from_run_id"], identity)
                self.assertNotIn("messages", payload)
                self.assertNotIn("skills", payload)
            with mock.patch.object(client, "_stream_sse", return_value=iter([])) as stream:
                client.stream_run("new", resume_from_run_id=identity)
                self.assertNotIn("messages", stream.call_args.args[2])
                client.stream_agent("reviewer", "new",
                                    resume_from_run_id=identity)
                self.assertNotIn("messages", stream.call_args.args[2])
            with self.assertRaises(PyAgentClientError):
                client.run("new", resume_from_run_id=identity, messages=[])
            with self.assertRaises(PyAgentClientError):
                client.stream_run(
                    "new", resume_from_run_id=identity, skills=[])
        with mock.patch.object(client, "_request_json", return_value={"enabled": False, "resume_supported": True}) as send:
            with self.assertRaises(PyAgentClientError):
                client.run("new", resume_from_run_id=identity)
            # never POST to unsupported/disabled servers
            self.assertEqual(send.call_count, 1)
        with mock.patch.object(client, "_request_json", return_value={}) as send:
            with self.assertRaises(PyAgentClientError):
                client.run("new", resume_from_run_id=identity)
            self.assertEqual(send.call_count, 1)


if __name__ == "__main__":
    unittest.main()
