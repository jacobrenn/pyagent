from __future__ import annotations

import asyncio
import contextlib
import copy
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
from uuid import uuid4

from pyagent.agent import Agent
from pyagent.agent_definitions import AgentDefinitionStore, build_agent_from_definition
from pyagent.config import AppConfig
from pyagent.llm_client import OpenAICompatibleClient, OpenAIResponsesClient
from pyagent.main import main
from pyagent.resume import restore_session
from pyagent.session_store import SessionError, SessionStore, format_sessions, validate_messages
from pyagent.tools import ToolRegistry, ToolSpec
from pyagent.ui import PyAgentApp
from test_run_logging import FakeClient, records


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.profiles = self.root / "profiles.json"
        self.profiles.write_text(json.dumps({
            "default_profile": "test",
            "profiles": {
                "test": {"provider": "ollama", "model": "test-model", "base_url": "http://localhost:11434"},
                "other": {"provider": "openai", "model": "other-model", "base_url": "http://localhost:8000/v1"},
            },
        }))
        env = mock.patch.dict(os.environ, {
            "PYAGENT_LOGGING": "on",
            "PYAGENT_USER_DIR": str(self.root / "user"),
            "PYAGENT_USER_TOOLS_ENABLED": "off",
            "PYAGENT_MODEL_PROFILES_PATH": str(self.profiles),
            "PYAGENT_SYSTEM_PROMPT_PATH": str(self.root / "system.txt"),
        }, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.clients = []

        def build(profile, **kwargs):
            client = FakeClient()
            client.profile = profile
            self.clients.append(client)
            return client
        patch = mock.patch(
            "pyagent.agent.build_chat_client", side_effect=build)
        patch.start()
        self.addCleanup(patch.stop)
        self.store = SessionStore()

    def agent(self, streams=None, **changes):
        config = AppConfig.from_env()
        for key, value in changes.items():
            setattr(config, key, value)
        agent = Agent(config=config, tool_registry=ToolRegistry(
            []), workspace=self.root)
        if streams is not None:
            agent.client.streams = iter(streams)
        self.addCleanup(agent.close)
        return agent

    def saved(self, **changes):
        agent = self.agent(**changes)
        list(agent.run("original prompt"))
        return agent

    def restore(self, identity, **kwargs):
        restored = restore_session(identity, store=self.store, **kwargs)
        self.addCleanup(restored.agent.close)
        return restored

    def rewrite(self, agent, edit):
        log = records(agent.last_log_path)
        edit(log)
        agent.last_log_path.write_text(
            "".join(json.dumps(r) + "\n" for r in log))

    def test_empty_store_does_not_create_logs(self):
        self.assertEqual(self.store.list(), [])
        self.assertIn("No saved runs", format_sessions([]))
        self.assertFalse(self.store.root.exists())

    def test_restores_checkpoint_not_repeated_request_records(self):
        source = self.saved()
        before = source.last_log_path.read_bytes()
        (self.root / "AGENTS.md").write_text("Current trusted project instructions")
        with mock.patch.object(FakeClient, "chat_stream") as chat, mock.patch.object(ToolRegistry, "execute") as execute:
            restored = self.restore(source.run_id)
            chat.assert_not_called()
            execute.assert_not_called()
        agent = restored.agent
        self.assertEqual(agent.messages[1:], source.messages[1:])
        self.assertIn("Current trusted project instructions",
                      agent.messages[0]["content"])
        self.assertEqual(agent.parent_run_id, source.run_id)
        self.assertIsNone(agent._run_log)
        self.assertEqual(len(self.store.list()), 1)
        captured = []

        def chunks(messages, tools=None):
            captured.append(copy.deepcopy(messages))
            yield {"content": "continued"}
        agent.client.chat_stream = chunks
        list(agent.run("new prompt"))
        self.assertEqual([m["content"] for m in captured[0] if m["role"] == "user"], [
                         "original prompt", "new prompt"])
        self.assertNotEqual(agent.run_id, source.run_id)
        self.assertEqual(records(agent.last_log_path)[
                         0]["attributes"]["parent_run_id"], source.run_id)
        self.assertEqual(source.last_log_path.read_bytes(), before)

    def test_branches_and_missing_ancestors(self):
        source = self.saved()
        branches = [self.restore(source.run_id).agent for _ in range(2)]
        for agent in branches:
            list(agent.run("branch"))
            self.assertEqual(self.store.lineage(agent.run_id),
                             ([agent.run_id, source.run_id], None))
        self.assertNotEqual(branches[0].run_id, branches[1].run_id)
        source.last_log_path.unlink()
        restored = self.restore(branches[0].run_id)
        self.assertEqual(restored.agent.parent_run_id, branches[0].run_id)
        chain, warning = self.store.lineage(branches[0].run_id)
        self.assertEqual(chain, [branches[0].run_id])
        self.assertIn(source.run_id, warning)

    def test_listing_groups_only_explicit_links_and_does_not_keep_histories(self):
        first = self.saved()
        second = self.restore(first.run_id).agent
        list(second.run("child"))
        unrelated = self.saved()
        runs = {r.run_id: r for r in self.store.list()}
        self.assertEqual(runs[second.run_id].conversation_id, first.run_id)
        self.assertEqual(
            runs[unrelated.run_id].conversation_id, unrelated.run_id)
        self.assertTrue(all(not r.messages for r in runs.values()))
        text = format_sessions(list(runs.values()))
        self.assertEqual(text.count(f"Conversation {first.run_id}"), 1)

    def test_parent_cycle_is_reported_without_looping(self):
        first = self.saved()
        second = self.restore(first.run_id).agent
        list(second.run("second"))
        self.rewrite(first, lambda log: log[0]["attributes"].update(
            parent_run_id=second.run_id))
        chain, warning = self.store.lineage(second.run_id)
        self.assertEqual(len(chain), 2)
        self.assertIn("Cyclic", warning)

    def test_logging_off_and_reset_do_not_create_or_infer_ancestry(self):
        source = self.saved()
        config = replace(AppConfig.from_env(), logging_enabled=False)
        agent = self.restore(source.run_id, config=config).agent
        list(agent.run("not saved"))
        self.assertEqual(len(self.store.list()), 1)
        self.assertIsNone(agent.last_log_path)
        agent.reset()
        self.assertIsNone(agent.parent_run_id)

    def test_current_history_limit_applies_without_changing_source(self):
        source = self.saved()
        saved = self.store.load(source.run_id)
        config = replace(AppConfig.from_env(), max_history_messages=1)
        restored = self.restore(source.run_id, config=config)
        self.assertLess(len(restored.agent.messages), len(saved.messages))
        self.assertTrue(
            any("trimmed" in warning for warning in restored.warnings))
        self.assertEqual(self.store.load(
            source.run_id).messages, saved.messages)

    def test_tools_are_not_replayed_and_history_works_in_all_adapters(self):
        source = self.agent([
            [{"tool_calls": [{"function": {"name": "echo", "arguments": {}}}]}],
            [{"content": "done"}],
        ])
        handler = mock.Mock(return_value="original result")
        source.tool_registry = ToolRegistry([ToolSpec(name="echo", description="echo", parameters={
                                            "type": "object", "properties": {}}, handler=handler)])
        source.tools = source.tool_registry.definitions()
        list(source.run("use tool"))
        handler.assert_called_once()
        restored = self.restore(source.run_id)
        handler.assert_called_once()
        messages = restored.agent.messages
        call = next(m for m in messages if m.get(
            "tool_calls"))["tool_calls"][0]
        output = next(m for m in messages if m["role"] == "tool")
        self.assertEqual(call["id"], output["tool_call_id"])
        self.assertTrue(any("Ollama" in w for w in restored.warnings))
        profile = replace(restored.agent.current_profile(), provider="openai")
        chat = OpenAICompatibleClient(profile)
        self.addCleanup(chat.close)
        prepared = chat._prepare_messages(messages)
        self.assertTrue(
            any(m.get("content") == "original result" for m in prepared))
        responses = OpenAIResponsesClient(
            replace(profile, api_mode="responses"))
        self.addCleanup(responses.close)
        _, prepared = responses._prepare_input(messages)
        self.assertTrue(any(m.get(
            "type") == "function_call_output" and m["output"] == "original result" for m in prepared))

    def test_invalid_tool_history_is_rejected(self):
        call = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call-1", "function": {"name": "echo", "arguments": {}}}]}
        output = {"role": "tool", "name": "echo",
                  "tool_call_id": "call-1", "content": "done"}
        for messages in ([call], [output], [call, {**output, "tool_call_id": "wrong"}], [call, output, output]):
            with self.subTest(messages=messages), self.assertRaises(SessionError):
                validate_messages(messages, provider="openai")

    def test_unresumable_outcomes_legacy_and_partial_logs(self):
        unstarted = self.agent()
        unstarted.run("never started").close()
        cancelled = self.agent()
        run = cancelled.run("cancel")
        next(run)
        run.close()
        failed = self.agent([[{"error": "failure"}]])
        list(failed.run("fail"))
        incomplete = self.saved()
        log = records(incomplete.last_log_path)
        incomplete.last_log_path.write_text(
            "".join(json.dumps(r) + "\n" for r in log[:-1]))
        legacy = self.store.root / "session-old.jsonl"
        legacy.write_text("{}\n")
        for reference in (unstarted.run_id, cancelled.run_id, failed.run_id, incomplete.run_id, legacy.name):
            with self.subTest(reference=reference):
                saved = self.store.inspect(reference)
                self.assertFalse(saved.resumable)
                with self.assertRaises(SessionError):
                    self.store.load(reference)
        self.assertEqual(len(self.store.list()), 5)

    def test_schema_identity_sequence_and_message_corruption(self):
        edits = [
            lambda log: log[0].update(schema_version=99),
            lambda log: log[1].update(run_id=str(uuid4())),
            lambda log: log[1].update(sequence=99),
            lambda log: log.append(copy.deepcopy(log[-1])),
            lambda log: log[-1]["attributes"].update(
                messages=[{"role": "invalid", "content": "x"}]),
            lambda log: log[-1]["attributes"].update(
                messages=[{"role": [], "content": "x"}]),
            lambda log: log[-1]["attributes"].update(outcome={}),
            lambda log: log[0]["attributes"].update(
                parent_run_id=log[0]["run_id"]),
        ]
        for edit in edits:
            source = self.saved()
            self.rewrite(source, edit)
            with self.assertRaises(SessionError):
                self.store.load(source.run_id)
        source = self.saved()
        source.last_log_path.write_bytes(
            source.last_log_path.read_bytes()[:-4])
        with self.assertRaisesRegex(SessionError, "truncated"):
            self.store.load(source.run_id)
        self.assertTrue(all(not r.resumable for r in self.store.list()))

    def test_file_record_and_message_limits(self):
        source = self.saved()
        for name in ("MAX_FILE_BYTES", "MAX_RECORD_BYTES", "MAX_MESSAGES", "MAX_JSON_DEPTH", "MAX_JSON_NODES"):
            with self.subTest(limit=name), mock.patch("pyagent.session_store." + name, 1), self.assertRaises(SessionError):
                self.store.load(source.run_id)

    def test_paths_symlinks_and_duplicate_ids_are_rejected(self):
        source = self.saved()
        for reference in ("../profiles.jsonl", str(source.last_log_path), "not-an-id"):
            with self.subTest(reference=reference), self.assertRaises(SessionError):
                self.store.load(reference)
        symlink = self.store.root / "session-link.jsonl"
        symlink.symlink_to(self.profiles)
        with self.assertRaisesRegex(SessionError, "symlink"):
            self.store.inspect(symlink.name)
        duplicate = self.store.root / f"run-copy-{source.run_id}.jsonl"
        duplicate.write_bytes(source.last_log_path.read_bytes())
        with self.assertRaisesRegex(SessionError, "found 2"):
            self.store.load(source.run_id)
        with self.assertRaises(SessionError):
            self.store.load(source.last_log_path.name)

    def test_log_changes_during_read_are_rejected(self):
        source = self.saved()
        original = self.store._read

        def changing(stream, path):
            saved = original(stream, path)
            with path.open("ab") as writer:
                writer.write(b"\n")
            return saved
        with mock.patch.object(self.store, "_read", side_effect=changing), self.assertRaisesRegex(SessionError, "changed"):
            self.store.load(source.run_id)

    def test_saved_system_and_credentials_are_not_restored(self):
        source = self.saved()

        def alter(log):
            log[-1]["attributes"]["messages"][0]["content"] = "UNTRUSTED SAVED SYSTEM"
            log[0]["attributes"]["metadata"]["api_key"] = "do-not-load"
            log[0]["attributes"]["metadata"]["base_url"] = "https://untrusted.invalid"
        self.rewrite(source, alter)
        restored = self.restore(source.run_id)
        self.assertNotIn("UNTRUSTED SAVED SYSTEM",
                         restored.agent.messages[0]["content"])
        self.assertEqual(restored.agent.current_profile(
        ).base_url, "http://localhost:11434")
        self.assertIsNone(restored.agent.current_profile().api_key)
        self.assertTrue(restored.agent.restrict_workspace)

    def test_skills_use_live_contents_and_survive_tui_context_reload(self):
        skills = self.root / "skills"
        skills.mkdir()
        skill = skills / "guide.md"
        skill.write_text("Old skill instructions")
        source = self.saved()
        self.rewrite(source, lambda log: log[0]["attributes"]["metadata"].update(
            skills=["project:skills/guide.md"]))
        skill.write_text("New skill instructions")
        restored = self.restore(source.run_id)
        self.assertIn("New skill instructions",
                      restored.agent.messages[0]["content"])
        app = PyAgentApp(resumed_session=restored)
        self.assertEqual(app.loaded_user_skills, ["project:skills/guide.md"])
        app._reload_context_details()
        self.assertIn("New skill instructions",
                      app.agent.messages[0]["content"])
        self.assertIsNone(app.agent.parent_run_id)  # explicit context reset

    def test_missing_runtime_resources_fail_before_execution(self):
        source = self.saved()
        for kwargs in ({"profile": "missing"}, {"cwd": str(self.root / "missing")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.restore(source.run_id, **kwargs)
        self.rewrite(source, lambda log: log[0]["attributes"]["metadata"].update(
            skills=["project:skills/missing.md"]))
        with self.assertRaisesRegex(SessionError, "skill"):
            self.restore(source.run_id)
        self.assertEqual(len(self.store.list()), 1)

    def test_ad_hoc_overrides_and_current_safety_policy(self):
        source = self.saved(bash_enabled=False, bash_readonly_mode=True)
        other = self.root / "other-workspace"
        other.mkdir()
        restored = self.restore(
            source.run_id, profile="other", model="override", cwd=str(other))
        self.assertEqual(restored.agent.workspace, other)
        self.assertEqual(restored.agent.current_profile().name, "other")
        self.assertEqual(restored.agent.current_profile().model, "override")
        self.assertFalse(restored.agent.config.bash_enabled)
        self.assertTrue(restored.agent.config.bash_readonly_mode)
        self.assertTrue(any("workspace" in w for w in restored.warnings))
        # Current policy can further restrict a saved enabled setting.
        other_source = self.saved()
        restored = self.restore(other_source.run_id, config=replace(
            AppConfig.from_env(), tools_enabled=False))
        self.assertEqual(restored.agent.tools, [])

    def test_named_revision_relative_workspace_and_locks(self):
        workspace = self.root / "repo"
        workspace.mkdir()
        definitions = AgentDefinitionStore()
        definitions.create(
            {"name": "reviewer", "profile": "test", "workspace": "repo", "tools": []})
        source, _, _ = build_agent_from_definition("reviewer", cwd=self.root)
        self.addCleanup(source.close)
        list(source.run("review"))
        definitions.update("reviewer", {"model": "different-model"})
        restored = self.restore(source.run_id)
        self.assertEqual(restored.definition.revision, 1)
        self.assertEqual(restored.agent.workspace, workspace)
        self.assertTrue(restored.agent.restrict_workspace)
        self.assertEqual(restored.agent.allowed_tools, frozenset())
        app = PyAgentApp(resumed_session=restored)
        self.assertIsNotNone(
            app._definition_command_lock_message("/model", ["other"]))
        for kwargs in ({"profile": "other"}, {"model": "other"}, {"cwd": str(workspace)}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(SessionError, "overrides"):
                self.restore(source.run_id, **kwargs)
        definitions.delete("reviewer")
        with self.assertRaises(ValueError):
            self.restore(source.run_id)

    def test_named_resume_uses_selected_stores_user_directory(self):
        user_dir = self.root / "other-user"
        definitions = AgentDefinitionStore(user_dir)
        definitions.create(
            {"name": "private", "workspace": str(self.root), "tools": []})
        source, _, _ = build_agent_from_definition(
            "private", user_dir=user_dir)
        self.addCleanup(source.close)
        list(source.run("review"))
        restored = restore_session(source.run_id, store=SessionStore(user_dir))
        self.addCleanup(restored.agent.close)
        self.assertEqual(restored.definition.name, "private")
        self.assertEqual(Path(restored.agent.config.user_dir), user_dir)

    def test_named_absolute_workspace_cannot_be_changed_by_log(self):
        definitions = AgentDefinitionStore()
        definitions.create(
            {"name": "fixed", "workspace": str(self.root), "tools": []})
        source, _, _ = build_agent_from_definition("fixed")
        self.addCleanup(source.close)
        list(source.run("review"))
        other = self.root / "other"
        other.mkdir()
        self.rewrite(source, lambda log: log[0]["attributes"]["metadata"].update(
            workspace=str(other)))
        with self.assertRaisesRegex(ValueError, "absolute workspace"):
            self.restore(source.run_id)

    def test_named_inherited_profile_and_model_survive_default_changes(self):
        definitions = AgentDefinitionStore()
        definitions.create(
            {"name": "inherited", "workspace": str(self.root), "tools": []})
        source, _, _ = build_agent_from_definition("inherited")
        self.addCleanup(source.close)
        list(source.run("review"))
        profiles = json.loads(self.profiles.read_text())
        profiles["default_profile"] = "other"
        profiles["profiles"]["test"]["model"] = "new-default-model"
        self.profiles.write_text(json.dumps(profiles))
        restored = self.restore(source.run_id)
        self.assertEqual(restored.agent.current_profile().name, "test")
        self.assertEqual(restored.agent.current_profile().model, "test-model")

    def test_restore_failure_closes_allocated_agent(self):
        source = self.saved()
        with mock.patch.object(Agent, "load_messages", side_effect=ValueError("invalid")), self.assertRaises(ValueError):
            restore_session(source.run_id)
        self.assertTrue(self.clients[-1].closed)
        self.assertFalse(source.client.closed)

    def test_cli_list_show_and_single_prompt_resume(self):
        source = self.saved()
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            main(["sessions", "list"])
        self.assertIn(source.run_id, stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            main(["sessions", "show", source.run_id])
        self.assertEqual(json.loads(stdout.getvalue())
                         ["messages"], source.messages)
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            main(["resume", source.run_id, "--prompt", "continue"])
        self.assertEqual(stdout.getvalue(), "Done")
        self.assertIn("Resumed checkpoint", stderr.getvalue())
        self.assertTrue(self.clients[-1].closed)
        child = next(r for r in self.store.list() if r.run_id != source.run_id)
        self.assertEqual(child.parent_run_id, source.run_id)

    def test_cli_interactive_resume_and_invalid_selection(self):
        source = self.saved()
        apps = []

        def make_app(**kwargs):
            restored = kwargs["resumed_session"]
            app = mock.Mock(agent=restored.agent)
            apps.append(app)
            return app
        with mock.patch("pyagent.ui.PyAgentApp", side_effect=make_app):
            main(["resume", source.run_id])
        apps[0].run.assert_called_once()
        self.assertTrue(apps[0].agent.client.closed)
        self.assertEqual(len(self.store.list()), 1)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            main(["resume", str(uuid4()), "--prompt", "continue"])
        self.assertEqual(exc.exception.code, 2)

    def test_named_chat_closes_current_agent_after_session_switch(self):
        original = self.agent()
        replacement = self.agent()
        definitions = AgentDefinitionStore()
        definition = definitions.create({"name": "reviewer", "tools": []})
        app = mock.Mock(agent=original)

        def switch():
            original.close(reason="session_replaced")
            app.agent = replacement
        app.run.side_effect = switch
        with mock.patch("pyagent.agent_definitions.build_agent_from_definition", return_value=(original, definition, None)), mock.patch("pyagent.ui.PyAgentApp", return_value=app):
            main(["agents", "chat", "reviewer"])
        self.assertTrue(original.client.closed)
        self.assertTrue(replacement.client.closed)

    def test_cli_resume_respects_logging_off_and_ad_hoc_overrides(self):
        source = self.saved()
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, {"PYAGENT_LOGGING": "off"}), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            main(["resume", source.run_id, "--profile", "other",
                 "--model", "override", "--prompt", "continue"])
        self.assertEqual(stdout.getvalue(), "Done")
        self.assertIn("Logging is off", stderr.getvalue())
        self.assertEqual(self.clients[-1].profile.name, "other")
        self.assertEqual(self.clients[-1].profile.model, "override")
        self.assertEqual(len(self.store.list()), 1)

    def test_tui_startup_restores_history_without_new_log(self):
        source = self.saved()
        restored = self.restore(source.run_id)
        app = PyAgentApp(resumed_session=restored)

        async def exercise():
            async with app.run_test() as pilot:
                await pilot.pause()
                self.assertEqual(app.input_history, ["original prompt"])
                self.assertEqual(len(self.store.list()), 1)
                self.assertEqual(app.agent.messages[1:], source.messages[1:])
                await app.process_message("continue")
                self.assertEqual(records(app.agent.last_log_path)[
                                 0]["attributes"]["parent_run_id"], source.run_id)
        asyncio.run(exercise())

    def test_tui_confirmation_busy_failure_and_successful_swap(self):
        source = self.saved()
        old = self.agent()
        old.add_message("user", "unsaved conversation")
        app = PyAgentApp(agent=old)

        async def exercise():
            async with app.run_test() as pilot:
                await pilot.pause()
                app._handle_slash_command(f"/resume {source.run_id}")
                self.assertIs(app.agent, old)
                self.assertFalse(app.is_processing)
                app.is_processing = True
                app._handle_slash_command(f"/resume {source.run_id} --confirm")
                self.assertIs(app.agent, old)
                app.is_processing = False
                app._handle_slash_command(f"/resume {uuid4()} --confirm")
                await app.workers.wait_for_complete()
                self.assertIs(app.agent, old)
                self.assertFalse(old.client.closed)
                self.assertEqual(
                    old.messages[-1]["content"], "unsaved conversation")
                app._handle_logging_command(["off"])
                app._handle_slash_command(f"/resume {source.run_id} --confirm")
                await app.workers.wait_for_complete()
                self.assertIsNot(app.agent, old)
                self.assertTrue(old.client.closed)
                self.assertFalse(app.agent.logging_enabled)
                self.assertEqual(app.agent.parent_run_id, source.run_id)
                self.assertEqual(app.input_history, ["original prompt"])
                self.assertEqual(len(self.store.list()), 1)
                app.agent.close()
        asyncio.run(exercise())

    def test_tui_cancelled_restore_closes_candidate_without_swapping(self):
        source = self.saved()
        old = self.agent()
        app = PyAgentApp(agent=old)
        candidate = self.restore(source.run_id)
        entered, release = threading.Event(), threading.Event()

        def blocked(*args, **kwargs):
            entered.set()
            release.wait(5)
            return candidate

        async def exercise():
            async with app.run_test() as pilot:
                await pilot.pause()
                with mock.patch("pyagent.resume.restore_session", side_effect=blocked):
                    task = asyncio.create_task(
                        app._session_action("/resume", [source.run_id]))
                    self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                    task.cancel()
                    release.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                self.assertIs(app.agent, old)
                self.assertTrue(candidate.agent.client.closed)
                self.assertFalse(old.client.closed)
        try:
            asyncio.run(exercise())
        finally:
            release.set()

    def test_tui_extension_start_failure_keeps_original_agent(self):
        source = self.saved()
        old = self.agent()
        app = PyAgentApp(agent=old)
        candidate = self.restore(source.run_id)

        async def exercise():
            async with app.run_test() as pilot:
                await pilot.pause()
                with mock.patch("pyagent.resume.restore_session", return_value=candidate), mock.patch.object(candidate.agent, "load_extensions", side_effect=ValueError("startup failed")):
                    app._handle_slash_command(f"/resume {source.run_id}")
                    await app.workers.wait_for_complete()
                self.assertIs(app.agent, old)
                self.assertFalse(old.client.closed)
                self.assertTrue(candidate.agent.client.closed)
        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
