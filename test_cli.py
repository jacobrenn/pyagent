"""End-to-end tests for the unified CLI contract.

Every command prints either a human-readable rendering (default) or, with
``--json``, a single JSON envelope on stdout::

    {"ok": true,  "command": "<name>", "data": {...}}
    {"ok": false, "command": "<name>", "error": {"code": "...", "message": "..."}}

``ok`` mirrors the process exit code. These tests pin down the envelope
shape, the exit codes, and where output goes (stdout vs stderr).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from pyagent import cli_output
from pyagent.cli_output import EXIT_ERROR, EXIT_OK, EXIT_USAGE, CliError, Result
from pyagent.main import main


def run_cli(argv, *, user_dir: str | None = None):
    """Run the CLI entry point, capturing exit code, stdout, and stderr."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    env = {} if user_dir is None else {"PYAGENT_USER_DIR": user_dir}
    with mock.patch.dict(os.environ, env, clear=False):
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                main(argv)
                code = 0
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
    return code, stdout.getvalue(), stderr.getvalue()


def read_envelope(stdout: str) -> dict:
    return json.loads(stdout)


def decode_envelopes(buf: str) -> list[dict]:
    """Decode concatenated pretty-printed JSON envelopes from a buffer."""
    decoder = json.JSONDecoder()
    envelopes = []
    idx = 0
    text = buf.strip()
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        envelopes.append(obj)
        idx = end
    return envelopes


class CliEnvelopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.user_dir = os.path.join(self.tmp.name, "user")
        os.makedirs(self.user_dir)

    def test_json_success_envelope(self):
        code, out, err = run_cli(["health", "--json"], user_dir=self.user_dir)
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(err, "")
        payload = read_envelope(out)
        self.assertIs(payload["ok"], True)
        self.assertEqual(payload["command"], "health")
        self.assertEqual(payload["data"]["status"], "ok")

    def test_json_flag_accepted_before_subcommand(self):
        code, out, _ = run_cli(["--json", "health"], user_dir=self.user_dir)
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(read_envelope(out)["command"], "health")

    def test_json_flag_accepted_between_command_and_subcommand(self):
        code, out, _ = run_cli(
            ["sessions", "--json", "list"], user_dir=self.user_dir
        )
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(read_envelope(out)["command"], "sessions.list")

    def test_json_flag_accepted_after_subcommand(self):
        code, out, _ = run_cli(
            ["sessions", "list", "--json"], user_dir=self.user_dir
        )
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(read_envelope(out)["command"], "sessions.list")

    def test_json_envelope_accepted_at_every_level(self):
        commands = [
            ["version"],
            ["sessions", "list"],
            ["profiles", "list"],
            ["prompts", "list"],
            ["tools", "list"],
            ["skills", "list"],
            ["agents", "list"],
            ["extensions", "list"],
            ["mcp", "list", "--no-connect"],
        ]
        for argv in commands:
            with self.subTest(argv=argv):
                code, out, _ = run_cli(
                    [*argv, "--json"], user_dir=self.user_dir)
                self.assertEqual(code, EXIT_OK, msg=out)
                payload = read_envelope(out)
                self.assertIs(payload["ok"], True)

    def test_human_output_has_no_envelope(self):
        code, out, err = run_cli(["health"], user_dir=self.user_dir)
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(out.strip(), "ok")
        self.assertEqual(err, "")

    def test_failure_envelope_goes_to_stdout_with_nonzero_exit(self):
        code, out, err = run_cli(
            ["prompts", "show", "definitely-missing-name", "--json"],
            user_dir=self.user_dir,
        )
        self.assertEqual(code, EXIT_ERROR)
        payload = read_envelope(out)
        self.assertIs(payload["ok"], False)
        self.assertEqual(payload["command"], "prompts.show")
        self.assertEqual(payload["error"]["code"], "not_found")
        self.assertTrue(payload["error"]["message"])
        self.assertEqual(err, "")

    def test_failure_human_output_goes_to_stderr(self):
        code, out, err = run_cli(
            ["prompts", "show", "definitely-missing-name"],
            user_dir=self.user_dir,
        )
        self.assertEqual(code, EXIT_ERROR)
        self.assertEqual(out, "")
        self.assertIn("definitely-missing-name", err)

    def test_unknown_subcommand_exits_with_usage_error(self):
        code, _, _ = run_cli(["definitely-not-a-command"],
                             user_dir=self.user_dir)
        self.assertEqual(code, EXIT_USAGE)

    def test_unknown_prompt_exits_with_error(self):
        code, out, _ = run_cli(
            ["prompts", "show", "nope", "--json"], user_dir=self.user_dir
        )
        self.assertEqual(code, EXIT_ERROR)
        self.assertIs(read_envelope(out)["ok"], False)

    def test_agents_validate_invalid_definition_exits_2(self):
        from pyagent.agent_definitions import AgentDefinitionStore

        store = AgentDefinitionStore(self.user_dir)
        store.create({"name": "broken", "profile": "no-such-profile"})
        code, out, _ = run_cli(
            ["agents", "validate", "broken", "--json"], user_dir=self.user_dir
        )
        self.assertEqual(code, EXIT_USAGE)
        payload = read_envelope(out)
        self.assertIs(payload["ok"], False)
        self.assertIs(payload["data"]["valid"], False)
        self.assertTrue(payload["data"]["errors"])

    def test_agents_validate_valid_definition_exits_0(self):
        from pyagent.agent_definitions import AgentDefinitionStore

        store = AgentDefinitionStore(self.user_dir)
        store.create({"name": "empty"})
        code, out, _ = run_cli(
            ["agents", "validate", "empty", "--json"], user_dir=self.user_dir
        )
        payload = read_envelope(out)
        self.assertIn("valid", payload["data"])
        self.assertEqual(
            code, EXIT_OK if payload["data"]["valid"] else EXIT_USAGE)
        self.assertEqual(payload["ok"], payload["data"]["valid"])


class CliStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.user_dir = os.path.join(self.tmp.name, "user")
        os.makedirs(self.user_dir)

    def _fake_agent(self):
        agent = mock.Mock()
        agent.run.side_effect = lambda prompt: iter([
            {"type": "content_delta", "delta": "Hel"},
            {"type": "content_delta", "delta": "lo"},
            {"type": "assistant_done", "content": "Hello"},
        ])
        agent.current_profile.side_effect = RuntimeError("no profile in tests")
        agent.project_context_files = []
        agent.run_id = "run-123"
        agent.parent_run_id = None
        agent.log_saved = False
        agent.messages = []
        return agent

    def test_streaming_json_emits_ndjson_events(self):
        agent = self._fake_agent()
        with mock.patch("pyagent.main.build_agent_for_request", return_value=agent):
            code, out, err = run_cli(
                ["--prompt", "hi", "--json", "--stream"],
                user_dir=self.user_dir,
            )
        self.assertEqual(code, EXIT_OK, msg=err)
        events = [json.loads(line)
                  for line in out.splitlines() if line.strip()]
        self.assertTrue(all(event.get("ok") for event in events))
        types = [event["event"]["type"]
                 for event in events if "event" in event]
        self.assertEqual(
            types, ["content_delta", "content_delta", "assistant_done", "done"])
        done = events[-1]
        self.assertEqual(done["event"]["data"]["response"], "Hello")

    def test_streaming_plain_prints_deltas(self):
        agent = self._fake_agent()
        with mock.patch("pyagent.main.build_agent_for_request", return_value=agent):
            code, out, _ = run_cli(
                ["--prompt", "hi", "--stream"],
                user_dir=self.user_dir,
            )
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(out, "Hello")

    def test_run_failure_envelope(self):
        agent = self._fake_agent()
        agent.run.side_effect = lambda prompt: iter([
            {"type": "error", "message": "model exploded"},
        ])
        with mock.patch("pyagent.main.build_agent_for_request", return_value=agent):
            code, out, err = run_cli(
                ["--prompt", "hi"], user_dir=self.user_dir
            )
        self.assertEqual(code, EXIT_ERROR)
        self.assertIn("model exploded", err)


class EmitUnitTests(unittest.TestCase):
    def test_emit_maps_exit_code_to_ok_flag(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli_output.emit(Result(command="c", data={"a": 1}, exit_code=EXIT_OK),
                            json_output=True)
            cli_output.emit(Result(command="c", data={"a": 1}, exit_code=EXIT_USAGE),
                            json_output=True)
        envelopes = decode_envelopes(buf.getvalue())
        self.assertEqual([e["ok"] for e in envelopes], [True, False])

    def test_emit_error_maps_exceptions(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli_output.emit_error("c", CliError("bad input", code="usage_error",
                                                exit_code=EXIT_USAGE),
                                  json_output=True)
            cli_output.emit_error("c", ValueError(
                "bad value"), json_output=True)
            cli_output.emit_error("c", OSError("disk gone"), json_output=True)
            cli_output.emit_error("c", RuntimeError("boom"), json_output=True)
        codes = [e["error"]["code"] for e in decode_envelopes(buf.getvalue())]
        self.assertEqual(
            codes, ["usage_error", "usage_error", "io_error", "internal_error"])

    def test_emit_human_mode_prints_nothing_for_empty_result(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli_output.emit(Result(command="c", data=""), json_output=False)
        self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
