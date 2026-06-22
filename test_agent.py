from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import tempfile
import textwrap
import time
import unittest

from fastapi.testclient import TestClient
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pyagent.agent import Agent
from pyagent.config import AppConfig
from pyagent.extensions.bus import Ctx, EventBus, NoOpLog
from pyagent.extensions.loader import (
    MAX_EXTENSION_SKILLS_CHARS,
    _discover,
    collect_skill_text,
    load_all,
    load_one,
    unload_one,
)
from pyagent.session_logger import SessionLogger
from pyagent.external_tools import (
    DEFAULT_DESCRIBE_TIMEOUT_SECONDS,
    DEFAULT_INVOKE_TIMEOUT_SECONDS,
    ExternalToolHandler,
    build_external_tool_specs,
    discover_external_tools,
)
from pyagent.llm_client import OpenAICompatibleClient, OllamaClient, build_chat_client
from pyagent.model_profiles import (
    ModelProfile,
    load_profile_store,
    save_profile_store,
    update_profile_store,
)
from pyagent.project_context import (
    GLOBAL_SCOPE,
    PROJECT_SCOPE,
    discover_project_instruction_files,
    discover_project_skill_files,
    discover_user_global_instruction_files,
    load_full_context,
    load_project_context,
    resolve_user_skill,
)
from pyagent.scaffold import ScaffoldError, create_user_tool
from pyagent.tools import (
    BUILTIN_ORIGIN,
    EXTERNAL_ORIGIN,
    ToolSpec,
    append_file,
    bash,
    create_default_tool_registry,
    edit_file,
    find_files,
    list_files,
    list_skills,
    load_skills,
    search_text,
)
# ... existing imports ...
from pyagent.ui import ChatMessage, PyAgentApp
from pyagent.user_runtime import RunnerStatus

from pyagent.main import build_agent_for_request, main as main_entry, run_single_shot
from urllib import error

# ... (all existing test classes) ...


class ApiTests(unittest.TestCase):
    def test_health_endpoint_returns_ok(self) -> None:
        from pyagent.api import create_app

        client = TestClient(create_app())

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_run_endpoint_returns_agent_response_payload(self) -> None:
        from pyagent.api import create_app

        mock_agent = mock.Mock()
        mock_agent.run.return_value = [
            {"type": "content_delta", "delta": "Hello "},
            {"type": "assistant_done", "content": "Hello World!"},
        ]
        mock_agent.current_profile.return_value = SimpleNamespace(
            name="p1", provider="ollama", model="m1"
        )
        mock_agent.project_context_files = ["AGENTS.md", "skills/testing.md"]
        mock_agent.messages = [
            {"role": "system", "content": "ctx"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello World!"},
        ]
        prior_messages = [{"role": "user", "content": "Earlier"}]

        client = TestClient(create_app())
        with mock.patch("pyagent.main.build_agent_for_request", return_value=mock_agent) as mock_build:
            response = client.post(
                "/run",
                json={
                    "message": "Hi",
                    "messages": prior_messages,
                    "profile": "p1",
                    "model": "m1",
                    "cwd": "/tmp",
                    "skills": ["foo.md"],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "response": "Hello World!",
                "profile": "p1",
                "provider": "ollama",
                "model": "m1",
                "messages": mock_agent.messages,
                "context_files": ["AGENTS.md", "skills/testing.md"],
            },
        )
        mock_build.assert_called_once_with(
            profile="p1", model="m1", cwd="/tmp", skills=["foo.md"]
        )
        mock_agent.load_messages.assert_called_once_with(prior_messages)
        mock_agent.run.assert_called_once_with("Hi")

    def test_run_endpoint_rejects_invalid_request_body(self) -> None:
        from pyagent.api import create_app

        client = TestClient(create_app())

        response = client.post("/run", json={"message": ""})

        self.assertEqual(response.status_code, 422)

    def test_run_endpoint_returns_400_for_bad_request_configuration(self) -> None:
        from pyagent.api import create_app

        client = TestClient(create_app())
        with mock.patch(
            "pyagent.main.build_agent_for_request",
            side_effect=ValueError("Unknown skill: nope.md"),
        ):
            response = client.post(
                "/run",
                json={"message": "Hi", "skills": ["nope.md"]},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "Unknown skill: nope.md"})

    def test_run_endpoint_returns_400_for_invalid_messages(self) -> None:
        from pyagent.api import create_app

        mock_agent = mock.Mock()
        mock_agent.load_messages.side_effect = ValueError(
            "each message must include a non-empty string role"
        )

        client = TestClient(create_app())
        with mock.patch("pyagent.main.build_agent_for_request", return_value=mock_agent):
            response = client.post(
                "/run",
                json={"message": "Hi", "messages": [{"content": "oops"}]},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json(),
            {"detail": "each message must include a non-empty string role"},
        )


class PyAgentClientTests(unittest.TestCase):
    def test_run_includes_messages_payload(self) -> None:
        from pyagent.client import PyAgentClient

        client = PyAgentClient()
        response_payload = {
            "response": "Hello World!",
            "profile": "p1",
            "provider": "ollama",
            "model": "m1",
            "messages": [{"role": "assistant", "content": "Hello World!"}],
            "context_files": ["AGENTS.md"],
        }

        with mock.patch.object(client, "_request_json", return_value=response_payload) as mock_request:
            result = client.run(
                "Hi",
                messages=[{"role": "user", "content": "Earlier"}],
                profile="p1",
                model="m1",
                cwd="/tmp",
                skills=["foo.md"],
            )

        self.assertEqual(result.response, "Hello World!")
        mock_request.assert_called_once_with(
            "POST",
            "/run",
            {
                "message": "Hi",
                "messages": [{"role": "user", "content": "Earlier"}],
                "profile": "p1",
                "model": "m1",
                "cwd": "/tmp",
                "skills": ["foo.md"],
            },
        )


class AgentMessageLoadingTests(unittest.TestCase):
    def test_load_messages_ignores_system_messages_and_preserves_current_system(self) -> None:
        agent = Agent()
        original_system = agent.messages[0]["content"]

        agent.load_messages(
            [
                {"role": "system", "content": "ignore me"},
                {"role": "user", "content": "Earlier"},
                {"role": "assistant", "content": "Sure"},
            ]
        )

        self.assertEqual(agent.messages[0]["role"], "system")
        self.assertEqual(agent.messages[0]["content"], original_system)
        self.assertEqual(
            agent.messages[1:],
            [
                {"role": "user", "content": "Earlier"},
                {"role": "assistant", "content": "Sure"},
            ],
        )

    def test_load_messages_preserves_tool_metadata(self) -> None:
        agent = Agent()

        agent.load_messages(
            [
                {
                    "role": "assistant",
                    "content": "Running tool",
                    "tool_calls": [{"id": "call-1", "type": "function"}],
                },
                {
                    "role": "tool",
                    "content": "done",
                    "tool_call_id": "call-1",
                    "name": "list_files",
                },
            ]
        )

        self.assertEqual(agent.messages[1]["tool_calls"], [
                         {"id": "call-1", "type": "function"}])
        self.assertEqual(agent.messages[2]["tool_call_id"], "call-1")
        self.assertEqual(agent.messages[2]["name"], "list_files")

    def test_load_messages_rejects_invalid_role(self) -> None:
        agent = Agent()

        with self.assertRaisesRegex(ValueError, "non-empty string role"):
            agent.load_messages([{"content": "oops"}])


class MainCliTests(unittest.TestCase):
    def test_single_shot_mode_prints_response_and_exits(self) -> None:
        # Mock the Agent to avoid real LLM calls
        with mock.patch("pyagent.main.Agent") as MockAgent, mock.patch("pyagent.main.load_full_context", return_value=("ctx", [])):
            mock_agent_instance = MockAgent.return_value
            # Simulate the generator returned by agent.run()
            mock_agent_instance.run.return_value = [
                {"type": "content_delta", "delta": "Hello "},
                {"type": "content_delta", "delta": "World!"},
                {"type": "assistant_done", "content": "Hello World!"},
            ]

            # Mock sys.argv to simulate: pyagent --prompt "Hi"
            with mock.patch("sys.argv", ["pyagent", "--prompt", "Hi"]):
                with mock.patch("sys.stdout") as mock_stdout:
                    # We expect sys.exit(0) to be called
                    with self.assertRaises(SystemExit) as cm:
                        main_entry()

                    self.assertEqual(cm.exception.code, 0)
                    # The current implementation in main.py only prints the final content
                    # from the assistant_done event.
                    mock_stdout.write.assert_called_with("Hello World!")

    def test_single_shot_mode_passes_profile_model_and_loaded_context(self) -> None:
        context_sources = [SimpleNamespace(label="~/.pyagent/AGENTS.md")]
        with mock.patch("pyagent.main.Agent") as MockAgent, mock.patch(
            "pyagent.main.load_full_context",
            return_value=("ctx", context_sources),
        ) as mock_load_context:
            mock_agent_instance = MockAgent.return_value
            mock_agent_instance.run.return_value = [
                {"type": "assistant_done", "content": "Done"}
            ]

            # Simulate: pyagent --profile my-profile --model my-model --prompt "Hi"
            with mock.patch("sys.argv", ["pyagent", "--profile", "my-profile", "--model", "my-model", "--prompt", "Hi"]):
                with mock.patch("sys.stdout"):
                    with self.assertRaises(SystemExit):
                        main_entry()

                    mock_load_context.assert_called_once()
                    # Verify Agent was instantiated with the correct overrides
                    MockAgent.assert_called_once_with(
                        profile="my-profile",
                        model="my-model",
                        project_context="ctx",
                        project_context_files=["~/.pyagent/AGENTS.md"],
                    )

    def test_single_shot_mode_loads_requested_skills(self) -> None:
        context_sources = [SimpleNamespace(label="~/.pyagent/skills/foo.md")]
        with mock.patch("pyagent.main.Agent") as MockAgent, mock.patch(
            "pyagent.main.resolve_available_skill",
            side_effect=[SimpleNamespace(id="user:foo.md"), SimpleNamespace(
                id="user:folder/bar.skill")],
        ) as mock_resolve_skill, mock.patch(
            "pyagent.main.load_full_context",
            return_value=("ctx", context_sources),
        ) as mock_load_context:
            mock_agent_instance = MockAgent.return_value
            mock_agent_instance.run.return_value = [
                {"type": "assistant_done", "content": "Done"}
            ]

            with mock.patch(
                "sys.argv",
                ["pyagent", "--skills", "foo.md,folder/bar.skill", "--prompt", "Hi"],
            ):
                with mock.patch("sys.stdout"):
                    with self.assertRaises(SystemExit) as cm:
                        main_entry()

            self.assertEqual(cm.exception.code, 0)
            self.assertEqual(mock_resolve_skill.call_count, 2)
            mock_load_context.assert_called_once_with(
                mock.ANY,
                loaded_user_skills=["user:foo.md", "user:folder/bar.skill"],
            )

    def test_single_shot_mode_rejects_unknown_skills(self) -> None:
        with mock.patch("pyagent.main.resolve_available_skill", return_value=None), mock.patch("sys.argv", ["pyagent", "--skills", "missing.md", "--prompt", "Hi"]):
            with mock.patch("sys.stderr") as mock_stderr:
                with self.assertRaises(SystemExit) as cm:
                    main_entry()

        self.assertEqual(cm.exception.code, 2)
        mock_stderr.write.assert_any_call(
            "Unknown skill(s): missing.md\nSkills must be specified as scoped IDs (user:<path> or project:<path>) or as existing user-skill paths relative to ~/.pyagent/skills/.\n"
        )

    def test_skills_flag_without_prompt_exits_with_error(self) -> None:
        with mock.patch("sys.argv", ["pyagent", "--skills", "foo.md"]):
            with mock.patch("sys.stderr") as mock_stderr:
                with self.assertRaises(SystemExit) as cm:
                    main_entry()

        self.assertEqual(cm.exception.code, 2)
        mock_stderr.write.assert_called_with(
            "--skills is currently supported only with --prompt\n"
        )

    def test_interactive_mode_launches_app(self) -> None:
        fake_ui_module = SimpleNamespace(PyAgentApp=mock.Mock())
        mock_app_instance = fake_ui_module.PyAgentApp.return_value

        with mock.patch.dict(sys.modules, {"pyagent.ui": fake_ui_module}, clear=False):
            with mock.patch("sys.argv", ["pyagent"]):
                main_entry()

        fake_ui_module.PyAgentApp.assert_called_once()
        mock_app_instance.run.assert_called_once()

    def test_run_single_shot_builds_agent_and_returns_final_content(self) -> None:
        mock_agent = mock.Mock()
        mock_agent.run.return_value = [
            {"type": "content_delta", "delta": "Hello "},
            {"type": "assistant_done", "content": "Hello World!"},
        ]
        with mock.patch("pyagent.main.build_agent_for_request", return_value=mock_agent) as mock_build:
            response = run_single_shot(
                prompt="Hi", profile="p1", model="m1", skills=["foo.md"])

        self.assertEqual(response, "Hello World!")
        mock_build.assert_called_once_with(
            profile="p1", model="m1", skills=["foo.md"])

    def test_build_agent_for_request_loads_context_and_passes_labels(self) -> None:
        context_sources = [SimpleNamespace(label="~/.pyagent/AGENTS.md")]
        with mock.patch("pyagent.main.resolve_available_skill", return_value=SimpleNamespace(id="user:foo.md")), mock.patch(
            "pyagent.main.load_full_context",
            return_value=("ctx", context_sources),
        ) as mock_load_context, mock.patch("pyagent.main.Agent") as MockAgent:
            build_agent_for_request(
                profile="prof", model="mod", cwd="/tmp", skills=["foo.md"])

        mock_load_context.assert_called_once_with(
            "/tmp", loaded_user_skills=["user:foo.md"])
        MockAgent.assert_called_once_with(
            profile="prof",
            model="mod",
            project_context="ctx",
            project_context_files=["~/.pyagent/AGENTS.md"],
        )

    def test_serve_subcommand_runs_uvicorn(self) -> None:
        fake_uvicorn = SimpleNamespace(run=mock.Mock())
        fake_api_module = SimpleNamespace(
            create_app=mock.Mock(return_value="app-instance"))
        with mock.patch.dict(sys.modules, {"uvicorn": fake_uvicorn, "pyagent.api": fake_api_module}, clear=False):
            main_entry(["serve", "--host", "0.0.0.0", "--port", "9000"])

        fake_api_module.create_app.assert_called_once_with()
        fake_uvicorn.run.assert_called_once_with(
            "app-instance", host="0.0.0.0", port=9000)

    def test_serve_subcommand_exits_when_dependencies_missing(self) -> None:
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "uvicorn":
                raise ImportError("No module named 'uvicorn'")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=fake_import), mock.patch("sys.stderr") as mock_stderr:
            with self.assertRaises(SystemExit) as cm:
                main_entry(["serve"])

        self.assertEqual(cm.exception.code, 2)
        mock_stderr.write.assert_called()

    def test_web_subcommand_runs_textual_serve_server(self) -> None:
        fake_server_instance = mock.Mock()
        fake_server_class = mock.Mock(return_value=fake_server_instance)
        fake_textual_serve_module = SimpleNamespace(Server=fake_server_class)

        with mock.patch.dict(
            sys.modules,
            {"textual_serve.server": fake_textual_serve_module},
            clear=False,
        ):
            main_entry(
                [
                    "web",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "9001",
                    "--profile",
                    "local-qwen",
                    "--model",
                    "qwen2.5-coder:7b",
                ]
            )

        fake_server_class.assert_called_once_with(
            command="python -m pyagent --profile local-qwen --model qwen2.5-coder:7b",
            host="0.0.0.0",
            port=9001,
            title="PyAgent",
        )
        fake_server_instance.serve.assert_called_once_with()

    def test_web_subcommand_exits_when_dependencies_missing(self) -> None:
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "textual_serve.server":
                raise ImportError("No module named 'textual_serve'")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=fake_import), mock.patch("sys.stderr") as mock_stderr:
            with self.assertRaises(SystemExit) as cm:
                main_entry(["web"])

        self.assertEqual(cm.exception.code, 2)
        mock_stderr.write.assert_called()

# ... (remaining file) ...


class DummyClient:
    def __init__(self, responses: list[list[dict]]):
        self.responses = responses
        self.calls = 0
        self.seen_tools: list[object] = []

    def chat_stream(self, messages, tools=None):
        self.seen_tools.append(tools)
        response = self.responses[self.calls]
        self.calls += 1
        for chunk in response:
            yield chunk


class ModelListClient:
    def __init__(self, payload: dict[str, object]):
        self.payload = payload

    def list_models(self):
        return self.payload


class FakeOpenAICompletions:
    def __init__(self, stream=None):
        self.stream = stream or []
        self.last_create = None

    def create(self, **kwargs):
        self.last_create = kwargs
        return self.stream


class FakeOpenAIModels:
    def __init__(self, response=None):
        self.response = response or SimpleNamespace(data=[])

    def list(self):
        return self.response


class FakeOpenAIClient:
    def __init__(self, stream=None, models_response=None):
        self.chat = SimpleNamespace(
            completions=FakeOpenAICompletions(stream=stream))
        self.models = FakeOpenAIModels(response=models_response)
        self.closed = False

    def close(self):
        self.closed = True


def make_chunk(content=None, tool_calls=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content, tool_calls=tool_calls or []),
                finish_reason=None,
                index=0,
            )
        ]
    )


def make_tool_call_delta(index: int, id: str | None = None, name: str | None = None, arguments: str | None = None):
    return SimpleNamespace(
        index=index,
        id=id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class PyAgentUiContextTests(unittest.TestCase):
    def test_context_status_reports_default_sources_and_loaded_skills(self) -> None:
        app = PyAgentApp()
        app.project_context = "global and project context"
        app.context_sources = [
            SimpleNamespace(scope=GLOBAL_SCOPE, label="~/.pyagent/AGENTS.md"),
            SimpleNamespace(scope=GLOBAL_SCOPE,
                            label="~/.pyagent/skills/manual.md"),
            SimpleNamespace(scope=PROJECT_SCOPE, label="AGENTS.md"),
            SimpleNamespace(scope=PROJECT_SCOPE, label="skills/project.md"),
        ]
        app.project_context_files = [
            source.label for source in app.context_sources]
        app.agent.project_context = app.project_context
        app.agent.project_context_files = app.project_context_files
        app.loaded_user_skills = ["user:manual.md", "project:skills/project.md"]

        status = app._context_status_text()

        self.assertIn("Loaded instruction context:", status)
        self.assertIn("- User-global default sources:", status)
        self.assertIn("  - `~/.pyagent/AGENTS.md`", status)
        self.assertIn("- Project default sources:", status)
        self.assertIn("  - `AGENTS.md`", status)
        self.assertIn("- Skills loaded into system prompt this session: `2`", status)
        self.assertIn("  - `user:manual.md`", status)
        self.assertIn("  - `project:skills/project.md`", status)
        self.assertIn("- Loaded user skill sources:", status)
        self.assertIn("  - `~/.pyagent/skills/manual.md`", status)
        self.assertIn("- Loaded project skill sources:", status)
        self.assertIn("  - `skills/project.md`", status)


class AgentTests(unittest.TestCase):
    def test_tools_disabled_omits_tool_definitions_and_updates_system_prompt(self) -> None:
        config = AppConfig(max_iterations=1, tools_enabled=False)
        agent = Agent(
            config=config, tool_registry=create_default_tool_registry(config))
        agent.client = DummyClient([[{"content": "Plain answer"}]])

        events = list(agent.run("Answer without tools"))

        self.assertEqual(agent.tools, [])
        self.assertIn("Tool calling is disabled", agent.messages[0]["content"])
        self.assertEqual(agent.client.seen_tools, [None])
        self.assertEqual(
            events[-1], {"type": "assistant_done", "content": "Plain answer"})

    def test_builtin_tools_disabled_omits_builtins_but_keeps_tool_calling_enabled(self) -> None:
        config = AppConfig(max_iterations=1, builtin_tools_enabled=False)
        external = ToolSpec(
            name="echo_tool",
            description="Echo tool",
            parameters={"type": "object", "properties": {}},
            handler=lambda **_: "external",
        )
        agent = Agent(
            config=config,
            tool_registry=create_default_tool_registry(
                config, external_specs=[external]),
        )
        agent.client = DummyClient([[{"content": "Plain answer"}]])

        events = list(agent.run("Answer with external tools only"))

        self.assertEqual(
            agent.tool_registry.names_by_origin(BUILTIN_ORIGIN), [])
        self.assertEqual(
            agent.tool_registry.names_by_origin(EXTERNAL_ORIGIN), ["echo_tool"])
        advertised_names = [
            tool["function"]["name"] for tool in agent.client.seen_tools[0]
        ]
        self.assertEqual(advertised_names, ["echo_tool"])
        self.assertIn("Built-in tools are disabled",
                      agent.messages[0]["content"])
        self.assertNotIn("Tool calling is disabled",
                         agent.messages[0]["content"])
        self.assertEqual(
            events[-1], {"type": "assistant_done", "content": "Plain answer"})

    def test_appends_tool_result_when_model_stops_after_intro(self) -> None:
        config = AppConfig(max_iterations=3)
        agent = Agent(
            config=config, tool_registry=create_default_tool_registry(config))
        agent.client = DummyClient(
            [
                [
                    {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "list_files",
                                    "arguments": {"path": ".", "max_depth": 1},
                                },
                            }
                        ]
                    }
                ],
                [
                    {
                        "content": "The files and directories in the current working directory are:",
                    }
                ],
            ]
        )

        events = list(
            agent.run("What files are available in the current working directory?"))
        assistant_deltas = "".join(
            event.get("delta", "") for event in events if event.get("type") == "content_delta"
        )

        self.assertIn(
            "The files and directories in the current working directory are:",
            assistant_deltas,
        )
        self.assertIn("```text", assistant_deltas)
        self.assertIn("README.md", assistant_deltas)

    def test_negative_max_iterations_allows_unbounded_tool_loop(self) -> None:
        config = AppConfig(max_iterations=-1)
        agent = Agent(
            config=config, tool_registry=create_default_tool_registry(config))
        agent.client = DummyClient(
            [
                [
                    {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "list_files",
                                    "arguments": {"path": ".", "max_depth": 1},
                                },
                            }
                        ]
                    }
                ],
                [{"content": "Finished"}],
            ]
        )

        events = list(agent.run("Inspect the repository"))

        self.assertEqual(agent.client.calls, 2)
        self.assertEqual(
            events[-1], {"type": "assistant_done", "content": "Finished"})

    def test_trim_history_drops_orphaned_tool_messages(self) -> None:
        config = AppConfig(max_history_messages=3)
        agent = Agent(
            config=config, tool_registry=create_default_tool_registry(config))
        agent.messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "older question"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "search_text",
                            "arguments": {"query": "PyAgent"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "search_text",
                "content": "result",
                "tool_call_id": "call_1",
            },
            {"role": "assistant", "content": "older answer"},
            {"role": "user", "content": "latest question"},
        ]

        agent._trim_history()

        self.assertEqual(
            [message["role"] for message in agent.messages],
            ["system", "assistant", "user"],
        )
        self.assertEqual(agent.messages[-1]["content"], "latest question")

    def test_set_model_updates_agent_and_client(self) -> None:
        agent = Agent(config=AppConfig(),
                      tool_registry=create_default_tool_registry(AppConfig()))

        agent.set_model("new-model")

        self.assertEqual(agent.current_profile().model, "new-model")
        self.assertEqual(agent.client.model, "new-model")

    def test_set_profile_rebuilds_client(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = os.path.join(temp_dir, "profiles.json")
            with open(profile_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "default_profile": "local",
                        "profiles": {
                            "local": {
                                "provider": "ollama",
                                "model": "qwen2.5-coder:7b",
                                "base_url": "http://localhost:11434",
                            },
                            "remote": {
                                "provider": "openai_compatible",
                                "model": "gpt-4.1-mini",
                                "base_url": "https://example.com/v1",
                                "api_key": "secret",
                            },
                        },
                    },
                    file,
                )

            agent = Agent(
                config=AppConfig(model_profiles_path=profile_path),
                tool_registry=create_default_tool_registry(
                    AppConfig(model_profiles_path=profile_path)),
            )
            agent.set_profile("remote")

        self.assertEqual(agent.current_profile().name, "remote")
        self.assertEqual(agent.current_profile().provider, "openai_compatible")
        self.assertEqual(agent.client.model, "gpt-4.1-mini")

    def test_available_models_uses_normalized_client_response(self) -> None:
        agent = Agent(config=AppConfig(),
                      tool_registry=create_default_tool_registry(AppConfig()))
        agent.client = ModelListClient(
            {"models": ["llama3.1", "qwen2.5-coder:7b"]})

        names, error = agent.available_models()

        self.assertIsNone(error)
        self.assertEqual(names, ["llama3.1", "qwen2.5-coder:7b"])

    def test_reload_profiles_refreshes_store_from_disk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = os.path.join(temp_dir, "profiles.json")
            with open(profile_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "default_profile": "local",
                        "profiles": {
                            "local": {
                                "provider": "ollama",
                                "model": "qwen2.5-coder:7b",
                                "base_url": "http://localhost:11434",
                            }
                        },
                    },
                    file,
                )

            config = AppConfig(model_profiles_path=profile_path)
            agent = Agent(
                config=config, tool_registry=create_default_tool_registry(config))

            with open(profile_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "default_profile": "remote",
                        "profiles": {
                            "remote": {
                                "provider": "openai_compatible",
                                "model": "gpt-4.1-mini",
                                "base_url": "https://example.com/v1",
                                "api_key": "secret",
                            }
                        },
                    },
                    file,
                )

            agent.reload_profiles()

        self.assertEqual(agent.current_profile().name, "remote")
        self.assertEqual(agent.current_profile().model, "gpt-4.1-mini")


class UiCommandTests(unittest.TestCase):
    def test_chat_message_detects_long_lines_in_fenced_code_blocks(self) -> None:
        message = ChatMessage(
            "assistant",
            "Here is code:\n\n```python\n" + ("x" * 200) + "\n```",
            finalized=True,
        )

        self.assertTrue(message._has_long_code_block_line())

    def test_unknown_command_suggests_close_match(self) -> None:
        app = PyAgentApp()

        message = app._unknown_command_message("/stats")

        self.assertIn("Did you mean `/status`?", message)

    def test_context_status_text_reports_loaded_files(self) -> None:
        app = PyAgentApp()
        app.agent.project_context = "alpha\nbeta"
        app.context_sources = [
            SimpleNamespace(scope=PROJECT_SCOPE, label="AGENTS.md"),
            SimpleNamespace(scope=PROJECT_SCOPE, label="skills/testing.md"),
        ]
        app.project_context_files = [source.label for source in app.context_sources]
        app.agent.project_context_files = app.project_context_files

        text = app._context_status_text()

        self.assertIn("Files loaded: `2`", text)
        self.assertIn("AGENTS.md", text)
        self.assertIn("skills/testing.md", text)

    def test_help_text_mentions_history_search_context_and_prompt_keys(self) -> None:
        app = PyAgentApp()

        text = app._command_help_text()

        self.assertIn("`/context`", text)
        self.assertIn("`/history search <text>`", text)
        self.assertIn("`/max_iterations <n|-1>`", text)
        self.assertIn("`/tools on|off`", text)
        self.assertIn("`Enter`", text)
        self.assertIn("`Shift+Enter`", text)
        self.assertIn("`Ctrl+P` / `Ctrl+N`", text)

    def test_tools_command_toggles_runtime_tools_and_resets_conversation(self) -> None:
        app = PyAgentApp()
        notes: list[str] = []
        statuses: list[str] = []
        app._add_system_note = notes.append
        app._set_status = statuses.append
        app.agent.add_message("user", "hello")

        handled = app._handle_slash_command("/tools off")

        self.assertTrue(handled)
        self.assertFalse(app.agent.config.tools_enabled)
        self.assertEqual(app.agent.tools, [])
        self.assertEqual(len(app.agent.messages), 1)
        self.assertIn("Tool calling is disabled",
                      app.agent.messages[0]["content"])
        self.assertIn("Tools disabled for this session", notes[-1])
        self.assertTrue(statuses)

        handled = app._handle_slash_command("/tools on")

        self.assertTrue(handled)
        self.assertTrue(app.agent.config.tools_enabled)
        self.assertGreater(len(app.agent.tools), 0)
        self.assertEqual(len(app.agent.messages), 1)
        self.assertNotIn("Tool calling is disabled",
                         app.agent.messages[0]["content"])
        self.assertIn("Tools enabled for this session", notes[-1])

    def test_status_command_reports_agent_tool_loop_max_iterations(self) -> None:
        app = PyAgentApp()
        app.agent.config.max_iterations = 7
        notes: list[str] = []
        app._add_system_note = notes.append

        handled = app._handle_slash_command("/status")

        self.assertTrue(handled)
        self.assertIn("Agent tool-loop max iterations: `7`", notes[-1])

    def test_max_iterations_command_updates_runtime_limit(self) -> None:
        app = PyAgentApp()
        notes: list[str] = []
        statuses: list[str] = []
        app._add_system_note = notes.append
        app._set_status = statuses.append

        handled = app._handle_slash_command("/max_iterations 20")

        self.assertTrue(handled)
        self.assertEqual(app.agent.config.max_iterations, 20)
        self.assertIn("Set agent tool-loop max iterations to `20`", notes[-1])
        self.assertTrue(statuses)

    def test_max_iterations_command_accepts_infinite(self) -> None:
        app = PyAgentApp()
        notes: list[str] = []
        app._add_system_note = notes.append
        app._set_status = lambda _text: None

        handled = app._handle_slash_command("/max_iterations -1")

        self.assertTrue(handled)
        self.assertEqual(app.agent.config.max_iterations, -1)
        self.assertIn("`-1` (infinite)", notes[-1])

    def test_max_iterations_command_rejects_invalid_values(self) -> None:
        app = PyAgentApp()
        notes: list[str] = []
        app._add_system_note = notes.append

        handled = app._handle_slash_command("/max_iterations 0")

        self.assertTrue(handled)
        self.assertIn("positive integer or `-1` for infinite", notes[-1])

    def test_skills_load_and_unload_resolve_nested_skill_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_path = Path(temp_dir) / "skills" / "python" / "lint.md"
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            skill_path.write_text("# lint\n", encoding="utf-8")

            app = PyAgentApp()
            app.agent.config.user_dir = temp_dir
            notes: list[str] = []
            statuses: list[str] = []
            app._add_system_note = notes.append
            app._set_status = statuses.append
            app._reload_context_details = lambda: "Reloaded context."

            self.assertIsNone(resolve_user_skill("python/lint", temp_dir))
            self.assertEqual(resolve_user_skill(
                "python/lint.md", temp_dir).label, "python/lint.md")

            handled = app._handle_slash_command("/skills load python/lint.md")

            self.assertTrue(handled)
            self.assertIn("user:python/lint.md", app.loaded_user_skills)
            self.assertIn("Loaded skill `user:python/lint.md`", notes[-1])
            self.assertTrue(statuses)

            handled = app._handle_slash_command(
                "/skills unload python/lint.md")

            self.assertTrue(handled)
            self.assertNotIn("user:python/lint.md", app.loaded_user_skills)
            self.assertIn("Unloaded skill `user:python/lint.md`", notes[-1])


class ConfigTests(unittest.TestCase):
    def test_from_env_reads_tools_enabled_flag(self) -> None:
        previous = os.environ.get("PYAGENT_TOOLS_ENABLED")
        try:
            os.environ["PYAGENT_TOOLS_ENABLED"] = "false"
            config = AppConfig.from_env()
        finally:
            if previous is None:
                os.environ.pop("PYAGENT_TOOLS_ENABLED", None)
            else:
                os.environ["PYAGENT_TOOLS_ENABLED"] = previous

        self.assertFalse(config.tools_enabled)

    def test_from_env_reads_builtin_and_user_tools_enabled_flags(self) -> None:
        previous_builtin = os.environ.get("PYAGENT_BUILTIN_TOOLS_ENABLED")
        previous_user = os.environ.get("PYAGENT_USER_TOOLS_ENABLED")
        try:
            os.environ["PYAGENT_BUILTIN_TOOLS_ENABLED"] = "false"
            os.environ["PYAGENT_USER_TOOLS_ENABLED"] = "false"
            config = AppConfig.from_env()
        finally:
            if previous_builtin is None:
                os.environ.pop("PYAGENT_BUILTIN_TOOLS_ENABLED", None)
            else:
                os.environ["PYAGENT_BUILTIN_TOOLS_ENABLED"] = previous_builtin
            if previous_user is None:
                os.environ.pop("PYAGENT_USER_TOOLS_ENABLED", None)
            else:
                os.environ["PYAGENT_USER_TOOLS_ENABLED"] = previous_user

        self.assertFalse(config.builtin_tools_enabled)
        self.assertFalse(config.user_tools_enabled)


class ClientTests(unittest.TestCase):
    def test_build_chat_client_chooses_provider_implementation(self) -> None:
        ollama_profile = ModelProfile(
            name="local",
            provider="ollama",
            model="llama3.1",
            base_url="http://localhost:11434",
        )
        openai_profile = ModelProfile(
            name="remote",
            provider="openai_compatible",
            model="gpt-4.1-mini",
            base_url="https://example.com/v1",
            api_key="secret",
        )

        self.assertIsInstance(build_chat_client(ollama_profile), OllamaClient)
        self.assertIsInstance(build_chat_client(
            openai_profile), OpenAICompatibleClient)

    def test_openai_compatible_stream_assembles_tool_calls(self) -> None:
        profile = ModelProfile(
            name="remote",
            provider="openai_compatible",
            model="gpt-4.1-mini",
            base_url="https://example.com/v1",
            api_key="secret",
        )
        client = OpenAICompatibleClient(profile=profile)
        fake_sdk_client = FakeOpenAIClient(
            stream=[
                make_chunk(content="Hello"),
                make_chunk(
                    tool_calls=[
                        make_tool_call_delta(
                            index=0,
                            id="call_1",
                            name="search_text",
                            arguments='{"query":',
                        )
                    ]
                ),
                make_chunk(
                    tool_calls=[
                        make_tool_call_delta(
                            index=0,
                            arguments='"PyAgent"}',
                        )
                    ]
                ),
            ]
        )
        client._client_factory = lambda **kwargs: fake_sdk_client

        chunks = list(client.chat_stream(messages=[], tools=[]))

        self.assertEqual(chunks[0], {"content": "Hello"})
        self.assertEqual(chunks[1]["tool_calls"][0]
                         ["function"]["name"], "search_text")
        self.assertEqual(
            chunks[1]["tool_calls"][0]["function"]["arguments"],
            '{"query":"PyAgent"}',
        )
        self.assertEqual(
            fake_sdk_client.chat.completions.last_create["model"], "gpt-4.1-mini")
        self.assertEqual(
            fake_sdk_client.chat.completions.last_create["messages"], [])

    def test_openai_compatible_list_models_parses_response(self) -> None:
        profile = ModelProfile(
            name="remote",
            provider="openai_compatible",
            model="gpt-4.1-mini",
            base_url="https://example.com/v1",
            api_key="secret",
        )
        client = OpenAICompatibleClient(profile=profile)
        client._client_factory = lambda **kwargs: FakeOpenAIClient(
            models_response=SimpleNamespace(
                data=[SimpleNamespace(id="gpt-4.1"),
                      SimpleNamespace(id="local-model")]
            )
        )

        payload = client.list_models()

        self.assertEqual(payload, {"models": ["gpt-4.1", "local-model"]})

    def test_openai_compatible_client_uses_empty_api_key_for_no_auth_servers(self) -> None:
        profile = ModelProfile(
            name="remote",
            provider="openai_compatible",
            model="gpt-4.1-mini",
            base_url="http://localhost:1234/v1",
        )
        client = OpenAICompatibleClient(profile=profile)
        captured: dict[str, object] = {}

        def factory(**kwargs):
            captured.update(kwargs)
            return FakeOpenAIClient(models_response=SimpleNamespace(data=[]))

        client._client_factory = factory
        payload = client.list_models()

        self.assertEqual(payload, {"models": []})
        self.assertEqual(captured["api_key"], "")
        self.assertEqual(captured["base_url"], "http://localhost:1234/v1")

    def test_openai_compatible_prepare_messages_skips_invalid_tool_sequences(self) -> None:
        profile = ModelProfile(
            name="remote",
            provider="openai_compatible",
            model="gpt-4.1-mini",
            base_url="https://example.com/v1",
            api_key="secret",
        )
        client = OpenAICompatibleClient(profile=profile)

        prepared = client._prepare_messages(
            [
                {"role": "system", "content": "system"},
                {
                    "role": "tool",
                    "name": "search_text",
                    "content": "orphaned",
                    "tool_call_id": "call_orphaned",
                },
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "search_text", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "search_text",
                    "content": "kept",
                    "tool_call_id": "call_1",
                },
            ]
        )

        self.assertEqual([message["role"] for message in prepared], [
                         "system", "user", "assistant", "tool"])
        self.assertEqual(prepared[-1]["content"], "kept")

    def test_rebuild_client_closes_previous_client(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = os.path.join(temp_dir, "profiles.json")
            with open(profile_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "default_profile": "local",
                        "profiles": {
                            "local": {
                                "provider": "ollama",
                                "model": "qwen2.5-coder:7b",
                                "base_url": "http://localhost:11434",
                            }
                        },
                    },
                    file,
                )

            agent = Agent(
                config=AppConfig(model_profiles_path=profile_path),
                tool_registry=create_default_tool_registry(
                    AppConfig(model_profiles_path=profile_path)),
            )
            old_client = FakeOpenAIClient()
            agent.client = old_client

            agent._rebuild_client()

        self.assertTrue(old_client.closed)


class ProfileStoreTests(unittest.TestCase):
    def test_load_profile_store_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "profiles.json")
            with open(path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "default_profile": "vllm-local",
                        "profiles": {
                            "vllm-local": {
                                "provider": "vllm",
                                "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
                                "base_url": "http://localhost:8000/v1",
                                "api_key_env": "VLLM_API_KEY",
                            }
                        },
                    },
                    file,
                )

            store = load_profile_store(path)

        self.assertEqual(store.default_profile, "vllm-local")
        self.assertEqual(store.get().provider, "openai_compatible")
        self.assertEqual(store.get().api_key_env, "VLLM_API_KEY")

    def test_update_and_save_profile_store_round_trips_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "profiles.json")
            store = load_profile_store(path)
            profile = ModelProfile(
                name="openai-test",
                provider="openai_compatible",
                model="gpt-4.1-mini",
                base_url="https://example.com/v1",
                api_key_env="OPENAI_API_KEY",
                headers={"X-Test": "1"},
            )

            update_profile_store(store, profile, make_default=True)
            save_profile_store(store)
            reloaded = load_profile_store(path)

        self.assertEqual(reloaded.default_profile, "openai-test")
        self.assertEqual(reloaded.get().provider, "openai_compatible")
        self.assertEqual(reloaded.get().headers, {"X-Test": "1"})


class ToolTests(unittest.TestCase):
    def test_list_files_does_not_duplicate_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            os.makedirs(os.path.join(temp_dir, "nested"), exist_ok=True)
            with open(os.path.join(temp_dir, "file.txt"), "w", encoding="utf-8") as file:
                file.write("hello")

            result = list_files(temp_dir, max_depth=2)

        self.assertEqual(result.count("nested/"), 1)
        self.assertIn("file.txt", result)

    def test_find_files_supports_substring_and_glob(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            open(os.path.join(temp_dir, "alpha.py"),
                 "w", encoding="utf-8").close()
            open(os.path.join(temp_dir, "beta.txt"),
                 "w", encoding="utf-8").close()

            substring_result = find_files("alpha", path=temp_dir)
            glob_result = find_files("*.txt", path=temp_dir)

        self.assertIn("alpha.py", substring_result)
        self.assertIn("beta.txt", glob_result)

    def test_search_text_returns_matching_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "notes.py"), "w", encoding="utf-8") as file:
                file.write("first line\nPyAgentApp lives here\nlast line\n")

            result = search_text("PyAgentApp", path=temp_dir, glob="*.py")

        self.assertIn("notes.py:2: PyAgentApp lives here", result)

    def test_list_and_load_skills_tools_use_scoped_ids_without_mutating_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            user_dir = root / "user"
            project_dir = root / "project"
            (user_dir / "skills" / ".private").mkdir(parents=True)
            project_dir.mkdir()
            (project_dir / "skills").mkdir()
            (user_dir / "skills" / "python.md").write_text(
                "# Python\nUse Python guidance.", encoding="utf-8"
            )
            (user_dir / "skills" / ".private" / "hidden.skill").write_text(
                "# Hidden\nHidden user skill.", encoding="utf-8"
            )
            (project_dir / "skills" / "review.md").write_text(
                "# Review\nProject review checklist.", encoding="utf-8"
            )
            config = AppConfig(user_dir=str(user_dir))

            listed = list_skills(config=config, cwd=str(project_dir))
            loaded = load_skills(
                skills=["user:python.md", "project:skills/review.md"],
                config=config,
                cwd=str(project_dir),
            )

        self.assertIn("user:python.md", listed)
        self.assertIn("user:.private/hidden.skill", listed)
        self.assertIn("project:skills/review.md", listed)
        self.assertIn("Use Python guidance.", loaded)
        self.assertIn("Project review checklist.", loaded)
        self.assertIn("system prompt", loaded)

    def test_edit_file_supports_multiple_replacements(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "sample.txt")
            with open(path, "w", encoding="utf-8") as file:
                file.write("hello world\nalpha beta\n")

            result = edit_file(
                path,
                edits=[
                    {"old_text": "hello", "new_text": "goodbye"},
                    {"old_text": "beta", "new_text": "gamma"},
                ],
            )

            with open(path, "r", encoding="utf-8") as file:
                updated = file.read()

        self.assertIn("Successfully edited", result)
        self.assertEqual(updated, "goodbye world\nalpha gamma\n")

    def test_append_file_appends_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "sample.txt")
            with open(path, "w", encoding="utf-8") as file:
                file.write("hello")

            append_file(path, " world")

            with open(path, "r", encoding="utf-8") as file:
                updated = file.read()

        self.assertEqual(updated, "hello world")

    def test_bash_blocks_dangerous_commands(self) -> None:
        config = AppConfig(bash_blocked_substrings=("rm -rf",))
        result = bash("rm -rf .", config=config)
        self.assertIn("blocked by shell safety policy", result)

    def test_bash_readonly_mode_allows_safe_commands_and_blocks_mutating_ones(self) -> None:
        config = AppConfig(
            bash_readonly_mode=True,
            bash_readonly_prefixes=("pwd", "ls"),
            bash_blocked_substrings=(),
        )
        allowed = bash("pwd", config=config)
        blocked = bash("echo hello", config=config)

        self.assertIn("exit_code: 0", allowed)
        self.assertIn("read-only mode", blocked)


class ProjectContextTests(unittest.TestCase):
    def test_discovers_agents_separately_from_project_skill_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = os.path.realpath(temp_dir)
            os.makedirs(os.path.join(
                base_path, "skills", "python"), exist_ok=True)
            with open(os.path.join(base_path, "AGENTS.md"), "w", encoding="utf-8") as file:
                file.write("project rules")
            with open(os.path.join(base_path, "skills", "python", "testing.md"), "w", encoding="utf-8") as file:
                file.write("testing skill")
            with open(os.path.join(base_path, "local.skill"), "w", encoding="utf-8") as file:
                file.write("local skill")

            instruction_files = [path.relative_to(base_path).as_posix()
                                 for path in discover_project_instruction_files(base_path)]
            skill_files = [path.relative_to(base_path).as_posix()
                           for path in discover_project_skill_files(base_path)]

        self.assertEqual(instruction_files, ["AGENTS.md"])
        self.assertEqual(skill_files, ["local.skill", "skills/python/testing.md"])

    def test_load_project_context_and_agent_reset_include_project_instructions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "AGENTS.md"), "w", encoding="utf-8") as file:
                file.write("Always run tests after edits.")

            context, files = load_project_context(temp_dir)
            agent = Agent(
                config=AppConfig(),
                tool_registry=create_default_tool_registry(AppConfig()),
                project_context=context,
                project_context_files=files,
            )

        self.assertEqual(files, ["AGENTS.md"])
        self.assertIn("Always run tests after edits.", context)
        self.assertIn("Always run tests after edits.",
                      agent.messages[0]["content"])

    def test_project_skills_are_not_loaded_by_default_but_can_be_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)
            (project_dir / "AGENTS.md").write_text("project rules", encoding="utf-8")
            (project_dir / "skills").mkdir()
            (project_dir / "skills" / "review.md").write_text(
                "# Review\nProject review checklist", encoding="utf-8"
            )

            default_context, default_sources = load_full_context(project_dir)
            explicit_context, explicit_sources = load_full_context(
                project_dir,
                loaded_user_skills=["project:skills/review.md"],
            )

        self.assertIn("project rules", default_context)
        self.assertNotIn("Project review checklist", default_context)
        self.assertEqual([source.label for source in default_sources], ["AGENTS.md"])
        self.assertIn("Project review checklist", explicit_context)
        self.assertEqual(
            [source.label for source in explicit_sources],
            ["AGENTS.md", "skills/review.md"],
        )


def _write_external_tool_script(
    target_dir: Path,
    name: str,
    *,
    body: str | None = None,
    sleep_seconds: float = 0.0,
    fail_describe: bool = False,
    invalid_json: bool = False,
) -> Path:
    """Write a hand-parsed test tool script that mimics the external-tool contract."""
    target_dir.mkdir(parents=True, exist_ok=True)
    script_path = target_dir / f"{name}.py"
    describe_payload = json.dumps(
        {
            "name": name,
            "description": f"Test tool {name}",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Echo input"},
                },
                "required": ["text"],
            },
            "version": "1",
        }
    )
    if invalid_json:
        describe_output = "this-is-not-json"
    else:
        describe_output = describe_payload

    script_template = textwrap.dedent(
        f'''
        import json
        import sys
        import time

        SLEEP_SECONDS = {sleep_seconds!r}

        def main():
            if SLEEP_SECONDS:
                time.sleep(SLEEP_SECONDS)

            if len(sys.argv) < 2:
                sys.exit(2)

            command = sys.argv[1]
            if command == "describe":
                if {fail_describe!r}:
                    print("describe error", file=sys.stderr)
                    sys.exit(3)
                print({describe_output!r})
                return

            if command == "invoke":
                idx = sys.argv.index("--args")
                args = json.loads(sys.argv[idx + 1])
                print({body!r}.format(**args) if {body!r} else "echo:" + args.get("text", ""))
                return

            sys.exit(2)

        if __name__ == "__main__":
            main()
        '''
    ).lstrip()
    script_path.write_text(script_template, encoding="utf-8")
    return script_path


def _python_runner_command() -> list[str]:
    return [sys.executable]


def _python_runner_status() -> RunnerStatus:
    return RunnerStatus(
        name="python", available=True, executable=sys.executable, message=None
    )


class ExternalToolDiscoveryTests(unittest.TestCase):
    def test_discovers_scripts_and_caches_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)
            tools_dir = user_dir / "tools"
            _write_external_tool_script(tools_dir, "echo_tool")

            first = discover_external_tools(
                user_dir=user_dir,
                runner_command=_python_runner_command(),
                runner_status=_python_runner_status(),
                describe_timeout=5.0,
            )
            cache_path = user_dir / "tools" / ".cache" / "manifests.json"

            self.assertEqual(len(first.loaded), 1)
            self.assertEqual(first.loaded[0].manifest.name, "echo_tool")
            self.assertTrue(cache_path.is_file())

            with mock.patch("pyagent.external_tools._describe_script") as describe_mock:
                describe_mock.side_effect = AssertionError(
                    "should hit the schema cache for unchanged scripts"
                )
                second = discover_external_tools(
                    user_dir=user_dir,
                    runner_command=_python_runner_command(),
                    runner_status=_python_runner_status(),
                    describe_timeout=5.0,
                )

            self.assertEqual(len(second.loaded), 1)
            self.assertEqual(second.loaded[0].manifest.name, "echo_tool")

    def test_cache_invalidates_when_script_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)
            tools_dir = user_dir / "tools"
            script = _write_external_tool_script(tools_dir, "echo_tool")

            discover_external_tools(
                user_dir=user_dir,
                runner_command=_python_runner_command(),
                runner_status=_python_runner_status(),
                describe_timeout=5.0,
            )

            time.sleep(0.01)
            script.write_text(
                script.read_text(encoding="utf-8") + "\n# modification\n",
                encoding="utf-8",
            )

            with mock.patch(
                "pyagent.external_tools._describe_script",
                wraps=__import__("pyagent.external_tools", fromlist=[
                                 "_describe_script"])._describe_script,
            ) as describe_mock:
                discover_external_tools(
                    user_dir=user_dir,
                    runner_command=_python_runner_command(),
                    runner_status=_python_runner_status(),
                    describe_timeout=5.0,
                )

            self.assertEqual(describe_mock.call_count, 1)

    def test_describe_timeout_marks_tool_broken(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)
            tools_dir = user_dir / "tools"
            _write_external_tool_script(
                tools_dir, "slow_tool", sleep_seconds=2.0)

            result = discover_external_tools(
                user_dir=user_dir,
                runner_command=_python_runner_command(),
                runner_status=_python_runner_status(),
                describe_timeout=0.5,
                cache_enabled=False,
            )

            self.assertEqual(len(result.loaded), 0)
            self.assertEqual(len(result.broken), 1)
            self.assertIn("timed out", result.broken[0].error or "")

    def test_invalid_describe_json_marks_tool_broken(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)
            tools_dir = user_dir / "tools"
            _write_external_tool_script(
                tools_dir, "bad_tool", invalid_json=True)

            result = discover_external_tools(
                user_dir=user_dir,
                runner_command=_python_runner_command(),
                runner_status=_python_runner_status(),
                describe_timeout=5.0,
                cache_enabled=False,
            )

            self.assertEqual(len(result.loaded), 0)
            self.assertEqual(len(result.broken), 1)
            self.assertIn("did not emit valid JSON",
                          result.broken[0].error or "")

    def test_missing_runner_disables_external_tools_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)
            tools_dir = user_dir / "tools"
            _write_external_tool_script(tools_dir, "echo_tool")

            unavailable = RunnerStatus(
                name="uv",
                available=False,
                executable=None,
                message="uv was not found on PATH.",
            )
            result = discover_external_tools(
                user_dir=user_dir,
                runner_command=_python_runner_command(),
                runner_status=unavailable,
                describe_timeout=5.0,
            )

            self.assertFalse(result.runner_available)
            self.assertEqual(len(result.loaded), 0)
            self.assertEqual(len(result.broken), 1)
            self.assertIn("uv was not found", result.broken[0].error or "")

    def test_disabled_directory_listed_but_not_registered(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)
            tools_dir = user_dir / "tools"
            disabled_dir = tools_dir / "disabled"
            _write_external_tool_script(tools_dir, "echo_tool")
            _write_external_tool_script(disabled_dir, "shelved_tool")

            result = discover_external_tools(
                user_dir=user_dir,
                runner_command=_python_runner_command(),
                runner_status=_python_runner_status(),
                describe_timeout=5.0,
            )

            loaded_names = [entry.manifest.name for entry in result.loaded]
            disabled_paths = [str(entry.script_path)
                              for entry in result.disabled]

            self.assertEqual(loaded_names, ["echo_tool"])
            self.assertEqual(len(result.disabled), 1)
            self.assertTrue(any(path.endswith("shelved_tool.py")
                            for path in disabled_paths))


class ExternalToolHandlerTests(unittest.TestCase):
    def test_invoke_returns_stdout_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)
            tools_dir = user_dir / "tools"
            script = _write_external_tool_script(tools_dir, "echo_tool")

            handler = ExternalToolHandler(
                script,
                runner_command=_python_runner_command(),
                invoke_timeout=10.0,
            )
            result = handler(text="hello")

        self.assertEqual(result.strip(), "echo:hello")

    def test_invoke_timeout_returns_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)
            tools_dir = user_dir / "tools"
            script = _write_external_tool_script(
                tools_dir, "slow_tool", sleep_seconds=3.0
            )

            handler = ExternalToolHandler(
                script,
                runner_command=_python_runner_command(),
                invoke_timeout=0.5,
            )
            start = time.monotonic()
            result = handler(text="hello")
            elapsed = time.monotonic() - start

        self.assertIn("exceeded its", result)
        self.assertIn("timeout", result)
        self.assertLess(elapsed, 5.0)


class ToolRegistryCompositionTests(unittest.TestCase):
    def test_builtin_wins_on_collision_and_collisions_are_reported(self) -> None:
        config = AppConfig()
        external = ToolSpec(
            name="list_files",
            description="Should NOT replace the built-in",
            parameters={"type": "object", "properties": {}},
            handler=lambda **_: "external",
        )
        registry = create_default_tool_registry(
            config, external_specs=[external])

        self.assertEqual(registry.origin("list_files"), BUILTIN_ORIGIN)
        self.assertNotEqual(registry.execute("list_files", {}), "external")
        self.assertEqual(len(registry.collisions()), 1)
        self.assertEqual(registry.collisions()[0].name, "list_files")

    def test_builtin_tools_can_be_disabled_while_external_tools_register(self) -> None:
        config = AppConfig(builtin_tools_enabled=False)
        external = ToolSpec(
            name="echo_tool",
            description="Echo tool",
            parameters={"type": "object", "properties": {}},
            handler=lambda **_: "external",
        )
        registry = create_default_tool_registry(
            config, external_specs=[external])

        self.assertEqual(registry.names_by_origin(BUILTIN_ORIGIN), [])
        self.assertEqual(registry.names_by_origin(
            EXTERNAL_ORIGIN), ["echo_tool"])
        self.assertEqual(registry.execute("echo_tool", {}), "external")

    def test_disabled_builtin_name_can_be_used_by_external_tool(self) -> None:
        config = AppConfig(builtin_tools_enabled=False)
        external = ToolSpec(
            name="list_files",
            description="External replacement",
            parameters={"type": "object", "properties": {}},
            handler=lambda **_: "external",
        )
        registry = create_default_tool_registry(
            config, external_specs=[external])

        self.assertEqual(registry.origin("list_files"), EXTERNAL_ORIGIN)
        self.assertEqual(registry.execute("list_files", {}), "external")
        self.assertEqual(registry.collisions(), [])

    def test_external_specs_register_and_track_origin_and_source(self) -> None:
        config = AppConfig()
        handler = lambda **kwargs: f"echo:{kwargs.get('text', '')}"
        handler.script_path = "/tmp/echo_tool.py"  # type: ignore[attr-defined]
        external = ToolSpec(
            name="echo_tool",
            description="Echo tool",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=handler,
        )
        registry = create_default_tool_registry(
            config, external_specs=[external])

        self.assertIn("echo_tool", registry.names())
        self.assertEqual(registry.origin("echo_tool"), EXTERNAL_ORIGIN)
        self.assertEqual(registry.source("echo_tool"), "/tmp/echo_tool.py")
        self.assertEqual(registry.names_by_origin(
            EXTERNAL_ORIGIN), ["echo_tool"])
        self.assertEqual(registry.execute(
            "echo_tool", {"text": "hi"}), "echo:hi")


class UserGlobalContextTests(unittest.TestCase):
    def test_load_full_context_layers_global_and_project(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir) / "userhome"
            (user_dir / "skills").mkdir(parents=True)
            (user_dir / "AGENTS.md").write_text(
                "Always check the user-global agent file.", encoding="utf-8"
            )
            (user_dir / "skills" / "global_skill.md").write_text(
                "Global skill content.", encoding="utf-8"
            )

            project_dir = Path(temp_dir) / "project"
            project_dir.mkdir()
            (project_dir / "AGENTS.md").write_text(
                "Project-only rule.", encoding="utf-8"
            )

            text, sources = load_full_context(project_dir, user_dir=user_dir)

            scopes = {source.scope for source in sources}
            labels = [source.label for source in sources]

            self.assertEqual(scopes, {GLOBAL_SCOPE, PROJECT_SCOPE})
            self.assertTrue(any(label.startswith("~/.pyagent/")
                            for label in labels))
            self.assertIn("AGENTS.md", labels)
            self.assertIn("Always check the user-global agent file.", text)
            self.assertIn("Project-only rule.", text)
            self.assertIn("User-global instructions", text)
            self.assertIn("Project-specific instructions", text)

    def test_discover_user_global_skips_missing_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            files = discover_user_global_instruction_files(Path(temp_dir))
        self.assertEqual(files, [])

    def test_load_project_context_compatibility_returns_labelled_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir) / "userhome"
            user_dir.mkdir()
            (user_dir / "AGENTS.md").write_text("global", encoding="utf-8")

            project_dir = Path(temp_dir) / "project"
            project_dir.mkdir()
            (project_dir / "AGENTS.md").write_text("project", encoding="utf-8")

            text, files = load_project_context(project_dir, user_dir=user_dir)

        self.assertIn("global", text)
        self.assertIn("project", text)
        self.assertEqual(len(files), 2)
        self.assertTrue(any(label.startswith("~/.pyagent/")
                        for label in files))


class PackagingTests(unittest.TestCase):
    def test_pyproject_includes_pyagent_subpackages(self) -> None:
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('include = ["pyagent*"]', pyproject)


class ScaffoldTests(unittest.TestCase):
    def test_create_user_tool_writes_compilable_script(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = create_user_tool("demo_tool", user_dir=temp_dir)

            self.assertTrue(path.is_file())
            content = path.read_text(encoding="utf-8")
            self.assertIn("describe", content)
            self.assertIn("invoke", content)
            compile(content, str(path), "exec")

    def test_create_user_tool_refuses_overwrite_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            create_user_tool("demo_tool", user_dir=temp_dir)
            with self.assertRaises(ScaffoldError):
                create_user_tool("demo_tool", user_dir=temp_dir)

    def test_create_user_tool_rejects_invalid_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ScaffoldError):
                create_user_tool("9bad-name!", user_dir=temp_dir)


class UiToolsCommandTests(unittest.TestCase):
    def _make_app_with_temp_user_dir(self, temp_dir: str) -> PyAgentApp:
        config = AppConfig(user_dir=temp_dir, user_tools_enabled=True)
        app = PyAgentApp.__new__(PyAgentApp)
        # Skip Textual __init__ entirely; we only exercise pure-Python helpers.
        from pyagent.agent import Agent

        agent = Agent(config=config)
        app.agent = agent
        app.project_context = ""
        app.context_sources = []
        app.project_context_files = []
        app.is_processing = False
        app.debug_visible = False
        app.input_history = []
        app.input_history_index = None
        app.input_history_draft = ""
        return app

    def test_tools_command_lists_builtin_and_external_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = self._make_app_with_temp_user_dir(temp_dir)
            notes: list[str] = []
            app._add_system_note = notes.append

            handled = app._handle_slash_command("/tools")

            self.assertTrue(handled)
            self.assertIn("Built-in tools enabled: `True`", notes[-1])
            self.assertIn("User-tools enabled: `True`", notes[-1])
            self.assertIn("Built-in tools:", notes[-1])
            self.assertIn("External tools", notes[-1])

    def test_tools_command_shows_no_builtins_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppConfig(
                user_dir=temp_dir,
                user_tools_enabled=True,
                builtin_tools_enabled=False,
            )
            app = PyAgentApp.__new__(PyAgentApp)
            from pyagent.agent import Agent

            app.agent = Agent(config=config)
            notes: list[str] = []
            app._add_system_note = notes.append

            handled = app._handle_slash_command("/tools")

            self.assertTrue(handled)
            self.assertIn("Built-in tools enabled: `False`", notes[-1])
            self.assertIn("Built-in tools:\n- _none_", notes[-1])

    def test_tools_new_creates_script_and_warns_on_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = self._make_app_with_temp_user_dir(temp_dir)
            notes: list[str] = []
            app._add_system_note = notes.append

            self.assertTrue(app._handle_slash_command("/tools new my_tool"))
            self.assertIn("Created starter tool", notes[-1])
            self.assertTrue(
                (Path(temp_dir) / "tools" / "my_tool.py").is_file()
            )

            self.assertTrue(app._handle_slash_command("/tools new my_tool"))
            self.assertIn("Could not create tool", notes[-1])

    def test_tools_disable_then_enable_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tools_dir = Path(temp_dir) / "tools"
            tools_dir.mkdir(parents=True)
            (tools_dir / "echo.py").write_text("# placeholder", encoding="utf-8")

            app = self._make_app_with_temp_user_dir(temp_dir)
            notes: list[str] = []
            app._add_system_note = notes.append

            self.assertTrue(app._handle_slash_command("/tools disable echo"))
            self.assertIn("Moved tool", notes[-1])
            self.assertTrue((tools_dir / "disabled" / "echo.py").is_file())
            self.assertFalse((tools_dir / "echo.py").exists())

            self.assertTrue(app._handle_slash_command("/tools enable echo"))
            self.assertTrue((tools_dir / "echo.py").is_file())
            self.assertFalse((tools_dir / "disabled" / "echo.py").exists())

    def test_tools_open_returns_path_or_friendly_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tools_dir = Path(temp_dir) / "tools"
            tools_dir.mkdir(parents=True)
            (tools_dir / "echo.py").write_text("# placeholder", encoding="utf-8")

            app = self._make_app_with_temp_user_dir(temp_dir)
            notes: list[str] = []
            app._add_system_note = notes.append

            self.assertTrue(app._handle_slash_command("/tools open echo"))
            self.assertIn("Tool script path", notes[-1])

            self.assertTrue(app._handle_slash_command(
                "/tools open does_not_exist"))
            self.assertIn("No tool named", notes[-1])

    def test_help_text_mentions_new_tool_subcommands(self) -> None:
        app = PyAgentApp()
        text = app._command_help_text()
        self.assertIn("`/tools reload`", text)
        self.assertIn("`/tools new <name>`", text)
        self.assertIn("`/tools enable <name>`", text)
        self.assertIn("`/tools open <name>`", text)


class AgentExternalToolsTests(unittest.TestCase):
    def test_agent_constructor_handles_missing_user_dir_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppConfig(user_dir=temp_dir, user_tools_enabled=True)
            agent = Agent(config=config)

            self.assertEqual(
                agent.tool_registry.names_by_origin(EXTERNAL_ORIGIN), [])
            self.assertGreater(
                len(agent.tool_registry.names_by_origin(BUILTIN_ORIGIN)), 0
            )

    def test_reload_external_tools_picks_up_new_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppConfig(
                user_dir=temp_dir,
                user_tools_enabled=True,
                tool_runner="python",
            )
            agent = Agent(config=config)
            self.assertEqual(
                agent.tool_registry.names_by_origin(EXTERNAL_ORIGIN), [])

            tools_dir = Path(temp_dir) / "tools"
            _write_external_tool_script(tools_dir, "echo_tool")

            with mock.patch(
                "pyagent.agent.discover_external_tools",
                lambda **kwargs: discover_external_tools(
                    user_dir=kwargs.get("user_dir", temp_dir),
                    runner_command=_python_runner_command(),
                    runner_status=_python_runner_status(),
                    describe_timeout=kwargs.get("describe_timeout", 5.0),
                ),
            ), mock.patch(
                "pyagent.agent.default_runner_command",
                lambda runner=None: _python_runner_command(),
            ):
                discovery = agent.reload_external_tools()

            self.assertIsNotNone(discovery)
            self.assertEqual(
                agent.tool_registry.names_by_origin(
                    EXTERNAL_ORIGIN), ["echo_tool"]
            )


class ExternalToolExtraDirsTests(unittest.TestCase):
    def test_discover_scans_extra_tool_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)
            ext_tools = user_dir / "extensions" / "myext" / "tools"
            _write_external_tool_script(ext_tools, "ext_tool")
            result = discover_external_tools(
                user_dir=user_dir,
                runner_command=_python_runner_command(),
                runner_status=_python_runner_status(),
                describe_timeout=5.0,
                extra_tool_dirs=[ext_tools],
            )
            self.assertEqual(len(result.loaded), 1)
            self.assertEqual(result.loaded[0].manifest.name, "ext_tool")

    def test_discover_ignores_extra_dir_when_not_passed(self) -> None:
        # Same files, but no extra_tool_dirs -> not discovered (hidden).
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)
            ext_tools = user_dir / "extensions" / "myext" / "tools"
            _write_external_tool_script(ext_tools, "ext_tool")
            result = discover_external_tools(
                user_dir=user_dir,
                runner_command=_python_runner_command(),
                runner_status=_python_runner_status(),
                describe_timeout=5.0,
            )
            self.assertEqual(len(result.loaded), 0)


class ExtensionColocationTests(unittest.TestCase):
    """Colocated extension scripts/skills/tools + load-gated discovery."""

    def _make_ext_package(self, user_dir: Path, name: str) -> Path:
        pkg = user_dir / "extensions" / name
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "__init__.py").write_text(
            "def register(bus, name):\n"
            "    @bus.on('turn_start')\n"
            "    def h(payload, ctx): pass\n",
            encoding="utf-8",
        )
        _write_external_tool_script(pkg / "tools", "ext_only_tool")
        skills = pkg / "skills"
        skills.mkdir(parents=True, exist_ok=True)
        (skills / "ext_skill.md").write_text("EXT SKILL TEXT")
        return pkg

    def _patch_discover(self, user_dir: Path):
        def _disc(**kwargs):
            return discover_external_tools(
                user_dir=kwargs.get("user_dir", user_dir),
                runner_command=_python_runner_command(),
                runner_status=_python_runner_status(),
                describe_timeout=kwargs.get("describe_timeout", 5.0),
                extra_tool_dirs=kwargs.get("extra_tool_dirs") or [],
            )
        return _disc

    def test_tool_and_skill_hidden_until_extension_loaded(self) -> None:
        from pyagent.extensions.manager import handle_extension_command
        from pyagent.tools import list_skills

        with tempfile.TemporaryDirectory() as tmp:
            user_dir = Path(tmp)
            self._make_ext_package(user_dir, "myext")
            config = AppConfig(user_dir=tmp, user_tools_enabled=True, tool_runner="python")
            agent = Agent(config=config)

            # Before loading: tool not registered, skill not in list_skills.
            self.assertNotIn("ext_only_tool", agent.tool_registry.names())
            self.assertNotIn("ext_skill", list_skills(scope="user", config=config, cwd=tmp))

            with mock.patch("pyagent.agent.discover_external_tools", self._patch_discover(user_dir)), \
                 mock.patch("pyagent.agent.default_runner_command", lambda runner=None: _python_runner_command()):
                agent.load_extensions(start_session=False)
                self.assertIn("myext", agent.bus.loaded_extensions())
                self.assertIn("ext_only_tool", agent.tool_registry.names())

                # Unloading hides the tool again.
                handle_extension_command(agent, ["unload", "myext"])
                self.assertNotIn("myext", agent.bus.loaded_extensions())
                self.assertNotIn("ext_only_tool", agent.tool_registry.names())

    def test_colocated_skill_resolves_and_is_hidden_from_list_skills(self) -> None:
        from pyagent.tools import list_skills

        with tempfile.TemporaryDirectory() as tmp:
            user_dir = Path(tmp)
            self._make_ext_package(user_dir, "myext")
            config = AppConfig(user_dir=tmp, tools_enabled=True)
            agent = Agent(config=config, tool_registry=create_default_tool_registry(config))
            # Colocated skill is never listed by list_skills.
            self.assertNotIn("ext_skill", list_skills(scope="user", config=config, cwd=tmp))
            # But resolves when the owning extension declares it.
            agent._this_skills = {"myext/ext_skill"}
            text = collect_skill_text(agent)
            self.assertIn("EXT SKILL TEXT", text)

    def test_colocated_skill_not_resolved_when_extension_name_differs(self) -> None:
        # A skill entry for a different extension must not resolve against
        # myext's colocated skill dir.
        with tempfile.TemporaryDirectory() as tmp:
            user_dir = Path(tmp)
            self._make_ext_package(user_dir, "myext")
            config = AppConfig(user_dir=tmp, tools_enabled=True)
            agent = Agent(config=config, tool_registry=create_default_tool_registry(config))
            agent._this_skills = {"other/ext_skill"}
            self.assertEqual(collect_skill_text(agent), "")


def demo() -> None:
    agent = Agent(model="llama3.1")
    prompt = (
        "Please create a file named 'test_agent_demo.txt' with the text "
        "'Hello from the Python Agent!', then read it back to verify."
    )
    print(f"Prompt: {prompt}\n")

    for event in agent.run(prompt):
        event_type = event.get("type")
        if event_type == "content_delta":
            print(event["delta"], end="", flush=True)
        elif event_type in {"tool_call", "tool_result", "error"}:
            print(f"\n[{event_type}] {event}\n")


class ModelProfileCompatibilityTests(unittest.TestCase):
    def test_profile_from_dict_accepts_legacy_http_kwargs_alias(self) -> None:
        payload = {
            "default_profile": "legacy",
            "profiles": {
                "legacy": {
                    "provider": "openai",
                    "model": "gpt-test",
                    "base_url": "https://example.invalid/v1",
                    "http_kwargs": {"verify": False},
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            store = load_profile_store(str(path))

        self.assertEqual(store.get("legacy").httpx_kwargs, {"verify": False})

    def test_profile_from_dict_prefers_httpx_kwargs_over_legacy_alias(self) -> None:
        payload = {
            "default_profile": "legacy",
            "profiles": {
                "legacy": {
                    "provider": "openai",
                    "model": "gpt-test",
                    "base_url": "https://example.invalid/v1",
                    "http_kwargs": {"verify": False},
                    "httpx_kwargs": {"verify": True},
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            store = load_profile_store(str(path))

        self.assertEqual(store.get("legacy").httpx_kwargs, {"verify": True})


class OpenAICompatibleClientHttpTests(unittest.TestCase):
    def test_openai_client_does_not_build_httpx_client_without_kwargs(self) -> None:
        profile = ModelProfile(
            name="test",
            provider="openai",
            model="gpt-test",
            base_url="https://example.invalid/v1",
        )
        client = OpenAICompatibleClient(profile)
        fake_sdk_client = FakeOpenAIClient()

        with mock.patch("pyagent.llm_client.httpx.Client") as mock_httpx_client:
            client._client_factory = mock.Mock(return_value=fake_sdk_client)
            built = client._get_client()

        self.assertIs(built, fake_sdk_client)
        self.assertIsNone(client._http_client)
        mock_httpx_client.assert_not_called()
        client._client_factory.assert_called_once_with(
            api_key="",
            base_url="https://example.invalid/v1",
            default_headers=None,
            timeout=300.0,
            max_retries=2,
        )

    def test_openai_client_builds_httpx_client_when_kwargs_are_present(self) -> None:
        profile = ModelProfile(
            name="test",
            provider="openai",
            model="gpt-test",
            base_url="https://example.invalid/v1",
            httpx_kwargs={"verify": False},
        )
        client = OpenAICompatibleClient(profile)
        fake_sdk_client = FakeOpenAIClient()
        fake_http_client = mock.Mock()

        with mock.patch("pyagent.llm_client.httpx.Client", return_value=fake_http_client) as mock_httpx_client:
            client._client_factory = mock.Mock(return_value=fake_sdk_client)
            built = client._get_client()

        self.assertIs(built, fake_sdk_client)
        self.assertIs(client._http_client, fake_http_client)
        mock_httpx_client.assert_called_once_with(verify=False)
        client._client_factory.assert_called_once_with(
            api_key="",
            base_url="https://example.invalid/v1",
            default_headers=None,
            timeout=300.0,
            max_retries=2,
            http_client=fake_http_client,
        )


class _FakeUrlOpenResponse:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class ClientTests(unittest.TestCase):
    def test_health_returns_status_payload(self) -> None:
        from pyagent.client import PyAgentClient

        with mock.patch(
            "pyagent.client.request.urlopen",
            return_value=_FakeUrlOpenResponse('{"status": "ok"}'),
        ) as mock_urlopen:
            client = PyAgentClient("http://127.0.0.1:8000/")

            response = client.health()

        self.assertEqual(response, {"status": "ok"})
        req = mock_urlopen.call_args.args[0]
        self.assertEqual(req.full_url, "http://127.0.0.1:8000/health")
        self.assertEqual(req.get_method(), "GET")

    def test_is_healthy_returns_false_when_server_fails(self) -> None:
        from pyagent.client import PyAgentClient

        with mock.patch(
            "pyagent.client.request.urlopen",
            side_effect=error.URLError("connection refused"),
        ):
            client = PyAgentClient()
            self.assertFalse(client.is_healthy())

    def test_run_returns_typed_response(self) -> None:
        from pyagent.client import PyAgentClient, RunResponse

        payload = json.dumps(
            {
                "response": "Hello World!",
                "profile": "p1",
                "provider": "ollama",
                "model": "m1",
                "messages": [],
                "context_files": ["AGENTS.md"],
            }
        )
        with mock.patch(
            "pyagent.client.request.urlopen",
            return_value=_FakeUrlOpenResponse(payload),
        ) as mock_urlopen:
            client = PyAgentClient(
                "http://127.0.0.1:8000", headers={"X-Test": "yes"})

            response = client.run(
                "Hi",
                profile="p1",
                model="m1",
                cwd="/tmp",
                skills=["foo.md"],
            )

        self.assertEqual(
            response,
            RunResponse(
                response="Hello World!",
                profile="p1",
                provider="ollama",
                model="m1",
                messages=[],
                context_files=["AGENTS.md"],
            ),
        )
        req = mock_urlopen.call_args.args[0]
        self.assertEqual(req.full_url, "http://127.0.0.1:8000/run")
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.headers["Content-type"], "application/json")
        self.assertEqual(req.headers["X-test"], "yes")
        self.assertEqual(
            json.loads(req.data.decode("utf-8")),
            {
                "message": "Hi",
                "messages": [],
                "profile": "p1",
                "model": "m1",
                "cwd": "/tmp",
                "skills": ["foo.md"],
            },
        )

    def test_run_raises_clear_error_for_http_error_detail(self) -> None:
        from pyagent.client import PyAgentClient, PyAgentClientError

        http_error = error.HTTPError(
            url="http://127.0.0.1:8000/run",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=None,
        )
        http_error.read = lambda: b'{"detail": "Unknown skill: nope.md"}'
        with mock.patch(
            "pyagent.client.request.urlopen",
            side_effect=http_error,
        ):
            client = PyAgentClient()
            with self.assertRaises(PyAgentClientError) as cm:
                client.run("Hi")

        self.assertIn("HTTP 400", str(cm.exception))
        self.assertIn("Unknown skill: nope.md", str(cm.exception))

    def test_run_raises_clear_error_for_non_json_http_error(self) -> None:
        from pyagent.client import PyAgentClient, PyAgentClientError

        http_error = error.HTTPError(
            url="http://127.0.0.1:8000/run",
            code=500,
            msg="Server Error",
            hdrs=None,
            fp=None,
        )
        http_error.read = lambda: b"kaboom"
        with mock.patch(
            "pyagent.client.request.urlopen",
            side_effect=http_error,
        ):
            client = PyAgentClient()
            with self.assertRaises(PyAgentClientError) as cm:
                client.run("Hi")

        self.assertIn("HTTP 500", str(cm.exception))
        self.assertIn("kaboom", str(cm.exception))

    def test_run_raises_clear_error_for_connection_failure(self) -> None:
        from pyagent.client import PyAgentClient, PyAgentClientError

        with mock.patch(
            "pyagent.client.request.urlopen",
            side_effect=error.URLError("connection refused"),
        ):
            client = PyAgentClient()
            with self.assertRaises(PyAgentClientError) as cm:
                client.run("Hi")

        self.assertIn("Could not connect to PyAgent server", str(cm.exception))

    def test_run_raises_clear_error_for_timeout(self) -> None:
        from pyagent.client import PyAgentClient, PyAgentClientError

        with mock.patch(
            "pyagent.client.request.urlopen",
            side_effect=error.URLError(socket.timeout("timed out")),
        ):
            client = PyAgentClient()
            with self.assertRaises(PyAgentClientError) as cm:
                client.run("Hi")

        self.assertIn("Timed out", str(cm.exception))

    def test_run_raises_clear_error_for_invalid_json_response(self) -> None:
        from pyagent.client import PyAgentClient, PyAgentClientError

        with mock.patch(
            "pyagent.client.request.urlopen",
            return_value=_FakeUrlOpenResponse("not-json"),
        ):
            client = PyAgentClient()
            with self.assertRaises(PyAgentClientError) as cm:
                client.run("Hi")

        self.assertIn("invalid JSON", str(cm.exception))

    def test_run_raises_clear_error_when_required_field_missing(self) -> None:
        from pyagent.client import PyAgentClient, PyAgentClientError

        with mock.patch(
            "pyagent.client.request.urlopen",
            return_value=_FakeUrlOpenResponse('{"profile": "p1"}'),
        ):
            client = PyAgentClient()
            with self.assertRaises(PyAgentClientError) as cm:
                client.run("Hi")

        self.assertIn("missing required field", str(cm.exception))


class SessionLoggerEventTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _load_lines(self, logger: SessionLogger) -> list[dict]:
        logger.close()
        with open(logger.path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def test_log_event_records_extension_event(self) -> None:
        logger = SessionLogger(log_dir=self.tmpdir)
        logger.log_event("tool_call", "audit", {"name": "bash"}, {"blocked": True})
        lines = self._load_lines(logger)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["severityText"], "DEBUG")
        self.assertEqual(lines[0]["body"], "Extension event: tool_call (audit)")
        self.assertEqual(lines[0]["attributes"]["event"], "tool_call")
        self.assertEqual(lines[0]["attributes"]["extension"], "audit")
        self.assertEqual(lines[0]["attributes"]["payload"], {"name": "bash"})
        self.assertEqual(lines[0]["attributes"]["result"], {"blocked": True})

    def test_log_event_without_extension(self) -> None:
        logger = SessionLogger(log_dir=self.tmpdir)
        logger.log_event("agent_start", None, {})
        lines = self._load_lines(logger)
        self.assertNotIn("extension", lines[0]["attributes"])
        self.assertEqual(lines[0]["body"], "Extension event: agent_start")

    def test_log_extension_skills_records_keys(self) -> None:
        logger = SessionLogger(log_dir=self.tmpdir)
        logger.log_extension_skills(["foo", "bar"], 1234)
        lines = self._load_lines(logger)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["attributes"]["event_type"], "extension_skills_loaded")
        self.assertEqual(lines[0]["attributes"]["keys"], ["foo", "bar"])
        self.assertEqual(lines[0]["attributes"]["text_chars"], 1234)

    def test_log_turn_body_includes_tool_names(self) -> None:
        logger = SessionLogger(log_dir=self.tmpdir)
        output = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "bash"}},
                {"function": {"name": "search_text"}},
            ]},
            {"role": "tool", "name": "bash", "content": "ok"},
        ]
        logger.log_turn(1, [], output)
        lines = self._load_lines(logger)
        self.assertIn("bash", lines[0]["body"])
        self.assertIn("search_text", lines[0]["body"])

    def test_log_turn_body_without_tools(self) -> None:
        logger = SessionLogger(log_dir=self.tmpdir)
        logger.log_turn(1, [], [{"role": "assistant", "content": "hi"}])
        lines = self._load_lines(logger)
        self.assertNotIn(" — ", lines[0]["body"])

    def test_log_skill_load_records_event(self) -> None:
        logger = SessionLogger(log_dir=self.tmpdir)
        logger.log_skill_load("python")
        lines = self._load_lines(logger)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["body"], "Skill loaded: python")
        self.assertEqual(lines[0]["attributes"]["event_type"], "skill_load")
        self.assertEqual(lines[0]["attributes"]["skill_id"], "python")

    def test_log_skill_unload_records_event(self) -> None:
        logger = SessionLogger(log_dir=self.tmpdir)
        logger.log_skill_unload("python")
        lines = self._load_lines(logger)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["body"], "Skill unloaded: python")
        self.assertEqual(lines[0]["attributes"]["event_type"], "skill_unload")
        self.assertEqual(lines[0]["attributes"]["skill_id"], "python")


class EventBusLoggingTests(unittest.TestCase):
    def test_emit_logs_event_to_logger(self) -> None:
        calls: list[tuple[str, str | None, dict, Any]] = []

        class CaptureLog:
            def log_event(self, event: str, extension: str | None, payload: dict, result: Any = None) -> None:
                calls.append((event, extension, dict(payload), result))

        bus = EventBus(CaptureLog())
        ctx = Ctx(add_skill=lambda _k: None, log=CaptureLog())

        @bus.on("tool_call", extension="sg")
        def gate(payload: dict, ctx: Ctx) -> dict:
            return {"blocked": True}

        out = bus.emit("tool_call", {"name": "bash"}, ctx)
        self.assertTrue(out.get("blocked"))
        self.assertEqual(len(calls), 1)
        event, ext, payload, result = calls[0]
        self.assertEqual(event, "tool_call")
        self.assertEqual(ext, "sg")
        self.assertEqual(payload, {"name": "bash"})
        self.assertEqual(result, {"blocked": True})

    def test_emit_without_handlers_logs_event(self) -> None:
        calls: list[tuple[str, str | None, Any, Any]] = []

        class CaptureLog:
            def log_event(self, event: str, extension: str | None, payload: dict, result: Any = None) -> None:
                calls.append((event, extension, payload, result))

        bus = EventBus(CaptureLog())
        ctx = Ctx(add_skill=lambda _k: None, log=CaptureLog())
        out = bus.emit("no_handlers", {"a": 1}, ctx)
        self.assertEqual(out, {"a": 1})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "no_handlers")
        self.assertIsNone(calls[0][1])

    def test_noop_log_event_does_nothing(self) -> None:
        log = NoOpLog()
        log.log_event("x", "y", {})
        log.log_extension_skills(["x"], 1)
        # No assertion needed: reaching this line means no exception was raised.



# --- extension system: event bus, loader, skill injection, loop wiring ---
# (consolidated from test_extensions.py)
class TestEventBus(unittest.TestCase):
    def setUp(self) -> None:
        self.log = NoOpLog()
        self.bus = EventBus(self.log)
        self.ctx = Ctx(add_skill=lambda _k: None, log=self.log)

    def test_emit_no_handlers_returns_payload_unchanged(self) -> None:
        out = self.bus.emit("nothing", {"a": 1}, self.ctx)
        self.assertEqual(out, {"a": 1})

    def test_returned_dict_merges_into_payload(self) -> None:
        @self.bus.on("tool_call", extension="sg")
        def h(payload, ctx):
            return {"blocked": True, "reason": "no"}

        out = self.bus.emit("tool_call", {"name": "bash", "input": {}}, self.ctx)
        self.assertTrue(out["blocked"])
        self.assertEqual(out["reason"], "no")

    def test_in_place_mutation_visible_to_next_handler(self) -> None:
        @self.bus.on("e", extension="a")
        def a(payload, ctx):
            payload["touched"] = True

        @self.bus.on("e", extension="b")
        def b(payload, ctx):
            return {"seen_b": True}

        out = self.bus.emit("e", {}, self.ctx)
        self.assertTrue(out["touched"])
        self.assertTrue(out["seen_b"])

    def test_handlers_run_in_load_order(self) -> None:
        order: list[str] = []

        @self.bus.on("e", extension="a")
        def a(payload, ctx):
            order.append("a")
            return {"val": "A"}

        @self.bus.on("e", extension="b")
        def b(payload, ctx):
            order.append("b")
            return {"val": payload["val"] + "+B"}

        out = self.bus.emit("e", {}, self.ctx)
        self.assertEqual(order, ["a", "b"])
        self.assertEqual(out["val"], "A+B")

    def test_veto_first_wins_short_circuits(self) -> None:
        ran: list[str] = []

        @self.bus.on("tool_call", extension="a")
        def a(payload, ctx):
            ran.append("a")
            return {"blocked": True, "reason": "no"}

        @self.bus.on("tool_call", extension="b")
        def b(payload, ctx):
            ran.append("b")
            return {"blocked": False}

        out = self.bus.emit("tool_call", {"name": "x"}, self.ctx)
        self.assertEqual(ran, ["a"])
        self.assertTrue(out["blocked"])

    def test_non_veto_last_writer_wins(self) -> None:
        @self.bus.on("before_agent_start", extension="a")
        def a(payload, ctx):
            return {"system_prompt": "A"}

        @self.bus.on("before_agent_start", extension="b")
        def b(payload, ctx):
            return {"system_prompt": "B"}

        out = self.bus.emit("before_agent_start", {}, self.ctx)
        self.assertEqual(out["system_prompt"], "B")

    def test_handler_exception_is_isolated(self) -> None:
        @self.bus.on("e", extension="boom")
        def boom(payload, ctx):
            raise RuntimeError("kaboom")

        @self.bus.on("e", extension="after")
        def after(payload, ctx):
            return {"ran_after": True}

        out = self.bus.emit("e", {"keep": 1}, self.ctx)
        self.assertTrue(out["ran_after"])
        self.assertEqual(out["keep"], 1)

    def test_scoped_view_auto_tags_and_off_extension(self) -> None:
        sb = self.bus.scoped("ext")

        @sb.on("e")
        def h(payload, ctx):
            return {"v": 1}

        self.bus.emit("e", {}, self.ctx)
        self.assertIn("ext", self.bus.loaded_extensions())
        self.bus.off_extension("ext")
        self.assertEqual(self.bus.loaded_extensions(), [])

    def test_clear_removes_all(self) -> None:
        self.bus.on("e", lambda p, c: None, extension="x")
        self.bus.clear()
        self.assertEqual(self.bus.loaded_extensions(), [])

    def test_ctx_extension_set_per_handler(self) -> None:
        seen: list[str] = []

        @self.bus.on("e", extension="alpha")
        def a(payload, ctx):
            seen.append(ctx.extension)

        @self.bus.on("e", extension="beta")
        def b(payload, ctx):
            seen.append(ctx.extension)

        self.bus.emit("e", {}, self.ctx)
        self.assertEqual(seen, ["alpha", "beta"])

    def test_async_handler_return_is_ignored(self) -> None:
        # Synchronous-only contract: a coroutine return is not a dict, so it
        # is treated as a no-op pass-through (not executed).
        @self.bus.on("e", extension="x")
        async def h(payload, ctx):
            return {"v": 1}

        out = self.bus.emit("e", {}, self.ctx)
        self.assertEqual(out, {})


# --- Ctx / add_skill ---


class TestCtx(unittest.TestCase):
    def test_add_skill_calls_callback(self) -> None:
        added: list[str] = []
        ctx = Ctx(add_skill=added.append, log=NoOpLog())
        ctx.add_skill("compaction")
        self.assertEqual(added, ["compaction"])


# --- loader ---


def _write_ext(ext_dir: Path, name: str, body: str) -> None:
    pkg = ext_dir / name
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(body)


class TestLoader(unittest.TestCase):
    def test_discover_packages_and_single_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ext_dir = Path(tmp) / "extensions"
            ext_dir.mkdir()
            _write_ext(ext_dir, "pkg", "def register(b, n): pass\n")
            (ext_dir / "single.py").write_text("def register(b, n): pass\n")
            (ext_dir / "ignore.txt").write_text("nope")
            self.assertEqual(_discover(ext_dir), ["pkg", "single"])

    def test_load_all_imports_and_registers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ext_dir = Path(tmp) / "extensions"
            ext_dir.mkdir()
            _write_ext(ext_dir, "greet", textwrap.dedent("""
                def register(bus, name):
                    @bus.on('tool_call')
                    def h(payload, ctx):
                        return {'blocked': True, 'reason': name}
            """))
            log = NoOpLog()
            bus = EventBus(log)
            loaded, failed = load_all(bus, ext_dir, log)
            self.assertEqual(loaded, ["greet"])
            self.assertEqual(failed, [])
            ctx = Ctx(add_skill=lambda _k: None, log=log)
            out = bus.emit("tool_call", {"name": "x", "input": {}}, ctx)
            self.assertTrue(out["blocked"])
            self.assertEqual(out["reason"], "greet")

    def test_missing_register_factory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ext_dir = Path(tmp) / "extensions"
            ext_dir.mkdir()
            _write_ext(ext_dir, "bad", "X = 1\n")
            log = NoOpLog()
            bus = EventBus(log)
            loaded, failed = load_all(bus, ext_dir, log)
            self.assertEqual(loaded, [])
            self.assertEqual(len(failed), 1)
            self.assertEqual(failed[0][0], "bad")
            self.assertIn("register", failed[0][1])

    def test_broken_extension_does_not_block_others(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ext_dir = Path(tmp) / "extensions"
            ext_dir.mkdir()
            _write_ext(ext_dir, "good", "def register(b, n): pass\n")
            _write_ext(ext_dir, "bad", "def register(b, n): raise RuntimeError('boom')\n")
            log = NoOpLog()
            bus = EventBus(log)
            loaded, failed = load_all(bus, ext_dir, log)
            self.assertIn("good", loaded)
            self.assertIn("bad", [n for n, _ in failed])

    def test_load_one_then_unload_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ext_dir = Path(tmp) / "extensions"
            ext_dir.mkdir()
            _write_ext(ext_dir, "ext", "def register(b, n):\n  b.on('e', lambda p, c: None)\n")
            log = NoOpLog()
            bus = EventBus(log)
            load_one(bus, "ext", ext_dir, log)
            self.assertEqual(bus.loaded_extensions(), ["ext"])
            unload_one(bus, "ext")
            self.assertEqual(bus.loaded_extensions(), [])

    def test_reload_is_idempotent_no_duplicate_handlers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ext_dir = Path(tmp) / "extensions"
            ext_dir.mkdir()
            _write_ext(ext_dir, "ext", textwrap.dedent("""
                def register(bus, name):
                    @bus.on('turn_end')
                    def h(payload, ctx): return None
            """))
            log = NoOpLog()
            bus = EventBus(log)
            load_one(bus, "ext", ext_dir, log)
            self.assertEqual(len(bus.handlers("turn_end")), 1)
            load_one(bus, "ext", ext_dir, log)  # reload: clears prior first
            self.assertEqual(len(bus.handlers("turn_end")), 1)

    def test_missing_dir_is_noop(self) -> None:
        log = NoOpLog()
        bus = EventBus(log)
        loaded, failed = load_all(bus, Path("/nonexistent/extensions"), log)
        self.assertEqual(loaded, [])
        self.assertEqual(failed, [])


# --- skill injection (collect_skill_text) ---


class TestSkillInjection(unittest.TestCase):
    def _agent(self, tmp: str) -> Agent:
        config = AppConfig(user_dir=tmp, tools_enabled=True)
        return Agent(config=config, tool_registry=create_default_tool_registry(config))

    def test_no_skills_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(collect_skill_text(self._agent(tmp)), "")

    def test_injects_active_skill_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(tmp)
            skills_dir = Path(tmp) / "extensions" / "compaction" / "skills"
            skills_dir.mkdir(parents=True)
            (skills_dir / "compaction.md").write_text("# Compact me\nDo the thing.")
            agent._this_skills = {"compaction/compaction"}
            text = collect_skill_text(agent)
            self.assertIn("Extension skill: compaction", text)
            self.assertIn("Do the thing.", text)

    def test_injects_active_skill_text_legacy_fallback(self) -> None:
        # The old ~/.pyagent/skills/extensions/<key>.md location still resolves
        # as a fallback for the declaring extension.
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(tmp)
            skills_dir = Path(tmp) / "skills" / "extensions"
            skills_dir.mkdir(parents=True)
            (skills_dir / "compaction.md").write_text("# Compact me\nDo the thing.")
            agent._this_skills = {"compaction/compaction"}
            text = collect_skill_text(agent)
            self.assertIn("Extension skill: compaction", text)
            self.assertIn("Do the thing.", text)

    def test_missing_skill_file_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(tmp)
            agent._this_skills = {"nonexistent/nonexistent"}
            self.assertEqual(collect_skill_text(agent), "")

    def test_budget_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(tmp)
            skills_dir = Path(tmp) / "extensions" / "big" / "skills"
            skills_dir.mkdir(parents=True)
            big = "X" * (MAX_EXTENSION_SKILLS_CHARS + 500)
            (skills_dir / "big.md").write_text(big)
            agent._this_skills = {"big/big"}
            text = collect_skill_text(agent)
            self.assertLessEqual(len(text), MAX_EXTENSION_SKILLS_CHARS + 200)
            self.assertIn("[truncated", text)

    def test_skills_hidden_from_normal_discovery(self) -> None:
        # ~/.pyagent/skills/extensions/<key>.md must NOT appear in list_skills.
        from pyagent.tools import list_skills
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = Path(tmp) / "skills" / "extensions"
            skills_dir.mkdir(parents=True)
            (skills_dir / "secret.md").write_text("# secret ext skill")
            agent = self._agent(tmp)
            out = list_skills(scope="user", config=agent.config, cwd=tmp)
            self.assertNotIn("secret", out)


# --- agent loop wiring ---


class RecordingClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0
        self.seen_messages: list[list[dict]] = []
        self.profile = type("P", (), {"model": "dummy-model"})()

    def chat_stream(self, messages, tools=None):
        self.seen_messages.append(list(messages))
        response = self.responses[self.calls]
        self.calls += 1
        for chunk in response:
            yield chunk

    def list_models(self):
        return {"models": []}


def _make_agent(responses, **cfg_overrides):
    config = AppConfig(max_iterations=2, **cfg_overrides)
    agent = Agent(config=config, tool_registry=create_default_tool_registry(config))
    agent.client = RecordingClient(responses)
    return agent


class TestAgentLoopWiring(unittest.TestCase):
    def test_emits_fire_without_handlers_unchanged(self):
        agent = _make_agent([[{"content": "hi"}]])
        events = list(agent.run("hello"))
        self.assertEqual(events[-1], {"type": "assistant_done", "content": "hi"})

    def test_input_transform(self):
        agent = _make_agent([[{"content": "ok"}]])

        @agent.bus.on("input", extension="t")
        def transform(payload, ctx):
            return {"action": "transform", "text": "rewritten"}

        list(agent.run("original"))
        self.assertEqual(agent.client.seen_messages[0][1]["content"], "rewritten")

    def test_input_handled_skips_llm(self):
        agent = _make_agent([[{"content": "nope"}]])

        @agent.bus.on("input", extension="t")
        def handled(payload, ctx):
            return {"action": "handled"}

        events = list(agent.run("skip me"))
        self.assertEqual(agent.client.calls, 0)
        self.assertEqual(events[-1], {"type": "assistant_done", "content": ""})

    def test_before_agent_start_augments_prompt(self):
        agent = _make_agent([[{"content": "ok"}]])

        @agent.bus.on("before_agent_start", extension="t")
        def inject(payload, ctx):
            return {"system_prompt": payload["system_prompt"] + "\n\nEXTRA"}

        list(agent.run("hi"))
        self.assertIn("EXTRA", agent.messages[0]["content"])

    def test_context_event_prunes_request_only(self):
        agent = _make_agent([[{"content": "ok"}]])

        @agent.bus.on("context", extension="t")
        def prune(payload, ctx):
            return {"messages": payload["messages"][-2:]}

        list(agent.run("hi"))
        sent = agent.client.seen_messages[0]
        self.assertEqual(len(sent), 2)
        self.assertGreater(len(agent.messages), 2)

    def test_tool_call_blocked(self):
        agent = _make_agent([
            [{"tool_calls": [{"id": "c1", "function": {
                "name": "list_files", "arguments": {"path": "."}}}]}],
            [{"content": "done"}],
        ])

        @agent.bus.on("tool_call", extension="t")
        def gate(payload, ctx):
            return {"blocked": True, "reason": "nope"}

        list(agent.run("list"))
        tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("nope", tool_msgs[0]["content"])

    def test_tool_result_redact(self):
        agent = _make_agent([
            [{"tool_calls": [{"id": "c1", "function": {
                "name": "list_files", "arguments": {"path": "."}}}]}],
            [{"content": "done"}],
        ])

        @agent.bus.on("tool_result", extension="t")
        def redact(payload, ctx):
            return {"content": payload["content"] + "\n[redacted]"}

        list(agent.run("list"))
        tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
        self.assertIn("[redacted]", tool_msgs[0]["content"])

    def test_turn_end_carries_message_count(self):
        agent = _make_agent([[{"content": "ok"}]])
        seen: list[int] = []

        @agent.bus.on("turn_end", extension="t")
        def on_end(payload, ctx):
            seen.append(payload["message_count"])

        list(agent.run("hi"))
        self.assertTrue(seen)
        self.assertTrue(all(isinstance(c, int) for c in seen))

    def test_agent_end_fires(self):
        agent = _make_agent([[{"content": "ok"}]])
        seen: list[bool] = []

        @agent.bus.on("agent_end", extension="t")
        def done(payload, ctx):
            seen.append(True)

        list(agent.run("hi"))
        self.assertEqual(seen, [True])

    def test_model_select_emits_on_set_model(self):
        agent = _make_agent([[{"content": "ok"}]])
        seen: list[str] = []

        @agent.bus.on("model_select", extension="t")
        def on_model(payload, ctx):
            seen.append(payload["model"])

        agent.set_model("gpt-foo")
        self.assertEqual(seen, ["gpt-foo"])

    def test_handler_crash_does_not_halt_loop(self):
        agent = _make_agent([[{"content": "ok"}]])

        @agent.bus.on("turn_start", extension="t")
        def boom(payload, ctx):
            raise RuntimeError("kaboom")

        events = list(agent.run("hi"))
        self.assertEqual(events[-1], {"type": "assistant_done", "content": "ok"})


# --- ephemeral skill injection in the loop ---


class TestEphemeralSkills(unittest.TestCase):
    def test_skill_declared_this_turn_appears_next_turn_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = Path(tmp) / "extensions" / "ext" / "skills"
            skills_dir.mkdir(parents=True)
            (skills_dir / "guide.md").write_text("GUIDANCE TEXT")

            agent = _make_agent(
                [[{"content": "a"}], [{"content": "b"}]], user_dir=tmp
            )

            @agent.bus.on("turn_end", extension="ext")
            def declare(payload, ctx):
                ctx.add_skill("guide")

            # Turn 1: skill declared in turn_end; not yet in the prompt.
            list(agent.run("first"))
            self.assertNotIn("GUIDANCE TEXT", agent.messages[0]["content"])
            # Turn 2: skill injected this turn.
            list(agent.run("second"))
            self.assertIn("GUIDANCE TEXT", agent.messages[0]["content"])
            # Turn 3: still declared each turn_end, so persists.
            agent.client = RecordingClient([[{"content": "c"}]])
            list(agent.run("third"))
            self.assertIn("GUIDANCE TEXT", agent.messages[0]["content"])

    def test_skill_auto_expunges_when_no_longer_declared(self):
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = Path(tmp) / "extensions" / "ext" / "skills"
            skills_dir.mkdir(parents=True)
            (skills_dir / "guide.md").write_text("GUIDANCE TEXT")

            agent = _make_agent(
                [[{"content": "a"}], [{"content": "b"}]], user_dir=tmp
            )
            declare_once = {"done": False}

            @agent.bus.on("turn_end", extension="ext")
            def declare(payload, ctx):
                if not declare_once["done"]:
                    ctx.add_skill("guide")
                    declare_once["done"] = True

            list(agent.run("first"))   # declare
            list(agent.run("second"))  # injected
            self.assertIn("GUIDANCE TEXT", agent.messages[0]["content"])
            agent.client = RecordingClient([[{"content": "c"}]])
            list(agent.run("third"))   # no longer declared -> gone
            self.assertNotIn("GUIDANCE TEXT", agent.messages[0]["content"])

    def test_skill_declared_at_turn_start_appears_this_turn(self):
        # Regression for the one-turn-late race: a skill declared at
        # turn_start (the ollama_coder pattern) must reach the LLM in the SAME
        # turn, not the next one.
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = Path(tmp) / "extensions" / "ext" / "skills"
            skills_dir.mkdir(parents=True)
            (skills_dir / "guide.md").write_text("GUIDANCE TEXT")

            agent = _make_agent([[{"content": "a"}]], user_dir=tmp)

            @agent.bus.on("turn_start", extension="ext")
            def declare(payload, ctx):
                ctx.add_skill("guide")

            list(agent.run("first"))
            self.assertIn("GUIDANCE TEXT", agent.messages[0]["content"])

    def test_skill_declared_at_turn_start_reaches_llm_request(self):
        # The injected skill must be in the messages actually sent to the LLM,
        # not just in stored history.
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = Path(tmp) / "extensions" / "ext" / "skills"
            skills_dir.mkdir(parents=True)
            (skills_dir / "guide.md").write_text("GUIDANCE TEXT")

            agent = _make_agent([[{"content": "a"}]], user_dir=tmp)

            @agent.bus.on("turn_start", extension="ext")
            def declare(payload, ctx):
                ctx.add_skill("guide")

            list(agent.run("first"))
            sent = agent.client.seen_messages[0]
            self.assertEqual(sent[0]["role"], "system")
            self.assertIn("GUIDANCE TEXT", sent[0]["content"])

    def test_before_agent_start_augmentation_survives_turn_start_skill(self):
        # A before_agent_start prompt change must not be clobbered when a
        # turn_start handler also injects a skill in the same turn.
        with tempfile.TemporaryDirectory() as tmp:
            skills_dir = Path(tmp) / "extensions" / "ext" / "skills"
            skills_dir.mkdir(parents=True)
            (skills_dir / "guide.md").write_text("GUIDANCE TEXT")

            agent = _make_agent([[{"content": "a"}]], user_dir=tmp)

            @agent.bus.on("before_agent_start", extension="bas")
            def augment(payload, ctx):
                return {"system_prompt": payload["system_prompt"] + "\n\nEXTRA"}

            @agent.bus.on("turn_start", extension="ext")
            def declare(payload, ctx):
                ctx.add_skill("guide")

            list(agent.run("first"))
            content = agent.messages[0]["content"]
            self.assertIn("EXTRA", content)
            self.assertIn("GUIDANCE TEXT", content)


# --- manager (/extension commands) ---


class TestManager(unittest.TestCase):
    def _agent(self, tmp: str) -> Agent:
        config = AppConfig(user_dir=tmp, tools_enabled=True)
        agent = Agent(config=config, tool_registry=create_default_tool_registry(config))
        return agent

    def test_list_empty(self):
        from pyagent.extensions.manager import handle_extension_command
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(tmp)
            out = handle_extension_command(agent, [])
            self.assertIn("No extensions", out)

    def test_load_unload_reload_new(self):
        from pyagent.extensions.manager import handle_extension_command
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(tmp)
            # new
            out = handle_extension_command(agent, ["new", "demo"])
            self.assertIn("Created", out)
            self.assertTrue((Path(tmp) / "extensions" / "demo" / "__init__.py").exists())
            self.assertTrue((Path(tmp) / "extensions" / "demo" / "tools" / "demo.py").exists())
            # load
            out = handle_extension_command(agent, ["load", "demo"])
            self.assertIn("Loaded", out)
            self.assertIn("demo", agent.bus.loaded_extensions())
            # list
            out = handle_extension_command(agent, ["list"])
            self.assertIn("demo", out)
            self.assertIn("loaded", out)
            # unload
            out = handle_extension_command(agent, ["unload", "demo"])
            self.assertIn("Unloaded", out)
            self.assertNotIn("demo", agent.bus.loaded_extensions())
            # reload scans and loads all on disk
            handle_extension_command(agent, ["reload"])
            self.assertIn("demo", agent.bus.loaded_extensions())

    def test_unload_not_loaded(self):
        from pyagent.extensions.manager import handle_extension_command
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(tmp)
            out = handle_extension_command(agent, ["unload", "ghost"])
            self.assertIn("not loaded", out)

    def test_new_rejects_bad_name(self):
        from pyagent.extensions.manager import handle_extension_command
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(tmp)
            out = handle_extension_command(agent, ["new", "1bad"])
            self.assertIn("Tool name", out)


if __name__ == "__main__":
    unittest.main()
