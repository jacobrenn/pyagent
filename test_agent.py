from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pyagent.agent import Agent
from pyagent.config import AppConfig
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
    discover_user_global_instruction_files,
    load_full_context,
    load_project_context,
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
    search_text,
)
# ... existing imports ...
from pyagent.ui import ChatMessage, PyAgentApp
from pyagent.user_runtime import RunnerStatus
from pyagent.main import main as main_entry  # Add this import

# ... (all existing test classes) ...


class MainCliTests(unittest.TestCase):
    def test_single_shot_mode_prints_response_and_exits(self) -> None:
        # Mock the Agent to avoid real LLM calls
        with mock.patch("pyagent.main.Agent") as MockAgent:
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

    def test_single_shot_mode_passes_profile_and_model(self) -> None:
        with mock.patch("pyagent.main.Agent") as MockAgent:
            mock_agent_instance = MockAgent.return_value
            mock_agent_instance.run.return_value = [
                {"type": "assistant_done", "content": "Done"}
            ]

            # Simulate: pyagent --profile my-profile --model my-model --prompt "Hi"
            with mock.patch("sys.argv", ["pyagent", "--profile", "my-profile", "--model", "my-model", "--prompt", "Hi"]):
                with mock.patch("sys.stdout"):
                    with self.assertRaises(SystemExit):
                        main_entry()

                    # Verify Agent was instantiated with the correct overrides
                    MockAgent.assert_called_once_with(
                        profile="my-profile", model="my-model")

    def test_interactive_mode_launches_app(self) -> None:
        with mock.patch("pyagent.main.PyAgentApp") as MockApp:
            # Mock the run method of the app
            mock_app_instance = MockApp.return_value

            # Simulate: pyagent (no prompt)
            with mock.patch("sys.argv", ["pyagent"]):
                main_entry()

                MockApp.assert_called_once()
                mock_app_instance.run.assert_called_once()

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
        app.agent.project_context_files = ["AGENTS.md", "skills/testing.md"]

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
    def test_discovers_agents_and_skill_files(self) -> None:
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

            files = [path.relative_to(base_path).as_posix(
            ) for path in discover_project_instruction_files(base_path)]

        self.assertEqual(
            files, ["AGENTS.md", "local.skill", "skills/python/testing.md"])

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
                idx = sys.argv.index("--args-file")
                path = sys.argv[idx + 1]
                with open(path, "r", encoding="utf-8") as f:
                    args = json.load(f)
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
            self.assertIn("Built-in tools:", notes[-1])
            self.assertIn("External tools", notes[-1])

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


if __name__ == "__main__":
    unittest.main()


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
