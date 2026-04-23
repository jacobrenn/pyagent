from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from pyagent.agent import Agent
from pyagent.config import AppConfig
from pyagent.llm_client import OpenAICompatibleClient, OllamaClient, build_chat_client
from pyagent.model_profiles import (
    ModelProfile,
    load_profile_store,
    save_profile_store,
    update_profile_store,
)
from pyagent.project_context import discover_project_instruction_files, load_project_context
from pyagent.tools import (
    append_file,
    bash,
    create_default_tool_registry,
    edit_file,
    find_files,
    list_files,
    search_text,
)
from pyagent.ui import PyAgentApp


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
        self.assertEqual(events[-1], {"type": "assistant_done", "content": "Finished"})

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
            profile_path = os.path.join(temp_dir, "models.json")
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
            profile_path = os.path.join(temp_dir, "models.json")
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
            profile_path = os.path.join(temp_dir, "models.json")
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
            path = os.path.join(temp_dir, "models.json")
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
            path = os.path.join(temp_dir, "models.json")
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
