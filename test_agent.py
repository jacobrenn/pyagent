from __future__ import annotations

import json
import os
import tempfile
import unittest

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


class DummyClient:
    def __init__(self, responses: list[list[dict]]):
        self.responses = responses
        self.calls = 0

    def chat_stream(self, messages, tools=None):
        response = self.responses[self.calls]
        self.calls += 1
        for chunk in response:
            yield chunk


class ModelListClient:
    def __init__(self, payload: dict[str, object]):
        self.payload = payload

    def list_models(self):
        return self.payload


class FakeStreamResponse:
    def __init__(self, lines: list[str], json_payload: dict | None = None):
        self.lines = lines
        self.json_payload = json_payload or {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode: bool = True):
        for line in self.lines:
            yield line

    def json(self):
        return self.json_payload


class FakeSession:
    def __init__(self, post_response: FakeStreamResponse | None = None, get_response: FakeStreamResponse | None = None):
        self.post_response = post_response
        self.get_response = get_response
        self.last_post = None
        self.last_get = None

    def post(self, url, **kwargs):
        self.last_post = {"url": url, **kwargs}
        if self.post_response is None:
            raise AssertionError("Unexpected POST request")
        return self.post_response

    def get(self, url, **kwargs):
        self.last_get = {"url": url, **kwargs}
        if self.get_response is None:
            raise AssertionError("Unexpected GET request")
        return self.get_response


class AgentTests(unittest.TestCase):
    def test_appends_tool_result_when_model_stops_after_intro(self) -> None:
        config = AppConfig(max_iterations=3)
        agent = Agent(config=config, tool_registry=create_default_tool_registry(config))
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

        events = list(agent.run("What files are available in the current working directory?"))
        assistant_deltas = "".join(
            event.get("delta", "") for event in events if event.get("type") == "content_delta"
        )

        self.assertIn(
            "The files and directories in the current working directory are:",
            assistant_deltas,
        )
        self.assertIn("```text", assistant_deltas)
        self.assertIn("README.md", assistant_deltas)

    def test_set_model_updates_agent_and_client(self) -> None:
        agent = Agent(config=AppConfig(), tool_registry=create_default_tool_registry(AppConfig()))

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
                tool_registry=create_default_tool_registry(AppConfig(model_profiles_path=profile_path)),
            )
            agent.set_profile("remote")

        self.assertEqual(agent.current_profile().name, "remote")
        self.assertEqual(agent.current_profile().provider, "openai_compatible")
        self.assertEqual(agent.client.model, "gpt-4.1-mini")

    def test_available_models_uses_normalized_client_response(self) -> None:
        agent = Agent(config=AppConfig(), tool_registry=create_default_tool_registry(AppConfig()))
        agent.client = ModelListClient({"models": ["llama3.1", "qwen2.5-coder:7b"]})

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
            agent = Agent(config=config, tool_registry=create_default_tool_registry(config))

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
        self.assertIsInstance(build_chat_client(openai_profile), OpenAICompatibleClient)

    def test_openai_compatible_stream_assembles_tool_calls(self) -> None:
        profile = ModelProfile(
            name="remote",
            provider="openai_compatible",
            model="gpt-4.1-mini",
            base_url="https://example.com/v1",
            api_key="secret",
        )
        client = OpenAICompatibleClient(profile=profile)
        client.session = FakeSession(
            post_response=FakeStreamResponse(
                [
                    'data: {"choices":[{"delta":{"content":"Hello"}}]}',
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"search_text","arguments":"{\\"query\\":"}}]}}]}',
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"PyAgent\\"}"}}]}}]}',
                    'data: [DONE]',
                ]
            )
        )

        chunks = list(client.chat_stream(messages=[], tools=[]))

        self.assertEqual(chunks[0], {"content": "Hello"})
        self.assertEqual(chunks[1]["tool_calls"][0]["function"]["name"], "search_text")
        self.assertEqual(
            chunks[1]["tool_calls"][0]["function"]["arguments"],
            '{"query":"PyAgent"}',
        )

    def test_openai_compatible_list_models_parses_response(self) -> None:
        profile = ModelProfile(
            name="remote",
            provider="openai_compatible",
            model="gpt-4.1-mini",
            base_url="https://example.com/v1",
            api_key="secret",
        )
        client = OpenAICompatibleClient(profile=profile)
        client.session = FakeSession(
            get_response=FakeStreamResponse(
                [],
                json_payload={"data": [{"id": "gpt-4.1"}, {"id": "local-model"}]},
            )
        )

        payload = client.list_models()

        self.assertEqual(payload, {"models": ["gpt-4.1", "local-model"]})


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
            open(os.path.join(temp_dir, "alpha.py"), "w", encoding="utf-8").close()
            open(os.path.join(temp_dir, "beta.txt"), "w", encoding="utf-8").close()

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
            os.makedirs(os.path.join(base_path, "skills", "python"), exist_ok=True)
            with open(os.path.join(base_path, "AGENTS.md"), "w", encoding="utf-8") as file:
                file.write("project rules")
            with open(os.path.join(base_path, "skills", "python", "testing.md"), "w", encoding="utf-8") as file:
                file.write("testing skill")
            with open(os.path.join(base_path, "local.skill"), "w", encoding="utf-8") as file:
                file.write("local skill")

            files = [path.relative_to(base_path).as_posix() for path in discover_project_instruction_files(base_path)]

        self.assertEqual(files, ["AGENTS.md", "local.skill", "skills/python/testing.md"])

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
        self.assertIn("Always run tests after edits.", agent.messages[0]["content"])


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
