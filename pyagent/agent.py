from __future__ import annotations

import json
from typing import Any

from .config import AppConfig, SYSTEM_PROMPT
from .external_tools import (
    DiscoveryResult,
    build_external_tool_specs,
    default_runner_command,
    discover_external_tools,
)
from .llm_client import build_chat_client
from .model_profiles import ModelProfile, ProfileStore, load_profile_store, save_profile_store, update_profile_store
from .tools import ToolRegistry, create_default_tool_registry


class Agent:
    def __init__(
        self,
        model: str | None = None,
        profile: str | None = None,
        config: AppConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        project_context: str = "",
        project_context_files: list[str] | None = None,
        external_tool_discovery: DiscoveryResult | None = None,
    ):
        self.config = config or AppConfig.from_env()
        self.profile_store: ProfileStore = load_profile_store(
            self.config.model_profiles_path
        )
        self.active_profile_name = profile or self.config.default_profile or self.profile_store.default_profile
        self.model_override = model.strip() if model else None
        self.external_tool_discovery: DiscoveryResult | None = external_tool_discovery
        if tool_registry is None:
            self.external_tool_discovery = (
                external_tool_discovery
                if external_tool_discovery is not None
                else self._discover_external_tools()
            )
            self.tool_registry = create_default_tool_registry(
                self.config,
                external_specs=self._external_specs_from_discovery(self.external_tool_discovery),
            )
        else:
            self.tool_registry = tool_registry
        self.tools = self.tool_registry.definitions() if self.config.tools_enabled else []
        self.project_context = project_context.strip()
        self.project_context_files = list(project_context_files or [])
        self._rebuild_client()
        self.reset()

    def _discover_external_tools(self) -> DiscoveryResult | None:
        if not self.config.user_tools_enabled:
            return None
        try:
            return discover_external_tools(
                user_dir=self.config.user_dir,
                runner=self.config.tool_runner,
                describe_timeout=self.config.user_tool_describe_timeout,
            )
        except Exception:
            return None

    def _external_specs_from_discovery(self, discovery: DiscoveryResult | None):
        if discovery is None:
            return None
        return build_external_tool_specs(
            discovery,
            invoke_timeout=self.config.user_tool_timeout,
            runner_command=default_runner_command(discovery.runner),
        )

    def reload_external_tools(self) -> DiscoveryResult | None:
        """Re-scan ``~/.pyagent/tools/`` and rebuild the tool registry.

        Tool calling state is preserved (`tools_enabled`, system prompt
        composition); only the registry contents change. The conversation
        is reset so the model sees a clean tool surface.
        """
        discovery = self._discover_external_tools()
        self.external_tool_discovery = discovery
        self.tool_registry = create_default_tool_registry(
            self.config,
            external_specs=self._external_specs_from_discovery(discovery),
        )
        self.tools = self.tool_registry.definitions() if self.config.tools_enabled else []
        self.reset()
        return discovery

    def _system_prompt(self) -> str:
        prompt = SYSTEM_PROMPT
        if not self.config.tools_enabled:
            prompt += (
                "\n\nTool calling is disabled for this session. Do not call tools. "
                "Answer using only the conversation and your built-in knowledge."
            )
        if not self.project_context:
            return prompt
        return f"{prompt}\n\n{self.project_context}"

    def _rebuild_client(self) -> None:
        existing_client = getattr(self, "client", None)
        if existing_client is not None:
            close = getattr(existing_client, "close", None)
            if callable(close):
                close()

        profile = self.profile_store.get(self.active_profile_name)
        if self.model_override:
            profile = ModelProfile(
                name=profile.name,
                provider=profile.provider,
                model=self.model_override,
                base_url=profile.base_url,
                api_key=profile.api_key,
                api_key_env=profile.api_key_env,
                headers=dict(profile.headers),
            )
        self.client = build_chat_client(
            profile, timeout=self.config.request_timeout)

    def current_profile(self) -> ModelProfile:
        return self.client.profile

    def profile_names(self) -> list[str]:
        return self.profile_store.names()

    def set_project_context(self, project_context: str, files: list[str] | None = None) -> None:
        self.project_context = project_context.strip()
        self.project_context_files = list(files or [])
        self.reset()

    def set_profile(self, profile: str) -> None:
        self.active_profile_name = profile.strip()
        self.model_override = None
        self._rebuild_client()

    def reload_profiles(self) -> None:
        current_profile_name = self.active_profile_name
        self.profile_store = load_profile_store(
            self.config.model_profiles_path)
        if current_profile_name not in self.profile_store.profiles:
            self.active_profile_name = self.profile_store.default_profile
            self.model_override = None
        self._rebuild_client()

    def save_profile(self, profile: ModelProfile, make_default: bool = False) -> None:
        update_profile_store(self.profile_store, profile,
                             make_default=make_default)
        save_profile_store(self.profile_store)
        self.profile_store = load_profile_store(self.profile_store.path)

    def set_model(self, model: str) -> None:
        self.model_override = model.strip()
        self._rebuild_client()

    def set_tools_enabled(self, enabled: bool) -> None:
        self.config.tools_enabled = enabled
        self.tools = self.tool_registry.definitions() if enabled else []
        self.reset()

    def available_models(self) -> tuple[list[str], str | None]:
        response = self.client.list_models()
        if "error" in response:
            return [], str(response["error"])

        names = [name for name in response.get(
            "models", []) if isinstance(name, str) and name]
        return names, None

    def reset(self) -> None:
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()}
        ]

    def add_message(self, role: str, content: str, **extra: Any) -> None:
        message = {"role": role, "content": content}
        message.update(extra)
        self.messages.append(message)
        self._trim_history()

    def _trim_history(self) -> None:
        max_history = self.config.max_history_messages
        if len(self.messages) <= max_history + 1:
            return

        system_message = self.messages[0]
        blocks: list[list[dict[str, Any]]] = []
        index = 1
        while index < len(self.messages):
            message = self.messages[index]
            role = message.get("role")

            if role == "tool":
                index += 1
                continue

            if role == "assistant" and message.get("tool_calls"):
                block = [message]
                index += 1
                while index < len(self.messages) and self.messages[index].get("role") == "tool":
                    block.append(self.messages[index])
                    index += 1
                blocks.append(block)
                continue

            blocks.append([message])
            index += 1

        kept_blocks: list[list[dict[str, Any]]] = []
        kept_count = 0
        for block in reversed(blocks):
            block_size = len(block)
            if kept_blocks and kept_count + block_size > max_history:
                break
            kept_blocks.append(block)
            kept_count += block_size
            if kept_count >= max_history:
                break

        trimmed_messages = [message for block in reversed(
            kept_blocks) for message in block]
        self.messages = [system_message, *trimmed_messages]

    def _normalize_arguments(self, arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                return parsed if isinstance(parsed, dict) else {"value": parsed}
            except json.JSONDecodeError:
                return {}
        return {}

    def _normalize_tool_call(
        self,
        tool_call: dict[str, Any],
        iteration: int,
        index: int,
    ) -> tuple[str, dict[str, Any], str]:
        function = tool_call.get("function", {})
        name = function.get("name", "")
        arguments = self._normalize_arguments(function.get("arguments", {}))
        tool_call_id = tool_call.get("id") or str(
            function.get("index", f"call_{iteration}_{index}")
        )
        return name, arguments, tool_call_id

    def _merge_tool_calls(
        self,
        existing: list[dict[str, Any]],
        incoming: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        ordered_keys: list[str] = []

        for tool_call in [*existing, *incoming]:
            function = tool_call.get("function", {})
            key = str(
                tool_call.get("id")
                or function.get("index")
                or f"{function.get('name', '')}:{json.dumps(function.get('arguments', {}), sort_keys=True, default=str)}"
            )
            if key not in merged:
                merged[key] = {
                    **tool_call,
                    "function": {
                        **function,
                        "arguments": function.get("arguments", {}),
                    },
                }
                ordered_keys.append(key)
                continue

            current = merged[key]
            current_function = current.setdefault("function", {})
            if function.get("name"):
                current_function["name"] = function["name"]

            incoming_arguments = function.get("arguments", {})
            existing_arguments = current_function.get("arguments", {})
            if isinstance(existing_arguments, dict) and isinstance(incoming_arguments, dict):
                current_function["arguments"] = {
                    **existing_arguments,
                    **incoming_arguments,
                }
            elif isinstance(existing_arguments, str) and isinstance(incoming_arguments, str):
                current_function["arguments"] = existing_arguments + \
                    incoming_arguments
            elif incoming_arguments:
                current_function["arguments"] = incoming_arguments

        return [merged[key] for key in ordered_keys]

    def _format_tool_results_for_fallback(self, tool_results: list[dict[str, Any]]) -> str:
        if not tool_results:
            return ""

        if len(tool_results) == 1:
            result = str(tool_results[0]["result"]).strip()
            return f"\n\n```text\n{result}\n```" if result else ""

        blocks: list[str] = []
        for tool_result in tool_results:
            result = str(tool_result["result"]).strip()
            if not result:
                continue
            blocks.append(
                f"**{tool_result['name']}**\n\n```text\n{result}\n```")
        if not blocks:
            return ""
        return "\n\nHere are the tool results:\n\n" + "\n\n".join(blocks)

    def _should_append_tool_results(self, content: str, tool_results: list[dict[str, Any]]) -> bool:
        if not tool_results:
            return False

        stripped = content.strip()
        if not stripped:
            return True

        return stripped.endswith(":")

    def run(self, user_input: str):
        self.add_message("user", user_input)
        yield {"type": "assistant_start"}
        recent_tool_results: list[dict[str, Any]] = []

        iteration = 0
        while self.config.max_iterations < 0 or iteration < self.config.max_iterations:
            full_response_parts: list[str] = []
            full_tool_calls: list[dict[str, Any]] = []
            yield {
                "type": "debug",
                "label": "iteration_start",
                "data": {"iteration": iteration + 1, "message_count": len(self.messages)},
            }

            for chunk in self.client.chat_stream(
                self.messages,
                tools=self.tools if self.config.tools_enabled else None,
            ):
                yield {"type": "debug", "label": "llm_chunk", "data": chunk}
                if "error" in chunk:
                    error_message = f"API Error: {chunk['error']}"
                    yield {"type": "error", "message": error_message}
                    return

                content = chunk.get("content") or ""
                if content:
                    full_response_parts.append(content)
                    yield {"type": "content_delta", "delta": content}

                tool_calls = chunk.get("tool_calls") or []
                if tool_calls:
                    full_tool_calls = self._merge_tool_calls(
                        full_tool_calls, tool_calls)
                    yield {
                        "type": "debug",
                        "label": "merged_tool_calls",
                        "data": full_tool_calls,
                    }

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(full_response_parts),
            }
            if full_tool_calls:
                assistant_message["tool_calls"] = full_tool_calls
            self.messages.append(assistant_message)
            self._trim_history()

            if not full_tool_calls:
                if self._should_append_tool_results(
                    assistant_message["content"], recent_tool_results
                ):
                    suffix = self._format_tool_results_for_fallback(
                        recent_tool_results)
                    if suffix:
                        assistant_message["content"] += suffix
                        self.messages[-1]["content"] = assistant_message["content"]
                        yield {
                            "type": "debug",
                            "label": "assistant_tool_result_fallback",
                            "data": {"tool_count": len(recent_tool_results)},
                        }
                        yield {"type": "content_delta", "delta": suffix}
                yield {
                    "type": "debug",
                    "label": "assistant_message_complete",
                    "data": assistant_message,
                }
                yield {
                    "type": "assistant_done",
                    "content": assistant_message["content"],
                }
                return

            for index, tool_call in enumerate(full_tool_calls):
                name, arguments, tool_call_id = self._normalize_tool_call(
                    tool_call,
                    iteration,
                    index,
                )
                if not name:
                    result = "Error: received malformed tool call from model."
                    self.messages.append(
                        {
                            "role": "tool",
                            "name": "<invalid>",
                            "content": result,
                            "tool_call_id": tool_call_id,
                        }
                    )
                    recent_tool_results.append(
                        {"name": "<invalid>", "arguments": {}, "result": result}
                    )
                    yield {"type": "tool_result", "name": "<invalid>", "result": result}
                    continue

                yield {"type": "tool_call", "name": name, "arguments": arguments}
                result = self.tool_registry.execute(name, arguments)
                yield {
                    "type": "debug",
                    "label": "tool_execution",
                    "data": {
                        "name": name,
                        "arguments": arguments,
                        "result_preview": result[:1000],
                    },
                }
                self.messages.append(
                    {
                        "role": "tool",
                        "name": name,
                        "content": result,
                        "tool_call_id": tool_call_id,
                    }
                )
                recent_tool_results.append(
                    {"name": name, "arguments": arguments, "result": result}
                )
                self._trim_history()
                yield {"type": "tool_result", "name": name, "result": result}

            iteration += 1

        warning = "Reached maximum iterations without a final answer."
        self.messages.append({"role": "assistant", "content": warning})
        yield {"type": "content_delta", "delta": f"\n\n{warning}"}
        yield {"type": "assistant_done", "content": warning}
