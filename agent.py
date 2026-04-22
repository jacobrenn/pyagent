from __future__ import annotations

import json
from typing import Any

from config import AppConfig, SYSTEM_PROMPT
from ollama_client import OllamaClient
from tools import ToolRegistry, create_default_tool_registry


class Agent:
    def __init__(
        self,
        model: str | None = None,
        config: AppConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        project_context: str = "",
        project_context_files: list[str] | None = None,
    ):
        self.config = config or AppConfig.from_env()
        if model is not None:
            self.config.model = model

        self.client = OllamaClient(
            model=self.config.model,
            base_url=self.config.base_url,
            timeout=self.config.request_timeout,
        )
        self.tool_registry = tool_registry or create_default_tool_registry(
            self.config)
        self.tools = self.tool_registry.definitions()
        self.project_context = project_context.strip()
        self.project_context_files = list(project_context_files or [])
        self.reset()

    def _system_prompt(self) -> str:
        if not self.project_context:
            return SYSTEM_PROMPT
        return f"{SYSTEM_PROMPT}\n\n{self.project_context}"

    def set_project_context(self, project_context: str, files: list[str] | None = None) -> None:
        self.project_context = project_context.strip()
        self.project_context_files = list(files or [])
        self.reset()

    def set_model(self, model: str) -> None:
        self.config.model = model.strip()
        self.client = OllamaClient(
            model=self.config.model,
            base_url=self.config.base_url,
            timeout=self.config.request_timeout,
        )

    def available_models(self) -> tuple[list[str], str | None]:
        response = self.client.list_models()
        if "error" in response:
            return [], str(response["error"])

        names: list[str] = []
        for model in response.get("models", []):
            if not isinstance(model, dict):
                continue
            name = model.get("model") or model.get("name")
            if isinstance(name, str) and name:
                names.append(name)
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
        self.messages = [self.messages[0], *self.messages[-max_history:]]

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

    def _normalize_tool_call(self, tool_call: dict[str, Any], iteration: int, index: int) -> tuple[str, dict[str, Any], str]:
        function = tool_call.get("function", {})
        name = function.get("name", "")
        arguments = self._normalize_arguments(function.get("arguments", {}))
        tool_call_id = tool_call.get("id") or str(
            function.get("index", f"call_{iteration}_{index}"))
        return name, arguments, tool_call_id

    def _merge_tool_calls(self, existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        ordered_keys: list[str] = []

        for tool_call in [*existing, *incoming]:
            function = tool_call.get("function", {})
            key = str(tool_call.get("id") or function.get(
                "index") or f"{function.get('name', '')}:{json.dumps(function.get('arguments', {}), sort_keys=True, default=str)}")
            if key not in merged:
                merged[key] = {
                    **tool_call,
                    "function": {
                        **function,
                        "arguments": self._normalize_arguments(function.get("arguments", {})),
                    },
                }
                ordered_keys.append(key)
                continue

            current = merged[key]
            current_function = current.setdefault("function", {})
            if function.get("name"):
                current_function["name"] = function["name"]

            incoming_arguments = self._normalize_arguments(
                function.get("arguments", {}))
            existing_arguments = current_function.get("arguments", {})
            if isinstance(existing_arguments, dict) and isinstance(incoming_arguments, dict):
                current_function["arguments"] = {
                    **existing_arguments, **incoming_arguments}
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

        for iteration in range(self.config.max_iterations):
            full_response_parts: list[str] = []
            full_tool_calls: list[dict[str, Any]] = []
            yield {"type": "debug", "label": "iteration_start", "data": {"iteration": iteration + 1, "message_count": len(self.messages)}}

            for chunk in self.client.chat_stream(self.messages, tools=self.tools):
                yield {"type": "debug", "label": "ollama_chunk", "data": chunk}
                if "error" in chunk:
                    error_message = f"API Error: {chunk['error']}"
                    yield {"type": "error", "message": error_message}
                    return

                message = chunk.get("message", {})
                content = message.get("content") or ""
                if content:
                    full_response_parts.append(content)
                    yield {"type": "content_delta", "delta": content}

                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    full_tool_calls = self._merge_tool_calls(
                        full_tool_calls, tool_calls)
                    yield {"type": "debug", "label": "merged_tool_calls", "data": full_tool_calls}

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(full_response_parts),
            }
            if full_tool_calls:
                assistant_message["tool_calls"] = full_tool_calls
            self.messages.append(assistant_message)
            self._trim_history()

            if not full_tool_calls:
                if self._should_append_tool_results(assistant_message["content"], recent_tool_results):
                    suffix = self._format_tool_results_for_fallback(
                        recent_tool_results)
                    if suffix:
                        assistant_message["content"] += suffix
                        self.messages[-1]["content"] = assistant_message["content"]
                        yield {"type": "debug", "label": "assistant_tool_result_fallback", "data": {"tool_count": len(recent_tool_results)}}
                        yield {"type": "content_delta", "delta": suffix}
                yield {"type": "debug", "label": "assistant_message_complete", "data": assistant_message}
                yield {"type": "assistant_done", "content": assistant_message["content"]}
                return

            for index, tool_call in enumerate(full_tool_calls):
                name, arguments, tool_call_id = self._normalize_tool_call(
                    tool_call, iteration, index)
                if not name:
                    result = "Error: received malformed tool call from model."
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_name": "<invalid>",
                            "content": result,
                            "tool_call_id": tool_call_id,
                        }
                    )
                    recent_tool_results.append(
                        {"name": "<invalid>", "arguments": {}, "result": result})
                    yield {"type": "tool_result", "name": "<invalid>", "result": result}
                    continue

                yield {"type": "tool_call", "name": name, "arguments": arguments}
                result = self.tool_registry.execute(name, arguments)
                yield {
                    "type": "debug",
                    "label": "tool_execution",
                    "data": {"name": name, "arguments": arguments, "result_preview": result[:1000]},
                }
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_name": name,
                        "content": result,
                        "tool_call_id": tool_call_id,
                    }
                )
                recent_tool_results.append(
                    {"name": name, "arguments": arguments, "result": result})
                self._trim_history()
                yield {"type": "tool_result", "name": name, "result": result}

        warning = "Reached maximum iterations without a final answer."
        self.messages.append({"role": "assistant", "content": warning})
        yield {"type": "content_delta", "delta": f"\n\n{warning}"}
        yield {"type": "assistant_done", "content": warning}
