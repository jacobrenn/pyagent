from __future__ import annotations

import json
from typing import Any, Iterable

import requests

from .model_profiles import ModelProfile


class BaseChatClient:
    def __init__(self, profile: ModelProfile, timeout: int = 300):
        self.profile = profile
        self.model = profile.model
        self.base_url = profile.base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.profile.headers}
        api_key = self.profile.resolved_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterable[dict[str, Any]]:
        raise NotImplementedError

    def list_models(self) -> dict[str, Any]:
        raise NotImplementedError


class OllamaClient(BaseChatClient):
    def __init__(self, profile: ModelProfile, timeout: int = 300):
        super().__init__(profile=profile, timeout=timeout)
        self.api_url = f"{self.base_url}/api/chat"
        self.tags_url = f"{self.base_url}/api/tags"

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for message in messages:
            item = {
                "role": message.get("role"),
                "content": message.get("content", ""),
            }
            tool_calls = message.get("tool_calls")
            if tool_calls:
                item["tool_calls"] = tool_calls
            if message.get("role") == "tool":
                tool_name = message.get("name") or message.get("tool_name")
                if tool_name:
                    item["tool_name"] = tool_name
                if message.get("tool_call_id"):
                    item["tool_call_id"] = message["tool_call_id"]
            prepared.append(item)
        return prepared

    def _payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._prepare_messages(messages),
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        return payload

    def list_models(self) -> dict[str, Any]:
        try:
            response = self.session.get(self.tags_url, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            return {"error": str(exc)}
        except ValueError as exc:
            return {"error": f"Invalid JSON response from Ollama: {exc}"}

        names: list[str] = []
        for model in payload.get("models", []):
            if not isinstance(model, dict):
                continue
            name = model.get("model") or model.get("name")
            if isinstance(name, str) and name:
                names.append(name)
        return {"models": names}

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterable[dict[str, Any]]:
        payload = self._payload(messages, tools, stream=True)
        collected_tool_calls: list[dict[str, Any]] = []
        try:
            with self.session.post(
                self.api_url,
                json=payload,
                stream=True,
                timeout=self.timeout,
                headers=self._headers(),
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError as exc:
                        yield {"error": f"Could not decode Ollama stream chunk: {exc}: {line!r}"}
                        return

                    message = chunk.get("message", {})
                    content = message.get("content") or ""
                    if content:
                        yield {"content": content}

                    tool_calls = message.get("tool_calls") or []
                    if tool_calls:
                        collected_tool_calls = _merge_tool_call_fragments(
                            collected_tool_calls,
                            tool_calls,
                        )

                if collected_tool_calls:
                    yield {"tool_calls": collected_tool_calls}
        except ValueError as exc:
            yield {"error": str(exc)}
        except requests.RequestException as exc:
            yield {"error": str(exc)}


class OpenAICompatibleClient(BaseChatClient):
    def __init__(self, profile: ModelProfile, timeout: int = 300):
        super().__init__(profile=profile, timeout=timeout)
        self.chat_url = f"{self.base_url}/chat/completions"
        self.models_url = f"{self.base_url}/models"

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for message in messages:
            item = {
                "role": message.get("role"),
                "content": message.get("content", ""),
            }
            if message.get("role") == "assistant" and message.get("tool_calls"):
                item["tool_calls"] = message["tool_calls"]
            elif message.get("role") == "tool":
                item["tool_call_id"] = message.get("tool_call_id")
                name = message.get("name") or message.get("tool_name")
                if name:
                    item["name"] = name
            prepared.append(item)
        return prepared

    def list_models(self) -> dict[str, Any]:
        try:
            response = self.session.get(
                self.models_url,
                timeout=self.timeout,
                headers=self._headers(),
            )
            response.raise_for_status()
            payload = response.json()
        except ValueError as exc:
            return {"error": str(exc)}
        except requests.RequestException as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": str(exc)}

        names: list[str] = []
        for model in payload.get("data", []):
            if not isinstance(model, dict):
                continue
            name = model.get("id") or model.get("name")
            if isinstance(name, str) and name:
                names.append(name)
        return {"models": names}

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterable[dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._prepare_messages(messages),
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        tool_call_fragments: dict[int, dict[str, Any]] = {}
        try:
            with self.session.post(
                self.chat_url,
                json=payload,
                stream=True,
                timeout=self.timeout,
                headers=self._headers(),
            ) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError as exc:
                        yield {"error": f"Could not decode OpenAI-compatible stream chunk: {exc}: {data!r}"}
                        return

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = _extract_openai_content(delta.get("content"))
                    if content:
                        yield {"content": content}

                    for tool_call in delta.get("tool_calls") or []:
                        index = int(tool_call.get("index", 0))
                        current = tool_call_fragments.setdefault(
                            index,
                            {
                                "id": tool_call.get("id") or f"call_{index}",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if tool_call.get("id"):
                            current["id"] = tool_call["id"]
                        function = tool_call.get("function") or {}
                        if function.get("name"):
                            current["function"]["name"] += str(function["name"])
                        if function.get("arguments"):
                            current["function"]["arguments"] += str(function["arguments"])

                if tool_call_fragments:
                    yield {
                        "tool_calls": [
                            tool_call_fragments[index]
                            for index in sorted(tool_call_fragments)
                        ]
                    }
        except ValueError as exc:
            yield {"error": str(exc)}
        except requests.RequestException as exc:
            yield {"error": str(exc)}


def build_chat_client(profile: ModelProfile, timeout: int = 300) -> BaseChatClient:
    provider = profile.resolved_provider()
    if provider == "ollama":
        return OllamaClient(profile=profile, timeout=timeout)
    if provider == "openai_compatible":
        return OpenAICompatibleClient(profile=profile, timeout=timeout)
    raise ValueError(f"Unsupported provider '{profile.provider}'")


def _extract_openai_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _merge_tool_call_fragments(
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
            existing_name = str(current_function.get("name", ""))
            incoming_name = str(function["name"])
            current_function["name"] = incoming_name if existing_name == incoming_name else existing_name or incoming_name

        existing_arguments = current_function.get("arguments", {})
        incoming_arguments = function.get("arguments", {})
        if isinstance(existing_arguments, dict) and isinstance(incoming_arguments, dict):
            current_function["arguments"] = {
                **existing_arguments,
                **incoming_arguments,
            }
        elif isinstance(existing_arguments, str) and isinstance(incoming_arguments, str):
            current_function["arguments"] = existing_arguments + incoming_arguments
        elif incoming_arguments:
            current_function["arguments"] = incoming_arguments

    return [merged[key] for key in ordered_keys]
