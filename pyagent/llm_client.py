from __future__ import annotations

import json
from typing import Any, Iterable

import openai
import requests
from openai import OpenAI

from .model_profiles import ModelProfile


class BaseChatClient:
    def __init__(self, profile: ModelProfile, timeout: int = 300):
        self.profile = profile
        self.model = profile.model
        self.base_url = profile.base_url.rstrip("/")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.profile.headers}
        api_key = self.profile.resolved_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def close(self) -> None:
        return None

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
        self.session = requests.Session()

    def close(self) -> None:
        self.session.close()

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
        self.models_url = f"{self.base_url}/models"
        self._client_factory = OpenAI
        self._sdk_client: OpenAI | None = None

    def close(self) -> None:
        if self._sdk_client is not None:
            self._sdk_client.close()
            self._sdk_client = None

    def _resolved_api_key_for_sdk(self) -> str:
        api_key = self.profile.resolved_api_key()
        return api_key if api_key is not None else ""

    def _get_client(self) -> OpenAI:
        if self._sdk_client is None:
            self._sdk_client = self._client_factory(
                api_key=self._resolved_api_key_for_sdk(),
                base_url=self.base_url,
                default_headers=self.profile.headers or None,
                timeout=float(self.timeout),
                max_retries=2,
            )
        return self._sdk_client

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
            payload = self._get_client().models.list()
        except ValueError as exc:
            return {"error": str(exc)}
        except openai.APIError as exc:
            return {"error": _format_openai_error(exc)}
        except Exception as exc:
            return {"error": str(exc)}

        names: list[str] = []
        for model in getattr(payload, "data", []) or []:
            name = getattr(model, "id", None) or getattr(model, "name", None)
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
            stream = self._get_client().chat.completions.create(**payload)
            for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue

                content = _extract_openai_content(getattr(delta, "content", None))
                if content:
                    yield {"content": content}

                for tool_call in getattr(delta, "tool_calls", None) or []:
                    index = int(getattr(tool_call, "index", 0) or 0)
                    current = tool_call_fragments.setdefault(
                        index,
                        {
                            "id": getattr(tool_call, "id", None) or f"call_{index}",
                            "type": getattr(tool_call, "type", None) or "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    tool_call_id = getattr(tool_call, "id", None)
                    if tool_call_id:
                        current["id"] = tool_call_id
                    tool_type = getattr(tool_call, "type", None)
                    if tool_type:
                        current["type"] = tool_type

                    function = getattr(tool_call, "function", None)
                    if function is None:
                        continue
                    function_name = getattr(function, "name", None)
                    if function_name:
                        current["function"]["name"] += str(function_name)
                    function_arguments = getattr(function, "arguments", None)
                    if function_arguments:
                        current["function"]["arguments"] += str(function_arguments)

            if tool_call_fragments:
                yield {
                    "tool_calls": [
                        tool_call_fragments[index]
                        for index in sorted(tool_call_fragments)
                    ]
                }
        except ValueError as exc:
            yield {"error": str(exc)}
        except openai.APIError as exc:
            yield {"error": _format_openai_error(exc)}
        except Exception as exc:
            yield {"error": str(exc)}


def build_chat_client(profile: ModelProfile, timeout: int = 300) -> BaseChatClient:
    provider = profile.resolved_provider()
    if provider == "ollama":
        return OllamaClient(profile=profile, timeout=timeout)
    if provider == "openai_compatible":
        return OpenAICompatibleClient(profile=profile, timeout=timeout)
    raise ValueError(f"Unsupported provider '{profile.provider}'")



def _format_openai_error(exc: openai.APIError) -> str:
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", None)
        request_id = getattr(exc, "request_id", None)
        parts = ["OpenAI-compatible API error"]
        if status is not None:
            parts.append(f"status={status}")
        if request_id:
            parts.append(f"request_id={request_id}")
        message = str(exc).strip()
        return ": ".join([" ".join(parts), message]) if message else " ".join(parts)
    if isinstance(exc, openai.APIConnectionError):
        return f"OpenAI-compatible connection error: {exc}"
    if isinstance(exc, openai.APITimeoutError):
        return f"OpenAI-compatible timeout error: {exc}"
    return f"OpenAI-compatible API error: {exc}"



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
                continue
            item_type = getattr(item, "type", None)
            text = getattr(item, "text", None)
            if item_type == "text" and isinstance(text, str):
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
