from __future__ import annotations

import json
import httpx
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
        self._http_client: httpx.Client | None = None

    def close(self) -> None:
        if self._sdk_client is not None:
            self._sdk_client.close()
            self._sdk_client = None
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None

    def _resolved_api_key_for_sdk(self) -> str:
        api_key = self.profile.resolved_api_key()
        return api_key if api_key is not None else ""

    def _client_args(self) -> dict[str, Any]:
        return {
            "api_key": self._resolved_api_key_for_sdk(),
            "base_url": self.base_url,
            "default_headers": self.profile.headers or None,
            "timeout": float(self.timeout),
            "max_retries": 2,
        }

    def _build_http_client(self) -> httpx.Client | None:
        if not self.profile.httpx_kwargs:
            return None
        return httpx.Client(**self.profile.httpx_kwargs)

    def _get_client(self) -> OpenAI:
        if self._sdk_client is None:
            client_args = self._client_args()
            self._http_client = self._build_http_client()
            if self._http_client is not None:
                client_args["http_client"] = self._http_client
            self._sdk_client = self._client_factory(**client_args)
        return self._sdk_client

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        pending_tool_call_ids: set[str] = set()
        pending_tool_results = False

        for message in messages:
            role = message.get("role")
            item = {
                "role": role,
                "content": message.get("content", ""),
            }

            if role == "assistant" and message.get("tool_calls"):
                tool_calls = message["tool_calls"]
                item["tool_calls"] = tool_calls
                prepared.append(item)
                pending_tool_call_ids = {
                    str(tool_call.get("id"))
                    for tool_call in tool_calls
                    if tool_call.get("id")
                }
                pending_tool_results = True
                continue

            if role == "tool":
                if not prepared:
                    continue
                previous = prepared[-1]
                if previous.get("role") not in {"assistant", "tool"}:
                    continue
                tool_call_id = message.get("tool_call_id")
                if pending_tool_call_ids and tool_call_id not in pending_tool_call_ids:
                    continue
                item["tool_call_id"] = tool_call_id
                name = message.get("name") or message.get("tool_name")
                if name:
                    item["name"] = name
                prepared.append(item)
                pending_tool_results = False
                continue

            if pending_tool_results and prepared and prepared[-1].get("role") == "assistant":
                prepared.pop()
            pending_tool_call_ids = set()
            pending_tool_results = False
            prepared.append(item)

        if pending_tool_results and prepared and prepared[-1].get("role") == "assistant":
            prepared.pop()

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

                content = _extract_openai_content(
                    getattr(delta, "content", None))
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
                        current["function"]["arguments"] += str(
                            function_arguments)

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


class OpenAIResponsesClient(OpenAICompatibleClient):
    """OpenAI-compatible client backed by the Responses API.

    PyAgent keeps provider-neutral, chat-shaped history internally. This client
    translates that history to Responses input items and translates streamed
    Responses events back to PyAgent's ``content`` / ``tool_calls`` contract.
    """

    def __init__(self, profile: ModelProfile, timeout: int = 300):
        super().__init__(profile=profile, timeout=timeout)
        self._last_response_output: list[dict[str, Any]] = []
        self._last_response_tool_call_ids: set[str] = set()

    def _cached_continuation_index(
        self,
        messages: list[dict[str, Any]],
    ) -> int | None:
        """Locate an immediate tool continuation matching the last response.

        Reusing raw Responses output items preserves reasoning items during a
        function-call loop without handing conversation ownership to
        ``previous_response_id``. That keeps local history masking and context
        extension behavior effective on every request.
        """
        if not self._last_response_output or not messages:
            return None

        index = len(messages) - 1
        if messages[index].get("role") != "tool":
            return None
        while index >= 0 and messages[index].get("role") == "tool":
            index -= 1
        if index < 0:
            return None

        assistant = messages[index]
        if assistant.get("role") != "assistant" or not assistant.get("tool_calls"):
            return None
        call_ids = {
            str(tool_call.get("id"))
            for tool_call in assistant["tool_calls"]
            if tool_call.get("id")
        }
        if call_ids != self._last_response_tool_call_ids:
            return None
        if _tool_call_signatures(assistant["tool_calls"]) != _tool_call_signatures(
            _tool_calls_from_response_output(self._last_response_output)
        ):
            return None
        if str(assistant.get("content", "")) != _text_from_response_output(
            self._last_response_output
        ):
            return None

        output_ids = {
            str(message.get("tool_call_id"))
            for message in messages[index + 1:]
            if message.get("role") == "tool" and message.get("tool_call_id")
        }
        return index if output_ids == call_ids else None

    def _prepare_input(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        instructions: list[str] = []
        input_items: list[dict[str, Any]] = []
        cached_index = self._cached_continuation_index(messages)

        pending_items: list[dict[str, Any]] = []
        pending_call_ids: set[str] = set()
        pending_output_ids: set[str] = set()

        def flush_pending() -> None:
            nonlocal pending_items, pending_call_ids, pending_output_ids
            if pending_call_ids and pending_call_ids.issubset(pending_output_ids):
                input_items.extend(pending_items)
            pending_items = []
            pending_call_ids = set()
            pending_output_ids = set()

        for index, message in enumerate(messages):
            role = message.get("role")
            content = str(message.get("content", ""))

            if role == "system":
                flush_pending()
                if content:
                    instructions.append(content)
                continue

            if role == "assistant" and message.get("tool_calls"):
                flush_pending()
                if index == cached_index:
                    pending_items = [dict(item) for item in self._last_response_output]
                    pending_call_ids = set(self._last_response_tool_call_ids)
                    continue

                if content:
                    pending_items.append(
                        {"role": "assistant", "content": content})
                for tool_index, tool_call in enumerate(message["tool_calls"]):
                    function = tool_call.get("function") or {}
                    name = str(function.get("name") or "")
                    if not name:
                        continue
                    call_id = str(
                        tool_call.get("id") or f"call_{index}_{tool_index}")
                    pending_call_ids.add(call_id)
                    pending_items.append(
                        {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": name,
                            "arguments": _stringify_function_arguments(
                                function.get("arguments", "{}")),
                        }
                    )
                if not pending_call_ids and pending_items:
                    input_items.extend(pending_items)
                    pending_items = []
                continue

            if role == "tool":
                call_id = str(message.get("tool_call_id") or "")
                if not pending_call_ids or call_id not in pending_call_ids:
                    continue
                pending_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": content,
                    }
                )
                pending_output_ids.add(call_id)
                continue

            flush_pending()
            if role in {"user", "assistant"}:
                input_items.append({"role": role, "content": content})

        flush_pending()
        instruction_text = "\n\n".join(instructions) or None
        return instruction_text, input_items

    def _prepare_tools(
        self,
        tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for tool in tools or []:
            if tool.get("type") != "function":
                continue
            function = tool.get("function")
            if not isinstance(function, dict):
                function = tool
            name = function.get("name")
            if not isinstance(name, str) or not name:
                continue
            item: dict[str, Any] = {
                "type": "function",
                "name": name,
                "parameters": function.get("parameters") or {},
                "strict": bool(function.get("strict", False)),
            }
            description = function.get("description")
            if isinstance(description, str) and description:
                item["description"] = description
            prepared.append(item)
        return prepared

    def _remember_response_output(self, output: list[Any]) -> None:
        serialized = [
            item
            for raw_item in output
            if (item := _response_item_to_dict(raw_item)) is not None
        ]
        call_ids = {
            str(item.get("call_id"))
            for item in serialized
            if item.get("type") == "function_call" and item.get("call_id")
        }
        if call_ids:
            self._last_response_output = serialized
            self._last_response_tool_call_ids = call_ids
        else:
            self._last_response_output = []
            self._last_response_tool_call_ids = set()

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterable[dict[str, Any]]:
        instructions, input_items = self._prepare_input(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "stream": True,
            "store": False,
        }
        if instructions:
            payload["instructions"] = instructions
        prepared_tools = self._prepare_tools(tools)
        if prepared_tools:
            payload["tools"] = prepared_tools

        self._last_response_output = []
        self._last_response_tool_call_ids = set()
        output_items: dict[int, Any] = {}
        completed_output: list[Any] | None = None
        tool_call_fragments: dict[int, dict[str, Any]] = {}

        try:
            stream = self._get_client().responses.create(**payload)
            for event in stream:
                event_type = getattr(event, "type", "")

                if event_type in {"response.output_text.delta", "response.refusal.delta"}:
                    delta = getattr(event, "delta", None)
                    if isinstance(delta, str) and delta:
                        yield {"content": delta}
                    continue

                if event_type in {
                    "response.output_item.added",
                    "response.output_item.done",
                }:
                    output_index = int(getattr(event, "output_index", 0) or 0)
                    item = getattr(event, "item", None)
                    if event_type == "response.output_item.done":
                        output_items[output_index] = item
                    item_payload = _response_item_to_dict(item)
                    if item_payload and item_payload.get("type") == "function_call":
                        tool_call_fragments[output_index] = {
                            "id": str(
                                item_payload.get("call_id")
                                or f"call_{output_index}"),
                            "type": "function",
                            "function": {
                                "name": str(item_payload.get("name") or ""),
                                "arguments": str(
                                    item_payload.get("arguments") or ""),
                            },
                        }
                    continue

                if event_type == "response.function_call_arguments.delta":
                    output_index = int(getattr(event, "output_index", 0) or 0)
                    current = tool_call_fragments.setdefault(
                        output_index,
                        {
                            "id": f"call_{output_index}",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    delta = getattr(event, "delta", None)
                    if delta:
                        current["function"]["arguments"] += str(delta)
                    continue

                if event_type == "response.function_call_arguments.done":
                    output_index = int(getattr(event, "output_index", 0) or 0)
                    current = tool_call_fragments.setdefault(
                        output_index,
                        {
                            "id": f"call_{output_index}",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    name = getattr(event, "name", None)
                    arguments = getattr(event, "arguments", None)
                    if name:
                        current["function"]["name"] = str(name)
                    if arguments is not None:
                        current["function"]["arguments"] = str(arguments)
                    continue

                if event_type == "response.completed":
                    response = getattr(event, "response", None)
                    completed_output = list(getattr(response, "output", None) or [])
                    continue

                if event_type == "error":
                    message = getattr(event, "message", None) or "Responses API stream error"
                    yield {"error": str(message)}
                    return

                if event_type in {"response.failed", "response.incomplete"}:
                    response = getattr(event, "response", None)
                    yield {
                        "error": _format_responses_terminal_error(
                            response,
                            incomplete=event_type == "response.incomplete",
                        )
                    }
                    return

            final_output = completed_output
            if final_output is None:
                final_output = [output_items[index] for index in sorted(output_items)]
            self._remember_response_output(final_output)

            calls_from_output = _tool_calls_from_response_output(final_output)
            tool_calls = calls_from_output or [
                tool_call_fragments[index]
                for index in sorted(tool_call_fragments)
                if tool_call_fragments[index]["function"]["name"]
            ]
            if tool_calls:
                yield {"tool_calls": tool_calls}
        except ValueError as exc:
            yield {"error": str(exc)}
        except openai.APIError as exc:
            yield {"error": _format_openai_error(exc)}
        except Exception as exc:
            yield {"error": str(exc)}


def build_chat_client(profile: ModelProfile, timeout: int = 300) -> BaseChatClient:
    provider = profile.resolved_provider()
    api_mode = profile.resolved_api_mode()
    if provider == "ollama":
        if api_mode != "chat_completions":
            raise ValueError(
                f"API mode '{api_mode}' cannot be used with provider 'ollama'."
            )
        return OllamaClient(profile=profile, timeout=timeout)
    if provider == "openai_compatible":
        if api_mode == "responses":
            return OpenAIResponsesClient(profile=profile, timeout=timeout)
        return OpenAICompatibleClient(profile=profile, timeout=timeout)
    raise ValueError(f"Unsupported provider '{profile.provider}'")


def _stringify_function_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments if arguments is not None else {})
    except (TypeError, ValueError):
        return "{}"


def _response_item_to_dict(item: Any) -> dict[str, Any] | None:
    if isinstance(item, dict):
        return dict(item)
    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        payload = model_dump(mode="json", exclude_none=True)
        return payload if isinstance(payload, dict) else None
    if item is None:
        return None

    item_type = getattr(item, "type", None)
    if not isinstance(item_type, str):
        return None
    payload: dict[str, Any] = {"type": item_type}
    for field_name in (
        "id",
        "call_id",
        "name",
        "arguments",
        "status",
        "role",
        "content",
    ):
        value = getattr(item, field_name, None)
        if value is not None:
            payload[field_name] = value
    return payload


def _tool_calls_from_response_output(output: list[Any]) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for index, raw_item in enumerate(output):
        item = _response_item_to_dict(raw_item)
        if not item or item.get("type") != "function_call":
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        tool_calls.append(
            {
                "id": str(item.get("call_id") or f"call_{index}"),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": _stringify_function_arguments(
                        item.get("arguments", "{}")),
                },
            }
        )
    return tool_calls


def _tool_call_signatures(
    tool_calls: list[dict[str, Any]],
) -> dict[str, tuple[str, str]]:
    signatures: dict[str, tuple[str, str]] = {}
    for tool_call in tool_calls:
        call_id = tool_call.get("id")
        function = tool_call.get("function") or {}
        if not call_id or not isinstance(function, dict):
            continue
        signatures[str(call_id)] = (
            str(function.get("name") or ""),
            _stringify_function_arguments(function.get("arguments", "{}")),
        )
    return signatures


def _text_from_response_output(output: list[Any]) -> str:
    parts: list[str] = []
    for raw_item in output:
        item = _response_item_to_dict(raw_item)
        if not item or item.get("type") != "message":
            continue
        content = item.get("content") or []
        if isinstance(content, str):
            parts.append(content)
            continue
        for part in content:
            part_payload = _response_item_to_dict(part)
            if not part_payload:
                continue
            text = part_payload.get("text") or part_payload.get("refusal")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _format_responses_terminal_error(response: Any, *, incomplete: bool) -> str:
    prefix = "Responses API returned an incomplete response" if incomplete else "Responses API request failed"
    if response is None:
        return prefix

    error = getattr(response, "error", None)
    message = getattr(error, "message", None)
    code = getattr(error, "code", None)
    if message:
        return f"{prefix}: {code}: {message}" if code else f"{prefix}: {message}"

    details = getattr(response, "incomplete_details", None)
    reason = getattr(details, "reason", None)
    return f"{prefix}: {reason}" if reason else prefix


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
            current_function["arguments"] = existing_arguments + \
                incoming_arguments
        elif incoming_arguments:
            current_function["arguments"] = incoming_arguments

    return [merged[key] for key in ordered_keys]
