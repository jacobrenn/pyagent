from __future__ import annotations

import json
from typing import Any, Iterable

import requests


class OllamaClient:
    def __init__(
        self,
        model: str = "gemma4:31b",
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_url = f"{self.base_url}/api/chat"
        self.session = requests.Session()

    def _payload(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        return payload

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        payload = self._payload(messages, tools, stream=False)
        try:
            response = self.session.post(
                self.api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            return {"error": str(exc)}
        except ValueError as exc:
            return {"error": f"Invalid JSON response from Ollama: {exc}"}

    def list_models(self) -> dict[str, Any]:
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            return {"error": str(exc)}
        except ValueError as exc:
            return {"error": f"Invalid JSON response from Ollama: {exc}"}

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterable[dict[str, Any]]:
        payload = self._payload(messages, tools, stream=True)
        try:
            with self.session.post(
                self.api_url,
                json=payload,
                stream=True,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        yield {"error": f"Could not decode Ollama stream chunk: {exc}: {line!r}"}
                        return
        except requests.RequestException as exc:
            yield {"error": str(exc)}
