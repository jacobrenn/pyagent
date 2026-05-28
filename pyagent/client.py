from __future__ import annotations

from dataclasses import dataclass, field
import json
import socket
from urllib import error, request


class PyAgentClientError(RuntimeError):
    """Raised when a PyAgent HTTP API request fails."""


@dataclass(slots=True)
class RunResponse:
    response: str
    profile: str
    provider: str
    model: str
    messages: list[dict]
    context_files: list[str] = field(default_factory=list)


class PyAgentClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        timeout: float = 60.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = dict(headers or {})

    def health(self) -> dict[str, str]:
        response = self._request_json("GET", "/health")
        if not isinstance(response, dict):
            raise PyAgentClientError(
                "PyAgent server returned an invalid health response")
        return {str(key): str(value) for key, value in response.items()}

    def is_healthy(self) -> bool:
        try:
            return self.health().get("status") == "ok"
        except PyAgentClientError:
            return False

    def run(
        self,
        message: str,
        *,
        messages: list[dict] | None = None,
        profile: str | None = None,
        model: str | None = None,
        cwd: str | None = None,
        skills: list[str] | None = None,
    ) -> RunResponse:
        payload = {
            "message": message,
            "messages": list(messages or []),
            "profile": profile,
            "model": model,
            "cwd": cwd,
            "skills": list(skills or []),
        }
        data = self._request_json("POST", "/run", payload)
        if not isinstance(data, dict):
            raise PyAgentClientError(
                "PyAgent server returned an invalid run response")
        try:
            return RunResponse(
                response=str(data["response"]),
                profile=str(data["profile"]),
                provider=str(data["provider"]),
                model=str(data["model"]),
                messages=data["messages"],
                context_files=[str(item)
                               for item in data.get("context_files", [])],
            )
        except KeyError as exc:
            raise PyAgentClientError(
                f"PyAgent server response is missing required field: {exc.args[0]}"
            ) from exc

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> object:
        url = f"{self.base_url}{path}"
        body: bytes | None = None
        headers = {"Accept": "application/json", **self.headers}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(url, data=body, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = self._extract_error_detail(exc)
            raise PyAgentClientError(
                f"PyAgent server returned HTTP {exc.code}: {detail}"
            ) from exc
        except error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, socket.timeout):
                raise PyAgentClientError(
                    f"Timed out connecting to PyAgent server at {url}"
                ) from exc
            raise PyAgentClientError(
                f"Could not connect to PyAgent server at {url}: {reason}"
            ) from exc
        except TimeoutError as exc:
            raise PyAgentClientError(
                f"Timed out connecting to PyAgent server at {url}"
            ) from exc

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PyAgentClientError(
                f"PyAgent server returned invalid JSON from {url}"
            ) from exc

    def _extract_error_detail(self, exc: error.HTTPError) -> str:
        try:
            raw = exc.read().decode("utf-8")
        except Exception:
            return exc.reason or "Unknown error"
        if not raw:
            return exc.reason or "Unknown error"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if isinstance(payload, dict) and "detail" in payload:
            return str(payload["detail"])
        return raw
