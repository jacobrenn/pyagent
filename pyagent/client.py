"""Synchronous HTTP client for the PyAgent API server.

The classes in this module provide a small, dependency-free wrapper around the
FastAPI server exposed by ``pyagent serve``.  The client intentionally uses the
standard library so applications can automate PyAgent without pulling FastAPI,
httpx, requests, or any other heavyweight friends into the room.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import socket
from typing import Any
from urllib import error, parse, request
import uuid


class PyAgentClientError(RuntimeError):
    """Raised when a PyAgent HTTP API request fails.

    This exception wraps transport failures, non-successful HTTP responses,
    malformed JSON payloads, and response-shape validation errors.  Callers can
    catch this single exception type for all client-level failures instead of
    dealing directly with ``urllib`` exceptions.
    """


@dataclass(slots=True)
class RunResponse:
    """Structured response returned by :meth:`PyAgentClient.run`.

    Attributes:
        response: Final assistant text produced by the agent run.
        profile: Name of the model profile used by the server.
        provider: Provider backing the selected profile, such as ``ollama`` or
            an OpenAI-compatible backend.
        api_mode: OpenAI API mode selected by the profile.
        model: Concrete model name used for the request.
        messages: Updated conversation history after the run, suitable for
            passing back into a subsequent :meth:`PyAgentClient.run` call.
        context_files: Project or user context files loaded for the request.
    """

    response: str
    profile: str
    provider: str
    model: str
    messages: list[dict]
    context_files: list[str] = field(default_factory=list)
    api_mode: str = "chat_completions"


@dataclass(slots=True)
class AgentRunResponse(RunResponse):
    """Result of running a stored agent-definition revision."""

    agent: str = ""
    revision: int = 1


class PyAgentClient:
    """Small synchronous client for the PyAgent HTTP API.

    ``PyAgentClient`` mirrors the resource-management and chat endpoints served
    by ``pyagent serve``.  It is designed for scripts, tests, notebooks, and
    lightweight integrations that need to drive PyAgent programmatically.

    The client performs JSON and multipart request construction, validates the
    broad shape of server responses, and normalizes network/server errors into
    :class:`PyAgentClientError`.  It does not maintain server-side state beyond
    the configured base URL, timeout, and default headers; conversation state is
    carried explicitly through the ``messages`` field returned by
    :meth:`run`.

    Example:
        >>> client = PyAgentClient("http://127.0.0.1:8000")
        >>> client.is_healthy()
        True
        >>> result = client.run("Summarize this repository.", cwd=".")
        >>> result.response
        '...'

    Args:
        base_url: Root URL of a running PyAgent API server.  A trailing slash is
            ignored so endpoint paths can be joined consistently.
        timeout: Socket timeout, in seconds, applied to each HTTP request.
        headers: Optional default headers to merge into every request.  These
            are useful for reverse proxies or authentication layers placed in
            front of the PyAgent API.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        timeout: float = 60.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize a PyAgent API client.

        Args:
            base_url: Base URL for the PyAgent API server.  The value is stored
                without a trailing slash.
            timeout: Maximum number of seconds to wait for a server response.
            headers: Headers to include on all JSON and multipart requests.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = dict(headers or {})

    def health(self) -> dict[str, str]:
        """Return the server health payload.

        Calls ``GET /health`` and expects a dictionary-like JSON response.  The
        current server returns ``{"status": "ok"}`` when it is alive and able
        to answer requests.

        Returns:
            Health response with keys and values coerced to strings.

        Raises:
            PyAgentClientError: If the request fails or the server returns an
                invalid response shape.
        """
        response = self._request_json("GET", "/health")
        if not isinstance(response, dict):
            raise PyAgentClientError(
                "PyAgent server returned an invalid health response")
        return {str(key): str(value) for key, value in response.items()}

    def is_healthy(self) -> bool:
        """Return whether the server reports an ``ok`` health status.

        This is a convenience wrapper around :meth:`health` that suppresses
        :class:`PyAgentClientError` and returns ``False`` for unreachable or
        unhealthy servers.
        """
        try:
            return self.health().get("status") == "ok"
        except PyAgentClientError:
            return False

    def version(self) -> dict[str, str]:
        """Return PyAgent server version metadata.

        Calls ``GET /version`` and returns the JSON object provided by the
        server.  The current response includes a ``version`` field.

        Returns:
            Version response with keys and values coerced to strings.

        Raises:
            PyAgentClientError: If the request fails or the server returns an
                invalid response shape.
        """
        response = self._request_json("GET", "/version")
        if not isinstance(response, dict):
            raise PyAgentClientError(
                "PyAgent server returned an invalid version response")
        return {str(key): str(value) for key, value in response.items()}

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
        """Run a single agent turn and return the final response.

        Calls ``POST /run`` with the user message and optional conversation
        state.  The API server constructs an agent, loads project context and
        requested skills, runs tools as needed, and returns the final assistant
        answer along with updated message history.

        Args:
            message: User prompt for this turn.  The server requires a
                non-empty string.
            messages: Existing conversation history to load before the new
                message.  Pass ``RunResponse.messages`` from a previous call to
                continue a conversation.
            profile: Optional named model profile to use for this run.
            model: Optional model override.  Profile/provider semantics are
                handled by the server.
            cwd: Optional working directory used for project context discovery
                and tool execution.
            skills: Optional skill identifiers to explicitly load into the
                system prompt for this request.

        Returns:
            A :class:`RunResponse` containing the assistant response, selected
            model details, updated messages, and loaded context files.

        Raises:
            PyAgentClientError: If the request fails, the server reports an
                agent error, or the response is missing required fields.
        """
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
                api_mode=str(data.get("api_mode", "chat_completions")),
                model=str(data["model"]),
                messages=data["messages"],
                context_files=[str(item)
                               for item in data.get("context_files", [])],
            )
        except KeyError as exc:
            raise PyAgentClientError(
                f"PyAgent server response is missing required field: {exc.args[0]}"
            ) from exc

    # Agent definitions ---------------------------------------------------

    def list_agents(self) -> dict[str, Any]:
        """List stored reusable agent definitions."""
        return self._expect_dict(
            self._request_json("GET", "/agents"), "agents list"
        )

    def create_agent(self, definition: dict[str, Any]) -> dict[str, Any]:
        """Create an agent definition from a JSON-compatible dictionary."""
        return self._expect_dict(
            self._request_json("POST", "/agents", definition), "agent create"
        )

    def show_agent(self, name: str, *, revision: int | None = None) -> dict[str, Any]:
        """Return the current or requested revision of an agent definition."""
        path = self._path_with_name("/agents", name)
        if revision is not None:
            path += "?" + parse.urlencode({"revision": revision})
        return self._expect_dict(self._request_json("GET", path), "agent")

    def list_agent_revisions(self, name: str) -> dict[str, Any]:
        """List immutable revisions stored for an agent definition."""
        return self._expect_dict(
            self._request_json(
                "GET", self._path_with_name("/agents", name, "/revisions")
            ),
            "agent revisions",
        )

    def update_agent(self, name: str, changes: dict[str, Any]) -> dict[str, Any]:
        """Create a new revision containing the supplied field changes."""
        return self._expect_dict(
            self._request_json(
                "PUT", self._path_with_name("/agents", name), changes
            ),
            "agent update",
        )

    def validate_agent(
        self,
        name: str,
        *,
        revision: int | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Resolve an agent definition against currently installed resources."""
        query = {
            key: value
            for key, value in {"revision": revision, "cwd": cwd}.items()
            if value is not None
        }
        path = self._path_with_name("/agents", name, "/validate")
        if query:
            path += "?" + parse.urlencode(query)
        return self._expect_dict(
            self._request_json("POST", path, {}), "agent validation"
        )

    def remove_agent(self, name: str) -> dict[str, Any]:
        """Remove an agent definition and its local revision history."""
        return self._expect_dict(
            self._request_json(
                "DELETE", self._path_with_name("/agents", name)
            ),
            "agent remove",
        )

    def run_agent(
        self,
        name: str,
        message: str,
        *,
        messages: list[dict] | None = None,
        revision: int | None = None,
        cwd: str | None = None,
    ) -> AgentRunResponse:
        """Run one turn using a stored agent-definition revision."""
        payload = {
            "message": message,
            "messages": list(messages or []),
            "revision": revision,
            "cwd": cwd,
        }
        data = self._request_json(
            "POST", self._path_with_name("/agents", name, "/run"), payload
        )
        if not isinstance(data, dict):
            raise PyAgentClientError(
                "PyAgent server returned an invalid agent run response"
            )
        try:
            return AgentRunResponse(
                response=str(data["response"]),
                profile=str(data["profile"]),
                provider=str(data["provider"]),
                api_mode=str(data.get("api_mode", "chat_completions")),
                model=str(data["model"]),
                messages=data["messages"],
                context_files=[str(item)
                               for item in data.get("context_files", [])],
                agent=str(data["agent"]),
                revision=int(data["revision"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PyAgentClientError(
                f"PyAgent server returned an invalid agent run response: {exc}"
            ) from exc

    # Prompts -------------------------------------------------------------

    def list_prompts(self) -> dict[str, Any]:
        """List installed prompt resources.

        Calls ``GET /prompts``.

        Returns:
            Prompt listing response, including the resource kind, storage root,
            and installed prompt items.
        """
        return self._expect_dict(self._request_json("GET", "/prompts"), "prompts list")

    def install_prompt(
        self,
        *,
        url: str | None = None,
        file_path: str | Path | None = None,
        name: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Install a reusable prompt from a URL or local file.

        Args:
            url: Remote prompt URL to fetch server-side.
            file_path: Local prompt file to upload using multipart form data.
            name: Optional installed resource name.  If omitted, the server
                derives one from the source.
            force: Replace an existing prompt with the same name.

        Returns:
            Installation metadata including label, path, byte count, and status
            message.

        Raises:
            PyAgentClientError: If neither or both of ``url`` and ``file_path``
                are provided, upload file reading fails, or the server rejects
                the install.
        """
        return self._install_resource("/prompts/install", url=url, file_path=file_path, name=name, force=force)

    def show_prompt(self, name: str) -> dict[str, Any]:
        """Return the content and metadata for an installed prompt.

        Args:
            name: Prompt label/name to fetch.  The value is URL-encoded for the
                request path.

        Returns:
            Prompt metadata and content from ``GET /prompts/{name}``.
        """
        return self._expect_dict(
            self._request_json("GET", self._path_with_name("/prompts", name)),
            "prompt",
        )

    def use_prompt(self, name: str) -> dict[str, Any]:
        """Activate an installed prompt as the user-global AGENTS.md file.

        Args:
            name: Installed prompt to copy into the active prompt destination.

        Returns:
            Action response describing the source and destination used by the
            server.
        """
        return self._expect_dict(
            self._request_json("POST", self._path_with_name(
                "/prompts", name, "/use"), {}),
            "prompt use",
        )

    def remove_prompt(self, name: str) -> dict[str, Any]:
        """Remove an installed prompt resource.

        Args:
            name: Prompt label/name to remove.

        Returns:
            Server action response for the deletion.
        """
        return self._expect_dict(
            self._request_json(
                "DELETE", self._path_with_name("/prompts", name)),
            "prompt remove",
        )

    # Skills --------------------------------------------------------------

    def list_skills(self, *, cwd: str | None = None) -> dict[str, Any]:
        """List skills available to the server.

        Calls ``GET /skills`` and optionally supplies a working directory so the
        server can include project-local skills in addition to user-global
        skills.

        Args:
            cwd: Optional project directory used for project skill discovery.

        Returns:
            Skills response including user directory, effective cwd, and skill
            metadata.
        """
        path = "/skills"
        if cwd is not None:
            path += "?" + parse.urlencode({"cwd": cwd})
        return self._expect_dict(self._request_json("GET", path), "skills list")

    def install_skill(
        self,
        *,
        url: str | None = None,
        file_path: str | Path | None = None,
        name: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Install a user-global skill from a URL or local file.

        Args:
            url: Remote skill URL to fetch server-side.
            file_path: Local skill file to upload using multipart form data.
            name: Optional installed resource name.  If omitted, the server
                derives one from the source.
            force: Replace an existing skill with the same name.

        Returns:
            Installation metadata including label, path, byte count, and status
            message.
        """
        return self._install_resource("/skills/install", url=url, file_path=file_path, name=name, force=force)

    def remove_skill(self, name: str) -> dict[str, Any]:
        """Remove an installed user-global skill.

        Args:
            name: Skill label/name to remove.

        Returns:
            Server action response for the deletion.
        """
        return self._expect_dict(
            self._request_json(
                "DELETE", self._path_with_name("/skills", name)),
            "skill remove",
        )

    # Tools ---------------------------------------------------------------

    def list_tools(self) -> dict[str, Any]:
        """List built-in and user-managed tools known to the server.

        Calls ``GET /tools``.  The response includes enabled/disabled status,
        runner availability, built-in tool specs, external tool specs, broken
        scripts, disabled scripts, and name collisions.

        Returns:
            Tools inventory response.
        """
        return self._expect_dict(self._request_json("GET", "/tools"), "tools list")

    def install_tool(
        self,
        *,
        url: str | None = None,
        file_path: str | Path | None = None,
        name: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Install a user-managed tool from a URL or local file.

        Args:
            url: Remote tool script URL to fetch server-side.
            file_path: Local tool script to upload using multipart form data.
            name: Optional installed resource name.  If omitted, the server
                derives one from the source.
            force: Replace an existing tool with the same name.

        Returns:
            Installation metadata including label, path, byte count, and status
            message.
        """
        return self._install_resource("/tools/install", url=url, file_path=file_path, name=name, force=force)

    def new_tool(self, name: str) -> dict[str, Any]:
        """Scaffold a new user-managed tool script on the server.

        Args:
            name: Tool name to scaffold.  The server validates the name and
                chooses the final script path.

        Returns:
            Action response describing the created tool file.
        """
        return self._expect_dict(
            self._request_json("POST", "/tools/new", {"name": name}),
            "tool new",
        )

    def tool_path(self, name: str) -> dict[str, Any]:
        """Return the filesystem path for a user-managed tool.

        Args:
            name: Tool label/name to resolve.

        Returns:
            Response containing the resolved tool name and absolute path.
        """
        return self._expect_dict(
            self._request_json("GET", self._path_with_name(
                "/tools", name, "/path")),
            "tool path",
        )

    def enable_tool(self, name: str) -> dict[str, Any]:
        """Enable a disabled user-managed tool.

        Args:
            name: Tool label/name to enable.

        Returns:
            Server action response for the enable operation.
        """
        return self._expect_dict(
            self._request_json("POST", self._path_with_name(
                "/tools", name, "/enable"), {}),
            "tool enable",
        )

    def disable_tool(self, name: str) -> dict[str, Any]:
        """Disable a user-managed tool without deleting its script.

        Args:
            name: Tool label/name to disable.

        Returns:
            Server action response for the disable operation.
        """
        return self._expect_dict(
            self._request_json("POST", self._path_with_name(
                "/tools", name, "/disable"), {}),
            "tool disable",
        )

    def remove_tool(self, name: str) -> dict[str, Any]:
        """Remove a user-managed tool script.

        Args:
            name: Tool label/name to remove.

        Returns:
            Server action response for the deletion.
        """
        return self._expect_dict(
            self._request_json("DELETE", self._path_with_name("/tools", name)),
            "tool remove",
        )

    # Extensions ----------------------------------------------------------

    def list_extensions(self) -> dict[str, Any]:
        """List enabled and disabled user extensions.

        Calls ``GET /extensions``.

        Returns:
            Extension inventory including extension directory and per-extension
            state.
        """
        return self._expect_dict(self._request_json("GET", "/extensions"), "extensions list")

    def new_extension(self, name: str, *, url: str | None = None) -> dict[str, Any]:
        """Create a new user extension scaffold.

        Args:
            name: Extension package/name to create.
            url: Optional remote source URL recorded or used by the server when
                creating the extension.

        Returns:
            Action response describing the created extension.
        """
        payload: dict[str, object] = {"name": name}
        if url is not None:
            payload["url"] = url
        return self._expect_dict(
            self._request_json("POST", "/extensions/new", payload),
            "extension new",
        )

    def enable_extension(self, name: str) -> dict[str, Any]:
        """Enable a disabled user extension.

        Args:
            name: Extension name to enable.

        Returns:
            Server action response for the enable operation.
        """
        return self._expect_dict(
            self._request_json("POST", self._path_with_name(
                "/extensions", name, "/enable"), {}),
            "extension enable",
        )

    def disable_extension(self, name: str) -> dict[str, Any]:
        """Disable a user extension without deleting its files.

        Args:
            name: Extension name to disable.

        Returns:
            Server action response for the disable operation.
        """
        return self._expect_dict(
            self._request_json("POST", self._path_with_name(
                "/extensions", name, "/disable"), {}),
            "extension disable",
        )

    def remove_extension(self, name: str) -> dict[str, Any]:
        """Remove a user extension from the server's extension directory.

        Args:
            name: Extension name to remove.

        Returns:
            Server action response for the deletion.
        """
        return self._expect_dict(
            self._request_json(
                "DELETE", self._path_with_name("/extensions", name)),
            "extension remove",
        )

    # Request helpers -----------------------------------------------------

    def _install_resource(
        self,
        endpoint: str,
        *,
        url: str | None,
        file_path: str | Path | None,
        name: str | None,
        force: bool,
    ) -> dict[str, Any]:
        """Install a prompt, skill, or tool using the shared install contract.

        Exactly one of ``url`` and ``file_path`` must be supplied.  URL installs
        are sent as JSON so the server can fetch the resource.  File installs
        are sent as multipart form data because, sadly, bytes do not teleport.

        Args:
            endpoint: Install endpoint to call, such as ``/skills/install``.
            url: Remote resource URL, or ``None`` for file uploads.
            file_path: Local path to upload, or ``None`` for URL installs.
            name: Optional installed resource name.
            force: Whether the server should overwrite existing resources.

        Returns:
            Validated dictionary response from the install endpoint.

        Raises:
            PyAgentClientError: If the source arguments are invalid, the upload
                cannot be read, or the server response is invalid.
        """
        if bool(url) == bool(file_path):
            raise PyAgentClientError(
                "Provide exactly one of `url` or `file_path` for install operations."
            )
        if url is not None:
            payload: dict[str, object] = {"url": url, "force": force}
            if name is not None:
                payload["name"] = name
            return self._expect_dict(self._request_json("POST", endpoint, payload), "install")
        return self._expect_dict(
            self._request_multipart(
                "POST",
                endpoint,
                file_path=Path(file_path or ""),
                fields={
                    **({"name": name} if name is not None else {}),
                    "force": "true" if force else "false",
                },
            ),
            "install",
        )

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> object:
        """Send an HTTP request with an optional JSON body.

        Args:
            method: HTTP method to use.
            path: API path, including any query string, relative to
                :attr:`base_url`.
            payload: Optional JSON-serializable dictionary to encode as the
                request body.  When provided, ``Content-Type`` is set to
                ``application/json``.

        Returns:
            Decoded JSON response object.

        Raises:
            PyAgentClientError: If the request fails or the response is not
                valid JSON.
        """
        body: bytes | None = None
        headers = {"Accept": "application/json", **self.headers}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        return self._send_request(method, path, body=body, headers=headers)

    def _request_multipart(
        self,
        method: str,
        path: str,
        *,
        file_path: Path,
        fields: dict[str, str],
    ) -> object:
        """Send a multipart/form-data request containing one file upload.

        The multipart body is assembled manually to keep the client free of
        third-party HTTP dependencies.

        Args:
            method: HTTP method to use.
            path: API path relative to :attr:`base_url`.
            file_path: Local file to read and upload under the field name
                ``file``.
            fields: Additional string form fields to include before the file.

        Returns:
            Decoded JSON response object.

        Raises:
            PyAgentClientError: If the upload file cannot be read, the request
                fails, or the response is invalid JSON.
        """
        try:
            file_data = file_path.expanduser().read_bytes()
        except OSError as exc:
            raise PyAgentClientError(
                f"Could not read upload file {file_path}: {exc}"
            ) from exc

        boundary = f"pyagent-{uuid.uuid4().hex}"
        chunks: list[bytes] = []
        for key, value in fields.items():
            chunks.append(
                (
                    f"--{boundary}\r\n"
                    f"Content-Disposition: form-data; name=\"{self._escape_multipart(key)}\"\r\n\r\n"
                    f"{value}\r\n"
                ).encode("utf-8")
            )
        chunks.append(
            (
                f"--{boundary}\r\n"
                f"Content-Disposition: form-data; name=\"file\"; "
                f"filename=\"{self._escape_multipart(file_path.name)}\"\r\n"
                "Content-Type: application/octet-stream\r\n\r\n"
            ).encode("utf-8")
        )
        chunks.append(file_data)
        chunks.append(b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
        headers = {
            "Accept": "application/json",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            **self.headers,
        }
        return self._send_request(method, path, body=b"".join(chunks), headers=headers)

    def _send_request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None,
        headers: dict[str, str],
    ) -> object:
        """Perform the low-level HTTP request and decode the JSON response.

        Args:
            method: HTTP method to use.
            path: API path relative to :attr:`base_url`.
            body: Optional raw request body bytes.
            headers: Complete request headers for this request.

        Returns:
            Decoded JSON response object.

        Raises:
            PyAgentClientError: If the server returns an HTTP error, the server
                cannot be reached, the request times out, or the response body
                cannot be parsed as JSON.
        """
        url = f"{self.base_url}{path}"
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
        """Extract a readable error detail from an HTTP error response.

        The FastAPI server typically returns ``{"detail": "..."}`` for
        errors.  If that shape is unavailable, this method falls back to the raw
        response body or HTTP reason phrase.

        Args:
            exc: HTTP error raised by ``urllib.request.urlopen``.

        Returns:
            Human-readable error detail for inclusion in
            :class:`PyAgentClientError`.
        """
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

    def _path_with_name(self, base: str, name: str, suffix: str = "") -> str:
        """Build an endpoint path containing a URL-encoded resource name.

        Args:
            base: Endpoint prefix, such as ``/tools``.
            name: Resource name to trim of leading/trailing slashes and encode
                as one path segment.
            suffix: Optional path suffix appended after the encoded name.

        Returns:
            Combined endpoint path.
        """
        encoded = parse.quote(name.strip("/"), safe="")
        return f"{base}/{encoded}{suffix}"

    @staticmethod
    def _escape_multipart(value: str) -> str:
        """Escape a value for use inside a multipart Content-Disposition header.

        Args:
            value: Header parameter value to escape.

        Returns:
            Value with backslashes and double quotes escaped.
        """
        return value.replace("\\", "\\\\").replace('"', "\\\"")

    @staticmethod
    def _expect_dict(response: object, label: str) -> dict[str, Any]:
        """Validate that a decoded response is a dictionary.

        Args:
            response: Decoded JSON response to validate.
            label: Human-readable endpoint label used in error messages.

        Returns:
            ``response`` typed as ``dict[str, Any]`` when valid.

        Raises:
            PyAgentClientError: If ``response`` is not a dictionary.
        """
        if not isinstance(response, dict):
            raise PyAgentClientError(
                f"PyAgent server returned an invalid {label} response")
        return response
