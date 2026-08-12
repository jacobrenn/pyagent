from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Any
from urllib import parse

from .config import AppConfig
from .external_tools import (
    build_external_tool_specs,
    default_runner_command,
    discover_external_tools,
    find_tool_script,
    move_tool_script,
)
from .project_context import list_available_skills
from .resources import (
    PROMPT_KIND,
    SKILL_KIND,
    TOOL_KIND,
    ManagedResource,
    ResourceKind,
    install_resource,
    install_resource_bytes,
    list_resources,
    remove_resource,
    resolve_resource,
    resource_dir,
)
from .scaffold import ScaffoldError, create_user_tool
from .tools import BUILTIN_ORIGIN, EXTERNAL_ORIGIN, create_default_tool_registry
from .user_runtime import resolve_user_dir, user_extensions_dir

try:
    from fastapi import FastAPI, HTTPException, Request
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - exercised via CLI guard
    raise RuntimeError(
        "FastAPI support is not installed. Install `pyagent-harness[api]` "
        "or install `fastapi` and `uvicorn` to use `pyagent serve`."
    ) from exc


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    profile: str | None = None
    model: str | None = None
    cwd: str | None = None
    skills: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    response: str
    profile: str
    provider: str
    model: str
    messages: list[dict]
    context_files: list[str] = Field(default_factory=list)


class VersionResponse(BaseModel):
    version: str


class ResourceItem(BaseModel):
    label: str
    path: str


class ResourceListResponse(BaseModel):
    kind: str
    root: str
    items: list[ResourceItem]


class ResourceInstallResponse(BaseModel):
    kind: str
    label: str
    path: str
    bytes_written: int
    message: str


class ResourceContentResponse(BaseModel):
    kind: str
    label: str
    path: str
    content: str


class ResourceActionResponse(BaseModel):
    message: str
    path: str | None = None


class PromptUseResponse(BaseModel):
    message: str
    source: str
    destination: str


class SkillInfo(BaseModel):
    id: str
    scope: str
    label: str
    title: str
    preview: str
    path: str
    error: str | None = None


class SkillsResponse(BaseModel):
    cwd: str
    user_dir: str
    skills: list[SkillInfo]


class ToolRunnerInfo(BaseModel):
    name: str
    available: bool | None = None
    message: str | None = None


class ToolInfo(BaseModel):
    name: str
    origin: str
    source: str | None = None
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolFileInfo(BaseModel):
    label: str
    path: str
    disabled: bool = False


class ExternalToolProblem(BaseModel):
    script_path: str
    error: str | None = None


class ExternalToolDisabled(BaseModel):
    script_path: str


class ToolCollisionResponse(BaseModel):
    name: str
    external_path: str | None = None


class ToolsResponse(BaseModel):
    tools_enabled: bool
    builtin_tools_enabled: bool
    user_tools_enabled: bool
    user_dir: str
    runner: ToolRunnerInfo
    builtin: list[ToolInfo]
    external: list[ToolInfo]
    files: list[ToolFileInfo]
    broken: list[ExternalToolProblem] = Field(default_factory=list)
    disabled: list[ExternalToolDisabled] = Field(default_factory=list)
    collisions: list[ToolCollisionResponse] = Field(default_factory=list)
    discovery_error: str | None = None


class CreateToolRequest(BaseModel):
    name: str = Field(..., min_length=1)


class ToolPathResponse(BaseModel):
    name: str
    path: str


class ExtensionItem(BaseModel):
    name: str
    state: str
    path: str


class ExtensionsResponse(BaseModel):
    user_dir: str
    extensions_dir: str
    enabled: list[ExtensionItem]
    disabled: list[ExtensionItem]


class ExtensionNewRequest(BaseModel):
    name: str = Field(..., min_length=1)
    url: str | None = None


@dataclass(frozen=True, slots=True)
class _ParsedInstallRequest:
    source_url: str | None
    upload_bytes: bytes | None
    upload_name: str | None
    name: str | None
    force: bool


app = FastAPI(title="PyAgent API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    from .main import get_version

    return VersionResponse(version=get_version())


@app.post("/run", response_model=ChatResponse)
def run(request: ChatRequest) -> ChatResponse:
    from .main import build_agent_for_request

    try:
        agent = build_agent_for_request(
            profile=request.profile,
            model=request.model,
            cwd=request.cwd,
            skills=request.skills,
        )
        agent.load_messages(request.messages)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    agent.load_extensions()
    final_response = ""
    for event in agent.run(request.message):
        if event.get("type") == "error":
            message = str(event.get("message") or "Agent run failed")
            raise HTTPException(status_code=502, detail=message)
        if event.get("type") == "assistant_done":
            final_response = event.get("content", "")
            break
    else:
        raise HTTPException(
            status_code=500,
            detail="Agent run finished without a final assistant response.",
        )

    profile = agent.current_profile()
    return ChatResponse(
        response=final_response,
        profile=profile.name,
        provider=profile.provider,
        model=profile.model,
        messages=agent.messages,
        context_files=list(agent.project_context_files),
    )


@app.get("/prompts", response_model=ResourceListResponse)
def list_prompts() -> ResourceListResponse:
    return _resource_list_response(PROMPT_KIND)


@app.post("/prompts/install", response_model=ResourceInstallResponse)
async def install_prompt(request: Request) -> ResourceInstallResponse:
    return await _install_managed_resource(PROMPT_KIND, request)


@app.post("/prompts/{name:path}/use", response_model=PromptUseResponse)
def use_prompt(name: str) -> PromptUseResponse:
    resource = _resolve_resource_or_404(PROMPT_KIND, name)
    config = AppConfig.from_env()
    destination = Path(config.system_prompt_path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(f".{destination.name}.tmp")
        shutil.copyfile(resource.path, tmp)
        shutil.move(str(tmp), str(destination))
    except OSError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not copy {resource.path} to {destination}: {exc}",
        ) from exc
    return PromptUseResponse(
        message=f"Copied prompt {resource.path} to active system prompt {destination}",
        source=str(resource.path),
        destination=str(destination),
    )


@app.get("/prompts/{name:path}", response_model=ResourceContentResponse)
def show_prompt(name: str) -> ResourceContentResponse:
    return _read_text_resource(PROMPT_KIND, name)


@app.delete("/prompts/{name:path}", response_model=ResourceActionResponse)
def delete_prompt(name: str) -> ResourceActionResponse:
    removed = _remove_resource_or_404(PROMPT_KIND, name)
    return ResourceActionResponse(message=f"Removed prompt {removed}", path=str(removed))


@app.get("/skills", response_model=SkillsResponse)
def list_skills(cwd: str | None = None) -> SkillsResponse:
    base_cwd = str(Path(cwd or os.getcwd()).expanduser().resolve())
    config = AppConfig.from_env()
    try:
        available = list_available_skills(base_cwd, user_dir=config.user_dir)
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SkillsResponse(
        cwd=base_cwd,
        user_dir=str(resolve_user_dir(config.user_dir)),
        skills=[
            SkillInfo(
                id=skill.id,
                scope=skill.scope,
                label=skill.label,
                title=skill.title,
                preview=skill.preview,
                path=str(skill.path),
                error=skill.error,
            )
            for skill in available
        ],
    )


@app.post("/skills/install", response_model=ResourceInstallResponse)
async def install_skill(request: Request) -> ResourceInstallResponse:
    return await _install_managed_resource(SKILL_KIND, request)


@app.delete("/skills/{name:path}", response_model=ResourceActionResponse)
def delete_skill(name: str) -> ResourceActionResponse:
    target = _user_managed_resource_target(name, kind="skill")
    removed = _remove_resource_or_404(SKILL_KIND, target)
    return ResourceActionResponse(message=f"Removed skill {removed}", path=str(removed))


@app.get("/tools", response_model=ToolsResponse)
def list_tools() -> ToolsResponse:
    config = AppConfig.from_env()
    discovery = None
    discovery_error: str | None = None
    if config.user_tools_enabled:
        try:
            discovery = discover_external_tools(
                user_dir=config.user_dir,
                runner=config.tool_runner,
                describe_timeout=config.user_tool_describe_timeout,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            discovery_error = str(exc)

    external_specs = (
        build_external_tool_specs(
            discovery,
            invoke_timeout=config.user_tool_timeout,
            runner_command=default_runner_command(discovery.runner),
        )
        if discovery is not None
        else None
    )
    registry = create_default_tool_registry(
        config, external_specs=external_specs)
    definitions = {
        item["function"]["name"]: item["function"]
        for item in registry.definitions()
        if isinstance(item.get("function"), dict)
    }

    runner = ToolRunnerInfo(
        name=discovery.runner if discovery is not None else config.tool_runner,
        available=discovery.runner_available if discovery is not None else None,
        message=discovery.runner_message if discovery is not None else None,
    )

    builtin = [
        _tool_info_response(registry, definitions, name, BUILTIN_ORIGIN)
        for name in sorted(registry.names_by_origin(BUILTIN_ORIGIN))
    ]
    external = [
        _tool_info_response(registry, definitions, name, EXTERNAL_ORIGIN)
        for name in sorted(registry.names_by_origin(EXTERNAL_ORIGIN))
    ]
    files = [
        ToolFileInfo(
            label=resource.label,
            path=str(resource.path),
            disabled="disabled" in Path(resource.label).parts[:-1],
        )
        for resource in list_resources(TOOL_KIND, user_dir=config.user_dir)
    ]

    broken = []
    disabled = []
    if discovery is not None:
        broken = [
            ExternalToolProblem(script_path=str(
                entry.script_path), error=entry.error)
            for entry in discovery.broken
        ]
        disabled = [
            ExternalToolDisabled(script_path=str(entry.script_path))
            for entry in discovery.disabled
        ]

    return ToolsResponse(
        tools_enabled=config.tools_enabled,
        builtin_tools_enabled=config.builtin_tools_enabled,
        user_tools_enabled=config.user_tools_enabled,
        user_dir=str(resolve_user_dir(config.user_dir)),
        runner=runner,
        builtin=builtin,
        external=external,
        files=files,
        broken=broken,
        disabled=disabled,
        collisions=[
            ToolCollisionResponse(
                name=collision.name,
                external_path=collision.external_path,
            )
            for collision in registry.collisions()
        ],
        discovery_error=discovery_error,
    )


@app.post("/tools/install", response_model=ResourceInstallResponse)
async def install_tool(request: Request) -> ResourceInstallResponse:
    return await _install_managed_resource(TOOL_KIND, request)


@app.post("/tools/new", response_model=ResourceActionResponse)
def new_tool(request: CreateToolRequest) -> ResourceActionResponse:
    config = AppConfig.from_env()
    try:
        created = create_user_tool(request.name, user_dir=config.user_dir)
    except ScaffoldError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ResourceActionResponse(
        message=f"Created starter tool at {created}",
        path=str(created),
    )


@app.get("/tools/{name:path}/path", response_model=ToolPathResponse)
def tool_path(name: str) -> ToolPathResponse:
    config = AppConfig.from_env()
    located = find_tool_script(name, user_dir=config.user_dir)
    if located is None:
        raise HTTPException(
            status_code=404,
            detail=f"No tool named {name!r} was found under {config.user_dir}/tools/.",
        )
    return ToolPathResponse(name=name, path=str(located))


@app.post("/tools/{name:path}/enable", response_model=ResourceActionResponse)
def enable_tool(name: str) -> ResourceActionResponse:
    return _move_tool(name, enable=True)


@app.post("/tools/{name:path}/disable", response_model=ResourceActionResponse)
def disable_tool(name: str) -> ResourceActionResponse:
    return _move_tool(name, enable=False)


@app.delete("/tools/{name:path}", response_model=ResourceActionResponse)
def delete_tool(name: str) -> ResourceActionResponse:
    removed = _remove_resource_or_404(TOOL_KIND, name)
    return ResourceActionResponse(message=f"Removed tool {removed}", path=str(removed))


@app.get("/extensions", response_model=ExtensionsResponse)
def list_extensions() -> ExtensionsResponse:
    from .extensions.loader import _discover

    config = AppConfig.from_env()
    resolved_user_dir = resolve_user_dir(config.user_dir)
    ext_dir = user_extensions_dir(resolved_user_dir)
    disabled_dir = ext_dir / "disabled"
    enabled_names = _discover(ext_dir)
    disabled_names = _discover(disabled_dir)
    return ExtensionsResponse(
        user_dir=str(resolved_user_dir),
        extensions_dir=str(ext_dir),
        enabled=[
            ExtensionItem(
                name=name,
                state="enabled",
                path=str(_extension_path(ext_dir, name)),
            )
            for name in enabled_names
        ],
        disabled=[
            ExtensionItem(
                name=name,
                state="disabled",
                path=str(_extension_path(disabled_dir, name)),
            )
            for name in disabled_names
        ],
    )


@app.post("/extensions/new", response_model=ResourceActionResponse)
def new_extension(request: ExtensionNewRequest) -> ResourceActionResponse:
    from .extensions.manager import _cmd_new

    message = _run_extension_action(
        lambda agent: _cmd_new(agent, request.name, request.url)
    )
    return ResourceActionResponse(message=message)


@app.post("/extensions/{name}/enable", response_model=ResourceActionResponse)
def enable_extension(name: str) -> ResourceActionResponse:
    from .extensions.manager import _cmd_enable

    message = _run_extension_action(lambda agent: _cmd_enable(agent, name))
    return ResourceActionResponse(message=message)


@app.post("/extensions/{name}/disable", response_model=ResourceActionResponse)
def disable_extension(name: str) -> ResourceActionResponse:
    from .extensions.manager import _cmd_disable

    message = _run_extension_action(lambda agent: _cmd_disable(agent, name))
    return ResourceActionResponse(message=message)


@app.delete("/extensions/{name}", response_model=ResourceActionResponse)
def delete_extension(name: str) -> ResourceActionResponse:
    from .extensions.manager import _cmd_remove

    message = _run_extension_action(lambda agent: _cmd_remove(agent, name))
    return ResourceActionResponse(message=message)


def _resource_item(resource: ManagedResource) -> ResourceItem:
    return ResourceItem(label=resource.label, path=str(resource.path))


def _resource_list_response(kind: ResourceKind) -> ResourceListResponse:
    return ResourceListResponse(
        kind=kind.name,
        root=str(resource_dir(kind)),
        items=[_resource_item(resource) for resource in list_resources(kind)],
    )


def _resolve_resource_or_404(kind: ResourceKind, target: str) -> ManagedResource:
    resource = resolve_resource(kind, target)
    if resource is None:
        raise HTTPException(
            status_code=404,
            detail=f"No {kind.name} named {target!r} was found under {resource_dir(kind)}.",
        )
    return resource


def _remove_resource_or_404(kind: ResourceKind, target: str) -> Path:
    try:
        return remove_resource(kind, target)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _read_text_resource(kind: ResourceKind, target: str) -> ResourceContentResponse:
    resource = _resolve_resource_or_404(kind, target)
    try:
        content = resource.path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not read {resource.path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read {resource.path} as UTF-8 text: {exc}",
        ) from exc
    return ResourceContentResponse(
        kind=kind.name,
        label=resource.label,
        path=str(resource.path),
        content=content,
    )


def _is_http_url(value: str) -> bool:
    return parse.urlparse(value).scheme.lower() in {"http", "https"}


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


async def _parse_install_request(request: Request) -> _ParsedInstallRequest:
    content_type = request.headers.get(
        "content-type", "").split(";", 1)[0].strip().lower()
    if content_type in {"application/json", ""} or content_type.endswith("+json"):
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid JSON body: {exc}") from exc
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=400, detail="Install request body must be a JSON object.")
        source_url = str(payload.get(
            "url") or payload.get("source") or "").strip()
        if not source_url:
            raise HTTPException(
                status_code=400,
                detail="JSON install requests must include an HTTP(S) URL in `url`.",
            )
        if not _is_http_url(source_url):
            raise HTTPException(
                status_code=400,
                detail="API installs accept only HTTP(S) URLs in JSON. Use multipart file upload for local files.",
            )
        return _ParsedInstallRequest(
            source_url=source_url,
            upload_bytes=None,
            upload_name=None,
            name=_optional_string(payload.get("name")),
            force=_parse_bool(payload.get("force")),
        )

    if content_type == "multipart/form-data":
        try:
            form = await request.form()
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Could not parse multipart form: {exc}") from exc
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(
                status_code=400,
                detail="Multipart install requests must include a file field named `file`.",
            )
        filename = str(getattr(upload, "filename", "") or "").strip()
        try:
            data = await upload.read()
        finally:
            close = getattr(upload, "close", None)
            if callable(close):
                await close()
        return _ParsedInstallRequest(
            source_url=None,
            upload_bytes=data,
            upload_name=filename,
            name=_optional_string(form.get("name")),
            force=_parse_bool(form.get("force")),
        )

    raise HTTPException(
        status_code=415,
        detail="Use application/json with an HTTP(S) `url`, or multipart/form-data with a `file` field.",
    )


async def _install_managed_resource(kind: ResourceKind, request: Request) -> ResourceInstallResponse:
    parsed = await _parse_install_request(request)
    try:
        if parsed.source_url is not None:
            result = install_resource(
                kind,
                parsed.source_url,
                name=parsed.name,
                force=parsed.force,
            )
        else:
            result = install_resource_bytes(
                kind,
                parsed.upload_bytes or b"",
                source_name=parsed.upload_name or "",
                name=parsed.name,
                force=parsed.force,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    root = resource_dir(kind)
    try:
        label = result.destination.relative_to(root).as_posix()
    except ValueError:
        label = result.destination.name
    return ResourceInstallResponse(
        kind=kind.name,
        label=label,
        path=str(result.destination),
        bytes_written=result.bytes_written,
        message=(
            f"Installed {kind.name} `{result.destination.name}` to {result.destination} "
            f"({result.bytes_written} bytes)"
        ),
    )


def _user_managed_resource_target(target: str, *, kind: str) -> str:
    if target.startswith("project:"):
        raise HTTPException(
            status_code=400,
            detail=f"Only user-managed {kind}s can be removed through this endpoint.",
        )
    if target.startswith("user:"):
        return target.split(":", 1)[1]
    return target


def _tool_info_response(registry: Any, definitions: dict[str, dict[str, Any]], name: str, origin: str) -> ToolInfo:
    definition = definitions.get(name, {})
    return ToolInfo(
        name=name,
        origin=origin,
        source=registry.source(name),
        description=definition.get("description"),
        parameters=definition.get("parameters"),
    )


def _move_tool(name: str, *, enable: bool) -> ResourceActionResponse:
    config = AppConfig.from_env()
    new_path, error = move_tool_script(
        name, user_dir=config.user_dir, enable=enable)
    if error or new_path is None:
        raise HTTPException(
            status_code=404, detail=error or "Unknown tool move error.")
    verb = "Enabled" if enable else "Disabled"
    return ResourceActionResponse(message=f"{verb} tool `{Path(new_path).name}` at {new_path}", path=str(new_path))


def _extension_path(ext_dir: Path, name: str) -> Path:
    for candidate in (ext_dir / name, ext_dir / f"{name}.py"):
        if candidate.exists():
            return candidate.resolve()
    return (ext_dir / name).resolve()


class _ExtensionApiBus:
    @staticmethod
    def loaded_extensions() -> list[str]:
        return []

    @staticmethod
    def clear() -> None:
        return None


class _ExtensionApiLog:
    @staticmethod
    def error(_message: str, _details: Any | None = None) -> None:
        return None


class _ExtensionApiAgent:
    def __init__(self) -> None:
        self.config = AppConfig.from_env()
        self.bus = _ExtensionApiBus()
        self._ext_log = _ExtensionApiLog()

    @staticmethod
    def _rebuild_external_tools() -> None:
        return None


def _extension_action_failed(message: str) -> bool:
    lowered = message.lower()
    failure_markers = (
        "already exists",
        "not found",
        "failed",
        "an error occurred",
        "cannot create",
        "subdirectory",
    )
    return any(marker in lowered for marker in failure_markers)


def _run_extension_action(action: Any) -> str:
    agent = _ExtensionApiAgent()
    try:
        message = str(action(agent))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if _extension_action_failed(message):
        raise HTTPException(status_code=400, detail=message)
    return message


def create_app() -> Any:
    return app
