from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

from .config import AppConfig
from .external_tools import (
    build_external_tool_specs,
    default_runner_command,
    discover_external_tools,
)
from .model_profiles import load_profile_store
from .project_context import load_full_context, resolve_available_skill
from .resources import PROMPT_KIND, resolve_resource
from .tools import create_default_tool_registry
from .user_runtime import (
    ensure_user_subdir,
    resolve_user_dir,
    user_agent_definitions_dir,
    user_extensions_dir,
)


AGENT_DEFINITION_SCHEMA_VERSION = 1
_AGENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_EDITABLE_FIELDS = {
    "description",
    "profile",
    "model",
    "prompt",
    "skills",
    "tools",
    "workspace",
    "max_iterations",
    "labels",
    "capabilities",
}
_STORED_FIELDS = {
    "schema_version",
    "name",
    "revision",
    "created_at",
    "updated_at",
    *_EDITABLE_FIELDS,
}


class AgentDefinitionError(ValueError):
    """Raised when an agent definition cannot be parsed or persisted."""


@dataclass(frozen=True, slots=True)
class AgentDefinition:
    schema_version: int
    name: str
    revision: int
    created_at: str
    updated_at: str
    description: str = ""
    profile: str | None = None
    model: str | None = None
    prompt: str | None = None
    skills: tuple[str, ...] = ()
    tools: tuple[str, ...] | None = None
    workspace: str | None = None
    max_iterations: int | None = None
    labels: dict[str, str] | None = None
    capabilities: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["skills"] = list(self.skills)
        payload["tools"] = list(self.tools) if self.tools is not None else None
        payload["labels"] = dict(self.labels or {})
        payload["capabilities"] = list(self.capabilities)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentDefinition":
        if not isinstance(payload, dict):
            raise AgentDefinitionError(
                "Agent definition must be a JSON object.")
        unknown = sorted(set(payload) - _STORED_FIELDS)
        if unknown:
            raise AgentDefinitionError(
                f"Unknown agent definition field(s): {', '.join(unknown)}"
            )

        schema_version = _integer(payload.get(
            "schema_version", AGENT_DEFINITION_SCHEMA_VERSION), "schema_version")
        if schema_version != AGENT_DEFINITION_SCHEMA_VERSION:
            raise AgentDefinitionError(
                f"Unsupported agent definition schema_version {schema_version}; "
                f"expected {AGENT_DEFINITION_SCHEMA_VERSION}."
            )

        name = _agent_name(payload.get("name"))
        revision = _integer(payload.get("revision", 1), "revision")
        if revision < 1:
            raise AgentDefinitionError(
                "Agent definition revision must be at least 1.")

        created_at = _required_string(payload.get("created_at"), "created_at")
        updated_at = _required_string(payload.get("updated_at"), "updated_at")
        description = _optional_string(payload.get("description")) or ""
        profile = _optional_string(payload.get("profile"))
        model = _optional_string(payload.get("model"))
        prompt = _optional_string(payload.get("prompt"))
        workspace = _optional_string(payload.get("workspace"))
        skills = _string_tuple(payload.get("skills", []), "skills")
        capabilities = _string_tuple(payload.get(
            "capabilities", []), "capabilities")

        raw_tools = payload.get("tools")
        tools = None if raw_tools is None else _string_tuple(
            raw_tools, "tools")

        raw_max_iterations = payload.get("max_iterations")
        max_iterations = None
        if raw_max_iterations is not None:
            max_iterations = _integer(raw_max_iterations, "max_iterations")
            if max_iterations != -1 and max_iterations < 1:
                raise AgentDefinitionError(
                    "max_iterations must be -1 or an integer greater than zero."
                )

        labels = _labels(payload.get("labels", {}))
        return cls(
            schema_version=schema_version,
            name=name,
            revision=revision,
            created_at=created_at,
            updated_at=updated_at,
            description=description,
            profile=profile,
            model=model,
            prompt=prompt,
            skills=skills,
            tools=tools,
            workspace=workspace,
            max_iterations=max_iterations,
            labels=labels,
            capabilities=capabilities,
        )


@dataclass(frozen=True, slots=True)
class AgentDefinitionValidation:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    resolved: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "resolved": self.resolved,
        }


class AgentDefinitionStore:
    """Versioned JSON storage rooted at ``~/.pyagent/agents/``."""

    def __init__(self, user_dir: str | os.PathLike[str] | None = None) -> None:
        self.user_dir = resolve_user_dir(user_dir)
        self.root = user_agent_definitions_dir(self.user_dir)
        self.revisions_root = self.root / ".revisions"

    def list(self) -> list[AgentDefinition]:
        if not self.root.is_dir():
            return []
        definitions: list[AgentDefinition] = []
        for path in sorted(self.root.glob("*.json")):
            definitions.append(self._read(path))
        return definitions

    def get(self, name: str, revision: int | None = None) -> AgentDefinition:
        normalized = _agent_name(name)
        path = (
            self.root / f"{normalized}.json"
            if revision is None
            else self._revision_path(normalized, revision)
        )
        if not path.is_file():
            suffix = f" revision {revision}" if revision is not None else ""
            raise AgentDefinitionError(
                f"No agent definition named {normalized!r}{suffix} was found under {self.root}."
            )
        return self._read(path)

    def revisions(self, name: str) -> list[AgentDefinition]:
        normalized = _agent_name(name)
        if not (self.root / f"{normalized}.json").is_file():
            raise AgentDefinitionError(
                f"No agent definition named {normalized!r} was found under {self.root}."
            )
        history = self.revisions_root / normalized
        if not history.is_dir():
            return []
        return [self._read(path) for path in sorted(history.glob("*.json"))]

    def create(self, payload: dict[str, Any]) -> AgentDefinition:
        name = _agent_name(payload.get("name"))
        current_path = self.root / f"{name}.json"
        if current_path.exists():
            raise AgentDefinitionError(
                f"Agent definition {name!r} already exists.")

        now = _timestamp()
        definition = AgentDefinition.from_dict(
            {
                **payload,
                "schema_version": AGENT_DEFINITION_SCHEMA_VERSION,
                "name": name,
                "revision": 1,
                "created_at": now,
                "updated_at": now,
            }
        )
        self._write_revision_and_current(definition)
        return definition

    def update(self, name: str, changes: dict[str, Any]) -> AgentDefinition:
        current = self.get(name)
        unknown = sorted(set(changes) - _EDITABLE_FIELDS)
        if unknown:
            raise AgentDefinitionError(
                f"Unknown or immutable agent definition field(s): {', '.join(unknown)}"
            )
        if not changes:
            raise AgentDefinitionError(
                "At least one agent definition field must be updated.")

        normalized_changes = dict(changes)
        for field_name in ("skills", "capabilities"):
            if field_name in normalized_changes and normalized_changes[field_name] is None:
                normalized_changes[field_name] = []
        if "labels" in normalized_changes and normalized_changes["labels"] is None:
            normalized_changes["labels"] = {}

        payload = current.to_dict()
        payload.update(normalized_changes)
        payload["revision"] = current.revision + 1
        payload["updated_at"] = _timestamp()
        updated = AgentDefinition.from_dict(payload)
        self._write_revision_and_current(updated)
        return updated

    def delete(self, name: str) -> Path:
        normalized = _agent_name(name)
        current_path = self.root / f"{normalized}.json"
        if not current_path.is_file():
            raise AgentDefinitionError(
                f"No agent definition named {normalized!r} was found under {self.root}."
            )
        try:
            current_path.unlink()
            history = self.revisions_root / normalized
            if history.exists():
                shutil.rmtree(history)
        except OSError as exc:
            raise AgentDefinitionError(
                f"Could not remove agent definition {normalized!r}: {exc}"
            ) from exc
        return current_path

    def _revision_path(self, name: str, revision: int) -> Path:
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise AgentDefinitionError(
                "Agent definition revision must be at least 1.")
        return self.revisions_root / name / f"{revision:08d}.json"

    def _read(self, path: Path) -> AgentDefinition:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise AgentDefinitionError(
                f"Could not read {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise AgentDefinitionError(
                f"Could not parse {path}: {exc}") from exc
        try:
            return AgentDefinition.from_dict(payload)
        except AgentDefinitionError as exc:
            raise AgentDefinitionError(
                f"Invalid agent definition {path}: {exc}") from exc

    def _write_revision_and_current(self, definition: AgentDefinition) -> None:
        ensure_user_subdir(self.root)
        revision_path = self._revision_path(
            definition.name, definition.revision)
        ensure_user_subdir(revision_path.parent)
        if revision_path.exists():
            raise AgentDefinitionError(
                f"Agent definition revision already exists: {revision_path}"
            )
        payload = definition.to_dict()
        _atomic_write_json(revision_path, payload)
        try:
            _atomic_write_json(self.root / f"{definition.name}.json", payload)
        except AgentDefinitionError:
            try:
                revision_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise


def validate_agent_definition(
    definition: AgentDefinition,
    *,
    cwd: str | os.PathLike[str] | None = None,
    config: AppConfig | None = None,
) -> AgentDefinitionValidation:
    runtime_config = config or AppConfig.from_env()
    errors: list[str] = []
    warnings: list[str] = []
    resolved: dict[str, Any] = {}

    workspace = _resolve_workspace(definition, cwd)
    resolved["workspace"] = str(workspace)
    if not workspace.is_dir():
        errors.append(f"Workspace directory does not exist: {workspace}")

    try:
        store = load_profile_store(runtime_config.model_profiles_path)
        profile = store.get(definition.profile)
        if definition.model:
            profile = replace(profile, model=definition.model)
        profile.resolved_provider()
        api_mode = profile.resolved_api_mode()
        profile.resolved_api_key()
        resolved["profile"] = {
            "name": profile.name,
            "provider": profile.provider,
            "api_mode": api_mode,
            "model": profile.model,
            "base_url": profile.base_url,
        }
    except ValueError as exc:
        errors.append(str(exc))

    if definition.prompt:
        prompt = resolve_resource(
            PROMPT_KIND, definition.prompt, user_dir=runtime_config.user_dir
        )
        if prompt is None:
            errors.append(f"Unknown prompt resource: {definition.prompt}")
        else:
            resolved["prompt"] = {
                "label": prompt.label, "path": str(prompt.path)}
    else:
        resolved["prompt"] = {
            "label": "<active>",
            "path": str(Path(runtime_config.system_prompt_path).expanduser().resolve()),
        }

    resolved_skills: list[str] = []
    if workspace.is_dir():
        for skill_ref in definition.skills:
            skill = resolve_available_skill(
                skill_ref, workspace, user_dir=runtime_config.user_dir
            )
            if skill is None:
                errors.append(f"Unknown skill resource: {skill_ref}")
                continue
            resolved_skills.append(skill.id)
    resolved["skills"] = resolved_skills

    available_tools: list[str] = []
    try:
        discovery = None
        external_specs = None
        if runtime_config.user_tools_enabled:
            from .extensions.loader import _discover, loaded_extension_tool_dirs

            extension_dir = user_extensions_dir(
                resolve_user_dir(runtime_config.user_dir)
            )
            extension_tool_dirs = loaded_extension_tool_dirs(
                extension_dir, _discover(extension_dir)
            )
            discovery = discover_external_tools(
                user_dir=runtime_config.user_dir,
                runner=runtime_config.tool_runner,
                describe_timeout=runtime_config.user_tool_describe_timeout,
                extra_tool_dirs=extension_tool_dirs,
            )
            external_specs = build_external_tool_specs(
                discovery,
                invoke_timeout=runtime_config.user_tool_timeout,
                runner_command=default_runner_command(discovery.runner),
                cwd=workspace,
            )
            if discovery.runner_message:
                warnings.append(discovery.runner_message)
            for entry in discovery.broken:
                warnings.append(
                    f"External tool {entry.script_path.name}: {entry.error}")
        registry = create_default_tool_registry(
            runtime_config,
            external_specs=external_specs,
            workspace=workspace,
            restrict_workspace=True,
        )
        available_tools = registry.names()
    except Exception as exc:
        warnings.append(f"Could not discover external tools: {exc}")
        registry = create_default_tool_registry(
            runtime_config,
            workspace=workspace,
            restrict_workspace=True,
        )
        available_tools = registry.names()

    requested_tools = definition.tools
    if requested_tools is None:
        resolved_tools = available_tools if runtime_config.tools_enabled else []
    else:
        missing_tools = [
            name for name in requested_tools if name not in available_tools]
        for tool_name in missing_tools:
            errors.append(f"Unknown or disabled tool: {tool_name}")
        resolved_tools = [
            name for name in requested_tools if name in available_tools]
        if requested_tools and not runtime_config.tools_enabled:
            errors.append("Tool calling is disabled by server configuration.")
    resolved["tools"] = resolved_tools
    resolved["max_iterations"] = (
        definition.max_iterations
        if definition.max_iterations is not None
        else runtime_config.max_iterations
    )

    return AgentDefinitionValidation(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(dict.fromkeys(warnings)),
        resolved=resolved,
    )


def build_agent_from_definition(
    name: str,
    *,
    revision: int | None = None,
    cwd: str | os.PathLike[str] | None = None,
    user_dir: str | os.PathLike[str] | None = None,
):
    """Resolve a stored definition and construct a workspace-bound Agent."""
    from .agent import Agent

    store = AgentDefinitionStore(user_dir)
    definition = store.get(name, revision=revision)
    config = AppConfig.from_env()
    if user_dir is not None:
        config.user_dir = str(resolve_user_dir(user_dir))

    validation = validate_agent_definition(definition, cwd=cwd, config=config)
    if not validation.valid:
        details = "; ".join(validation.errors)
        raise AgentDefinitionError(
            f"Agent definition {definition.name!r} is invalid: {details}"
        )

    if definition.max_iterations is not None:
        config.max_iterations = definition.max_iterations
    if definition.prompt:
        config.system_prompt_path = validation.resolved["prompt"]["path"]
    if definition.tools == ():
        config.tools_enabled = False

    workspace = Path(validation.resolved["workspace"])
    skill_ids = list(validation.resolved["skills"])
    project_context, context_sources = load_full_context(
        workspace,
        user_dir=config.user_dir,
        loaded_user_skills=skill_ids,
    )
    agent = Agent(
        profile=definition.profile,
        model=definition.model,
        config=config,
        project_context=project_context,
        project_context_files=[source.label for source in context_sources],
        allowed_tools=list(
            definition.tools) if definition.tools is not None else None,
        workspace=workspace,
        restrict_workspace=True,
    )
    return agent, definition, validation


def _resolve_workspace(
    definition: AgentDefinition,
    cwd: str | os.PathLike[str] | None,
) -> Path:
    base = Path(cwd or os.getcwd()).expanduser().resolve()
    if not definition.workspace:
        return base
    configured = Path(definition.workspace).expanduser()
    if configured.is_absolute():
        return configured.resolve()
    return (base / configured).resolve()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2,
                       sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise AgentDefinitionError(f"Could not write {path}: {exc}") from exc


def _agent_name(value: Any) -> str:
    name = _required_string(value, "name")
    if not _AGENT_NAME.fullmatch(name):
        raise AgentDefinitionError(
            "Agent name must start with a letter or underscore and contain only "
            "letters, digits, underscores, or hyphens."
        )
    return name


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AgentDefinitionError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AgentDefinitionError(
            "Optional text fields must be strings or null.")
    return value.strip() or None


def _integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AgentDefinitionError(f"{field_name} must be an integer.")
    return value


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise AgentDefinitionError(
            f"{field_name} must be an array of strings.")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise AgentDefinitionError(
                f"{field_name} must contain only non-empty strings."
            )
        text = item.strip()
        if text not in seen:
            normalized.append(text)
            seen.add(text)
    return tuple(normalized)


def _labels(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise AgentDefinitionError(
            "labels must be an object of string values.")
    labels: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise AgentDefinitionError(
                "labels must use non-empty string keys.")
        if not isinstance(item, str):
            raise AgentDefinitionError(
                "labels must contain only string values.")
        labels[key.strip()] = item
    return labels
