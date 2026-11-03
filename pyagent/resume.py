"""Construct a fresh, policy-bound agent from a validated saved checkpoint."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .agent import Agent
from .agent_definitions import AgentDefinition, AgentDefinitionValidation, build_agent_from_definition
from .config import AppConfig
from .project_context import ContextSource, load_full_context, resolve_available_skill
from .run_logging import safe_endpoint
from .session_store import SavedRun, SessionError, SessionStore


@dataclass
class RestoredSession:
    agent: Agent
    checkpoint: SavedRun
    definition: AgentDefinition | None = None
    validation: AgentDefinitionValidation | None = None
    context_sources: list[ContextSource] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def notice(self) -> str:
        profile = self.agent.current_profile()
        lines = [f"Resumed checkpoint {self.checkpoint.run_id}.",
                 f"Runtime: {profile.name} / {profile.model} / {profile.resolved_api_mode()}",
                 f"Workspace: {self.agent.workspace}",
                 "Current instructions/resources apply; filesystem and extension/provider state were not restored.",
                 "File tools are workspace-bound; bash safety policy still applies but is not a sandbox."]
        if self.definition is not None:
            lines.append(
                f"Agent {self.definition.name} revision {self.definition.revision} remains pinned; its settings are read-only.")
        lines.extend(self.warnings)
        if not self.agent.logging_enabled:
            lines.append(
                "Logging is off: subsequent prompts will not be saved. Use /logging on or PYAGENT_LOGGING=on.")
        return "\n".join(lines)


def _text(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SessionError(f"Checkpoint is missing valid {key} metadata.")
    return value


def _strings(metadata: dict[str, Any], key: str) -> list[str]:
    value = metadata.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise SessionError(f"Invalid {key} metadata.")
    return value


def restore_session(
    reference: str,
    *,
    store: SessionStore | None = None,
    config: AppConfig | None = None,
    profile: str | None = None,
    model: str | None = None,
    cwd: str | None = None,
) -> RestoredSession:
    """Load history only. No model request or historical extension/tool replay.

    Installed extensions retain their normal startup lifecycle in the caller.
    Saved settings may restrict current policy, but never enable permissions.
    """
    config = replace(config) if config is not None else AppConfig.from_env()
    store = store or SessionStore(config.user_dir)
    config.user_dir = str(store.user_dir)
    saved = store.load(reference)
    metadata = saved.metadata
    saved_profile = _text(metadata, "profile")
    saved_model = _text(metadata, "model")
    saved_workspace = _text(metadata, "workspace")
    if not Path(saved_workspace).is_absolute():
        raise SessionError(
            "Saved workspace must be an absolute resolved path.")
    workspace = Path(cwd or saved_workspace).expanduser().resolve()
    if not workspace.is_dir():
        raise SessionError(
            f"Workspace does not exist: {workspace}. For ad hoc runs, specify --cwd.")
    skills = _strings(metadata, "skills")
    tools = _strings(metadata, "tools")
    settings = metadata.get("settings", {})
    if not isinstance(settings, dict):
        raise SessionError("Invalid runtime settings metadata.")
    for key in ("tools_enabled", "builtin_tools_enabled", "user_tools_enabled", "bash_enabled"):
        if key in settings:
            if type(settings[key]) is not bool:
                raise SessionError(f"Invalid {key} setting.")
            setattr(config, key, getattr(config, key) and settings[key])
    if "bash_readonly_mode" in settings:
        if type(settings["bash_readonly_mode"]) is not bool:
            raise SessionError("Invalid bash_readonly_mode setting.")
        config.bash_readonly_mode = config.bash_readonly_mode or settings["bash_readonly_mode"]

    name = metadata.get("agent")
    if name is not None:
        if not isinstance(name, str) or not name:
            raise SessionError("Invalid saved agent name.")
        if profile is not None or model is not None or cwd is not None:
            raise SessionError(
                "Named-agent resumes use the saved revision/workspace; runtime overrides are not allowed.")
        revision = metadata.get("agent_revision")
        if type(revision) is not int or revision < 1:
            raise SessionError(
                "Named-agent checkpoint must specify a revision.")

    agent = None
    try:
        definition = validation = None
        config.default_profile = profile or saved_profile
        if name is not None:
            agent, definition, validation = build_agent_from_definition(
                name, revision=revision, config=config, resolved_workspace=workspace)
            if definition.profile and definition.profile != saved_profile:
                raise SessionError(
                    "Saved profile conflicts with the pinned agent definition.")
            if definition.model and definition.model != saved_model:
                raise SessionError(
                    "Saved model conflicts with the pinned agent definition.")
            if not definition.model and agent.current_profile().model != saved_model:
                agent.set_model(saved_model)
            skills = list(validation.resolved["skills"])
            _, sources = load_full_context(
                workspace, user_dir=config.user_dir, loaded_user_skills=skills)
        else:
            resolved_skills = []
            for key in skills:
                skill = resolve_available_skill(
                    key, workspace, user_dir=config.user_dir)
                if skill is None:
                    raise SessionError(
                        f"Saved skill is unavailable: {key}. Restore the resource before resuming.")
                resolved_skills.append(skill.id)
            skills = resolved_skills
            context, sources = load_full_context(
                workspace, user_dir=config.user_dir, loaded_user_skills=skills)
            agent = Agent(
                profile=profile or saved_profile, model=model or saved_model,
                config=config, project_context=context,
                project_context_files=[source.label for source in sources],
                workspace=workspace, allowed_tools=tools, restrict_workspace=True,
            )
        # Restrict even after extension tool discovery; logs never enable tools.
        allowed = set(tools)
        if agent.allowed_tools is not None:
            allowed.intersection_update(agent.allowed_tools)
        agent.allowed_tools = frozenset(allowed)
        agent.tool_registry = agent.tool_registry.restricted(
            agent.allowed_tools)
        agent.tools = agent.tool_registry.definitions() if agent.config.tools_enabled else []
        agent.run_metadata["skills"] = skills
        warnings = list(saved.warnings)
        active = agent.current_profile()
        for key, actual in (
            ("profile", active.name), ("model", active.model),
            ("provider", active.provider), ("api_mode", active.resolved_api_mode()),
            ("base_url", safe_endpoint(active.base_url)
             ), ("workspace", str(workspace)),
        ):
            if metadata.get(key) != actual:
                warnings.append(f"Runtime changed: {key} now uses {actual}.")
        if sorted(tools) != sorted(item["function"]["name"] for item in agent.tools):
            warnings.append(
                "Available tools differ from the saved run; current restrictions apply.")
        for key in ("max_iterations", "max_history_messages", "max_previous_tool_result_chars",
                    "tools_enabled", "builtin_tools_enabled", "user_tools_enabled", "bash_enabled", "bash_readonly_mode"):
            if key in settings and settings[key] != getattr(agent.config, key):
                warnings.append(
                    f"Current {key} is {getattr(agent.config, key)} (saved: {settings[key]}).")
        agent.load_messages(saved.messages)
        if len(agent.messages) < len(saved.messages):
            warnings.append(
                "Restored model history was trimmed to the current history limit.")
        # load_messages/reset deliberately clear ancestry; establish it only
        # after validated restoration. No run/file is prepared until a prompt.
        agent.parent_run_id = saved.run_id
        return RestoredSession(agent, saved, definition, validation, sources, warnings)
    except Exception:
        if agent is not None:
            agent.close(reason="resume_failed")
        raise
