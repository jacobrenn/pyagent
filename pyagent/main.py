#!/usr/bin/env python
"""PyAgent command line interface.

Output contract
---------------
Every command either prints a human-readable rendering (default) or, with
``--json``, a single JSON envelope on stdout. See ``pyagent/cli_output.py``
for the full contract. In short: ``ok`` in the envelope matches the exit
code (``0`` success, ``1`` runtime failure, ``2`` usage/validation error).
"""

from __future__ import annotations

from importlib.metadata import version
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
from typing import Any

from .agent import Agent
from .cli_output import (
    EXIT_ERROR,
    EXIT_OK,
    EXIT_USAGE,
    CliError,
    Result,
    emit,
    emit_error,
    note,
    render_detail,
    render_table,
)
from .cli_run import run_and_report
from .config import AppConfig
from .model_profiles import load_profile_store
from .project_context import load_full_context, resolve_available_skill
from .resources import install_resource, kind_for_name, list_resources, remove_resource, resolve_resource, resource_dir


def get_version():
    try:
        return version('pyagent-harness')
    except Exception:
        return 'Unknown'


def _parse_skills_arg(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _validate_skills(
    skills: list[str],
    *,
    cwd: str | None = None,
    user_dir: str | None = None,
) -> list[str]:
    missing: list[str] = []
    validated: list[str] = []
    seen: set[str] = set()
    base_cwd = cwd or os.getcwd()
    for skill_name in skills:
        skill = resolve_available_skill(
            skill_name, base_cwd, user_dir=user_dir)
        if skill is None:
            missing.append(skill_name)
            continue
        if skill.id in seen:
            continue
        seen.add(skill.id)
        validated.append(skill.id)
    if missing:
        missing_list = ", ".join(missing)
        raise CliError(
            f"Unknown skill(s): {missing_list}\n"
            "Skills must be specified as scoped IDs (user:<path> or project:<path>) "
            "or as existing user-skill paths relative to ~/.pyagent/skills/.",
            code="unknown_skill",
            exit_code=EXIT_USAGE,
        )
    return validated


def build_agent_for_request(
    *,
    profile: str | None = None,
    model: str | None = None,
    cwd: str | None = None,
    skills: list[str] | None = None,
) -> Agent:
    base_cwd = cwd or os.getcwd()
    validated_skills = _validate_skills(skills or [], cwd=base_cwd)
    project_context, context_sources = load_full_context(
        base_cwd,
        loaded_user_skills=validated_skills,
    )
    project_context_files = [source.label for source in context_sources]
    agent = Agent(
        profile=profile,
        model=model,
        project_context=project_context,
        project_context_files=project_context_files,
        workspace=base_cwd,
    )
    agent.run_metadata["skills"] = validated_skills
    return agent


def run_single_shot(
    *,
    prompt: str,
    profile: str | None = None,
    model: str | None = None,
    skills: list[str] | None = None,
) -> str:
    agent = build_agent_for_request(
        profile=profile,
        model=model,
        skills=skills,
    )
    try:
        _, summary = run_and_report(
            agent, prompt, command="run", quiet=True, source="cli")
    finally:
        agent.close(reason="single_shot_complete")
    return str(summary.get("response", ""))


def run_agent_definition(
    *,
    name: str,
    prompt: str,
    revision: int | None = None,
    cwd: str | None = None,
) -> str:
    from .agent_definitions import build_agent_from_definition

    agent, _, _ = build_agent_from_definition(name, revision=revision, cwd=cwd)
    try:
        code, summary = run_and_report(
            agent, prompt, command="agents.run", quiet=True, source="cli")
    finally:
        agent.close(reason="single_shot_complete")
    if code != EXIT_OK:
        raise ValueError(summary.get("error") or "Agent run failed")
    return str(summary.get("response", ""))


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _json_mode(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "json", False))


def _command_name(args: argparse.Namespace) -> str:
    command = getattr(args, "command", None)
    if command is None:
        return "run" if getattr(args, "prompt", None) is not None else "tui"
    for attr in ("session_action", "profile_action", "agent_action", "mcp_action",
                 "ext_action", "resource_action"):
        sub = getattr(args, attr, None)
        if sub:
            return f"{command}.{sub}"
    return str(command)


def _result(command: str, data: Any, human: Any = None, exit_code: int = EXIT_OK) -> Result:
    return Result(command=command, data=data, human=human, exit_code=exit_code)


def _yes_no(value: bool | None) -> str:
    if value is None:
        return "-"
    return "yes" if value else "no"


# ---------------------------------------------------------------------------
# Sessions / resume
# ---------------------------------------------------------------------------

def _session_summary_dict(saved) -> dict[str, Any]:
    from .session_views import saved_run_summary

    return saved_run_summary(saved)


def _session_detail_dict(saved, ancestors: list[str], lineage_warning: str | None) -> dict[str, Any]:
    from .session_views import saved_run_detail

    detail = saved_run_detail(saved)
    detail["ancestors"] = list(ancestors)
    detail["lineage_warning"] = lineage_warning
    return detail


def _handle_sessions_command(args: argparse.Namespace) -> Result:
    from .resume import restore_session
    from .session_store import SessionStore, format_sessions

    store = SessionStore(AppConfig.from_env().user_dir)

    if args.command == "sessions":
        if args.session_action == "capabilities":
            enabled = bool(getattr(AppConfig.from_env(),
                           "session_api_enabled", False))
            data = {"resume_supported": True, "enabled": enabled}
            return _result(
                "sessions.capabilities",
                data,
                lambda: render_detail("Session capabilities", [
                    ("resume_supported", "yes"),
                    ("enabled", _yes_no(enabled)),
                ]),
            )
        if args.session_action == "list":
            if args.limit < 1:
                raise CliError("--limit must be positive",
                               code="usage_error", exit_code=EXIT_USAGE)
            if args.offset < 0:
                raise CliError("--offset must be zero or positive",
                               code="usage_error", exit_code=EXIT_USAGE)
            all_runs = store.list()
            runs = [run for run in all_runs if run.run_id is not None]
            page = runs[args.offset:args.offset + args.limit]
            data = {
                "runs": [_session_summary_dict(run) for run in page],
                "total": len(runs),
                "unavailable_count": len(all_runs) - len(runs),
                "offset": args.offset,
                "limit": args.limit,
            }

            def human() -> str:
                text = format_sessions(page, args.limit)
                if args.offset:
                    text += f"\n(offset {args.offset}, total {len(runs)})"
                return text

            return _result("sessions.list", data, human)

        # sessions show
        saved = store.inspect(args.run_id)
        chain, warning = store.lineage(args.run_id)
        detail = _session_detail_dict(saved, chain[1:], warning)

        def human() -> str:
            lines = [
                f"Run:         {detail.get('run_id')}",
                f"Timestamp:   {detail.get('timestamp')}",
                f"Outcome:     {detail.get('outcome')} ({'resumable' if detail.get('resumable') else detail.get('reason')})",
                f"Profile:     {detail.get('profile') or '-'}",
                f"Model:       {detail.get('model') or '-'}",
                f"Workspace:   {detail.get('workspace') or '-'}",
                f"Agent:       {detail.get('agent') or '-'}"
                + (f" (revision {detail.get('agent_revision')})" if detail.get("agent_revision") else ""),
                f"Parent:      {detail.get('parent_run_id') or '-'}",
                f"Conversation:{detail.get('conversation_id') or '-'}",
                f"Prompt:      {detail.get('prompt_preview') or ''}",
                f"Messages:    {len(detail.get('messages') or [])} (use --json to view full history)",
            ]
            if detail.get("skills"):
                lines.append(f"Skills:      {', '.join(detail['skills'])}")
            for warning_line in detail.get("warnings") or []:
                lines.append(f"Warning:     {warning_line}")
            if detail.get("ancestors"):
                lines.append(f"Ancestors:   {', '.join(detail['ancestors'])}")
            if detail.get("lineage_warning"):
                lines.append(f"Lineage:     {detail['lineage_warning']}")
            return "\n".join(lines)

        return _result("sessions.show", detail, human)

    # resume
    if args.skills is not None:
        raise CliError(
            "Resume restores saved skills; --skills cannot override them.",
            code="usage_error",
            exit_code=EXIT_USAGE,
        )
    restored = restore_session(
        args.run_id, store=store, profile=args.profile, model=args.model, cwd=args.cwd)
    if args.prompt is None:
        # Interactive TUI; not part of the JSON contract.
        from .ui import PyAgentApp

        app = PyAgentApp(resumed_session=restored)
        try:
            app.run()
        finally:
            app.agent.close(reason="tui_exit")
        return None  # type: ignore[return-value]

    if not args.prompt.strip():
        restored.agent.close(reason="resume_complete")
        raise CliError("--prompt must not be empty",
                       code="usage_error", exit_code=EXIT_USAGE)

    agent = restored.agent
    warnings = list(restored.warnings)
    note(restored.notice())
    try:
        code, summary = run_and_report(
            agent,
            args.prompt,
            command="resume",
            json_output=_json_mode(args),
            stream=getattr(args, "stream", False),
            with_messages=getattr(args, "with_messages", False),
            source="cli",
            warnings=warnings,
            extra={"resumed_from": args.run_id},
        )
    finally:
        agent.close(reason="resume_complete")
    if code != EXIT_OK:
        # run_and_report already rendered the failure (envelope or stderr).
        return Result(command="resume", data=summary,
                      exit_code=EXIT_ERROR, already_emitted=True)
    return Result(command="resume", data=summary, exit_code=EXIT_OK, already_emitted=True)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def _parse_agent_labels(values: list[str] | None) -> dict[str, str]:
    labels: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise CliError(f"Invalid label {value!r}; expected KEY=VALUE.",
                           code="usage_error", exit_code=EXIT_USAGE)
        key, item = value.split("=", 1)
        if not key.strip():
            raise CliError("Agent label keys must not be empty.",
                           code="usage_error", exit_code=EXIT_USAGE)
        labels[key.strip()] = item
    return labels


def _agent_changes_from_args(args: argparse.Namespace, *, creating: bool) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    fields = (
        "description",
        "profile",
        "model",
        "prompt_resource",
        "workspace",
        "max_iterations",
    )
    for field_name in fields:
        value = getattr(args, field_name, None)
        if value is not None:
            target = "prompt" if field_name == "prompt_resource" else field_name
            changes[target] = value
    if creating and "description" not in changes:
        changes["description"] = ""
    if getattr(args, "skill", None) is not None:
        changes["skills"] = args.skill
    if getattr(args, "no_tools", False):
        changes["tools"] = []
    elif getattr(args, "tool", None) is not None:
        changes["tools"] = args.tool
    if getattr(args, "capability", None) is not None:
        changes["capabilities"] = args.capability
    if getattr(args, "label", None) is not None:
        changes["labels"] = _parse_agent_labels(args.label)

    # --clear-* flags (update only)
    clear_map = {
        "clear_profile": "profile",
        "clear_model": "model",
        "clear_prompt": "prompt",
        "clear_workspace": "workspace",
        "clear_max_iterations": "max_iterations",
        "clear_description": "description",
    }
    for flag, field in clear_map.items():
        if getattr(args, flag, False):
            changes[field] = "" if field == "description" else None
    if getattr(args, "clear_skills", False):
        changes["skills"] = []
    if getattr(args, "clear_tools", False):
        changes["tools"] = None
    if getattr(args, "clear_capabilities", False):
        changes["capabilities"] = []
    if getattr(args, "clear_labels", False):
        changes["labels"] = {}
    if getattr(args, "tool", None) is not None and getattr(args, "clear_tools", False):
        raise CliError("--tool and --clear-tools cannot be used together.",
                       code="usage_error", exit_code=EXIT_USAGE)
    return changes


def _agent_summary_line(item: Any) -> list[Any]:
    return [
        item.name,
        item.revision,
        item.profile or "<default>",
        item.model or "<profile>",
        item.workspace or ".",
    ]


def _handle_agents_command(args: argparse.Namespace) -> Result | None:
    from .agent_definitions import (
        AgentDefinitionError,
        AgentDefinitionStore,
        validate_agent_definition,
    )

    store = AgentDefinitionStore()
    try:
        if args.agent_action == "list":
            definitions = store.list()
            data = {
                "root": str(store.root),
                "agents": [item.to_dict() for item in definitions],
            }

            def human() -> str:
                if not definitions:
                    return f"No agent definitions found in {store.root}."
                rows = [_agent_summary_line(item) for item in definitions]
                return render_table(["Name", "Revision", "Profile", "Model", "Workspace"], rows)

            return _result("agents.list", data, human)

        if args.agent_action == "show":
            definition = store.get(args.name, revision=args.revision)
            data = definition.to_dict()

            def human() -> str:
                return json.dumps(data, indent=2, sort_keys=True)

            return _result("agents.show", data, human)

        if args.agent_action == "revisions":
            revisions = store.revisions(args.name)
            data = {
                "name": args.name,
                "revisions": [item.to_dict() for item in revisions],
            }

            def human() -> str:
                if not revisions:
                    return f"No revisions found for `{args.name}`."
                rows = [[item.revision, item.updated_at,
                         item.profile or "<default>", item.model or "<profile>"]
                        for item in revisions]
                return render_table(["Revision", "Updated", "Profile", "Model"], rows)

            return _result("agents.revisions", data, human)

        if args.agent_action == "create":
            payload = {"name": args.name,
                       **_agent_changes_from_args(args, creating=True)}
            definition = store.create(payload)
            data = definition.to_dict()
            return _result(
                "agents.create",
                data,
                lambda: f"Created agent `{definition.name}` revision {definition.revision} at {store.root}.",
            )

        if args.agent_action == "update":
            definition = store.update(
                args.name, _agent_changes_from_args(args, creating=False)
            )
            data = definition.to_dict()
            return _result(
                "agents.update",
                data,
                lambda: f"Updated agent `{definition.name}` to revision {definition.revision}.",
            )

        if args.agent_action == "validate":
            definition = store.get(args.name, revision=args.revision)
            result = validate_agent_definition(definition, cwd=args.cwd)
            data = {
                "name": definition.name,
                "revision": definition.revision,
                **result.to_dict(),
            }

            def human() -> str:
                lines = [f"Agent `{definition.name}` revision {definition.revision}: "
                         + ("valid" if result.valid else "INVALID")]
                for error in result.errors:
                    lines.append(f"  error: {error}")
                for warning in result.warnings:
                    lines.append(f"  warning: {warning}")
                return "\n".join(lines)

            return _result(
                "agents.validate",
                data,
                human,
                exit_code=EXIT_OK if result.valid else EXIT_USAGE,
            )

        if args.agent_action == "remove":
            removed = store.delete(args.name)
            data = {"name": args.name, "path": str(removed)}
            return _result(
                "agents.remove",
                data,
                lambda: f"Removed agent definition {removed}.",
            )

        if args.agent_action == "run":
            from .agent_definitions import build_agent_from_definition

            agent, definition, _validation = build_agent_from_definition(
                args.name,
                revision=args.revision,
                cwd=args.cwd,
            )
            try:
                code, summary = run_and_report(
                    agent,
                    args.message,
                    command="agents.run",
                    json_output=_json_mode(args),
                    stream=getattr(args, "stream", False),
                    with_messages=getattr(args, "with_messages", False),
                    source="cli",
                    extra={"agent": definition.name,
                           "revision": definition.revision},
                )
            finally:
                agent.close(reason="single_shot_complete")
            if code != EXIT_OK:
                return Result(command="agents.run", data=summary,
                              exit_code=EXIT_ERROR, already_emitted=True)
            return Result(command="agents.run", data=summary,
                          exit_code=EXIT_OK, already_emitted=True)

        if args.agent_action == "chat":
            from .agent_definitions import build_agent_from_definition
            from .ui import PyAgentApp

            agent, definition, validation = build_agent_from_definition(
                args.name,
                revision=args.revision,
                cwd=args.cwd,
            )
            app = None
            try:
                app = PyAgentApp(
                    agent=agent,
                    agent_definition=definition,
                    agent_validation=validation,
                )
                app.run()
            finally:
                (app.agent if app is not None else agent).close(reason="tui_exit")
            return None

    except AgentDefinitionError as exc:
        raise CliError(str(exc), code="agent_definition_error",
                       exit_code=EXIT_USAGE) from exc

    raise CliError(f"Unknown agents action: {args.agent_action}",
                   code="usage_error", exit_code=EXIT_USAGE)


# ---------------------------------------------------------------------------
# Resources (skills / tools / prompts)
# ---------------------------------------------------------------------------

def _resource_payload(kind_name: str) -> tuple[Any, Any]:
    kind = kind_for_name(kind_name)
    return kind, resource_dir(kind)


def _resource_list_data(kind_name: str) -> dict[str, Any]:
    kind, root = _resource_payload(kind_name)
    resources = list_resources(kind)
    return {
        "kind": kind.name,
        "root": str(root),
        "items": [{"label": r.label, "path": str(r.path)} for r in resources],
    }


def _resource_human_list(kind_name: str) -> str:
    kind, root = _resource_payload(kind_name)
    resources = list_resources(kind)
    if not resources:
        return f"No {kind.name}s installed in {root}"
    lines = [f"{kind.name.title()}s in {root}:"]
    lines.extend(f"- {r.label}" for r in resources)
    return "\n".join(lines)


def _read_installed_resource(kind_name: str, target: str) -> tuple[str, Any]:
    kind, root = _resource_payload(kind_name)
    resource = resolve_resource(kind, target)
    if resource is None:
        raise CliError(
            f"No {kind.name} named {target!r} was found under {root}.",
            code="not_found",
            exit_code=EXIT_ERROR,
        )
    try:
        content = resource.path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CliError(
            f"Could not read {resource.path}: {exc}", code="io_error") from exc
    except UnicodeDecodeError as exc:
        raise CliError(f"Could not read {resource.path} as UTF-8 text: {exc}",
                       code="decode_error") from exc
    return content, resource


def _activate_system_prompt(target: str) -> tuple[Path, Path]:
    kind, root = _resource_payload("prompts")
    resource = resolve_resource(kind, target)
    if resource is None:
        raise CliError(
            f"No {kind.name} named {target!r} was found under {root}.",
            code="not_found",
            exit_code=EXIT_ERROR,
        )

    config = AppConfig.from_env()
    destination = Path(config.system_prompt_path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(f".{destination.name}.tmp")
        shutil.copyfile(resource.path, tmp)
        shutil.move(str(tmp), str(destination))
    except OSError as exc:
        raise CliError(f"Could not copy {resource.path} to {destination}: {exc}",
                       code="io_error") from exc
    return resource.path, destination


def _tool_file_display_name(name: str) -> str:
    file_name = Path(name.strip()).name
    return file_name if file_name.endswith(".py") else f"{file_name}.py"


def _handle_tool_toggle_command(args: argparse.Namespace) -> Result:
    from .external_tools import move_tool_script

    enable = args.resource_action == "enable"
    new_path, error = move_tool_script(args.name, enable=enable)
    if error is not None:
        raise CliError(error, code="tool_toggle_failed", exit_code=EXIT_ERROR)

    verb = "Enabled" if enable else "Disabled"
    display_name = _tool_file_display_name(args.name)
    data = {"name": display_name, "path": str(new_path), "enabled": enable}
    return _result(
        f"tools.{args.resource_action}",
        data,
        lambda: f"{verb} tool `{display_name}` at {new_path}",
    )


def _handle_resource_command(args: argparse.Namespace) -> Result:
    kind = kind_for_name(args.command)
    action = args.resource_action
    command = f"{args.command}.{action}"

    if kind.name == "tool" and action in {"enable", "disable"}:
        return _handle_tool_toggle_command(args)

    if action == "list":
        data = _resource_list_data(args.command)
        return _result(command, data, lambda: _resource_human_list(args.command))

    if action == "install":
        result = install_resource(
            kind,
            args.source,
            name=args.name,
            force=args.force,
        )
        data = {
            "kind": kind.name,
            "label": result.destination.name,
            "path": str(result.destination),
            "bytes_written": result.bytes_written,
        }
        return _result(
            command,
            data,
            lambda: (
                f"Installed {kind.name} `{result.destination.name}` to {result.destination} "
                f"({result.bytes_written} bytes)"
            ),
        )

    if action == "show":
        content, resource = _read_installed_resource(args.command, args.name)
        data = {
            "kind": kind.name,
            "label": resource.label,
            "path": str(resource.path),
            "content": content,
        }
        return _result(command, data, lambda: content)

    if action == "use":
        source, destination = _activate_system_prompt(args.name)
        data = {"source": str(source), "destination": str(destination)}
        return _result(
            command,
            data,
            lambda: f"Copied prompt {source} to active system prompt {destination}",
        )

    if action == "remove":
        removed = remove_resource(kind, args.name)
        data = {"kind": kind.name, "path": str(removed)}
        return _result(command, data, lambda: f"Removed {kind.name} {removed}")

    raise CliError(f"Unknown {kind.name} action: {action}",
                   code="usage_error", exit_code=EXIT_USAGE)


# ---------------------------------------------------------------------------
# Skills (available skills, not just installed files)
# ---------------------------------------------------------------------------

def _skill_entry(skill: Any) -> dict[str, Any]:
    return {
        "id": skill.id,
        "scope": skill.scope,
        "label": skill.label,
        "title": skill.title,
        "preview": skill.preview,
        "error": skill.error,
        "path": str(skill.path),
    }


def _handle_skills_list(args: argparse.Namespace) -> Result:
    from .project_context import list_available_skills

    cwd = args.cwd or os.getcwd()
    skills = list_available_skills(cwd)
    if getattr(args, "user_only", False):
        skills = [s for s in skills if s.scope == "user"]
    data = {
        "cwd": cwd,
        "skills": [_skill_entry(skill) for skill in skills],
    }

    def human() -> str:
        if not skills:
            return f"No skills found (cwd: {cwd})."
        lines = [f"Available skills (cwd: {cwd}):"]
        for skill in skills:
            marker = f"  error: {skill.error}" if skill.error else ""
            title = f" — {skill.title}" if skill.title and skill.title != skill.label else ""
            lines.append(f"- {skill.id}{title}{marker}")
        return "\n".join(lines)

    return _result("skills.list", data, human)


# ---------------------------------------------------------------------------
# Tools (registry view)
# ---------------------------------------------------------------------------

def _handle_tools_list(args: argparse.Namespace) -> Result:
    from .tool_report import tools_report

    report = tools_report(AppConfig.from_env())

    def human() -> str:
        lines = [
            f"Tools (enabled={_yes_no(report['tools_enabled'])}, "
            f"builtin={_yes_no(report['builtin_tools_enabled'])}, "
            f"user={_yes_no(report['user_tools_enabled'])})",
            f"User dir: {report['user_dir']}",
            f"Runner:   {report['runner']['name']}"
            + ("" if report['runner'].get('available')
               is not False else " (unavailable)"),
        ]
        if report.get("discovery_error"):
            lines.append(f"Discovery error: {report['discovery_error']}")
        if report.get("mcp_error"):
            lines.append(f"MCP error: {report['mcp_error']}")

        def section(title: str, items: list[dict[str, Any]]) -> None:
            lines.append("")
            lines.append(f"{title} ({len(items)}):")
            for item in items:
                desc = item.get("description") or ""
                lines.append(f"- {item['name']}" +
                             (f" — {desc}" if desc else ""))

        section("Built-in", report["builtin"])
        section("External", report["external"])
        section("MCP", report["mcp"])
        if report["files"]:
            lines.append("")
            lines.append(f"Files ({len(report['files'])}):")
            for entry in report["files"]:
                suffix = " [disabled]" if entry.get("disabled") else ""
                lines.append(f"- {entry['label']}{suffix}")
        if report["broken"]:
            lines.append("")
            lines.append(f"Broken ({len(report['broken'])}):")
            for entry in report["broken"]:
                lines.append(f"- {entry['script_path']}: {entry.get('error')}")
        if report["collisions"]:
            lines.append("")
            lines.append(f"Collisions ({len(report['collisions'])}):")
            for entry in report["collisions"]:
                lines.append(
                    f"- {entry['name']} (loser: {entry.get('loser_origin')})")
        return "\n".join(lines)

    return _result("tools.list", report, human)


def _handle_tools_new(args: argparse.Namespace) -> Result:
    from .scaffold import ScaffoldError, create_user_tool

    try:
        created = create_user_tool(
            args.name, user_dir=AppConfig.from_env().user_dir)
    except ScaffoldError as exc:
        raise CliError(str(exc), code="scaffold_failed",
                       exit_code=EXIT_ERROR) from exc
    data = {"path": str(created)}
    return _result(
        "tools.new",
        data,
        lambda: f"Created starter tool at `{created}`.\nEdit it, then run `pyagent tools list` to see it.",
    )


def _handle_tools_path(args: argparse.Namespace) -> Result:
    from .external_tools import find_tool_script

    config = AppConfig.from_env()
    located = find_tool_script(args.name, user_dir=config.user_dir)
    if located is None:
        raise CliError(
            f"No tool named {args.name!r} was found under {config.user_dir}/tools/.",
            code="not_found",
            exit_code=EXIT_ERROR,
        )
    data = {"name": args.name, "path": str(located)}
    return _result("tools.path", data, lambda: str(located))


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

def _load_profile_store():
    config = AppConfig.from_env()
    try:
        return load_profile_store(config.model_profiles_path), config
    except (OSError, ValueError) as exc:
        raise CliError(str(exc), code="profile_store_error",
                       exit_code=EXIT_ERROR) from exc


def _profile_dict(profile: Any, default_profile: str) -> dict[str, Any]:
    from .profile_views import profile_to_dict

    return profile_to_dict(profile, default_profile)


def _save_profile_store(store) -> None:
    from .model_profiles import save_profile_store

    try:
        save_profile_store(store)
    except (OSError, ValueError) as exc:
        raise CliError(str(exc), code="profile_store_error",
                       exit_code=EXIT_ERROR) from exc


def _profile_human(profile: dict[str, Any]) -> str:
    lines = [
        f"Name:          {profile['name']}",
        f"Provider:      {profile['provider']}",
        f"API mode:      {profile['api_mode']}",
        f"Model:         {profile['model']}",
        f"Base URL:      {profile['base_url']}",
        f"API key env:   {profile.get('api_key_env') or '-'}",
        f"Inline key:    {'yes' if profile.get('has_inline_api_key') else 'no'}",
        f"Default:       {'yes' if profile.get('is_default') else 'no'}",
    ]
    headers = profile.get("headers") or {}
    if headers:
        lines.append("Headers:")
        for key, value in headers.items():
            lines.append(f"  {key}: {value}")
    redacted = profile.get("redacted_headers") or []
    if redacted:
        lines.append(f"Redacted headers: {', '.join(redacted)}")
    if profile.get("httpx_kwargs"):
        lines.append(
            f"httpx_kwargs:  {json.dumps(profile['httpx_kwargs'], sort_keys=True)}")
    return "\n".join(lines)


def _normalize_profile_name(value: str) -> str:
    name = value.strip()
    if not name:
        raise CliError("Profile name must not be empty.",
                       code="usage_error", exit_code=EXIT_USAGE)
    if "/" in name or "\\" in name:
        raise CliError("Profile names must not contain path separators.",
                       code="usage_error", exit_code=EXIT_USAGE)
    return name


def _parse_headers(values: list[str] | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    for item in values or []:
        if ":" not in item:
            raise CliError(f"--header entries must be 'Name: Value', got {item!r}",
                           code="usage_error", exit_code=EXIT_USAGE)
        key, value = item.split(":", 1)
        key = key.strip()
        if not key:
            raise CliError(f"--header entry has an empty name: {item!r}",
                           code="usage_error", exit_code=EXIT_USAGE)
        headers[key] = value.strip()
    return headers


def _parse_json_object(raw: str, flag: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CliError(f"{flag} must be a JSON object: {exc}",
                       code="usage_error", exit_code=EXIT_USAGE) from exc
    if not isinstance(value, dict):
        raise CliError(f"{flag} must be a JSON object.",
                       code="usage_error", exit_code=EXIT_USAGE)
    return value


def _handle_profiles_command(args: argparse.Namespace) -> Result:
    from .model_profiles import (
        ModelProfile,
        default_base_url_for_provider,
        normalize_model_profile,
        remove_profile,
        set_default_profile,
        update_profile_store,
    )

    store, config = _load_profile_store()
    action = args.profile_action

    if action == "list":
        from .profile_views import profile_list_payload

        effective_default = config.default_profile or store.default_profile
        data = profile_list_payload(
            store,
            effective_default=effective_default,
            default_overridden_by_env=config.default_profile is not None,
        )

        def human() -> str:
            rows = []
            for profile in data["profiles"]:
                name = profile["name"]
                if name == data["default_profile"]:
                    name = f"* {name}"
                rows.append([name, profile["provider"], profile["api_mode"],
                             profile["model"], profile["base_url"]])
            header = f"Default profile: {data['default_profile']}"
            if data["default_overridden_by_env"]:
                header += f" (overridden by PYAGENT_PROFILE={config.default_profile})"
            if not rows:
                return f"{header}\n\nNo profiles configured."
            return f"{header}\n\n" + render_table(
                ["Name", "Provider", "API Mode", "Model", "Base URL"], rows
            )

        return _result("profiles.list", data, human)

    if action == "show":
        name = _normalize_profile_name(args.name)
        if name not in store.profiles:
            raise CliError(
                f"Unknown profile '{name}'. Available profiles: {', '.join(store.names()) or '<none>'}",
                code="profile_not_found",
                exit_code=EXIT_ERROR,
            )
        data = _profile_dict(store.get(name), store.default_profile)
        return _result("profiles.show", data, lambda: _profile_human(data))

    if action == "create":
        headers = _parse_headers(args.header)
        httpx_kwargs = _parse_json_object(
            args.httpx_kwargs, "--httpx-kwargs") if args.httpx_kwargs else {}
        try:
            base_url = args.base_url or default_base_url_for_provider(
                args.provider)
            profile = ModelProfile(
                name=_normalize_profile_name(args.name),
                provider=args.provider,
                model=args.model,
                base_url=base_url,
                api_mode=args.api_mode,
                api_key=args.api_key,
                api_key_env=args.api_key_env,
                headers=headers,
                httpx_kwargs=httpx_kwargs,
            )
            profile = normalize_model_profile(profile)
        except ValueError as exc:
            raise CliError(str(exc), code="invalid_profile",
                           exit_code=EXIT_USAGE) from exc
        update_profile_store(store, profile, make_default=args.make_default)
        _save_profile_store(store)
        data = _profile_dict(store.get(profile.name), store.default_profile)
        return _result(
            "profiles.create",
            data,
            lambda: f"Created profile `{profile.name}`"
            + (" (default)" if store.default_profile == profile.name else ""),
        )

    if action == "update":
        name = _normalize_profile_name(args.name)
        if name not in store.profiles:
            raise CliError(
                f"Unknown profile '{name}'. Available profiles: {', '.join(store.names()) or '<none>'}",
                code="profile_not_found",
                exit_code=EXIT_ERROR,
            )
        current = store.get(name)
        values: dict[str, Any] = {
            "provider": current.provider,
            "model": current.model,
            "base_url": current.base_url,
            "api_mode": current.api_mode,
            "api_key": current.api_key,
            "api_key_env": current.api_key_env,
            "headers": dict(current.headers),
            "httpx_kwargs": dict(current.httpx_kwargs),
        }
        for field_name in ("provider", "model", "base_url", "api_mode", "api_key", "api_key_env"):
            value = getattr(args, field_name, None)
            if value is not None:
                values[field_name] = value
        if args.header is not None:
            values["headers"] = _parse_headers(args.header)
        if args.httpx_kwargs is not None:
            values["httpx_kwargs"] = _parse_json_object(
                args.httpx_kwargs, "--httpx-kwargs")
        if getattr(args, "clear_api_key", False):
            values["api_key"] = None
        if getattr(args, "clear_api_key_env", False):
            values["api_key_env"] = None
        if getattr(args, "clear_headers", False):
            values["headers"] = {}
        if getattr(args, "clear_httpx_kwargs", False):
            values["httpx_kwargs"] = {}
        try:
            profile = normalize_model_profile(
                ModelProfile(name=name, **values))
        except ValueError as exc:
            raise CliError(str(exc), code="invalid_profile",
                           exit_code=EXIT_USAGE) from exc
        update_profile_store(store, profile)
        _save_profile_store(store)
        data = _profile_dict(store.get(name), store.default_profile)
        return _result("profiles.update", data, lambda: f"Updated profile `{name}`.")

    if action == "remove":
        name = _normalize_profile_name(args.name)
        try:
            remove_profile(store, name)
        except ValueError as exc:
            raise CliError(str(exc), code="profile_remove_failed",
                           exit_code=EXIT_ERROR) from exc
        _save_profile_store(store)
        data = {"name": name, "default_profile": store.default_profile}
        return _result("profiles.remove", data, lambda: f"Removed profile `{name}`.")

    if action == "default":
        name = _normalize_profile_name(args.name)
        try:
            set_default_profile(store, name)
        except ValueError as exc:
            raise CliError(str(exc), code="profile_not_found",
                           exit_code=EXIT_ERROR) from exc
        _save_profile_store(store)
        data = _profile_dict(store.get(name), store.default_profile)
        return _result("profiles.default", data, lambda: f"Default profile set to `{name}`.")

    if action == "models":
        name = _normalize_profile_name(args.name)
        if name not in store.profiles:
            raise CliError(
                f"Unknown profile '{name}'. Available profiles: {', '.join(store.names()) or '<none>'}",
                code="profile_not_found",
                exit_code=EXIT_ERROR,
            )
        profile = store.get(name)
        from .llm_client import build_chat_client

        try:
            profile.resolved_api_key()
            client = build_chat_client(profile, timeout=config.request_timeout)
        except ValueError as exc:
            raise CliError(str(exc), code="profile_error",
                           exit_code=EXIT_ERROR) from exc
        try:
            result = client.list_models()
        except Exception as exc:
            raise CliError(str(exc), code="upstream_error",
                           exit_code=EXIT_ERROR) from exc
        finally:
            try:
                client.close()
            except Exception:
                pass
        if not isinstance(result, dict):
            raise CliError("The model endpoint returned an invalid model listing.",
                           code="upstream_error", exit_code=EXIT_ERROR)
        if result.get("error"):
            raise CliError(
                str(result["error"]), code="upstream_error", exit_code=EXIT_ERROR)
        models = [str(m) for m in result.get("models", [])
                  if isinstance(m, str) and m]
        models = list(dict.fromkeys(models))
        data = {"profile": name, "models": models}

        def human() -> str:
            if not models:
                return f"No models reported by profile `{name}`."
            return "\n".join([f"Models for `{name}`:", *[f"- {m}" for m in models]])

        return _result("profiles.models", data, human)

    raise CliError(
        f"Unknown profiles action: {action}", code="usage_error", exit_code=EXIT_USAGE)


# ---------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------

def _mcp_config():
    from .mcp_config import McpConfigError, load_mcp_config

    try:
        return load_mcp_config()
    except McpConfigError as exc:
        raise CliError(str(exc), code="mcp_config_error",
                       exit_code=EXIT_ERROR) from exc


def _mcp_entry(server, status=None) -> dict[str, Any]:
    from .mcp_report import mcp_entry_dict

    return mcp_entry_dict(server, status)


def _mcp_status_map(config):
    from .mcp_report import mcp_status_map

    return mcp_status_map(config)


def _mcp_human_entry(entry: dict[str, Any]) -> list[str]:
    lines = [
        f"Name:      {entry['name']}",
        f"Transport: {entry['transport']}",
        f"Enabled:   {_yes_no(entry['enabled'])}",
        f"Status:    {entry.get('status') or 'unknown'}",
    ]
    if entry["transport"] == "stdio":
        lines.append(
            f"Command:   {entry.get('command') or ''} {' '.join(entry.get('args') or [])}".rstrip())
        if entry.get("cwd"):
            lines.append(f"CWD:       {entry['cwd']}")
        if entry.get("env"):
            lines.append("Env:       " + ", ".join(sorted(entry["env"])))
    else:
        lines.append(f"URL:       {entry.get('url') or ''}")
        if entry.get("headers"):
            lines.append("Headers:   " + ", ".join(sorted(entry["headers"])))
    if entry.get("missing_env"):
        lines.append(f"Missing env: {', '.join(entry['missing_env'])}")
    if entry.get("error"):
        lines.append(f"Error:     {entry['error']}")
    tools = entry.get("tools") or []
    if tools:
        lines.append(f"Tools ({len(tools)}):")
        lines.extend(f"  - mcp__{entry['name']}__{tool}" for tool in tools)
    return lines


def _handle_mcp_command(args: argparse.Namespace) -> Result:
    from .mcp_config import (
        McpConfigError,
        McpServerConfig,
        default_mcp_config_path,
        remove_server,
        server_from_dict,
        server_to_dict,
        set_server_enabled,
        upsert_server,
    )

    config = AppConfig.from_env()
    action = args.mcp_action

    if action == "list":
        parsed = _mcp_config()
        enabled = bool(getattr(config, "mcp_enabled", True))
        statuses: dict[str, Any] = {}
        status_error: str | None = None
        if not args.no_connect and enabled:
            statuses, status_error = _mcp_status_map(config)
        servers = [_mcp_entry(s, statuses.get(s.name)) for s in parsed.servers]
        data = {
            "enabled": enabled,
            "config_path": str(parsed.path),
            "servers": servers,
            "error": status_error,
        }

        def human() -> str:
            lines: list[str] = []
            if not enabled:
                lines.append(
                    "(MCP is disabled via PYAGENT_MCP_ENABLED=off; showing configuration only)")
            if not servers:
                lines.append(
                    f"No MCP servers configured in {parsed.path}.\n"
                    "Add one with: pyagent mcp add <name> --command <cmd> [--arg ...]\n"
                    "or:         pyagent mcp add <name> --url <url>"
                )
                return "\n".join(lines)
            rows = []
            for entry in servers:
                if not entry["enabled"]:
                    state, detail = "disabled", ""
                elif entry.get("status") == "ok":
                    state, detail = "ok", f"{entry.get('tool_count', 0)} tool(s)"
                elif entry.get("status") in (None, "unknown"):
                    state, detail = "unknown", "(not connected)"
                else:
                    state, detail = "error", entry.get("error") or ""
                target = (
                    f"{entry.get('command') or ''} {' '.join(entry.get('args') or [])}".strip(
                    )
                    if entry["transport"] == "stdio"
                    else (entry.get("url") or "")
                )
                rows.append(
                    [entry["name"], entry["transport"], state, detail, target])
            lines.append(render_table(
                ["Name", "Transport", "Status", "Tools/Error", "Target"], rows))
            if status_error:
                lines.append(
                    f"Warning: could not query MCP servers: {status_error}")
            lines.append(f"\nConfig: {parsed.path}")
            return "\n".join(lines)

        return _result("mcp.list", data, human)

    if action == "show":
        parsed = _mcp_config()
        server = parsed.get(args.name)
        if server is None:
            raise CliError(f"No MCP server named {args.name!r}.",
                           code="not_found", exit_code=EXIT_ERROR)
        statuses, _ = _mcp_status_map(config) if getattr(
            config, "mcp_enabled", True) else ({}, None)
        entry = _mcp_entry(server, statuses.get(server.name))
        return _result("mcp.show", entry, lambda: "\n".join(_mcp_human_entry(entry)))

    if action == "add":
        env: dict[str, str] = {}
        for item in args.env:
            if "=" not in item:
                raise CliError(f"--env entries must be KEY=VALUE, got {item!r}",
                               code="usage_error", exit_code=EXIT_USAGE)
            key, value = item.split("=", 1)
            key = key.strip()
            if not key:
                raise CliError(f"--env entry has an empty key: {item!r}",
                               code="usage_error", exit_code=EXIT_USAGE)
            env[key] = value
        headers: dict[str, str] = {}
        for item in args.headers:
            if ":" not in item:
                raise CliError(f"--header entries must be 'Name: Value', got {item!r}",
                               code="usage_error", exit_code=EXIT_USAGE)
            key, value = item.split(":", 1)
            headers[key.strip()] = value.strip()
        command = getattr(args, "exec_command", None)
        transport = "stdio" if command else "http"
        if command and args.headers:
            raise CliError("--header is only valid with --url",
                           code="usage_error", exit_code=EXIT_USAGE)
        if args.url and (args.env or args.args or args.cwd):
            raise CliError("--env/--arg/--cwd are only valid with --command",
                           code="usage_error", exit_code=EXIT_USAGE)
        try:
            server = McpServerConfig(
                name=args.name.strip(),
                transport=transport,
                enabled=not args.disabled,
                command=command.strip() if command else None,
                args=tuple(args.args),
                env=env,
                cwd=args.cwd,
                url=args.url.strip() if args.url else None,
                headers=headers,
            )
            if server.transport == "http" and not (args.url or "").strip().lower().startswith(("http://", "https://")):
                raise CliError("--url must start with http:// or https://",
                               code="usage_error", exit_code=EXIT_USAGE)
            upsert_server(server, user_dir=config.user_dir, replace=args.force)
        except McpConfigError as exc:
            raise CliError(str(exc), code="mcp_config_error",
                           exit_code=EXIT_USAGE) from exc
        state = "disabled" if args.disabled else "enabled"
        entry = _mcp_entry(server, None)
        data = {"server": entry, "config_path": str(default_mcp_config_path())}

        def human() -> str:
            lines = [
                f"Added MCP server `{server.name}` ({state}) in {default_mcp_config_path()}"]
            if not args.disabled:
                lines.append(f"Run `pyagent mcp test {server.name}` to verify it, "
                             "or restart pyagent / run /mcp reload in the TUI.")
            return "\n".join(lines)

        result = _result("mcp.add", data, human)
        if args.test:
            test_result = _mcp_test_result(config, server.name)
            if test_result is not None:
                return test_result
        return result

    if action == "update":
        parsed = _mcp_config()
        existing = parsed.get(args.name)
        if existing is None:
            raise CliError(f"No MCP server named {args.name!r}.",
                           code="not_found", exit_code=EXIT_ERROR)
        data = server_to_dict(existing)
        if getattr(args, "exec_command", None) is not None:
            data["command"] = args.exec_command
            data.pop("url", None)
        if args.url is not None:
            data["url"] = args.url
            data.pop("command", None)
            data.pop("args", None)
            data.pop("env", None)
            data.pop("cwd", None)
        if args.args is not None:
            data["args"] = list(args.args)
        if args.env is not None:
            env: dict[str, str] = {}
            for item in args.env:
                if "=" not in item:
                    raise CliError(f"--env entries must be KEY=VALUE, got {item!r}",
                                   code="usage_error", exit_code=EXIT_USAGE)
                key, value = item.split("=", 1)
                env[key.strip()] = value
            data["env"] = env
        if args.headers is not None:
            headers = {}
            for item in args.headers:
                if ":" not in item:
                    raise CliError(f"--header entries must be 'Name: Value', got {item!r}",
                                   code="usage_error", exit_code=EXIT_USAGE)
                key, value = item.split(":", 1)
                headers[key.strip()] = value.strip()
            data["headers"] = headers
        if args.cwd is not None:
            data["cwd"] = args.cwd
        if getattr(args, "clear_cwd", False):
            data.pop("cwd", None)
        if getattr(args, "clear_env", False):
            data.pop("env", None)
        if getattr(args, "clear_headers", False):
            data.pop("headers", None)
        if getattr(args, "clear_args", False):
            data.pop("args", None)
        if getattr(args, "enable", False):
            data["enabled"] = True
        if getattr(args, "disable", False):
            data["enabled"] = False
        try:
            updated = server_from_dict(args.name, data)
            upsert_server(updated, user_dir=config.user_dir, replace=True)
        except McpConfigError as exc:
            raise CliError(str(exc), code="mcp_config_error",
                           exit_code=EXIT_USAGE) from exc
        entry = _mcp_entry(updated, None)
        return _result("mcp.update", entry,
                       lambda: f"Updated MCP server `{updated.name}`.")

    if action == "remove":
        removed = remove_server(args.name, user_dir=config.user_dir)
        if not removed:
            raise CliError(f"No MCP server named {args.name!r}.",
                           code="not_found", exit_code=EXIT_ERROR)
        data = {"name": args.name}
        return _result("mcp.remove", data, lambda: f"Removed MCP server `{args.name}`.")

    if action in {"enable", "disable"}:
        enabled = action == "enable"
        try:
            updated = set_server_enabled(
                args.name, enabled, user_dir=config.user_dir)
        except McpConfigError as exc:
            raise CliError(str(exc), code="mcp_config_error",
                           exit_code=EXIT_USAGE) from exc
        if updated is None:
            raise CliError(f"No MCP server named {args.name!r}.",
                           code="not_found", exit_code=EXIT_ERROR)
        entry = _mcp_entry(updated, None)

        def human() -> str:
            lines = [
                f"MCP server `{updated.name}` {'enabled' if enabled else 'disabled'}."]
            if enabled:
                lines.append(f"Run `pyagent mcp test {updated.name}` to verify it, "
                             "or restart pyagent / run /mcp reload in the TUI.")
            return "\n".join(lines)

        return _result(f"mcp.{action}", entry, human)

    if action == "test":
        result = _mcp_test_result(config, args.name)
        if result is None:  # pragma: no cover - _mcp_test_result always returns
            raise CliError("MCP test failed",
                           code="mcp_test_failed", exit_code=EXIT_ERROR)
        return result

    if action == "tools":
        parsed = _mcp_config()
        targets = [
            s for s in parsed.servers if args.name is None or s.name == args.name]
        if args.name is not None and not targets:
            raise CliError(f"No MCP server named {args.name!r}.",
                           code="not_found", exit_code=EXIT_ERROR)
        from .mcp_runtime import get_mcp_manager

        manager = get_mcp_manager()
        results = []
        failures = 0
        for server in targets:
            if not server.enabled:
                results.append({"name": server.name, "status": "disabled",
                                "tool_count": 0, "tools": [], "error": None})
                continue
            result = manager.test_server(
                server.name, user_dir=config.user_dir,
                timeout=config.mcp_connect_timeout,
            )
            entry = {
                "name": server.name,
                "status": "ok" if result.get("ok") else "error",
                "tool_count": result.get("tool_count", 0),
                "tools": list(result.get("tools") or []),
                "latency_ms": result.get("latency_ms"),
                "error": result.get("error"),
            }
            if not result.get("ok"):
                failures += 1
            results.append(entry)
        data = {"servers": results}

        def human() -> str:
            lines: list[str] = []
            for entry in results:
                if entry["status"] == "disabled":
                    lines.append(f"{entry['name']}: disabled")
                    continue
                if entry["status"] == "ok":
                    lines.append(
                        f"{entry['name']}: {entry['tool_count']} tool(s) in {entry.get('latency_ms')} ms")
                    lines.extend(
                        f"  - mcp__{entry['name']}__{tool}" for tool in entry["tools"])
                else:
                    lines.append(
                        f"{entry['name']}: error — {entry.get('error')}")
            return "\n".join(lines)

        return _result("mcp.tools", data, human,
                       exit_code=EXIT_ERROR if failures else EXIT_OK)

    raise CliError(
        f"Unknown mcp action: {action}", code="usage_error", exit_code=EXIT_USAGE)


def _mcp_test_result(config: AppConfig, name: str) -> Result:
    from .mcp_runtime import get_mcp_manager

    result = get_mcp_manager().test_server(
        name, user_dir=config.user_dir, timeout=config.mcp_connect_timeout,
    )
    if result.get("ok"):
        data = {
            "server": name,
            "ok": True,
            "transport": result.get("transport"),
            "latency_ms": result.get("latency_ms"),
            "tool_count": result.get("tool_count", 0),
            "tools": list(result.get("tools") or []),
            "error": None,
        }

        def human() -> str:
            lines = [
                f"MCP server `{name}` OK — transport={data['transport']}, "
                f"{data['tool_count']} tool(s), handshake {data['latency_ms']} ms",
            ]
            lines.extend(f"  - mcp__{name}__{tool}" for tool in data["tools"])
            return "\n".join(lines)

        return _result("mcp.test", data, human)

    error = str(result.get("error") or "unknown error")
    data = {
        "server": name,
        "ok": False,
        "transport": result.get("transport"),
        "latency_ms": result.get("latency_ms"),
        "tool_count": 0,
        "tools": [],
        "error": error,
    }
    return _result(
        "mcp.test",
        data,
        lambda: f"MCP server `{name}` FAILED: {error}",
        exit_code=EXIT_ERROR,
    )


# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------

class _ExtensionCliBus:
    """Minimal bus shim for disk-only extension CLI commands."""

    @staticmethod
    def loaded_extensions() -> list[str]:
        return []

    @staticmethod
    def clear() -> None:
        return None


class _ExtensionCliLog:
    @staticmethod
    def error(_message: str, _details: Any | None = None) -> None:
        return None


class _ExtensionCliAgent:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.bus = _ExtensionCliBus()
        self._ext_log = _ExtensionCliLog()

    @staticmethod
    def _rebuild_external_tools() -> None:
        return None


def _extension_path(ext_dir: Path, name: str) -> Path:
    for candidate in (ext_dir / name, ext_dir / f"{name}.py"):
        if candidate.exists():
            return candidate.resolve()
    return (ext_dir / name).resolve()


def _extension_entry(name: str, state: str, path: Path) -> dict[str, Any]:
    return {"name": name, "state": state, "path": str(path)}


def _handle_extensions_command(args: argparse.Namespace) -> Result:
    from .extensions.loader import _discover
    from .extensions.manager import (
        _cmd_disable,
        _cmd_enable,
        _cmd_new,
        _cmd_remove,
        install_extension_from_git,
    )
    from .user_runtime import resolve_user_dir, user_extensions_dir

    config = AppConfig.from_env()
    action = args.ext_action
    agent = _ExtensionCliAgent(config)

    resolved_user_dir = resolve_user_dir(config.user_dir)
    ext_dir = user_extensions_dir(resolved_user_dir)
    disabled_dir = ext_dir / "disabled"

    if action == "list":
        enabled_names = _discover(ext_dir)
        disabled_names = _discover(disabled_dir)
        enabled = [_extension_entry(n, "enabled", _extension_path(
            ext_dir, n)) for n in enabled_names]
        disabled = [_extension_entry(n, "disabled", _extension_path(
            disabled_dir, n)) for n in disabled_names]
        data = {
            "user_dir": str(resolved_user_dir),
            "extensions_dir": str(ext_dir),
            "enabled": enabled,
            "disabled": disabled,
        }

        def human() -> str:
            if not enabled and not disabled:
                return f"No extensions found in `{ext_dir}`."
            lines = ["Extensions:"]
            for item in enabled:
                lines.append(f"- {item['name']} [enabled]")
            for item in disabled:
                lines.append(f"- {item['name']} [disabled]")
            return "\n".join(lines)

        return _result("extensions.list", data, human)

    if action == "new":
        message = _cmd_new(agent, args.name, args.url)
        failed = _extension_action_failed(message)
        data = {"name": args.name, "url": args.url, "message": message}
        return _result(
            "extensions.new",
            data,
            lambda: message,
            exit_code=EXIT_ERROR if failed else EXIT_OK,
        )

    if action == "install":
        source = args.source
        name = args.name
        force = args.force
        from urllib import parse as _urlparse

        from .extension_resources import (
            install_extension_bytes,
            install_extension_url,
        )

        is_url = _urlparse.urlparse(source).scheme.lower() in {"http", "https"}
        if is_url:
            suffix = Path(_urlparse.urlparse(source).path).suffix.lower()
            if suffix in {".py", ".zip"}:
                try:
                    result = install_extension_url(
                        source, name=name, force=force, user_dir=config.user_dir
                    )
                except ValueError as exc:
                    raise CliError(str(exc), code="install_failed",
                                   exit_code=EXIT_ERROR) from exc
                data = {
                    "name": result.name,
                    "path": str(result.destination),
                    "bytes_written": result.bytes_written,
                    "source": source,
                }
                return _result(
                    "extensions.install",
                    data,
                    lambda: (
                        f"Installed extension `{result.name}` to {result.destination} "
                        f"({result.bytes_written} bytes)"
                    ),
                )
            # Git repository URL
            if force:
                raise CliError(
                    "--force is supported for direct .py/.zip URLs and local files, "
                    "not Git repository installs.",
                    code="usage_error",
                    exit_code=EXIT_USAGE,
                )
            repo_name = name
            if not repo_name:
                raise CliError(
                    "Git repository extension installs require an explicit name.",
                    code="usage_error",
                    exit_code=EXIT_USAGE,
                )
            message = install_extension_from_git(
                repo_name, source, user_dir=config.user_dir)
            failed = _extension_action_failed(message)
            data = {"name": repo_name, "source": source, "message": message}
            return _result(
                "extensions.install",
                data,
                lambda: message,
                exit_code=EXIT_ERROR if failed else EXIT_OK,
            )

        # Local file
        path = Path(source).expanduser()
        if not path.is_file():
            raise CliError(f"Source file does not exist: {path}",
                           code="not_found", exit_code=EXIT_ERROR)
        try:
            data_bytes = path.read_bytes()
        except OSError as exc:
            raise CliError(f"Could not read {path}: {exc}",
                           code="io_error", exit_code=EXIT_ERROR) from exc
        try:
            result = install_extension_bytes(
                data_bytes,
                source_name=path.name,
                name=name,
                force=force,
                user_dir=config.user_dir,
            )
        except ValueError as exc:
            raise CliError(str(exc), code="install_failed",
                           exit_code=EXIT_ERROR) from exc
        data = {
            "name": result.name,
            "path": str(result.destination),
            "bytes_written": result.bytes_written,
            "source": str(path),
        }
        return _result(
            "extensions.install",
            data,
            lambda: (
                f"Installed extension `{result.name}` to {result.destination} "
                f"({result.bytes_written} bytes)"
            ),
        )

    if action == "enable":
        message = _cmd_enable(agent, args.name)
        failed = _extension_action_failed(message)
        data = {"name": args.name, "state": "enabled", "message": message}
        return _result(
            "extensions.enable",
            data,
            lambda: message,
            exit_code=EXIT_ERROR if failed else EXIT_OK,
        )

    if action == "disable":
        message = _cmd_disable(agent, args.name)
        failed = _extension_action_failed(message)
        data = {"name": args.name, "state": "disabled", "message": message}
        return _result(
            "extensions.disable",
            data,
            lambda: message,
            exit_code=EXIT_ERROR if failed else EXIT_OK,
        )

    if action == "remove":
        message = _cmd_remove(agent, args.name)
        failed = _extension_action_failed(message)
        data = {"name": args.name, "message": message}
        return _result(
            "extensions.remove",
            data,
            lambda: message,
            exit_code=EXIT_ERROR if failed else EXIT_OK,
        )

    raise CliError(f"Unknown extensions action: {action}",
                   code="usage_error", exit_code=EXIT_USAGE)


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


# ---------------------------------------------------------------------------
# Parser and dispatch
# ---------------------------------------------------------------------------

def _add_run_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stream", action="store_true",
        help="Stream assistant output as it arrives (plain text, or NDJSON with --json)",
    )
    parser.add_argument(
        "--with-messages", action="store_true",
        help="Include the full message history in the JSON result (implies --json payload size)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PyAgent",
        epilog="Every command supports --json for a machine-readable envelope. "
               "Exit codes: 0 success, 1 runtime failure, 2 usage/validation error.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"PyAgent: {get_version()}",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="Emit a JSON envelope on stdout instead of human-readable text")
    parser.add_argument("--json", action="store_true",
                        help="Emit a JSON envelope on stdout instead of human-readable text")
    parser.add_argument(
        "--profile", help="Saved model profile to use (overrides PYAGENT_PROFILE)")
    parser.add_argument(
        "--model", help="Model name override for the active profile")
    parser.add_argument(
        "--skills",
        help="Comma-separated skill IDs (user:<path> or project:<path>) or user skill paths (supported only with --prompt)",
    )
    parser.add_argument(
        "--cwd", help="Working directory for the agent (default: current directory)")
    parser.add_argument("--prompt", type=str,
                        help="Single prompt to run and exit")
    _add_run_flags(parser)

    subparsers = parser.add_subparsers(dest="command")

    def add_json(p: argparse.ArgumentParser) -> None:
        # Reuse the shared --json action so the flag works at any level.
        for action in common._actions:  # type: ignore[attr-defined]
            p._add_action(action)  # type: ignore[attr-defined]

    def add_leaf(parent, name, **kwargs):
        kwargs.setdefault("parents", [common])
        return parent.add_parser(name, **kwargs)

    # --- sessions ---
    sessions_parser = subparsers.add_parser(
        "sessions", help="Inspect saved run checkpoints")
    add_json(sessions_parser)
    sessions_actions = sessions_parser.add_subparsers(
        dest="session_action", required=True)
    sessions_list = sessions_actions.add_parser(
        "list", help="List saved runs and parent links", parents=[common])
    sessions_list.add_argument("--limit", type=int, default=20)
    sessions_list.add_argument("--offset", type=int, default=0)
    sessions_show = sessions_actions.add_parser(
        "show", help="Show checkpoint history and ancestry", parents=[common])
    sessions_show.add_argument(
        "run_id", help="Full run UUID (or log basename for inspection)")
    sessions_actions.add_parser(
        "capabilities", help="Show whether saved-session resume is supported and enabled", parents=[common])

    # --- resume ---
    resume_parser = subparsers.add_parser(
        "resume", help="Continue a completed saved checkpoint")
    add_json(resume_parser)
    resume_parser.add_argument("run_id", help="Full run UUID")
    resume_parser.add_argument(
        "--prompt", default=argparse.SUPPRESS, help="Continue once and exit")
    resume_parser.add_argument(
        "--profile", default=argparse.SUPPRESS, help="Ad hoc profile override")
    resume_parser.add_argument(
        "--model", default=argparse.SUPPRESS, help="Ad hoc model override")
    resume_parser.add_argument("--cwd", help="Ad hoc replacement workspace")
    _add_run_flags(resume_parser)

    # --- serve / web ---
    serve_parser = subparsers.add_parser(
        "serve", help="Run the PyAgent HTTP API")
    serve_parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind")
    serve_parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind")

    web_parser = subparsers.add_parser(
        "web", help="Run a textual server to access PyAgent via a web browser"
    )
    web_parser.add_argument(
        "--profile",
        help="Saved model profile to use (overrides PYAGENT_PROFILE)"
    )
    web_parser.add_argument(
        "--model",
        help="Model name override for the active profile"
    )
    web_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind"
    )
    web_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind"
    )

    # --- version ---
    version_parser = subparsers.add_parser(
        "version", help="Print the PyAgent version")
    add_json(version_parser)

    # --- health ---
    health_parser = subparsers.add_parser(
        "health", help="Check that PyAgent is installed and importable")
    add_json(health_parser)

    # --- profiles ---
    profiles_parser = subparsers.add_parser(
        "profiles", help="Manage model profiles"
    )
    add_json(profiles_parser)
    profiles_subparsers = profiles_parser.add_subparsers(
        dest="profile_action", required=True)
    profiles_subparsers.add_parser(
        "list", help="Show available profiles", parents=[common])

    profile_show = profiles_subparsers.add_parser(
        "show", help="Show one profile (secrets redacted)", parents=[common])
    profile_show.add_argument("name")

    create_parser = profiles_subparsers.add_parser(
        "create", help="Create a new profile", parents=[common])
    create_parser.add_argument("--name", required=True, help="Profile name")
    create_parser.add_argument(
        "--provider", required=True, help="Model provider")
    create_parser.add_argument(
        "--base-url", "--base_url", dest="base_url",
        help="Base URL for the provider (defaults from provider)")
    create_parser.add_argument("--model", required=True, help="Model name")
    create_parser.add_argument(
        "--api-mode", "--api_mode", dest="api_mode",
        default="chat_completions",
        help="OpenAI API mode: chat_completions (default) or responses",
    )
    create_parser.add_argument(
        "--api-key", "--api_key", dest="api_key",
        help="Inline API key (prefer --api-key-env for shared machines)")
    create_parser.add_argument(
        "--api-key-env", "--api_key_env", dest="api_key_env",
        help="Environment variable for API key")
    create_parser.add_argument(
        "--header", action="append", default=None, metavar="NAME: VALUE",
        help="Extra HTTP header (repeatable)")
    create_parser.add_argument(
        "--httpx-kwargs", "--httpx_kwargs", dest="httpx_kwargs",
        help="JSON object of extra httpx client kwargs")
    create_parser.add_argument(
        "--default", dest="make_default", action="store_true",
        help="Make the new profile the default")

    update_parser = profiles_subparsers.add_parser(
        "update", help="Update a profile (partial; unspecified fields are kept)", parents=[common])
    update_parser.add_argument("name")
    update_parser.add_argument("--provider")
    update_parser.add_argument("--model")
    update_parser.add_argument(
        "--base-url", "--base_url", dest="base_url")
    update_parser.add_argument(
        "--api-mode", "--api_mode", dest="api_mode")
    update_parser.add_argument(
        "--api-key", "--api_key", dest="api_key")
    update_parser.add_argument(
        "--api-key-env", "--api_key_env", dest="api_key_env")
    update_parser.add_argument(
        "--header", action="append", default=None, metavar="NAME: VALUE",
        help="Replace all headers with these (repeatable)")
    update_parser.add_argument(
        "--httpx-kwargs", "--httpx_kwargs", dest="httpx_kwargs",
        help="JSON object of extra httpx client kwargs (replaces existing)")
    update_parser.add_argument("--clear-api-key", action="store_true",
                               help="Remove the inline API key")
    update_parser.add_argument("--clear-api-key-env", action="store_true",
                               help="Remove the API key environment variable reference")
    update_parser.add_argument("--clear-headers", action="store_true",
                               help="Remove all custom headers")
    update_parser.add_argument("--clear-httpx-kwargs", action="store_true",
                               help="Remove all httpx kwargs")

    remove_parser = profiles_subparsers.add_parser(
        "remove", help="Remove a profile (cannot remove the last or default profile)", parents=[common])
    remove_parser.add_argument("name")

    default_parser = profiles_subparsers.add_parser(
        "default", help="Set the default profile", parents=[common])
    default_parser.add_argument("name")

    models_parser = profiles_subparsers.add_parser(
        "models", help="List models exposed by a profile's endpoint", parents=[common])
    models_parser.add_argument("name")

    # --- resources (skills / tools / prompts) ---
    def add_resource_parser(
        name: str,
        singular: str,
        *,
        show: bool = False,
        use: bool = False,
        list_help: str | None = None,
    ) -> None:
        resource_parser = subparsers.add_parser(
            name,
            help=f"Manage installed {name}",
        )
        add_json(resource_parser)
        resource_subparsers = resource_parser.add_subparsers(
            dest="resource_action",
            required=True,
        )
        list_parser = resource_subparsers.add_parser(
            "list",
            help=list_help or f"List installed {name}",
            parents=[common],
        )
        if name == "skills":
            list_parser.add_argument(
                "--cwd", default=None,
                help="Project directory used to discover project skills (default: current directory)",
            )
            list_parser.add_argument(
                "--user-only", action="store_true",
                help="Only list installed user skills, not project skills",
            )
        install_parser = resource_subparsers.add_parser(
            "install",
            help=f"Install a {singular} from a local file or URL",
            parents=[common],
        )
        install_parser.add_argument(
            "source",
            help="Local file path or URL to install",
        )
        install_parser.add_argument(
            "--name",
            help=f"Destination filename to use under ~/.pyagent/{name}/",
        )
        install_parser.add_argument(
            "--force",
            action="store_true",
            help="Overwrite an existing file with the same name",
        )
        if show:
            show_parser = resource_subparsers.add_parser(
                "show",
                help=f"Print an installed {singular}",
                parents=[common],
            )
            show_parser.add_argument(
                "name",
                help=f"Installed {singular} filename or relative path",
            )
        if use:
            use_parser = resource_subparsers.add_parser(
                "use",
                help=f"Copy an installed {singular} to the active system prompt file",
                parents=[common],
            )
            use_parser.add_argument(
                "name",
                help=f"Installed {singular} filename or relative path",
            )
        remove_parser = resource_subparsers.add_parser(
            "remove",
            help=f"Remove an installed {singular}",
            parents=[common],
        )
        remove_parser.add_argument(
            "name",
            help=f"Installed {singular} filename or relative path",
        )
        if name == "tools":
            for action, help_text in (
                ("enable", "Move a tool out of tools/disabled/"),
                ("disable", "Move a tool into tools/disabled/"),
            ):
                action_parser = resource_subparsers.add_parser(
                    action, help=help_text, parents=[common])
                action_parser.add_argument(
                    "name",
                    metavar="tool_file",
                    help="Tool filename/path or name, with or without .py",
                )
            new_parser = resource_subparsers.add_parser(
                "new", help="Scaffold a starter tool script", parents=[common])
            new_parser.add_argument("name", help="Tool name (without .py)")
            path_parser = resource_subparsers.add_parser(
                "path", help="Print the path of an installed tool", parents=[common])
            path_parser.add_argument(
                "name", help="Tool name, with or without .py")

    add_resource_parser("skills", "skill",
                        list_help="List available skills (user and project)")
    add_resource_parser("tools", "tool")
    add_resource_parser("prompts", "prompt", show=True, use=True)

    # --- agents ---
    agents_parser = subparsers.add_parser(
        "agents", help="Manage reusable agent definitions"
    )
    add_json(agents_parser)
    agents_subparsers = agents_parser.add_subparsers(
        dest="agent_action", required=True
    )
    agents_subparsers.add_parser(
        "list", help="List agent definitions", parents=[common])

    agent_show = agents_subparsers.add_parser(
        "show", help="Show an agent definition", parents=[common])
    agent_show.add_argument("name")
    agent_show.add_argument("--revision", type=int)

    agent_revisions = agents_subparsers.add_parser(
        "revisions", help="List immutable revisions for an agent definition",
        parents=[common],
    )
    agent_revisions.add_argument("name")

    def add_agent_definition_fields(agent_parser: argparse.ArgumentParser) -> None:
        agent_parser.add_argument("--description")
        agent_parser.add_argument("--profile")
        agent_parser.add_argument("--model")
        agent_parser.add_argument("--prompt-resource", dest="prompt_resource")
        agent_parser.add_argument("--skill", action="append")
        agent_parser.add_argument("--tool", action="append")
        agent_parser.add_argument(
            "--no-tools", action="store_true", help="Disable tool calling for this agent"
        )
        agent_parser.add_argument("--workspace")
        agent_parser.add_argument("--max-iterations", type=int)
        agent_parser.add_argument(
            "--label", action="append", metavar="KEY=VALUE")
        agent_parser.add_argument("--capability", action="append")

    def add_agent_clear_fields(agent_parser: argparse.ArgumentParser) -> None:
        agent_parser.add_argument("--clear-description", action="store_true",
                                  help="Clear the description")
        agent_parser.add_argument("--clear-profile", action="store_true",
                                  help="Clear the profile (inherit default)")
        agent_parser.add_argument("--clear-model", action="store_true",
                                  help="Clear the model override")
        agent_parser.add_argument("--clear-prompt", action="store_true",
                                  help="Clear the prompt resource")
        agent_parser.add_argument("--clear-workspace", action="store_true",
                                  help="Clear the workspace override")
        agent_parser.add_argument("--clear-max-iterations", action="store_true",
                                  help="Clear the iteration limit override")
        agent_parser.add_argument("--clear-skills", action="store_true",
                                  help="Clear all skills")
        agent_parser.add_argument("--clear-tools", action="store_true",
                                  help="Clear the tool allowlist (inherit all)")
        agent_parser.add_argument("--clear-capabilities", action="store_true",
                                  help="Clear all capabilities")
        agent_parser.add_argument("--clear-labels", action="store_true",
                                  help="Clear all labels")

    agent_create = agents_subparsers.add_parser(
        "create", help="Create an agent definition", parents=[common])
    agent_create.add_argument("name")
    add_agent_definition_fields(agent_create)

    agent_update = agents_subparsers.add_parser(
        "update", help="Create a new agent revision", parents=[common])
    agent_update.add_argument("name")
    add_agent_definition_fields(agent_update)
    add_agent_clear_fields(agent_update)

    agent_validate = agents_subparsers.add_parser(
        "validate", help="Resolve and validate an agent definition", parents=[common])
    agent_validate.add_argument("name")
    agent_validate.add_argument("--revision", type=int)
    agent_validate.add_argument("--cwd")

    agent_remove = agents_subparsers.add_parser(
        "remove", help="Remove an agent definition and revision history", parents=[common])
    agent_remove.add_argument("name")

    agent_run = agents_subparsers.add_parser(
        "run", help="Run one prompt with an agent definition", parents=[common])
    agent_run.add_argument("name")
    agent_run.add_argument("message")
    agent_run.add_argument("--revision", type=int)
    agent_run.add_argument("--cwd")
    _add_run_flags(agent_run)

    agent_chat = agents_subparsers.add_parser(
        "chat", help="Open an interactive TUI with an agent definition", parents=[common])
    agent_chat.add_argument("name")
    agent_chat.add_argument("--revision", type=int)
    agent_chat.add_argument(
        "--cwd",
        help="Base directory used to resolve a relative agent workspace",
    )

    # --- MCP ---
    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Manage MCP (Model Context Protocol) servers",
    )
    add_json(mcp_parser)
    mcp_subparsers = mcp_parser.add_subparsers(
        dest="mcp_action", required=True
    )
    mcp_list = mcp_subparsers.add_parser(
        "list", help="List configured MCP servers and their status", parents=[common]
    )
    mcp_list.add_argument(
        "--no-connect",
        action="store_true",
        help="Only show configured servers; do not connect to them",
    )
    mcp_show = mcp_subparsers.add_parser(
        "show", help="Show one MCP server (secrets redacted)", parents=[common])
    mcp_show.add_argument("name")

    mcp_add = mcp_subparsers.add_parser(
        "add", help="Add an MCP server", parents=[common])
    mcp_add.add_argument("name", help="Server name ([a-z0-9_-], max 32 chars)")
    transport_group = mcp_add.add_mutually_exclusive_group(required=True)
    transport_group.add_argument(
        "--command", dest="exec_command",
        help="Command for a stdio server (e.g. npx)")
    transport_group.add_argument(
        "--url", help="URL for a Streamable HTTP server")
    mcp_add.add_argument(
        "--arg", dest="args", action="append", default=[],
        help="Argument for --command (repeatable; use --arg=-y for dash-prefixed values)",
    )
    mcp_add.add_argument(
        "--env", dest="env", action="append", default=[], metavar="KEY=VALUE",
        help="Environment variable for a stdio server (repeatable)",
    )
    mcp_add.add_argument(
        "--header", dest="headers", action="append", default=[], metavar="NAME: VALUE",
        help="HTTP header for an --url server (repeatable)",
    )
    mcp_add.add_argument("--cwd", help="Working directory for a stdio server")
    mcp_add.add_argument(
        "--disabled", action="store_true", help="Add the server disabled")
    mcp_add.add_argument(
        "--force", action="store_true", help="Overwrite an existing server with the same name")
    mcp_add.add_argument(
        "--test", action="store_true", help="Connect to the server after adding it")

    mcp_update = mcp_subparsers.add_parser(
        "update", help="Update an MCP server (partial; unspecified fields are kept)", parents=[common])
    mcp_update.add_argument("name")
    mcp_update.add_argument(
        "--command", dest="exec_command",
        help="Command for a stdio server (switches transport to stdio)")
    mcp_update.add_argument(
        "--url", help="URL for a Streamable HTTP server (switches transport to http)")
    mcp_update.add_argument(
        "--arg", dest="args", action="append", default=None,
        help="Replace the argument list (repeatable)")
    mcp_update.add_argument(
        "--env", dest="env", action="append", default=None, metavar="KEY=VALUE",
        help="Replace the environment map (repeatable)")
    mcp_update.add_argument(
        "--header", dest="headers", action="append", default=None, metavar="NAME: VALUE",
        help="Replace the header map (repeatable)")
    mcp_update.add_argument(
        "--cwd", help="Working directory for a stdio server")
    mcp_update.add_argument("--clear-cwd", action="store_true")
    mcp_update.add_argument("--clear-args", action="store_true")
    mcp_update.add_argument("--clear-env", action="store_true")
    mcp_update.add_argument("--clear-headers", action="store_true")
    mcp_update.add_argument(
        "--enable", action="store_true", help="Enable the server")
    mcp_update.add_argument(
        "--disable", action="store_true", help="Disable the server")

    mcp_remove = mcp_subparsers.add_parser(
        "remove", help="Remove an MCP server", parents=[common])
    mcp_remove.add_argument("name")
    for _action in ("enable", "disable"):
        _p = mcp_subparsers.add_parser(
            _action, help=f"{_action.capitalize()} an MCP server", parents=[common])
        _p.add_argument("name")
    mcp_test = mcp_subparsers.add_parser(
        "test", help="Connect to a server and report its tools", parents=[common])
    mcp_test.add_argument("name")
    mcp_tools = mcp_subparsers.add_parser(
        "tools", help="List tools exposed by MCP servers", parents=[common])
    mcp_tools.add_argument(
        "name", nargs="?", default=None,
        help="Only show tools from this server",
    )

    # --- extensions ---
    extensions_parser = subparsers.add_parser(
        "extensions",
        help="Manage installed extensions on disk",
    )
    add_json(extensions_parser)
    extensions_subparsers = extensions_parser.add_subparsers(
        dest="ext_action",
        required=True,
    )
    extensions_subparsers.add_parser(
        "list",
        help="List enabled and disabled extensions",
        parents=[common],
    )
    for action, help_text in (
        ("enable", "Move an extension out of extensions/disabled/"),
        ("disable", "Move an extension into extensions/disabled/"),
        ("remove", "Permanently delete an extension from disk"),
    ):
        action_parser = extensions_subparsers.add_parser(
            action, help=help_text, parents=[common])
        action_parser.add_argument("name", help="Extension name")

    new_parser = extensions_subparsers.add_parser(
        "new", help="Scaffold or install an extension", parents=[common]
    )
    new_parser.add_argument("name", help="Extension name")
    new_parser.add_argument(
        "url", nargs="?", help="GitHub URL to install from")

    install_ext = extensions_subparsers.add_parser(
        "install",
        help="Install an extension from a local file, a direct .py/.zip URL, or a git repository",
        parents=[common],
    )
    install_ext.add_argument(
        "source", help="Local path, direct .py/.zip URL, or git repository URL")
    install_ext.add_argument(
        "--name", help="Extension name (required for git repositories)")
    install_ext.add_argument("--force", action="store_true",
                             help="Overwrite an existing extension (not supported for git installs)")

    return parser


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _dispatch(args: argparse.Namespace) -> Result | None:
    """Route parsed args to a handler. Returns None for interactive commands."""
    command = args.command

    if command == "version":
        data = {"version": get_version()}
        return _result("version", data, lambda: f"PyAgent: {data['version']}")

    if command == "health":
        data = {"status": "ok", "version": get_version()}
        return _result("health", data, lambda: "ok")

    if command in {"sessions", "resume"}:
        return _handle_sessions_command(args)

    if command == "skills":
        if args.resource_action == "list":
            return _handle_skills_list(args)
        return _handle_resource_command(args)

    if command == "tools":
        if args.resource_action == "list":
            return _handle_tools_list(args)
        if args.resource_action == "new":
            return _handle_tools_new(args)
        if args.resource_action == "path":
            return _handle_tools_path(args)
        return _handle_resource_command(args)

    if command == "prompts":
        return _handle_resource_command(args)

    if command == "agents":
        return _handle_agents_command(args)

    if command == "profiles":
        return _handle_profiles_command(args)

    if command == "mcp":
        return _handle_mcp_command(args)

    if command == "extensions":
        return _handle_extensions_command(args)

    if command == "serve":
        try:
            import uvicorn
            from .api import create_app
        except (ImportError, RuntimeError) as exc:
            raise CliError(str(exc), code="missing_dependency",
                           exit_code=EXIT_USAGE) from exc
        uvicorn.run(create_app(), host=args.host, port=args.port)
        return None

    if command == "web":
        try:
            from textual_serve.server import Server
        except (ImportError, RuntimeError) as exc:
            raise CliError(str(exc), code="missing_dependency",
                           exit_code=EXIT_USAGE) from exc

        command_parts = ["python", "-m", "pyagent"]
        if args.profile:
            command_parts += ["--profile", args.profile]
        if args.model:
            command_parts += ["--model", args.model]
        command_line = " ".join(shlex.quote(part) for part in command_parts)

        server = Server(
            command=command_line,
            host=args.host,
            port=args.port,
            title='PyAgent',
        )
        server.serve()
        return None

    # Top-level single-shot / interactive mode
    if args.prompt is not None:
        parsed_skills = _parse_skills_arg(args.skills)
        try:
            agent = build_agent_for_request(
                profile=args.profile,
                model=args.model,
                cwd=args.cwd,
                skills=parsed_skills,
            )
        except CliError:
            raise
        except ValueError as exc:
            raise CliError(str(exc), code="usage_error",
                           exit_code=EXIT_USAGE) from exc
        try:
            code, summary = run_and_report(
                agent,
                args.prompt,
                command="run",
                json_output=_json_mode(args),
                stream=getattr(args, "stream", False),
                with_messages=getattr(args, "with_messages", False),
                source="cli",
            )
        finally:
            agent.close(reason="single_shot_complete")
        if code != EXIT_OK:
            return Result(command="run", data=summary,
                          exit_code=EXIT_ERROR, already_emitted=True)
        return Result(command="run", data=summary,
                      human=lambda: summary["response"],
                      exit_code=EXIT_OK, already_emitted=True)

    # Interactive TUI
    from .ui import PyAgentApp

    app = PyAgentApp(profile=args.profile, model=args.model)
    try:
        app.run()
    finally:
        app.agent.close(reason="tui_exit")
    return None


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --skills only makes sense with --prompt
    if getattr(args, "skills", None) and getattr(args, "prompt", None) is None and args.command is None:
        note("--skills is currently supported only with --prompt")
        raise SystemExit(EXIT_USAGE)

    command_name = _command_name(args)
    try:
        result = _dispatch(args)
    except CliError as exc:
        raise SystemExit(emit_error(command_name, exc,
                         json_output=_json_mode(args))) from exc
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:  # noqa: BLE001 - last-resort guard for a stable CLI contract
        raise SystemExit(emit_error(command_name, exc,
                         json_output=_json_mode(args))) from exc

    if result is not None:
        emit(result, json_output=_json_mode(args))
        raise SystemExit(result.exit_code)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
