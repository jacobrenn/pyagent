#!/usr/bin/env python
from importlib.metadata import version
from tabulate import tabulate
import argparse
import os
from pathlib import Path
import shutil
import sys
from typing import Any

from .agent import Agent
from .config import AppConfig
from .model_profiles import load_profile_store
from .project_context import load_full_context, resolve_available_skill
from .resources import install_resource, kind_for_name, list_resources, remove_resource, resolve_resource, resource_dir


def get_version():
    try:
        return version('pyagent-harness')
    except Exception as e:
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
        raise ValueError(
            f"Unknown skill(s): {missing_list}\n"
            "Skills must be specified as scoped IDs (user:<path> or project:<path>) "
            "or as existing user-skill paths relative to ~/.pyagent/skills/."
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
    return Agent(
        profile=profile,
        model=model,
        project_context=project_context,
        project_context_files=project_context_files,
    )


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
    agent.load_extensions()
    response = ""
    for event in agent.run(prompt):
        if event.get("type") == "assistant_done":
            response = event["content"]
            break
    return response


def _print_resource_list(kind_name: str) -> None:
    kind = kind_for_name(kind_name)
    root = resource_dir(kind)
    resources = list_resources(kind)
    if not resources:
        sys.stdout.write(f"No {kind.name}s installed in {root}\n")
        sys.stdout.flush()
        return
    sys.stdout.write(f"{kind.name.title()}s in {root}:\n")
    for resource in resources:
        sys.stdout.write(f"- {resource.label}\n")
    sys.stdout.flush()


def _read_installed_resource(kind_name: str, target: str) -> str:
    kind = kind_for_name(kind_name)
    resource = resolve_resource(kind, target)
    if resource is None:
        root = resource_dir(kind)
        raise ValueError(
            f"No {kind.name} named {target!r} was found under {root}.")
    try:
        return resource.path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read {resource.path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Could not read {resource.path} as UTF-8 text: {exc}") from exc


def _activate_system_prompt(target: str) -> tuple[Path, Path]:
    kind = kind_for_name("prompts")
    resource = resolve_resource(kind, target)
    if resource is None:
        root = resource_dir(kind)
        raise ValueError(
            f"No {kind.name} named {target!r} was found under {root}.")

    config = AppConfig.from_env()
    destination = Path(config.system_prompt_path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(f".{destination.name}.tmp")
        shutil.copyfile(resource.path, tmp)
        shutil.move(str(tmp), str(destination))
    except OSError as exc:
        raise ValueError(
            f"Could not copy {resource.path} to {destination}: {exc}") from exc
    return resource.path, destination


def _tool_file_display_name(name: str) -> str:
    file_name = Path(name.strip()).name
    return file_name if file_name.endswith(".py") else f"{file_name}.py"


def _handle_tool_toggle_command(args: argparse.Namespace) -> None:
    from .external_tools import move_tool_script

    enable = args.resource_action == "enable"
    new_path, error = move_tool_script(args.name, enable=enable)
    if error is not None:
        sys.stderr.write(f"{error}\n")
        sys.stderr.flush()
        sys.exit(2)

    verb = "Enabled" if enable else "Disabled"
    display_name = _tool_file_display_name(args.name)
    sys.stdout.write(f"{verb} tool `{display_name}` at {new_path}\n")
    sys.stdout.flush()


def _handle_resource_command(args: argparse.Namespace) -> None:
    kind = kind_for_name(args.command)
    action = args.resource_action

    if kind.name == "tool" and action in {"enable", "disable"}:
        _handle_tool_toggle_command(args)
        return

    try:
        if action == "list":
            _print_resource_list(args.command)
            return
        if action == "install":
            result = install_resource(
                kind,
                args.source,
                name=args.name,
                force=args.force,
            )
            sys.stdout.write(
                f"Installed {kind.name} `{result.destination.name}` to {result.destination} "
                f"({result.bytes_written} bytes)\n"
            )
            sys.stdout.flush()
            return
        if action == "show":
            sys.stdout.write(_read_installed_resource(args.command, args.name))
            sys.stdout.flush()
            return
        if action == "use":
            source, destination = _activate_system_prompt(args.name)
            sys.stdout.write(
                f"Copied prompt {source} to active system prompt {destination}\n")
            sys.stdout.flush()
            return
        if action == "remove":
            removed = remove_resource(kind, args.name)
            sys.stdout.write(f"Removed {kind.name} {removed}\n")
            sys.stdout.flush()
            return
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        sys.stderr.flush()
        sys.exit(2)

    raise SystemExit(f"Unknown {kind.name} action: {action}")


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


def _handle_extensions_command(args: argparse.Namespace) -> None:
    from .extensions.manager import _cmd_disable, _cmd_enable, _cmd_remove
    from .extensions.loader import _discover
    from .user_runtime import resolve_user_dir, user_extensions_dir

    config = AppConfig.from_env()
    agent = _ExtensionCliAgent(config)

    if args.ext_action == "list":
        ext_dir = user_extensions_dir(resolve_user_dir(config.user_dir))
        disabled_dir = ext_dir / "disabled"
        on_disk = _discover(ext_dir)
        disabled = _discover(disabled_dir)
        if not on_disk and not disabled:
            print("No extensions found in `~/.pyagent/extensions/`.")
        else:
            lines = ["Extensions:"]
            for name in on_disk:
                lines.append(f"- {name} [enabled]")
            for name in disabled:
                lines.append(f"- {name} [disabled]")
            print("\n".join(lines))
        return
    if args.ext_action == "enable":
        print(_cmd_enable(agent, args.name))
        return
    if args.ext_action == "disable":
        print(_cmd_disable(agent, args.name))
        return
    if args.ext_action == "remove":
        print(_cmd_remove(agent, args.name))
        return
    raise SystemExit(f"Unknown extensions action: {args.ext_action}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run PyAgent")
    parser.add_argument(
        "--version",
        action="version",
        version=f"PyAgent: {get_version()}"
    )
    parser.add_argument(
        "--profile", help="Saved model profile to use (overrides PYAGENT_PROFILE)")
    parser.add_argument(
        "--model", help="Model name override for the active profile")
    parser.add_argument(
        "--skills",
        help="Comma-separated skill IDs (user:<path> or project:<path>) or user skill paths (supported only with --prompt)",
    )
    parser.add_argument("--prompt", type=str,
                        help="Single prompt to run and exit")
    subparsers = parser.add_subparsers(dest="command")
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
        help="Saved model provile to use (overrides PYAGENT_PROFILE)"
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
    profiles_parser = subparsers.add_parser(
        "profiles", help="Show available profiles"
    )

    def add_resource_parser(
        name: str,
        singular: str,
        *,
        show: bool = False,
        use: bool = False,
    ) -> None:
        resource_parser = subparsers.add_parser(
            name,
            help=f"Manage installed {name}",
        )
        resource_subparsers = resource_parser.add_subparsers(
            dest="resource_action",
            required=True,
        )
        resource_subparsers.add_parser(
            "list",
            help=f"List installed {name}",
        )
        install_parser = resource_subparsers.add_parser(
            "install",
            help=f"Install a {singular} from a local file or URL",
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
            )
            show_parser.add_argument(
                "name",
                help=f"Installed {singular} filename or relative path",
            )
        if use:
            use_parser = resource_subparsers.add_parser(
                "use",
                help=f"Copy an installed {singular} to the active system prompt file",
            )
            use_parser.add_argument(
                "name",
                help=f"Installed {singular} filename or relative path",
            )
        remove_parser = resource_subparsers.add_parser(
            "remove",
            help=f"Remove an installed {singular}",
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
                    action, help=help_text)
                action_parser.add_argument(
                    "name",
                    metavar="tool_file",
                    help="Tool filename/path or name, with or without .py",
                )

    add_resource_parser("skills", "skill")
    add_resource_parser("tools", "tool")
    add_resource_parser("prompts", "prompt", show=True, use=True)

    extensions_parser = subparsers.add_parser(
        "extensions",
        help="Manage installed extensions on disk",
    )
    extensions_subparsers = extensions_parser.add_subparsers(
        dest="ext_action",
        required=True,
    )
    extensions_subparsers.add_parser(
        "list",
        help="List enabled and disabled extensions",
    )
    for action, help_text in (
        ("enable", "Move an extension out of extensions/disabled/"),
        ("disable", "Move an extension into extensions/disabled/"),
        ("remove", "Permanently delete an extension from disk"),
    ):
        action_parser = extensions_subparsers.add_parser(
            action, help=help_text)
        action_parser.add_argument("name", help="Extension name")

    args = parser.parse_args(argv)

    if args.command in {"skills", "tools", "prompts"}:
        _handle_resource_command(args)
        return

    if args.command == "extensions":
        _handle_extensions_command(args)
        return

    if args.command == "profiles":
        try:
            config = AppConfig.from_env()
            profile_store = load_profile_store(config.model_profiles_path)

            default_profile_name = profile_store.default_profile
            rows = []
            for profile_name in profile_store.names():
                profile = profile_store.get(profile_name)
                name = profile.name
                if name == default_profile_name:
                    name = f"* {name}"
                row = [name, profile.model, profile.base_url]
                rows.append(row)
            table = tabulate(
                rows,
                headers=["Name", "Model", "Base URL"]
            )
            response = f"Default Profile: {default_profile_name}\n\nAll Profiles:\n{table}\n"

        except Exception as exc:
            sys.stderr.write(f"{exc}\n")
            sys.stderr.flush()
            sys.exit(2)

        sys.stdout.write(response)
        sys.stdout.flush()
        sys.exit(0)

    if args.command == "serve":
        try:
            import uvicorn
            from .api import create_app
        except (ImportError, RuntimeError) as exc:
            sys.stderr.write(f"{exc}\n")
            sys.stderr.flush()
            sys.exit(2)

        uvicorn.run(create_app(), host=args.host, port=args.port)
        return

    if args.command == "web":
        try:
            from textual_serve.server import Server
        except (ImportError, RuntimeError) as exc:
            sys.stderr.write(f"{exc}\n")
            sys.stderr.flush()
            sys.exit(2)

        command = 'python -m pyagent'
        if args.profile:
            command += f" --profile {args.profile}"
        if args.model:
            command += f" --model {args.model}"

        server = Server(
            command=command,
            host=args.host,
            port=args.port,
            title='PyAgent'
        )
        server.serve()
        return

    parsed_skills = _parse_skills_arg(args.skills)
    if parsed_skills and args.prompt is None:
        sys.stderr.write(
            "--skills is currently supported only with --prompt\n")
        sys.stderr.flush()
        sys.exit(2)

    if args.prompt is not None:
        try:
            response = run_single_shot(
                prompt=args.prompt,
                profile=args.profile,
                model=args.model,
                skills=parsed_skills,
            )
        except ValueError as exc:
            sys.stderr.write(f"{exc}\n")
            sys.stderr.flush()
            sys.exit(2)
        sys.stdout.write(response)
        sys.stdout.flush()
        sys.exit(0)

    # interactive mode
    from .ui import PyAgentApp

    app = PyAgentApp(profile=args.profile, model=args.model)
    app.run()


if __name__ == "__main__":
    main()
