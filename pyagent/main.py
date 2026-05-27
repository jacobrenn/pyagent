#!/usr/bin/env python
from importlib.metadata import version
import argparse
import os
import sys
from typing import Any

from .agent import Agent
from .project_context import load_full_context, resolve_user_skill


def get_version():
    try:
        return version('pyagent-harness')
    except Exception as e:
        return 'Unknown'


def _parse_skills_arg(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _validate_skills(skills: list[str], *, user_dir: str | None = None) -> list[str]:
    missing: list[str] = []
    validated: list[str] = []
    seen: set[str] = set()
    for skill_name in skills:
        skill = resolve_user_skill(skill_name, user_dir)
        if skill is None:
            missing.append(skill_name)
            continue
        if skill.label in seen:
            continue
        seen.add(skill.label)
        validated.append(skill.label)
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(
            f"Unknown skill(s): {missing_list}\n"
            "Skills must be specified as paths relative to ~/.pyagent/skills/"
        )
    return validated


def build_agent_for_request(
    *,
    profile: str | None = None,
    model: str | None = None,
    cwd: str | None = None,
    skills: list[str] | None = None,
) -> Agent:
    validated_skills = _validate_skills(skills or [])
    project_context, context_sources = load_full_context(
        cwd or os.getcwd(),
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
    response = ""
    for event in agent.run(prompt):
        if event.get("type") == "assistant_done":
            response = event["content"]
            break
    return response


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
        help="Comma-separated skill paths relative to ~/.pyagent/skills/ (supported only with --prompt)",
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
    args = parser.parse_args(argv)

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
