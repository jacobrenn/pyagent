#!/usr/bin/env python
from importlib.metadata import version
import argparse
import os
import sys

from .ui import PyAgentApp
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


def main() -> None:
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
    args = parser.parse_args()

    parsed_skills = _parse_skills_arg(args.skills)
    if parsed_skills and args.prompt is None:
        sys.stderr.write(
            "--skills is currently supported only with --prompt\n")
        sys.stderr.flush()
        sys.exit(2)

    if args.prompt is not None:
        try:
            validated_skills = _validate_skills(parsed_skills)
        except ValueError as exc:
            sys.stderr.write(f"{exc}\n")
            sys.stderr.flush()
            sys.exit(2)

        project_context, context_sources = load_full_context(
            os.getcwd(),
            loaded_user_skills=validated_skills,
        )
        project_context_files = [source.label for source in context_sources]

        # single-shot mode
        agent = Agent(
            profile=args.profile,
            model=args.model,
            project_context=project_context,
            project_context_files=project_context_files,
        )
        response = []
        for event in agent.run(args.prompt):
            if event.get("type") == "assistant_done":
                response = [
                    event["content"]
                ]
                break
        sys.stdout.write("".join(response))
        sys.stdout.flush()
        sys.exit(0)

    # interactive mode
    app = PyAgentApp(profile=args.profile, model=args.model)
    app.run()


if __name__ == "__main__":
    main()
