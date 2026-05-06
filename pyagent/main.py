#!/usr/bin/env python
import argparse
import sys
from .ui import PyAgentApp
from .agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PyAgent")
    parser.add_argument(
        "--profile", help="Saved model profile to use (overrides PYAGENT_PROFILE)")
    parser.add_argument(
        "--model", help="Model name override for the active profile")
    parser.add_argument("--prompt", type=str,
                        help="Single prompt to run and exit")
    args = parser.parse_args()

    if args.prompt is not None:
        # single-shot mode
        agent = Agent(profile=args.profile, model=args.model)
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
