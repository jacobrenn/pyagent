#!/usr/bin/env python

import argparse

from .ui import PyAgentApp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PyAgent")
    parser.add_argument(
        "--profile",
        help="Saved model profile to use (overrides PYAGENT_PROFILE)",
    )
    parser.add_argument(
        "--model",
        help="Model name override for the active profile",
    )
    args = parser.parse_args()

    app = PyAgentApp(profile=args.profile, model=args.model)
    app.run()


if __name__ == "__main__":
    main()
