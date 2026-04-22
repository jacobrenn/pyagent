import argparse

from ui import PyAgentApp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PyAgent")
    parser.add_argument(
        "--model",
        help="Ollama model to use (overrides PYAGENT_MODEL)",
    )
    args = parser.parse_args()

    app = PyAgentApp(model=args.model)
    app.run()


if __name__ == "__main__":
    main()
