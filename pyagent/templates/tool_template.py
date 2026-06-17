from __future__ import annotations


TOOL_SCRIPT_TEMPLATE = '''\
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click"]
# ///
"""PyAgent external tool: {tool_name}.

This script is auto-discovered by PyAgent when placed under
``~/.pyagent/tools/``. Each external tool is a runnable UV script
exposing two subcommands: ``describe`` and ``invoke``.

Add or replace dependencies in the ``# /// script`` block above and they
will be installed by uv on first run, in an isolated venv that does not
affect the PyAgent install itself.
"""

from __future__ import annotations

import json
import sys

import click


TOOL_NAME = "{tool_name}"
TOOL_DESCRIPTION = (
    "TODO: Describe what this tool does for the model. "
    "Be specific: the description and parameter schema are sent verbatim "
    "to the LLM."
)
TOOL_PARAMETERS = {{
    "type": "object",
    "properties": {{
        "input": {{
            "type": "string",
            "description": "TODO: Describe this argument.",
        }},
    }},
    "required": ["input"],
}}
TOOL_VERSION = "1"


def run_tool(*, input: str) -> str:
    """Replace the body of this function with your tool's logic.

    Return a string. PyAgent will forward it to the model as the tool
    result. Raise an exception or print to stderr + exit non-zero to
    signal an error.
    """
    return f"Echo: {{input}}"


@click.group()
def cli() -> None:
    """PyAgent external tool entry point."""


@cli.command()
def describe() -> None:
    """Print the JSON manifest used by PyAgent to register this tool."""
    click.echo(
        json.dumps(
            {{
                "name": TOOL_NAME,
                "description": TOOL_DESCRIPTION,
                "parameters": TOOL_PARAMETERS,
                "version": TOOL_VERSION,
            }},
            ensure_ascii=False,
        )
    )


@cli.command()
@click.option(
    "--args",
    "args_json",
    required=True,
    help="Stringified JSON object with the tool arguments (not a file path).",
)
def invoke(args_json: str) -> None:
    """Run the tool with arguments from a JSON string passed via ``--args``.

    ``--args`` takes a single stringified JSON object (not a file path),
    for example ``--args '{{"input": "value"}}'``.
    """
    try:
        arguments = json.loads(args_json)
    except json.JSONDecodeError as exc:
        click.echo(f"Failed to parse --args: {{exc}}", err=True)
        sys.exit(2)

    if not isinstance(arguments, dict):
        click.echo("--args must contain a JSON object.", err=True)
        sys.exit(2)

    try:
        result = run_tool(**arguments)
    except TypeError as exc:
        click.echo(f"Invalid tool arguments: {{exc}}", err=True)
        sys.exit(2)
    except Exception as exc:
        click.echo(f"Tool error: {{exc}}", err=True)
        sys.exit(1)

    click.echo(result if result is not None else "")


if __name__ == "__main__":
    cli()
'''


def render_tool_template(tool_name: str) -> str:
    safe_name = tool_name.strip()
    if not safe_name:
        raise ValueError("Tool name must not be empty.")
    return TOOL_SCRIPT_TEMPLATE.format(tool_name=safe_name)
