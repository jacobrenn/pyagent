#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click>=8.1"]
# ///
"""Reference PyAgent external tool: compact_context.

A **deterministic** context condenser paired with the `compaction` extension
(see `examples/extensions/compaction/`). The extension surfaces a skill that
tells the LLM to call this tool with older conversation text; this tool
returns a condensed version.

It is intentionally dependency-light and network-free (extensions_prd.md §2:
extensions and their tools are deterministic). The default strategy keeps the
first and last portions of the input verbatim and replaces the middle with a
count of the omitted characters — a lossy but predictable stub. Wire in a
real summarizer (e.g. shell out to a local model) by editing `_condense`.

Tool contract (matches every PyAgent external tool):
- ``uv run compact_context.py describe``
    Prints a JSON manifest (name, description, JSON-Schema parameters).
- ``uv run compact_context.py invoke --args <json>``
    Reads a JSON object with a ``text`` field; prints the condensed text.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import click


TOOL_NAME = "compact_context"
TOOL_DESCRIPTION = (
    "Condense older conversation text into a shorter form. Pass the prior "
    "messages you want compacted as `text`; the tool returns a condensed "
    "version with the middle replaced by an omission marker. Use proactively "
    "when the conversation is long."
)
TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "description": "The older conversation text to condense.",
        },
        "keep_head": {
            "type": "integer",
            "description": "Characters to keep verbatim at the start.",
            "default": 800,
        },
        "keep_tail": {
            "type": "integer",
            "description": "Characters to keep verbatim at the end.",
            "default": 800,
        },
    },
    "required": ["text"],
}
TOOL_VERSION = "0.1.0"


def _condense(text: str, keep_head: int, keep_tail: int) -> str:
    text = text.strip()
    if not text:
        return "[compact_context: empty input]"

    budget = max(0, keep_head) + max(0, keep_tail)
    if len(text) <= budget:
        return text

    head = text[: max(0, keep_head)]
    tail = text[len(text) - max(0, keep_tail):] if keep_tail else ""
    omitted = len(text) - len(head) - len(tail)
    return (
        f"{head}\n\n"
        f"[... {omitted} characters condensed by compact_context ...]\n\n"
        f"{tail}"
    )


@click.group()
def cli() -> None:
    """PyAgent external tool: compact_context."""


@cli.command()
def describe() -> None:
    """Print the JSON manifest used by PyAgent to register this tool."""
    click.echo(
        json.dumps(
            {
                "name": TOOL_NAME,
                "description": TOOL_DESCRIPTION,
                "parameters": TOOL_PARAMETERS,
                "version": TOOL_VERSION,
            },
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
    """Run the tool with arguments from a JSON string passed via ``--args``."""
    try:
        arguments = json.loads(args_json)
    except json.JSONDecodeError as exc:
        click.echo(f"Failed to parse --args: {exc}", err=True)
        sys.exit(2)
    if not isinstance(arguments, dict):
        click.echo("--args must contain a JSON object.", err=True)
        sys.exit(2)

    text = arguments.get("text")
    if not isinstance(text, str) or not text.strip():
        click.echo("Error: `text` must be a non-empty string.", err=True)
        sys.exit(2)

    keep_head = int(arguments.get("keep_head", 800))
    keep_tail = int(arguments.get("keep_tail", 800))
    click.echo(_condense(text, keep_head, keep_tail))


if __name__ == "__main__":
    cli()
