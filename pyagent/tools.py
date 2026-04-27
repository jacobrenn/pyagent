from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import os
import shutil
import subprocess
from typing import Any, Callable

from .config import AppConfig


MAX_TOOL_OUTPUT_CHARS = 12_000


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]

    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec]):
        self._specs = {spec.name: spec for spec in specs}

    def definitions(self) -> list[dict[str, Any]]:
        return [spec.definition() for spec in self._specs.values()]

    def has(self, name: str) -> bool:
        return name in self._specs

    def names(self) -> list[str]:
        return list(self._specs)

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._specs:
            return f"Error: tool '{name}' not found."

        try:
            return self._specs[name].handler(**arguments)
        except TypeError as exc:
            return f"Error calling tool '{name}': {exc}"
        except Exception as exc:  # pragma: no cover - defensive fallback
            return f"Unexpected error in tool '{name}': {exc}"


def _truncate(text: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n\n... [truncated {omitted} characters]"


def _command_matches_prefix(command: str, prefixes: tuple[str, ...]) -> bool:
    normalized = command.strip().lower()
    for prefix in prefixes:
        candidate = prefix.strip().lower()
        if normalized == candidate or normalized.startswith(f"{candidate} "):
            return True
    return False


def _validate_bash_command(command: str, config: AppConfig) -> str | None:
    stripped = command.strip()
    if not stripped:
        return "Error: empty bash command."

    normalized = f" {stripped.lower()} "
    if any(blocked.lower() in normalized for blocked in config.bash_blocked_substrings):
        return f"Error: command blocked by shell safety policy: {stripped}"

    if config.bash_readonly_mode and not _command_matches_prefix(stripped, config.bash_readonly_prefixes):
        allowed = ", ".join(config.bash_readonly_prefixes)
        return (
            "Error: bash is in read-only mode and this command is not allowed. "
            f"Allowed prefixes: {allowed}"
        )

    return None


def bash(command: str, timeout: int | None = None, config: AppConfig | None = None) -> str:
    """Execute a bash command and return exit code, stdout, and stderr."""
    runtime_config = config or AppConfig.from_env()
    if not runtime_config.bash_enabled:
        return "Error: bash tool is disabled by configuration."

    validation_error = _validate_bash_command(command, runtime_config)
    if validation_error:
        return validation_error

    effective_timeout = timeout if timeout is not None else runtime_config.bash_timeout_default

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired:
        return f"Command timed out after {effective_timeout} seconds."
    except Exception as exc:
        return f"Error executing bash command: {exc}"

    stdout = _truncate(result.stdout.strip()) or "<empty>"
    stderr = _truncate(result.stderr.strip()) or "<empty>"
    return f"exit_code: {result.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"


def list_files(path: str = ".", max_depth: int = 2) -> str:
    """List files and folders beneath a path up to a limited depth."""
    if not os.path.exists(path):
        return f"Error: path does not exist: {path}"

    base_path = os.path.abspath(path)
    lines: list[str] = [base_path]
    base_depth = base_path.count(os.sep)

    for root, dirs, files in os.walk(base_path):
        depth = root.count(os.sep) - base_depth
        dirs.sort()
        files.sort()

        if depth >= max_depth:
            dirs[:] = []

        for directory in dirs:
            lines.append(
                f"{os.path.relpath(os.path.join(root, directory), base_path)}/")
        for file_name in files:
            lines.append(os.path.relpath(
                os.path.join(root, file_name), base_path))

    return _truncate("\n".join(lines))


def _looks_like_glob(query: str) -> bool:
    return any(char in query for char in "*?[]")


def find_files(query: str, path: str = ".", max_results: int = 100) -> str:
    """Find files by substring or glob-style pattern."""
    if not os.path.exists(path):
        return f"Error: path does not exist: {path}"

    base_path = os.path.abspath(path)
    query_normalized = query.strip()
    if not query_normalized:
        return "Error: query must not be empty."

    matches: list[str] = []
    use_glob = _looks_like_glob(query_normalized)
    needle = query_normalized.lower()

    for root, dirs, files in os.walk(base_path):
        dirs.sort()
        files.sort()
        for file_name in files:
            rel_path = os.path.relpath(
                os.path.join(root, file_name), base_path)
            haystacks = (rel_path.lower(), file_name.lower())
            matched = any(fnmatch.fnmatch(value, needle) for value in haystacks) if use_glob else any(
                needle in value for value in haystacks
            )
            if matched:
                matches.append(rel_path)
                if len(matches) >= max_results:
                    return _truncate("\n".join(matches))

    return _truncate("\n".join(matches) if matches else "<no matches>")


def search_text(query: str, path: str = ".", glob: str = "*", max_results: int = 50) -> str:
    """Search for text inside files and return matching lines with file paths and line numbers."""
    if not os.path.exists(path):
        return f"Error: path does not exist: {path}"

    if not query.strip():
        return "Error: query must not be empty."

    base_path = os.path.abspath(path)
    ripgrep = shutil.which("rg")
    if ripgrep is not None:
        command = [
            ripgrep,
            "--line-number",
            "--with-filename",
            "--color",
            "never",
            "--glob",
            glob,
            "--max-count",
            str(max_results),
            query,
            ".",
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, cwd=base_path, timeout=30)
            if result.returncode in {0, 1}:
                output_lines: list[str] = []
                for line in result.stdout.splitlines():
                    parts = line.split(":", 2)
                    if len(parts) == 3:
                        rel_path = parts[0][2:] if parts[0].startswith(
                            "./") else parts[0]
                        output_lines.append(
                            f"{rel_path}:{parts[1]}: {parts[2]}")
                    else:
                        output_lines.append(line)
                output = "\n".join(output_lines).strip()
                return _truncate(output if output else "<no matches>")
        except Exception:
            pass

    matches: list[str] = []
    needle = query.lower()
    for root, dirs, files in os.walk(base_path):
        dirs.sort()
        files.sort()
        for file_name in files:
            rel_path = os.path.relpath(
                os.path.join(root, file_name), base_path)
            if not fnmatch.fnmatch(rel_path, glob) and not fnmatch.fnmatch(file_name, glob):
                continue
            absolute_path = os.path.join(root, file_name)
            try:
                with open(absolute_path, "r", encoding="utf-8") as file:
                    for line_number, line in enumerate(file, start=1):
                        if needle in line.lower():
                            matches.append(
                                f"{rel_path}:{line_number}: {line.rstrip()}")
                            if len(matches) >= max_results:
                                return _truncate("\n".join(matches))
            except (UnicodeDecodeError, OSError):
                continue

    return _truncate("\n".join(matches) if matches else "<no matches>")


def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
    """Read part or all of a UTF-8 text file."""
    if not os.path.exists(path):
        return f"Error: file does not exist: {path}"

    try:
        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except Exception as exc:
        return f"Error reading file {path}: {exc}"

    start_index = max(start_line - 1, 0)
    selected = lines[start_index:end_line]
    numbered = [f"{start_index + index + 1}: {line.rstrip()}" for index,
                line in enumerate(selected)]
    content = "\n".join(numbered)
    return content if content else "<empty>"


def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed."""
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)
        return f"Successfully wrote {len(content)} characters to {path}"
    except Exception as exc:
        return f"Error writing to file {path}: {exc}"


def append_file(path: str, content: str) -> str:
    """Append content to a file, creating parent directories if needed."""
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "a", encoding="utf-8") as file:
            file.write(content)
        return f"Successfully appended {len(content)} characters to {path}"
    except Exception as exc:
        return f"Error appending to file {path}: {exc}"


def edit_file(
    path: str,
    edits: list[dict[str, str]] | None = None,
    old_text: str | None = None,
    new_text: str | None = None,
) -> str:
    """Replace one or more unique exact text blocks in a file."""
    if not os.path.exists(path):
        return f"Error: file does not exist: {path}"

    try:
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
    except Exception as exc:
        return f"Error reading {path}: {exc}"

    requested_edits = list(edits or [])
    if old_text is not None:
        if new_text is None:
            return "Error: new_text is required when old_text is provided."
        requested_edits.append({"old_text": old_text, "new_text": new_text})

    if not requested_edits:
        return "Error: no edits were provided."

    replacements: list[tuple[int, int, str]] = []
    for index, requested_edit in enumerate(requested_edits, start=1):
        old = requested_edit.get("old_text")
        replacement = requested_edit.get("new_text")
        if old is None or replacement is None:
            return f"Error: edit #{index} must include old_text and new_text."

        match_count = content.count(old)
        if match_count == 0:
            return f"Error: old_text for edit #{index} not found in {path}"
        if match_count > 1:
            return (
                f"Error: old_text for edit #{index} matched {match_count} locations in {path}. "
                "Please provide a more specific snippet."
            )

        start = content.index(old)
        end = start + len(old)
        replacements.append((start, end, replacement))

    sorted_replacements = sorted(replacements, key=lambda item: item[0])
    for (_, current_end, _), (next_start, _, _) in zip(sorted_replacements, sorted_replacements[1:]):
        if next_start < current_end:
            return f"Error: requested edits overlap in {path}."

    updated = content
    for start, end, replacement in reversed(sorted_replacements):
        updated = f"{updated[:start]}{replacement}{updated[end:]}"

    try:
        with open(path, "w", encoding="utf-8") as file:
            file.write(updated)
    except Exception as exc:
        return f"Error writing {path}: {exc}"

    count = len(requested_edits)
    return f"Successfully edited {path} with {count} replacement{'s' if count != 1 else ''}"


def calculator(num1: str | float, num2: str | float, operation: str) -> str:
    """Run a simple arithmetic calculator on two numbers."""
    try:
        first = float(num1)
        second = float(num2)
    except ValueError as exc:
        return f"Invalid number: {exc}"

    aliases = {
        "+": "addition",
        "-": "subtraction",
        "*": "multiplication",
        "/": "division",
    }
    operation = aliases.get(operation, operation)

    if operation == "addition":
        return str(first + second)
    if operation == "subtraction":
        return str(first - second)
    if operation == "multiplication":
        return str(first * second)
    if operation == "division":
        if second == 0:
            return "Division by zero is not defined"
        return str(first / second)
    return f"Operation '{operation}' not supported"


def create_default_tool_registry(config: AppConfig | None = None) -> ToolRegistry:
    runtime_config = config or AppConfig.from_env()

    def bash_handler(command: str, timeout: int | None = None) -> str:
        return bash(command=command, timeout=timeout, config=runtime_config)

    return ToolRegistry(
        [
            ToolSpec(
                name="bash",
                description=(
                    "Execute a bash command in the current working directory. This tool may be restricted "
                    "by safety policy or read-only mode."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The bash command to run."},
                        "timeout": {
                            "type": ["integer", "null"],
                            "description": "Optional timeout in seconds.",
                            "default": runtime_config.bash_timeout_default,
                        },
                    },
                    "required": ["command"],
                },
                handler=bash_handler,
            ),
            ToolSpec(
                name="list_files",
                description="List files and folders under a path up to a limited depth.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The path to inspect.", "default": "."},
                        "max_depth": {
                            "type": "integer",
                            "description": "Maximum directory depth to descend.",
                            "default": 2,
                        },
                    },
                },
                handler=list_files,
            ),
            ToolSpec(
                name="find_files",
                description="Find files by substring or glob-style pattern.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Substring or glob-style filename query."},
                        "path": {"type": "string", "description": "Base path to search.", "default": "."},
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of matching paths to return.",
                            "default": 100,
                        },
                    },
                    "required": ["query"],
                },
                handler=find_files,
            ),
            ToolSpec(
                name="search_text",
                description="Search for text inside files and return matching lines with paths and line numbers.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Text to search for."},
                        "path": {"type": "string", "description": "Base path to search.", "default": "."},
                        "glob": {
                            "type": "string",
                            "description": "Optional glob filter for filenames.",
                            "default": "*",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of matching lines to return.",
                            "default": 50,
                        },
                    },
                    "required": ["query"],
                },
                handler=search_text,
            ),
            ToolSpec(
                name="read_file",
                description="Read the contents of a text file, optionally limited by line range.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The path to the file."},
                        "start_line": {
                            "type": "integer",
                            "description": "The first 1-based line number to read.",
                            "default": 1,
                        },
                        "end_line": {
                            "type": ["integer", "null"],
                            "description": "Optional last line number (inclusive-style slice endpoint).",
                            "default": None,
                        },
                    },
                    "required": ["path"],
                },
                handler=read_file,
            ),
            ToolSpec(
                name="write_file",
                description="Write content to a file, overwriting it if it already exists.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The path to the file."},
                        "content": {"type": "string", "description": "The content to write."},
                    },
                    "required": ["path", "content"],
                },
                handler=write_file,
            ),
            ToolSpec(
                name="append_file",
                description="Append content to the end of a file, creating it if needed.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The path to the file."},
                        "content": {"type": "string", "description": "The content to append."},
                    },
                    "required": ["path", "content"],
                },
                handler=append_file,
            ),
            ToolSpec(
                name="edit_file",
                description=(
                    "Replace one or more unique exact text blocks in a file. Prefer the edits array for multiple "
                    "replacements; old_text/new_text is also supported for compatibility."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The path to the file."},
                        "edits": {
                            "type": "array",
                            "description": "List of exact text replacements to apply.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "old_text": {
                                        "type": "string",
                                        "description": "The exact existing text to replace. It must be unique.",
                                    },
                                    "new_text": {
                                        "type": "string",
                                        "description": "The replacement text.",
                                    },
                                },
                                "required": ["old_text", "new_text"],
                            },
                        },
                        "old_text": {
                            "type": "string",
                            "description": "Single exact existing text block to replace.",
                        },
                        "new_text": {
                            "type": "string",
                            "description": "Replacement text for old_text.",
                        },
                    },
                    "required": ["path"],
                },
                handler=edit_file,
            ),
            ToolSpec(
                name="calculator",
                description="Run a simple arithmetic operation on two numbers.",
                parameters={
                    "type": "object",
                    "properties": {
                        "num1": {"type": "string", "description": "The first number."},
                        "num2": {"type": "string", "description": "The second number."},
                        "operation": {
                            "type": "string",
                            "description": "One of +, -, *, /, addition, subtraction, multiplication, division.",
                        },
                    },
                    "required": ["num1", "num2", "operation"],
                },
                handler=calculator,
            ),
        ]
    )


DEFAULT_TOOL_REGISTRY = create_default_tool_registry()

AVAILABLE_TOOLS = {
    name: spec.handler for name, spec in DEFAULT_TOOL_REGISTRY._specs.items()
}


def get_tool_definitions(registry: ToolRegistry | None = None) -> list[dict[str, Any]]:
    return (registry or DEFAULT_TOOL_REGISTRY).definitions()
