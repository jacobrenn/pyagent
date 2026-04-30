from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from .user_runtime import (
    DEFAULT_TOOL_RUNNER,
    DEFAULT_USER_DIR,
    TOOL_RUNNER_ENV_VAR,
    USER_DIR_ENV_VAR,
    resolve_user_dir,
)


def _parse_csv_env(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _default_profiles_path() -> str:
    return str(Path.home() / ".pyagent" / "models.json")


def _default_user_dir() -> str:
    return str(resolve_user_dir())


def _default_system_prompt_path() -> str:
    return str(Path.home() / ".pyagent" / "system_prompt.txt")


SYSTEM_PROMPT = """You are PyAgent, a capable coding assistant operating in a tool-use loop. You have access to tools to inspect the file system, read and write files, edit files, run shell commands, and any other tools that have been created to extend your toolset.

Prefer the tools that will get the job done, explain your reasoning briefly, and use tools when needed. Be precise when making code changes.

After you receive a tool result, continue the task and provide a complete user-facing answer; make sure that you fully answer user requests. If a tool directly answers the question, summarize the result and include the relevant output.
"""


@dataclass(slots=True)
class AppConfig:
    request_timeout: int = 300
    max_iterations: int = 10
    max_history_messages: int = 24
    stream_batch_interval: float = 0.05
    default_profile: str | None = None
    model_profiles_path: str = _default_profiles_path()
    system_prompt_path: str = _default_system_prompt_path()
    tools_enabled: bool = True
    bash_enabled: bool = True
    bash_readonly_mode: bool = False
    bash_timeout_default: int = 60
    bash_blocked_substrings: tuple[str, ...] = (
        "sudo ",
        "rm -",
        "shutdown",
        "reboot",
        "mkfs",
        " dd ",
        "dd if=",
    )
    bash_readonly_prefixes: tuple[str, ...] = (
        "pwd",
        "ls",
        "find",
        "rg",
        "grep",
        "git status",
        "git diff",
        "git log",
        "head",
        "tail",
        "wc",
        "which",
    )
    user_dir: str = DEFAULT_USER_DIR
    user_tools_enabled: bool = True
    user_tool_timeout: float = 60.0
    user_tool_describe_timeout: float = 10.0
    tool_runner: str = DEFAULT_TOOL_RUNNER

    @classmethod
    def from_env(cls) -> "AppConfig":
        defaults = cls()
        return cls(
            request_timeout=int(
                os.getenv("PYAGENT_REQUEST_TIMEOUT",
                          str(defaults.request_timeout))
            ),
            max_iterations=int(
                os.getenv("PYAGENT_MAX_ITERATIONS",
                          str(defaults.max_iterations))
            ),
            max_history_messages=int(
                os.getenv(
                    "PYAGENT_MAX_HISTORY_MESSAGES",
                    str(defaults.max_history_messages),
                )
            ),
            stream_batch_interval=float(
                os.getenv(
                    "PYAGENT_STREAM_BATCH_INTERVAL",
                    str(defaults.stream_batch_interval),
                )
            ),
            default_profile=os.getenv("PYAGENT_PROFILE") or None,
            model_profiles_path=os.getenv(
                "PYAGENT_MODEL_PROFILES_PATH",
                defaults.model_profiles_path,
            ),
            system_prompt_path=os.getenv(
                "PYAGENT_SYSTEM_PROMPT_PATH",
                defaults.system_prompt_path,
            ),
            tools_enabled=os.getenv("PYAGENT_TOOLS_ENABLED", str(
                defaults.tools_enabled)).lower()
            in {"1", "true", "yes", "on"},
            bash_enabled=os.getenv("PYAGENT_BASH_ENABLED", str(
                defaults.bash_enabled)).lower()
            in {"1", "true", "yes", "on"},
            bash_readonly_mode=os.getenv(
                "PYAGENT_BASH_READONLY_MODE", str(defaults.bash_readonly_mode)
            ).lower()
            in {"1", "true", "yes", "on"},
            bash_timeout_default=int(
                os.getenv(
                    "PYAGENT_BASH_TIMEOUT_DEFAULT",
                    str(defaults.bash_timeout_default),
                )
            ),
            bash_blocked_substrings=_parse_csv_env(
                os.getenv(
                    "PYAGENT_BASH_BLOCKED_SUBSTRINGS",
                    ",".join(defaults.bash_blocked_substrings),
                )
            ),
            bash_readonly_prefixes=_parse_csv_env(
                os.getenv(
                    "PYAGENT_BASH_READONLY_PREFIXES",
                    ",".join(defaults.bash_readonly_prefixes),
                )
            ),
            user_dir=os.getenv(USER_DIR_ENV_VAR, defaults.user_dir),
            user_tools_enabled=os.getenv(
                "PYAGENT_USER_TOOLS_ENABLED",
                str(defaults.user_tools_enabled),
            ).lower()
            in {"1", "true", "yes", "on"},
            user_tool_timeout=float(
                os.getenv(
                    "PYAGENT_USER_TOOL_TIMEOUT",
                    str(defaults.user_tool_timeout),
                )
            ),
            user_tool_describe_timeout=float(
                os.getenv(
                    "PYAGENT_USER_TOOL_DESCRIBE_TIMEOUT",
                    str(defaults.user_tool_describe_timeout),
                )
            ),
            tool_runner=os.getenv(TOOL_RUNNER_ENV_VAR, defaults.tool_runner),
        )
