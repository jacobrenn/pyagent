from __future__ import annotations

from dataclasses import dataclass
import os


def _parse_csv_env(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


SYSTEM_PROMPT = (
    "You are a capable coding assistant operating in a tool-use loop. You have access to tools to "
    "inspect the file system, search for files and text, read and write files, append to files, edit "
    "files, and run shell commands. Prefer dedicated search and file tools over shell commands when "
    "possible. Explain your reasoning briefly, use tools when needed, and be precise when making code "
    "changes. After you receive a tool result, continue the task and provide a complete user-facing "
    "answer. Do not stop after only saying that you will inspect something. If a tool directly answers "
    "the question, summarize the result and include the relevant output."
)


@dataclass(slots=True)
class AppConfig:
    model: str = "gemma4:31b"
    base_url: str = "http://localhost:11434"
    request_timeout: int = 300
    max_iterations: int = 10
    max_history_messages: int = 24
    stream_batch_interval: float = 0.05
    bash_enabled: bool = True
    bash_readonly_mode: bool = False
    bash_timeout_default: int = 60
    bash_blocked_substrings: tuple[str, ...] = (
        "sudo ",
        " rm ",
        "rm -",
        " mv ",
        "chmod ",
        "chown ",
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

    @classmethod
    def from_env(cls) -> "AppConfig":
        defaults = cls()
        return cls(
            model=os.getenv("PYAGENT_MODEL", defaults.model),
            base_url=os.getenv("PYAGENT_BASE_URL", defaults.base_url),
            request_timeout=int(
                os.getenv("PYAGENT_REQUEST_TIMEOUT", str(defaults.request_timeout))),
            max_iterations=int(
                os.getenv("PYAGENT_MAX_ITERATIONS", str(defaults.max_iterations))),
            max_history_messages=int(
                os.getenv("PYAGENT_MAX_HISTORY_MESSAGES",
                          str(defaults.max_history_messages))
            ),
            stream_batch_interval=float(
                os.getenv(
                    "PYAGENT_STREAM_BATCH_INTERVAL",
                    str(defaults.stream_batch_interval),
                )
            ),
            bash_enabled=os.getenv("PYAGENT_BASH_ENABLED", str(
                defaults.bash_enabled)).lower() in {"1", "true", "yes", "on"},
            bash_readonly_mode=os.getenv("PYAGENT_BASH_READONLY_MODE", str(
                defaults.bash_readonly_mode)).lower() in {"1", "true", "yes", "on"},
            bash_timeout_default=int(
                os.getenv("PYAGENT_BASH_TIMEOUT_DEFAULT",
                          str(defaults.bash_timeout_default))
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
        )
