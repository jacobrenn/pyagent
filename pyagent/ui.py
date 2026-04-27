from __future__ import annotations

import asyncio
from datetime import datetime
import json
import os
import shlex
from difflib import get_close_matches
import re

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Footer, Header, Label, Markdown, RichLog, Static, TextArea

from .agent import Agent
from .model_profiles import ModelProfile, default_base_url_for_provider
from .project_context import load_project_context


PROMPT_INPUT_MIN_HEIGHT = 8
PROMPT_INPUT_MAX_HEIGHT = 20


def _truncate(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


class ChatMessage(Vertical):
    CODE_BLOCK_PATTERN = re.compile(r"```(?:[^\n]*)\n(.*?)```", re.DOTALL)
    LONG_CODE_LINE_LIMIT = 120

    def __init__(
        self,
        role: str,
        content: str = "",
        finalized: bool = False,
        render_mode: str = "markdown",
    ):
        super().__init__()
        self.role = role
        self.content = content
        self.finalized = finalized
        self.render_mode = render_mode
        self._label = Label(self._label_text(), classes=f"role-label {role}")
        self._stream_widget = Label(
            content, markup=False, classes="stream-content")
        self._markdown_widget = Markdown(content, classes="markdown-content")
        self._plain_text_widget = Label(
            content, markup=False, classes="plain-text-content")

    def _label_text(self) -> str:
        names = {
            "user": "You",
            "assistant": "Assistant",
            "tool": "Tool",
            "system": "System",
        }
        return names.get(self.role, self.role.title())

    def _has_long_code_block_line(self) -> bool:
        for match in self.CODE_BLOCK_PATTERN.finditer(self.content):
            code = match.group(1)
            if any(len(line) > self.LONG_CODE_LINE_LIMIT for line in code.splitlines()):
                return True
        return False

    def compose(self) -> ComposeResult:
        yield self._label
        yield self._stream_widget
        yield self._markdown_widget
        yield self._plain_text_widget

    def on_mount(self) -> None:
        self._sync_mode()

    def _sync_mode(self) -> None:
        use_markdown = self.render_mode == "markdown"
        use_plain_text = self.finalized and use_markdown and self._has_long_code_block_line()
        self._stream_widget.display = (
            (not self.finalized) or not use_markdown) and not use_plain_text
        self._markdown_widget.display = self.finalized and use_markdown and not use_plain_text
        self._plain_text_widget.display = use_plain_text
        if use_plain_text:
            self._plain_text_widget.update(self.content)
        elif self.finalized and use_markdown:
            self._markdown_widget.update(self.content)
        else:
            self._stream_widget.update(self.content)

    def append_stream(self, chunk: str) -> None:
        self.content += chunk
        self.finalized = False
        self._sync_mode()

    def set_content(self, content: str, finalized: bool = False) -> None:
        self.content = content
        self.finalized = finalized
        self._sync_mode()

    def finalize(self) -> None:
        self.finalized = True
        self._sync_mode()


class PromptInput(TextArea):
    BINDINGS = [
        Binding("enter", "submit", "Send", show=False, priority=True),
        Binding("shift+enter", "insert_newline",
                "New line", show=False, priority=True),
        *TextArea.BINDINGS,
    ]

    class Submitted(Message):
        def __init__(self, prompt_input: "PromptInput", value: str) -> None:
            super().__init__()
            self.prompt_input = prompt_input
            self.value = value

    def action_submit(self) -> None:
        self.post_message(self.Submitted(self, self.text))

    def action_insert_newline(self) -> None:
        self.insert("\n")


class PyAgentApp(App):
    TITLE = "PyAgent"
    SUB_TITLE = "Multi-provider coding agent"
    BINDINGS = [
        Binding("up", "scroll_chat_up", "Scroll chat up", priority=True),
        Binding("down", "scroll_chat_down", "Scroll chat down", priority=True),
        Binding("pageup", "scroll_chat_page_up",
                "Scroll chat up", priority=True),
        Binding("pagedown", "scroll_chat_page_down",
                "Scroll chat down", priority=True),
        Binding("home", "scroll_chat_home", "Chat top", priority=True),
        Binding("end", "scroll_chat_end", "Chat bottom", priority=True),
        Binding("ctrl+p", "history_previous",
                "Previous prompt", priority=True),
        Binding("ctrl+n", "history_next", "Next prompt", priority=True),
        Binding("ctrl+l", "clear_chat", "Clear chat"),
        Binding("ctrl+d", "toggle_debug", "Toggle debug"),
        Binding("ctrl+c", "quit", "Quit"),
    ]

    CSS = """
    Screen {
        align: center middle;
        background: #0f1115;
    }

    #app-shell {
        width: 95%;
        max-width: 100%;
        height: 1fr;
        min-height: 24;
    }

    #status-bar {
        height: auto;
        color: #d7e3f4;
        background: #1a2230;
        padding: 0 1;
        margin-bottom: 1;
        border: round #3d4f6b;
    }

    #chat-container {
        height: 1fr;
        border: round #3d8b5c;
        padding: 0 1;
        background: #12161d;
        overflow-y: auto;
        scrollbar-gutter: stable;
    }

    #user-input {
        height: 5;   # matches the new minimum height
        margin-top: 1;
        border: round #3d4f6b;
    }

    #debug-log {
        height: 12;
        margin-top: 1;
        border: round #6a5acd;
        background: #10131a;
        color: #d7e3f4;
    }

    ChatMessage {
        height: auto;
        margin: 1 0;
        padding: 0 1 1 1;
        border-left: wide transparent;
    }

    ChatMessage.user-message {
        border-left: wide #46b37b;
    }

    ChatMessage.assistant-message {
        border-left: wide #5c93ff;
    }

    ChatMessage.tool-message {
        border-left: wide #d8a657;
    }

    ChatMessage.system-message {
        border-left: wide #8e9aaf;
    }

    .role-label {
        text-style: bold;
        margin-bottom: 1;
    }

    .user {
        color: #46b37b;
    }

    .assistant {
        color: #5c93ff;
    }

    .tool {
        color: #d8a657;
    }

    .system {
        color: #aab7cf;
    }

    .stream-content,
    .markdown-content,
    .plain-text-content {
        width: 1fr;
        height: auto;
        overflow-x: hidden;
    }
    """

    def __init__(self, profile: str | None = None, model: str | None = None):
        super().__init__()
        self.project_context, self.project_context_files = load_project_context(
            os.getcwd())
        self.agent = Agent(
            model=model,
            profile=profile,
            project_context=self.project_context,
            project_context_files=self.project_context_files,
        )
        self.is_processing = False
        self.debug_visible = False
        self.chat_auto_follow = True
        self._chat_follow_timer = None
        self.input_history: list[str] = []
        self.input_history_index: int | None = None
        self.input_history_draft = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="app-shell"):
            yield Static(id="status-bar")
            yield VerticalScroll(id="chat-container")
            yield RichLog(id="debug-log", max_lines=500, wrap=True, highlight=False, markup=False)
            yield PromptInput(
                placeholder="Ask the agent anything…  (/clear to reset)",
                id="user-input",
            )
        yield Footer()

    def on_mount(self) -> None:
        self._debug_log_widget().display = False
        self._chat_follow_timer = self.set_interval(
            0.1, self._auto_follow_chat, pause=False)
        self._set_status(self._ready_status())
        profile = self.agent.current_profile()
        self._add_message(
            "system",
            (
                "Welcome to **PyAgent**. Responses stream as plain text for speed and render as Markdown when complete.\n\n"
                f"Active profile: `{profile.name}`  \n"
                f"Provider: `{profile.provider}`  \n"
                f"Model: `{profile.model}`"
            ),
            finalized=True,
        )
        if self.project_context_files:
            loaded_files = "\n".join(
                f"- `{path}`" for path in self.project_context_files)
            self._add_message(
                "system",
                f"Loaded project instructions:\n{loaded_files}",
                finalized=True,
            )
        prompt_input = self.query_one("#user-input", PromptInput)
        prompt_input.border_title = " Prompt "
        prompt_input.border_subtitle = " Enter sends • Shift+Enter newline • Ctrl+P/N history "
        self._resize_prompt_input()
        self._log_debug("Debug log initialized.")

    def _status_summary(self) -> str:
        profile = self.agent.current_profile()
        context_count = len(self.agent.project_context_files)
        return (
            f"Profile: {profile.name} • Provider: {profile.provider} • Model: {profile.model}"
            f" • Tools: {'on' if self.agent.config.tools_enabled else 'off'}"
            f" • Context: {context_count}"
            f" • Debug: {'on' if self.debug_visible else 'off'}"
        )

    def _max_iterations_text(self) -> str:
        if self.agent.config.max_iterations < 0:
            return "`-1` (infinite)"
        return f"`{self.agent.config.max_iterations}`"

    def _ready_status(self) -> str:
        return f"Ready • {self._status_summary()}"

    def _set_status(self, text: str) -> None:
        self.query_one("#status-bar", Static).update(text)

    def _prompt_input(self) -> PromptInput:
        return self.query_one("#user-input", PromptInput)

    def _set_prompt_text(self, text: str) -> None:
        input_widget = self._prompt_input()
        input_widget.load_text(text)
        self._resize_prompt_input()

    def _resize_prompt_input(self) -> None:
        input_widget = self._prompt_input()
        line_count = max(1, input_widget.text.count("\n") + 1)
        target_height = max(PROMPT_INPUT_MIN_HEIGHT, min(
            PROMPT_INPUT_MAX_HEIGHT, line_count + 2))
        input_widget.styles.height = target_height

    def _chat_container(self) -> VerticalScroll:
        return self.query_one("#chat-container", VerticalScroll)

    def _compact_arguments(self, arguments: dict[str, object], max_chars: int = 120) -> str:
        if not arguments:
            return "no arguments"
        parts = [f"{key}={value!r}" for key, value in arguments.items()]
        return _truncate(", ".join(parts), max_chars)

    def _add_system_note(self, content: str) -> None:
        self._add_message("system", content, finalized=True)

    def _command_help_text(self) -> str:
        return (
            "Available commands and keys:\n"
            "\n"
            "Prompt and navigation\n"
            "- `Enter` — send the current prompt\n"
            "- `Shift+Enter` — insert a newline\n"
            "- `Ctrl+P` / `Ctrl+N` — move through prompt history\n"
            "- `↑` / `↓` / `PgUp` / `PgDn` / `Home` / `End` — scroll the chat transcript\n"
            "\n"
            "Session\n"
            "- `/clear` — clear the conversation\n"
            "- `/status` — show current configuration and runtime limits\n"
            "- `/max_iterations <n|-1>` — set the agent tool-loop iteration limit for this session\n"
            "- `/cwd` — show the current working directory\n"
            "\n"
            "Profiles and models\n"
            "- `/profiles` — list saved model profiles\n"
            "- `/profiles reload` — reload profiles from disk\n"
            "- `/profile` — show the current profile\n"
            "- `/profile <name>` — switch to another saved profile\n"
            "- `/profile add <name> provider=<provider> model=<model> ...` — create or update a profile\n"
            "- `/model` — show the current model and usage\n"
            "- `/model list` — list models from the current endpoint, if supported\n"
            "- `/model <name>` — override the current profile's model for this session\n"
            "\n"
            "Context and prompts\n"
            "- `/context` — show loaded project instruction sources\n"
            "- `/prompt` — show the active system prompt\n"
            "- `/history` — show recent prompt history\n"
            "- `/history search <text>` — search saved prompt history\n"
            "- `/reload_context` — reload `AGENTS.md` and local skill files\n"
            "\n"
            "Tools and debugging\n"
            "- `/tools` — show tool status and available tools\n"
            "- `/tools on|off` — enable or disable model tool calling for this session\n"
            "- `/debug on|off` — show or hide the debug pane"
        )

    def _context_status_text(self) -> str:
        files = self.agent.project_context_files
        total_chars = len(self.agent.project_context)
        if not files:
            return "Project context:\n- No `AGENTS.md` or skill files loaded."
        lines = [
            "Project context:",
            f"- Files loaded: `{len(files)}`",
            f"- Context size: `{total_chars}` characters",
            "- Sources:",
        ]
        lines.extend(f"  - `{path}`" for path in files)
        return "\n".join(lines)

    def _unknown_command_message(self, command: str) -> str:
        known_commands = [
            "/clear",
            "/help",
            "/tools",
            "/profiles",
            "/profile",
            "/model",
            "/status",
            "/max_iterations",
            "/cwd",
            "/history",
            "/prompt",
            "/context",
            "/reload_profiles",
            "/reload_context",
            "/debug",
        ]
        suggestion = get_close_matches(
            command, known_commands, n=1, cutoff=0.5)
        if suggestion:
            return f"Unknown command: `{command}`. Did you mean `{suggestion[0]}`? Use `/help` to see available commands."
        return f"Unknown command: `{command}`. Use `/help` to see available commands."

    def _record_history(self, user_input: str) -> None:
        if not user_input:
            return
        if not self.input_history or self.input_history[-1] != user_input:
            self.input_history.append(user_input)
        self.input_history_index = None
        self.input_history_draft = ""

    def _show_history_entry(self, index: int | None) -> None:
        if index is None:
            self.input_history_index = None
            self._set_prompt_text(self.input_history_draft)
            return
        self.input_history_index = index
        self._set_prompt_text(self.input_history[index])

    def _parse_command_parts(self, raw_input: str) -> tuple[list[str] | None, str | None]:
        try:
            return shlex.split(raw_input), None
        except ValueError as exc:
            return None, str(exc)

    def _parse_bool_option(self, value: str) -> bool:
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _parse_max_iterations(self, value: str) -> tuple[int | None, str | None]:
        try:
            iterations = int(value.strip())
        except ValueError:
            return None, "Max iterations must be an integer. Use `-1` for infinite."

        if iterations == 0 or iterations < -1:
            return None, "Max iterations must be a positive integer or `-1` for infinite."
        return iterations, None

    def _parse_profile_add_options(self, args: list[str]) -> tuple[dict[str, str | bool | dict[str, str]], str | None]:
        if not args:
            return {}, "Usage: `/profile add <name> provider=<provider> model=<model> [base_url=<url>] [api_key_env=<ENV>] [api_key=<KEY>] [default=true|false] [switch=true|false] [header.<Name>=<Value>]`"

        name = args[0].strip()
        if not name:
            return {}, "Profile name must not be empty."

        options: dict[str, str | bool | dict[str, str]] = {
            "name": name,
            "headers": {},
        }
        for token in args[1:]:
            if "=" not in token:
                return {}, f"Invalid option `{token}`. Expected key=value."
            key, value = token.split("=", 1)
            normalized_key = key.strip().lower()
            value = value.strip()
            if not normalized_key:
                return {}, f"Invalid option `{token}`."
            if normalized_key.startswith("header."):
                header_name = key[len("header."):].strip()
                if not header_name:
                    return {}, f"Invalid header option `{token}`."
                headers = options["headers"]
                assert isinstance(headers, dict)
                headers[header_name] = value
                continue
            if normalized_key in {"default", "switch"}:
                options[normalized_key] = self._parse_bool_option(value)
                continue
            options[normalized_key] = value

        provider = str(options.get("provider", "")).strip()
        model = str(options.get("model", "")).strip()
        if not provider or not model:
            return {}, "`provider` and `model` are required."
        return options, None

    def _handle_slash_command(self, raw_input: str) -> bool:
        if not raw_input.startswith("/"):
            return False

        parts, error = self._parse_command_parts(raw_input)
        if error:
            self._add_system_note(f"Could not parse command: `{error}`")
            return True
        if not parts:
            return True

        command = parts[0].lower()
        args = parts[1:]

        if command == "/help":
            self._add_system_note(self._command_help_text())
            return True

        if command == "/tools":
            if len(args) == 1 and args[0].lower() in {"on", "off"}:
                enabled = args[0].lower() == "on"
                self.agent.set_tools_enabled(enabled)
                state = "enabled" if enabled else "disabled"
                self._add_system_note(
                    f"Tools {state} for this session. Conversation reset so the updated system prompt takes effect."
                )
                self._set_status(self._ready_status())
                return True

            if args:
                self._add_system_note("Usage: `/tools` or `/tools on|off`")
                return True

            tool_names = "\n".join(
                f"- `{name}`" for name in self.agent.tool_registry.names())
            self._add_system_note(
                "Tools:\n"
                f"- Status: `{'on' if self.agent.config.tools_enabled else 'off'}`\n"
                "- Available tool definitions:\n"
                f"{tool_names}"
            )
            return True

        if command == "/profiles":
            if len(args) == 1 and args[0].lower() == "reload":
                try:
                    self.agent.reload_profiles()
                except ValueError as exc:
                    self._add_system_note(
                        f"Could not reload profiles: `{exc}`")
                    return True
                self._add_system_note(
                    f"Reloaded profiles from `{self.agent.profile_store.path}`. Active profile: `{self.agent.current_profile().name}`."
                )
                self._set_status(self._ready_status())
                return True

            current = self.agent.current_profile().name
            lines = []
            default_name = self.agent.profile_store.default_profile
            for name in self.agent.profile_names():
                profile = self.agent.profile_store.get(name)
                markers = []
                if name == current:
                    markers.append("current")
                if name == default_name:
                    markers.append("default")
                marker_text = f" ({', '.join(markers)})" if markers else ""
                auth = f"api_key_env={profile.api_key_env}" if profile.api_key_env else (
                    "inline api key" if profile.api_key else "no api key")
                lines.append(
                    f"- `{name}`{marker_text} — `{profile.provider}` • `{profile.model}` • `{profile.base_url}` • {auth}"
                )
            self._add_system_note(
                "Saved profiles:\n"
                + ("\n".join(lines) if lines else "<no profiles>")
                + f"\n\nProfile file: `{self.agent.profile_store.path}`"
            )
            return True

        if command == "/profile":
            if not args:
                profile = self.agent.current_profile()
                self._add_system_note(
                    "Current profile:\n"
                    f"- Name: `{profile.name}`\n"
                    f"- Provider: `{profile.provider}`\n"
                    f"- Model: `{profile.model}`\n"
                    f"- Base URL: `{profile.base_url}`\n"
                    f"- API key env: `{profile.api_key_env or '<none>'}`"
                )
                return True

            if args[0].lower() == "add":
                options, error = self._parse_profile_add_options(args[1:])
                if error:
                    self._add_system_note(error)
                    return True

                provider = str(options.get("provider", "")).strip()
                profile_name = str(options.get("name", "")).strip()
                model = str(options.get("model", "")).strip()
                headers = options.get("headers") or {}
                assert isinstance(headers, dict)
                make_default = bool(options.get("default", False))
                switch_to = bool(options.get("switch", False))
                try:
                    base_url = str(options.get(
                        "base_url") or default_base_url_for_provider(provider)).strip()
                    profile = ModelProfile(
                        name=profile_name,
                        provider=provider,
                        model=model,
                        base_url=base_url,
                        api_key=str(options.get("api_key", "")
                                    ).strip() or None,
                        api_key_env=str(options.get(
                            "api_key_env", "")).strip() or None,
                        headers={str(key): str(value)
                                 for key, value in headers.items()},
                    )
                    self.agent.save_profile(profile, make_default=make_default)
                    if switch_to:
                        self.agent.set_profile(profile.name)
                except ValueError as exc:
                    self._add_system_note(f"Could not save profile: `{exc}`")
                    return True

                details = [
                    f"Saved profile `{profile.name}`.",
                    f"Provider: `{profile.provider}`",
                    f"Model: `{profile.model}`",
                    f"Base URL: `{profile.base_url}`",
                    f"Profile file: `{self.agent.profile_store.path}`",
                ]
                if make_default:
                    details.append("Set as default profile.")
                if switch_to:
                    details.append("Switched to the new profile.")
                    self._set_status(self._ready_status())
                self._add_system_note("\n".join(details))
                return True

            profile_name = " ".join(args).strip()
            old_profile = self.agent.current_profile().name
            try:
                self.agent.set_profile(profile_name)
            except ValueError as exc:
                self._add_system_note(str(exc))
                return True
            new_profile = self.agent.current_profile()
            self._add_system_note(
                f"Switched profile from `{old_profile}` to `{new_profile.name}`. Conversation preserved."
            )
            self._set_status(self._ready_status())
            return True

        if command == "/model":
            profile = self.agent.current_profile()
            if not args:
                self._add_system_note(
                    "Current model:\n"
                    f"- Profile: `{profile.name}`\n"
                    f"- Provider: `{profile.provider}`\n"
                    f"- Model: `{profile.model}`\n"
                    "Usage:\n"
                    "- `/model list` — list models from the current endpoint\n"
                    "- `/model <name>` — override the active profile's model"
                )
                return True

            if len(args) == 1 and args[0].lower() == "list":
                model_names, error = self.agent.available_models()
                if error:
                    self._add_system_note(f"Could not list models: `{error}`")
                    return True
                if not model_names:
                    self._add_system_note(
                        "This endpoint did not report any available models.")
                    return True
                models_text = "\n".join(f"- `{name}`" for name in model_names)
                self._add_system_note(
                    f"Available models for `{profile.name}`:\n{models_text}")
                return True

            new_model = " ".join(args).strip()
            if not new_model:
                self._add_system_note(
                    "Usage: `/model list` or `/model <name>`")
                return True

            old_model = profile.model
            if new_model == old_model:
                self._add_system_note(f"Already using model `{new_model}`.")
                return True

            self.agent.set_model(new_model)
            self._add_system_note(
                f"Switched model from `{old_model}` to `{new_model}` within profile `{profile.name}`."
            )
            self._set_status(self._ready_status())
            return True

        if command == "/max_iterations":
            if len(args) != 1:
                self._add_system_note(
                    "Usage: `/max_iterations <positive integer|-1>`")
                return True

            max_iterations, error = self._parse_max_iterations(args[0])
            if error:
                self._add_system_note(error)
                return True

            self.agent.config.max_iterations = max_iterations
            self._add_system_note(
                "Set agent tool-loop max iterations to "
                f"{self._max_iterations_text()} for this session."
            )
            self._set_status(self._ready_status())
            return True

        if command == "/status":
            profile = self.agent.current_profile()
            context_count = len(self.agent.project_context_files)
            self._add_system_note(
                "Current status:\n"
                f"- {self._status_summary()}\n"
                f"- Base URL: `{profile.base_url}`\n"
                f"- Profile file: `{self.agent.profile_store.path}`\n"
                f"- Agent tool-loop max iterations: {self._max_iterations_text()}\n"
                f"- Bash enabled: `{self.agent.config.bash_enabled}`\n"
                f"- Bash read-only: `{self.agent.config.bash_readonly_mode}`\n"
                f"- Project instruction files loaded: `{context_count}`\n"
                f"- Working directory: `{os.getcwd()}`"
            )
            return True

        if command == "/cwd":
            self._add_system_note(
                f"Current working directory: `{os.getcwd()}`")
            return True

        if command == "/history":
            if args and args[0].lower() == "search":
                query = " ".join(args[1:]).strip()
                if not query:
                    self._add_system_note("Usage: `/history search <text>`")
                    return True
                matches = [
                    entry for entry in self.input_history
                    if query.lower() in entry.lower()
                ]
                if not matches:
                    self._add_system_note(
                        f"No prompt history entries matched `{query}`.")
                    return True
                history_lines = "\n".join(
                    f"- {_truncate(entry.replace(chr(10), ' ⏎ '), 160)}"
                    for entry in matches[-10:]
                )
                self._add_system_note(
                    f"Prompt history matches for `{query}`:\n{history_lines}")
                return True
            if not self.input_history:
                self._add_system_note("Prompt history is empty.")
            else:
                history_lines = "\n".join(
                    f"{index + 1}. {_truncate(entry.replace(chr(10), ' ⏎ '), 120)}"
                    for index, entry in enumerate(self.input_history[-10:])
                )
                self._add_system_note(
                    f"Recent prompts:\n{history_lines}\n\nTip: use `/history search <text>` to find an older prompt.")
            return True

        if command == "/prompt":
            system_prompt = self.agent.messages[0]["content"] if self.agent.messages else "<missing>"
            self._add_system_note(
                f"Active system prompt:\n\n```text\n{system_prompt}\n```")
            return True

        if command == "/context":
            self._add_system_note(self._context_status_text())
            return True

        if command == "/reload_profiles":
            try:
                self.agent.reload_profiles()
            except ValueError as exc:
                self._add_system_note(f"Could not reload profiles: `{exc}`")
                return True
            self._add_system_note(
                f"Reloaded profiles from `{self.agent.profile_store.path}`. Active profile: `{self.agent.current_profile().name}`."
            )
            self._set_status(self._ready_status())
            return True

        if command == "/reload_context":
            previous_files = set(self.agent.project_context_files)
            self.project_context, self.project_context_files = load_project_context(
                os.getcwd())
            self.agent.set_project_context(
                self.project_context, self.project_context_files)
            current_files = set(self.project_context_files)
            added = sorted(current_files - previous_files)
            removed = sorted(previous_files - current_files)
            details = [self._context_status_text()]
            if added:
                details.append(
                    "Added:\n" + "\n".join(f"- `{path}`" for path in added))
            if removed:
                details.append(
                    "Removed:\n" + "\n".join(f"- `{path}`" for path in removed))
            if not self.project_context_files:
                details = [
                    "Reloaded project instructions. No `AGENTS.md` or skill files were found."
                ]
            self._add_system_note("\n\n".join(details))
            self._set_status(self._ready_status())
            return True

        if command == "/debug":
            if not args:
                self._add_system_note(
                    f"Debug pane is currently `{'on' if self.debug_visible else 'off'}`."
                )
                return True
            arg = args[0].lower()
            if arg in {"on", "off"}:
                self.debug_visible = arg == "on"
                self._debug_log_widget().display = self.debug_visible
                self._add_system_note(
                    f"Debug pane {'enabled' if self.debug_visible else 'disabled'}."
                )
                return True
            self._add_system_note("Usage: `/debug on` or `/debug off`")
            return True

        self._add_system_note(self._unknown_command_message(command))
        return True

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id == "user-input":
            self._resize_prompt_input()

    def _debug_log_widget(self) -> RichLog:
        return self.query_one("#debug-log", RichLog)

    def _log_debug(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._debug_log_widget().write(f"[{timestamp}] {message}")

    def _format_debug_data(self, data: object, max_chars: int = 1500) -> str:
        try:
            text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        except TypeError:
            text = str(data)
        return _truncate(text, max_chars)

    def _is_chat_near_bottom(self, threshold: int = 3) -> bool:
        chat = self._chat_container()
        return (chat.max_scroll_y - chat.scroll_y) <= threshold

    def _event_targets_chat(self, event: events.MouseEvent) -> bool:
        widget = event.widget
        if widget is None:
            return False
        chat = self._chat_container()
        return widget is chat or chat in widget.ancestors

    def _schedule_chat_scroll_end(self, force: bool = False) -> None:
        if force or self.chat_auto_follow or self._is_chat_near_bottom():
            self.chat_auto_follow = True
            self.call_after_refresh(
                self._chat_container().scroll_end, animate=False)

    def _update_chat_auto_follow(self) -> None:
        self.chat_auto_follow = self._is_chat_near_bottom()

    def _auto_follow_chat(self) -> None:
        if not self.is_processing or not self.chat_auto_follow:
            return
        if not self._is_chat_near_bottom(threshold=8):
            self.chat_auto_follow = False
            return
        self._chat_container().scroll_end(animate=False)

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        if self._event_targets_chat(event):
            self.chat_auto_follow = False

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        if self._event_targets_chat(event):
            self.call_after_refresh(self._update_chat_auto_follow)

    def _add_message(
        self,
        role: str,
        content: str,
        finalized: bool = True,
        render_mode: str = "markdown",
    ) -> ChatMessage:
        should_autoscroll = self.chat_auto_follow or self._is_chat_near_bottom()
        message = ChatMessage(
            role, content, finalized=finalized, render_mode=render_mode)
        message.add_class(f"{role}-message")
        self._chat_container().mount(message)
        self._schedule_chat_scroll_end(force=should_autoscroll)
        return message

    async def action_scroll_chat_up(self) -> None:
        self._chat_container().scroll_up(animate=False)
        self.chat_auto_follow = False

    async def action_scroll_chat_down(self) -> None:
        self._chat_container().scroll_down(animate=False)
        self.call_after_refresh(self._update_chat_auto_follow)

    async def action_scroll_chat_page_up(self) -> None:
        self._chat_container().scroll_page_up(animate=False)
        self.chat_auto_follow = False

    async def action_scroll_chat_page_down(self) -> None:
        self._chat_container().scroll_page_down(animate=False)
        self.call_after_refresh(self._update_chat_auto_follow)

    async def action_scroll_chat_home(self) -> None:
        self._chat_container().scroll_home(animate=False)
        self.chat_auto_follow = False

    async def action_scroll_chat_end(self) -> None:
        self.chat_auto_follow = True
        self._schedule_chat_scroll_end(force=True)

    async def action_clear_chat(self) -> None:
        if self.is_processing:
            self.notify(
                "Wait for the current response to finish before clearing.")
            return
        self.agent.reset()
        await self._chat_container().remove_children()
        self._debug_log_widget().clear()
        self._add_message("system", "Conversation cleared.", finalized=True)
        self._log_debug("Conversation and debug log cleared.")
        self._set_status(self._ready_status())

    async def action_toggle_debug(self) -> None:
        self.debug_visible = not self.debug_visible
        self._debug_log_widget().display = self.debug_visible
        state = "shown" if self.debug_visible else "hidden"
        self._log_debug(f"Debug pane {state}.")
        self.notify(f"Debug pane {state}.")

    async def action_history_previous(self) -> None:
        if self.is_processing or not self.input_history:
            return
        if self.input_history_index is None:
            self.input_history_draft = self._prompt_input().text
            self._show_history_entry(len(self.input_history) - 1)
            return
        if self.input_history_index > 0:
            self._show_history_entry(self.input_history_index - 1)

    async def action_history_next(self) -> None:
        if self.is_processing or not self.input_history:
            return
        if self.input_history_index is None:
            return
        if self.input_history_index < len(self.input_history) - 1:
            self._show_history_entry(self.input_history_index + 1)
            return
        self._show_history_entry(None)

    async def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        raw_input = event.value.strip()
        if not raw_input:
            return

        if self.is_processing:
            self.notify("The agent is still working.")
            return

        self._set_prompt_text("")
        self.input_history_index = None

        if raw_input == "/clear":
            await self.action_clear_chat()
            return

        if self._handle_slash_command(raw_input):
            self.chat_auto_follow = True
            self._schedule_chat_scroll_end(force=True)
            return

        self._record_history(raw_input)
        input_widget = self._prompt_input()
        input_widget.disabled = True
        self.is_processing = True

        self.chat_auto_follow = True
        self._add_message("user", raw_input, finalized=True)
        self._log_debug(f"User prompt: {_truncate(raw_input, 300)}")
        self._set_status("Thinking…")

        self.run_worker(
            self.process_message(raw_input),
            exclusive=True,
            thread=False,
        )

    async def process_message(self, user_input: str) -> None:
        buffer: list[str] = []
        loop = asyncio.get_running_loop()
        last_flush = loop.time()
        generator = self.agent.run(user_input)
        assistant_message: ChatMessage | None = None

        def next_event():
            try:
                return next(generator), False
            except StopIteration:
                return None, True

        def ensure_assistant_message() -> ChatMessage:
            nonlocal assistant_message
            if assistant_message is None:
                assistant_message = self._add_message(
                    "assistant", "", finalized=False)
            return assistant_message

        def flush_buffer() -> None:
            nonlocal last_flush
            if not buffer:
                return
            should_autoscroll = self.chat_auto_follow or self._is_chat_near_bottom()
            ensure_assistant_message().append_stream("".join(buffer))
            buffer.clear()
            last_flush = loop.time()
            self._schedule_chat_scroll_end(force=should_autoscroll)

        try:
            while True:
                event, stopped = await asyncio.to_thread(next_event)
                if stopped:
                    break

                event_type = event.get("type")
                if event_type == "debug":
                    self._log_debug(
                        f"{event.get('label', 'event')}: {self._format_debug_data(event.get('data'))}"
                    )
                    continue

                if event_type == "assistant_start":
                    self._log_debug("assistant_start")
                    continue

                if event_type == "content_delta":
                    buffer.append(event.get("delta", ""))
                    now = loop.time()
                    if now - last_flush >= self.agent.config.stream_batch_interval or "\n" in event.get("delta", ""):
                        flush_buffer()
                    continue

                flush_buffer()

                if event_type == "tool_call":
                    if assistant_message is not None:
                        assistant_message.finalize()
                        assistant_message = None
                    self._log_debug(
                        f"tool_call {event.get('name', '<unknown>')}: {self._format_debug_data(event.get('arguments', {}), 1000)}"
                    )
                    tool_name = event.get("name", "<unknown>")
                    self._add_message(
                        "tool",
                        f"Tool • `{tool_name}`\n\n{self._compact_arguments(event.get('arguments', {}), 220)}",
                        finalized=True,
                        render_mode="markdown",
                    )
                    self._set_status(f"Running tool • {tool_name}")
                    continue

                if event_type == "tool_result":
                    result = event.get("result", "")
                    self._log_debug(
                        f"tool_result {event.get('name', '<unknown>')}: {_truncate(result, 1200)}"
                    )
                    tool_name = event.get("name", "<unknown>")
                    result_preview = _truncate(result, 800)
                    suffix = "\n\n_Output truncated for readability._" if len(
                        result) > len(result_preview) else ""
                    self._add_message(
                        "tool",
                        f"Tool result • `{tool_name}`\n\n```text\n{result_preview}\n```{suffix}",
                        finalized=True,
                        render_mode="markdown",
                    )
                    self._set_status("Thinking…")
                    continue

                if event_type == "error":
                    self._log_debug(
                        f"error: {event.get('message', 'Unknown error')}")
                    ensure_assistant_message().append_stream(
                        f"\n\n{event.get('message', 'Unknown error')}"
                    )
                    break

                if event_type == "assistant_done":
                    self._log_debug(
                        f"assistant_done: {_truncate(event.get('content', ''), 500)}"
                    )
                    continue

            flush_buffer()
        except Exception as exc:
            ensure_assistant_message().append_stream(f"\n\nError: {exc}")
        finally:
            if assistant_message is not None:
                assistant_message.finalize()
                self._schedule_chat_scroll_end()
            self.is_processing = False
            input_widget = self._prompt_input()
            input_widget.disabled = False
            self._resize_prompt_input()
            input_widget.focus()
            self._set_status(self._ready_status())
