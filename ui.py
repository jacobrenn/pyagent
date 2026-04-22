from __future__ import annotations

import asyncio
from datetime import datetime
import json
import os

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Footer, Header, Label, Markdown, RichLog, Static, TextArea

from agent import Agent
from project_context import load_project_context


PROMPT_INPUT_MIN_HEIGHT = 3
PROMPT_INPUT_MAX_HEIGHT = 8


def _truncate(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


class ChatMessage(Vertical):
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
        self._stream_widget = Static(
            content, markup=False, classes="stream-content")
        self._markdown_widget = Markdown(content, classes="markdown-content")

    def _label_text(self) -> str:
        names = {
            "user": "You",
            "assistant": "Assistant",
            "tool": "Tool",
            "system": "System",
        }
        return names.get(self.role, self.role.title())

    def compose(self) -> ComposeResult:
        yield self._label
        yield self._stream_widget
        yield self._markdown_widget

    def on_mount(self) -> None:
        self._sync_mode()

    def _sync_mode(self) -> None:
        use_markdown = self.render_mode == "markdown"
        self._stream_widget.display = (not self.finalized) or not use_markdown
        self._markdown_widget.display = self.finalized and use_markdown
        if self.finalized and use_markdown:
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
    SUB_TITLE = "Markdown-aware local coding agent"
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
        width: 90%;
        max-width: 120;
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
        height: 3;
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
    .markdown-content {
        width: 1fr;
        height: auto;
    }
    """

    def __init__(self, model: str | None = None):
        super().__init__()
        self.project_context, self.project_context_files = load_project_context(
            os.getcwd())
        self.agent = Agent(
            model=model,
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
        self._add_message(
            "system",
            "Welcome to **PyAgent**. Responses stream as plain text for speed and render as Markdown when complete.",
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

    def _ready_status(self) -> str:
        return f"Ready • Model: {self.agent.config.model}"

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

    def _handle_slash_command(self, raw_input: str) -> bool:
        if not raw_input.startswith("/"):
            return False

        parts = raw_input.split()
        command = parts[0].lower()
        args = parts[1:]

        if command == "/help":
            self._add_system_note(
                "Available commands:\n"
                "- `/clear` — clear the conversation\n"
                "- `/help` — show this help\n"
                "- `/tools` — list available tools\n"
                "- `/model` — show the current model and usage\n"
                "- `/model list` — list available Ollama models\n"
                "- `/model <name>` — switch to a different Ollama model\n"
                "- `/status` — show current configuration\n"
                "- `/cwd` — show the current working directory\n"
                "- `/history` — show recent prompt history\n"
                "- `/prompt` — show the active system prompt\n"
                "- `/debug on|off` — show or hide the debug pane\n"
                "- `/reload_context` — reload `AGENTS.md` and local skill files"
            )
            return True

        if command == "/tools":
            tool_names = "\n".join(
                f"- `{name}`" for name in self.agent.tool_registry.names())
            self._add_system_note(f"Available tools:\n{tool_names}")
            return True

        if command == "/model":
            if not args:
                self._add_system_note(
                    "Current model: "
                    f"`{self.agent.config.model}`\n"
                    "Usage:\n"
                    "- `/model list` — list available Ollama models\n"
                    "- `/model <name>` — switch models without clearing the conversation"
                )
                return True

            if len(args) == 1 and args[0].lower() == "list":
                model_names, error = self.agent.available_models()
                if error:
                    self._add_system_note(
                        f"Could not list Ollama models: `{error}`")
                    return True
                if not model_names:
                    self._add_system_note(
                        "Ollama did not report any installed models.")
                    return True
                models_text = "\n".join(f"- `{name}`" for name in model_names)
                self._add_system_note(
                    f"Available Ollama models:\n{models_text}")
                return True

            new_model = " ".join(args).strip()
            if not new_model:
                self._add_system_note(
                    "Usage: `/model list` or `/model <name>`")
                return True

            old_model = self.agent.config.model
            if new_model == old_model:
                self._add_system_note(f"Already using model `{new_model}`.")
                return True

            self.agent.set_model(new_model)
            self._add_system_note(
                f"Switched model from `{old_model}` to `{new_model}`. Conversation preserved."
            )
            self._set_status(self._ready_status())
            return True

        if command == "/status":
            context_count = len(self.agent.project_context_files)
            self._add_system_note(
                "Current status:\n"
                f"- Model: `{self.agent.config.model}`\n"
                f"- Base URL: `{self.agent.config.base_url}`\n"
                f"- Max iterations: `{self.agent.config.max_iterations}`\n"
                f"- Bash enabled: `{self.agent.config.bash_enabled}`\n"
                f"- Bash read-only: `{self.agent.config.bash_readonly_mode}`\n"
                f"- Project instruction files loaded: `{context_count}`\n"
                f"- Debug pane visible: `{self.debug_visible}`"
            )
            return True

        if command == "/cwd":
            self._add_system_note(
                f"Current working directory: `{os.getcwd()}`")
            return True

        if command == "/history":
            if not self.input_history:
                self._add_system_note("Prompt history is empty.")
            else:
                history_lines = "\n".join(
                    f"{index + 1}. {_truncate(entry.replace(chr(10), ' ⏎ '), 120)}"
                    for index, entry in enumerate(self.input_history[-10:])
                )
                self._add_system_note(f"Recent prompts:\n{history_lines}")
            return True

        if command == "/prompt":
            system_prompt = self.agent.messages[0]["content"] if self.agent.messages else "<missing>"
            self._add_system_note(
                f"Active system prompt:\n\n```text\n{system_prompt}\n```")
            return True

        if command == "/reload_context":
            self.project_context, self.project_context_files = load_project_context(
                os.getcwd())
            self.agent.set_project_context(
                self.project_context, self.project_context_files)
            if self.project_context_files:
                loaded_files = "\n".join(
                    f"- `{path}`" for path in self.project_context_files)
                self._add_system_note(
                    f"Reloaded project instructions:\n{loaded_files}")
            else:
                self._add_system_note(
                    "Reloaded project instructions. No `AGENTS.md` or skill files were found.")
            return True

        if command == "/debug":
            if not args:
                self._add_system_note(
                    f"Debug pane is currently `{'on' if self.debug_visible else 'off'}`.")
                return True
            arg = args[0].lower()
            if arg in {"on", "off"}:
                self.debug_visible = arg == "on"
                self._debug_log_widget().display = self.debug_visible
                self._add_system_note(
                    f"Debug pane {'enabled' if self.debug_visible else 'disabled'}.")
                return True
            self._add_system_note("Usage: `/debug on` or `/debug off`")
            return True

        self._add_system_note(
            f"Unknown command: `{command}`. Use `/help` to see available commands.")
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
                        f"{event.get('label', 'event')}: {self._format_debug_data(event.get('data'))}")
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
                    tool_name = event.get('name', '<unknown>')
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
                    tool_name = event.get('name', '<unknown>')
                    result_preview = _truncate(result, 2500)
                    self._add_message(
                        "tool",
                        f"Tool result • `{tool_name}`\n\n```text\n{result_preview}\n```",
                        finalized=True,
                        render_mode="markdown",
                    )
                    self._set_status("Thinking…")
                    continue

                if event_type == "error":
                    self._log_debug(
                        f"error: {event.get('message', 'Unknown error')}")
                    ensure_assistant_message().append_stream(
                        f"\n\n{event.get('message', 'Unknown error')}")
                    break

                if event_type == "assistant_done":
                    self._log_debug(
                        f"assistant_done: {_truncate(event.get('content', ''), 500)}")
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
