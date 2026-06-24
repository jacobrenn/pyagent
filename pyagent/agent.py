from __future__ import annotations

import copy
import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import AppConfig, SYSTEM_PROMPT
from .extensions import Ctx, EventBus, NoOpLog
from .external_tools import (
    DiscoveryResult,
    build_external_tool_specs,
    default_runner_command,
    discover_external_tools,
)
from .llm_client import build_chat_client
from .model_profiles import ModelProfile, ProfileStore, load_profile_store, save_profile_store, update_profile_store
from .tools import ToolRegistry, create_default_tool_registry


class Agent:
    def __init__(
        self,
        model: str | None = None,
        profile: str | None = None,
        config: AppConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        project_context: str = "",
        project_context_files: list[str] | None = None,
        external_tool_discovery: DiscoveryResult | None = None,
    ):
        self.config = config or AppConfig.from_env()
        self.profile_store: ProfileStore = load_profile_store(
            self.config.model_profiles_path
        )
        self.active_profile_name = profile or self.config.default_profile or self.profile_store.default_profile
        self.model_override = model.strip() if model else None
        self.external_tool_discovery: DiscoveryResult | None = external_tool_discovery
        if tool_registry is None:
            self.external_tool_discovery = (
                external_tool_discovery
                if external_tool_discovery is not None
                else self._discover_external_tools()
            )
            self.tool_registry = create_default_tool_registry(
                self.config,
                external_specs=self._external_specs_from_discovery(
                    self.external_tool_discovery),
            )
        else:
            self.tool_registry = tool_registry
        self.tools = self.tool_registry.definitions() if self.config.tools_enabled else []
        self.project_context = project_context.strip()
        self.project_context_files = list(project_context_files or [])
        self.prompt_file_created = False
        self.prompt_file_path = None

        # --- Extension event bus. Emits are no-ops until extensions register
        # handlers via load_extensions(). _ext_log is swapped by /logging. ---
        self._ext_log: Any = NoOpLog()
        self.bus = EventBus(self._ext_log)
        self._session_started = False
        # Ephemeral skill injection. Two buffers keep the lifecycle explicit:
        #   _this_skills  -> injected into the system prompt THIS turn.
        #   _next_skills  -> declared this turn (e.g. at turn_end), applied NEXT
        #                    turn (rotated into _this_skills at the top of run()).
        # ctx.add_skill routes through _add_skill, which targets _this_skills
        # by default but _next_skills during the turn_end emit. This makes a
        # skill declared at turn_start land in the SAME turn's prompt (the
        # system-prompt suffix is re-injected after turn_start fires), while a
        # skill declared at turn_end still applies next turn.
        self._this_skills: set[str] = set()
        self._next_skills: set[str] = set()
        self._skill_target: str = "next"
        self._last_logged_skills: set[str] = set()
        # Base system prompt (without the extension-skill suffix). Set in
        # reset()/run() after before_agent_start so _refresh_system_message can
        # re-append skills each iteration without clobbering base modifications.
        self._system_prompt_base_value: str = ""
        self._ext_ctx = Ctx(self._add_skill, self._ext_log)

        created, path = self._ensure_system_prompt_file()
        if created:
            self.prompt_file_created = True
            self.prompt_file_path = path

        self._rebuild_client()
        self.reset()

    def _discover_external_tools(self) -> DiscoveryResult | None:
        if not self.config.user_tools_enabled:
            return None
        try:
            return discover_external_tools(
                user_dir=self.config.user_dir,
                runner=self.config.tool_runner,
                describe_timeout=self.config.user_tool_describe_timeout,
                extra_tool_dirs=self._extension_tool_dirs(),
            )
        except Exception:
            return None

    def _external_specs_from_discovery(self, discovery: DiscoveryResult | None):
        if discovery is None:
            return None
        return build_external_tool_specs(
            discovery,
            invoke_timeout=self.config.user_tool_timeout,
            runner_command=default_runner_command(discovery.runner),
        )

    def reload_external_tools(self) -> DiscoveryResult | None:
        """Re-scan ``~/.pyagent/tools/`` and loaded extensions' tool dirs.

        Tool calling state is preserved (`tools_enabled`, system prompt
        composition); only the registry contents change. The conversation
        is reset so the model sees a clean tool surface.
        """
        discovery = self._rebuild_external_tools()
        self.reset()
        return discovery

    def _ensure_system_prompt_file(self) -> tuple[bool, str | None]:
        path = self.config.system_prompt_path
        if os.path.exists(path):
            return False, None

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(SYSTEM_PROMPT)
            return True, path
        except (IOError, OSError) as exc:
            return False, str(exc)

    def _system_prompt_base(self) -> str:
        """Base system prompt: file content + tool-disable notes + project
        context, **excluding** the extension-skill suffix.

        Skills are appended separately by ``_refresh_system_message`` so they
        can be re-injected after ``turn_start`` (where extensions typically
        declare skills for this turn) without rebuilding the whole prompt.
        """
        try:
            with open(self.config.system_prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            if not prompt:
                prompt = SYSTEM_PROMPT
        except (FileNotFoundError, IOError):
            prompt = SYSTEM_PROMPT

        if not self.config.tools_enabled:
            prompt += (
                "\n\nTool calling is disabled for this session. Do not call tools. "
                "Answer using only the conversation and your built-in knowledge."
            )
        elif not self.config.builtin_tools_enabled:
            prompt += (
                "\n\nBuilt-in tools are disabled for this session. "
                "Use only the external tools that are explicitly advertised. "
                "If no tools are available, answer without tools."
            )
        if self.project_context:
            prompt = f"{prompt}\n\n{self.project_context}"
        return prompt

    def _collect_extension_skills(self) -> str:
        """Return the extension-skill suffix for the keys active this turn.

        Logs the injected skill set via the session logger the first time a
        given set is seen (deduped so repeated re-injection within a turn does
        not spam the log). Missing/unreadable skill files are skipped by
        ``collect_skill_text``.
        """
        from .extensions.loader import collect_skill_text

        text = collect_skill_text(self)
        if text and self._this_skills != self._last_logged_skills:
            self._last_logged_skills = set(self._this_skills)
            log_skills = getattr(self._ext_log, "log_extension_skills", None)
            if callable(log_skills):
                log_skills(sorted(self._this_skills), len(text))
        return text

    def _refresh_system_message(self) -> None:
        """Recompose ``messages[0]`` = base prompt + this turn's skill suffix.

        Called after the base is finalized (post ``before_agent_start``) and
        again after every ``turn_start`` so skills declared at ``turn_start``
        reach the LLM in the SAME turn instead of one turn late. Preserves the
        base prompt (including any ``before_agent_start`` modifications); only
        the skill suffix is recomputed. Safe to call each iteration.
        """
        base = self._system_prompt_base_value
        skill_text = self._collect_extension_skills()
        content = f"{base}\n\n{skill_text}" if skill_text else base
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = content
        else:
            self.messages.insert(0, {"role": "system", "content": content})

    def _add_skill(self, key: str) -> None:
        """``ctx.add_skill`` target: route to this turn or next turn.

        Default target is ``next`` (so mid/late-turn declarations such as
        ``turn_end`` apply next turn, preserving the one-turn-lag contract for
        end-of-turn declarations). The agent sets the target to ``this`` around
        the pre-request emits (``input``, ``before_agent_start``,
        ``turn_start``) so skills declared there land in the current prompt.
        """
        ext = getattr(self._ext_ctx, "extension", "") or ""
        entry = f"{ext}/{key}" if ext else key
        if self._skill_target == "next":
            self._next_skills.add(entry)
        else:
            self._this_skills.add(entry)

    def _rebuild_client(self) -> None:
        existing_client = getattr(self, "client", None)
        if existing_client is not None:
            close = getattr(existing_client, "close", None)
            if callable(close):
                close()

        profile = self.profile_store.get(self.active_profile_name)
        if self.model_override:
            profile = ModelProfile(
                name=profile.name,
                provider=profile.provider,
                model=self.model_override,
                base_url=profile.base_url,
                api_key=profile.api_key,
                api_key_env=profile.api_key_env,
                headers=dict(profile.headers),
            )
        self.client = build_chat_client(
            profile, timeout=self.config.request_timeout)

    def current_profile(self) -> ModelProfile:
        return self.client.profile

    def profile_names(self) -> list[str]:
        return self.profile_store.names()

    def set_project_context(self, project_context: str, files: list[str] | None = None) -> None:
        self.project_context = project_context.strip()
        self.project_context_files = list(files or [])
        self.reset()

    def set_profile(self, profile: str) -> None:
        self.active_profile_name = profile.strip()
        self.model_override = None
        self._rebuild_client()
        self._emit_model_select(source="profile")

    def reload_profiles(self) -> None:
        current_profile_name = self.active_profile_name
        self.profile_store = load_profile_store(
            self.config.model_profiles_path)
        if current_profile_name not in self.profile_store.profiles:
            self.active_profile_name = self.profile_store.default_profile
            self.model_override = None
        self._rebuild_client()

    def save_profile(self, profile: ModelProfile, make_default: bool = False) -> None:
        update_profile_store(self.profile_store, profile,
                             make_default=make_default)
        save_profile_store(self.profile_store)
        self.profile_store = load_profile_store(self.profile_store.path)

    def set_model(self, model: str) -> None:
        self.model_override = model.strip()
        self._rebuild_client()
        self._emit_model_select(source="override")

    def set_tools_enabled(self, enabled: bool) -> None:
        self.config.tools_enabled = enabled
        self.tools = self.tool_registry.definitions() if enabled else []
        self.reset()

    def available_models(self) -> tuple[list[str], str | None]:
        response = self.client.list_models()
        if "error" in response:
            return [], str(response["error"])

        names = [name for name in response.get(
            "models", []) if isinstance(name, str) and name]
        return names, None

    def reset(self) -> None:
        self._system_prompt_base_value = self._system_prompt_base()
        self.messages: list[dict[str, Any]] = []
        self._refresh_system_message()

    def load_messages(self, messages: list[dict[str, Any]] | None) -> None:
        self.reset()
        if not messages:
            return
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("messages must contain only objects")
            normalized = self._normalize_message(message)
            if normalized is None:
                continue
            self.messages.append(normalized)
        self._trim_history()

    def _normalize_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(
                "each message must include a non-empty string role")
        if role == "system":
            return None
        if role not in {"user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {role}")

        normalized: dict[str, Any] = {
            "role": role,
            "content": str(message.get("content", "")),
        }

        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if tool_calls is not None:
                if not isinstance(tool_calls, list):
                    raise ValueError("assistant tool_calls must be a list")
                normalized["tool_calls"] = tool_calls
        elif role == "tool":
            tool_call_id = message.get("tool_call_id")
            if tool_call_id is not None:
                normalized["tool_call_id"] = str(tool_call_id)
            name = message.get("name")
            if name is not None:
                normalized["name"] = str(name)
            tool_name = message.get("tool_name")
            if tool_name is not None:
                normalized["tool_name"] = str(tool_name)

        return normalized

    def add_message(self, role: str, content: str, **extra: Any) -> None:
        message = {"role": role, "content": content}
        message.update(extra)
        self.messages.append(message)
        self._trim_history()

    # --- Extension event bus plumbing ---

    def attach_logger(self, logger: Any) -> None:
        """Bind (or unbind with ``None``) the SessionLogger used by the bus."""
        log = logger if logger is not None else NoOpLog()
        self._ext_log = log
        self.bus.set_log(log)
        self._ext_ctx.log = log

    def _extension_tool_dirs(self) -> list[Path]:
        """Tool dirs of currently *loaded* package extensions.

        Only loaded extensions contribute, which is what hides an unloaded
        extension's tools from discovery.
        """
        from .extensions.loader import loaded_extension_tool_dirs
        from .user_runtime import resolve_user_dir, user_extensions_dir

        ext_dir = user_extensions_dir(resolve_user_dir(self.config.user_dir))
        return loaded_extension_tool_dirs(ext_dir, self.bus.loaded_extensions())

    def _rebuild_external_tools(self) -> DiscoveryResult | None:
        """Re-scan external tools (user tools + loaded extensions' tools).

        Rebuilds the tool registry in place and refreshes ``self.tools``.
        Does not reset the conversation. Returns the discovery result.
        """
        discovery = self._discover_external_tools()
        self.external_tool_discovery = discovery
        self.tool_registry = create_default_tool_registry(
            self.config,
            external_specs=self._external_specs_from_discovery(discovery),
        )
        self.tools = self.tool_registry.definitions() if self.config.tools_enabled else []
        return discovery

    def load_extensions(self, *, start_session: bool = True) -> tuple[list[str], list[tuple[str, str]]]:
        """Load all extensions from ``~/.pyagent/extensions/``.

        Returns ``(loaded, failed)``. Idempotent on reload: handlers from a
        prior load of the same name are cleared first. After loading, the
        external-tool registry is rebuilt so colocated ``tools/`` subdirs of
        loaded extensions become discoverable. Entry points call this after
        constructing the agent.
        """
        from .extensions.loader import load_all
        from .user_runtime import resolve_user_dir, user_extensions_dir

        ext_dir = user_extensions_dir(resolve_user_dir(self.config.user_dir))
        loaded, failed = load_all(self.bus, ext_dir, self._ext_log)
        self._rebuild_external_tools()
        if start_session:
            self.start_session()
        return loaded, failed

    def start_session(self, reason: str = "startup") -> None:
        """Emit ``session_start`` once after extensions load."""
        if self._session_started:
            return
        self._session_started = True
        self.bus.emit("session_start", {"reason": reason}, self._ext_ctx)

    def shutdown_session(self, reason: str = "quit") -> None:
        """Emit ``session_shutdown``."""
        if not self._session_started:
            return
        self._session_started = False
        self.bus.emit("session_shutdown", {"reason": reason}, self._ext_ctx)

    def _emit_model_select(self, source: str) -> None:
        try:
            previous = getattr(self, "_prev_model", None)
            current = self.current_profile().model
        except Exception:
            return
        self.bus.emit(
            "model_select",
            {"model": current, "previous_model": previous, "source": source},
            self._ext_ctx,
        )
        self._prev_model = current

    def _trim_history(self) -> None:
        max_history = self.config.max_history_messages
        if len(self.messages) <= max_history + 1:
            return

        system_message = self.messages[0]
        blocks: list[list[dict[str, Any]]] = []
        index = 1
        while index < len(self.messages):
            message = self.messages[index]
            role = message.get("role")

            if role == "tool":
                index += 1
                continue

            if role == "assistant" and message.get("tool_calls"):
                block = [message]
                index += 1
                while index < len(self.messages) and self.messages[index].get("role") == "tool":
                    block.append(self.messages[index])
                    index += 1
                blocks.append(block)
                continue

            blocks.append([message])
            index += 1

        kept_blocks: list[list[dict[str, Any]]] = []
        kept_count = 0
        for block in reversed(blocks):
            block_size = len(block)
            if kept_blocks and kept_count + block_size > max_history:
                break
            kept_blocks.append(block)
            kept_count += block_size
            if kept_count >= max_history:
                break

        trimmed_messages = [message for block in reversed(
            kept_blocks) for message in block]
        self.messages = [system_message, *trimmed_messages]

    def _normalize_arguments(self, arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                return parsed if isinstance(parsed, dict) else {"value": parsed}
            except json.JSONDecodeError:
                return {}
        return {}

    def _normalize_tool_call(
        self,
        tool_call: dict[str, Any],
        iteration: int,
        index: int,
    ) -> tuple[str, dict[str, Any], str]:
        function = tool_call.get("function", {})
        name = function.get("name", "")
        arguments = self._normalize_arguments(function.get("arguments", {}))
        tool_call_id = tool_call.get("id") or str(
            function.get("index", f"call_{iteration}_{index}")
        )
        return name, arguments, tool_call_id

    def _merge_tool_calls(
        self,
        existing: list[dict[str, Any]],
        incoming: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        ordered_keys: list[str] = []

        for tool_call in [*existing, *incoming]:
            function = tool_call.get("function", {})
            key = str(
                tool_call.get("id")
                or function.get("index")
                or f"{function.get('name', '')}:{json.dumps(function.get('arguments', {}), sort_keys=True, default=str)}"
            )
            if key not in merged:
                merged[key] = {
                    **tool_call,
                    "function": {
                        **function,
                        "arguments": function.get("arguments", {}),
                    },
                }
                ordered_keys.append(key)
                continue

            current = merged[key]
            current_function = current.setdefault("function", {})
            if function.get("name"):
                current_function["name"] = function["name"]

            incoming_arguments = function.get("arguments", {})
            existing_arguments = current_function.get("arguments", {})
            if isinstance(existing_arguments, dict) and isinstance(incoming_arguments, dict):
                current_function["arguments"] = {
                    **existing_arguments,
                    **incoming_arguments,
                }
            elif isinstance(existing_arguments, str) and isinstance(incoming_arguments, str):
                current_function["arguments"] = existing_arguments + \
                    incoming_arguments
            elif incoming_arguments:
                current_function["arguments"] = incoming_arguments

        return [merged[key] for key in ordered_keys]

    def _format_tool_results_for_fallback(self, tool_results: list[dict[str, Any]]) -> str:
        if not tool_results:
            return ""

        if len(tool_results) == 1:
            result = str(tool_results[0]["result"]).strip()
            return f"\n\n```text\n{result}\n```" if result else ""

        blocks: list[str] = []
        for tool_result in tool_results:
            result = str(tool_result["result"]).strip()
            if not result:
                continue
            blocks.append(
                f"**{tool_result['name']}**\n\n```text\n{result}\n```")
        if not blocks:
            return ""
        return "\n\nHere are the tool results:\n\n" + "\n\n".join(blocks)

    def _should_append_tool_results(self, content: str, tool_results: list[dict[str, Any]]) -> bool:
        if not tool_results:
            return False

        stripped = content.strip()
        if not stripped:
            return True

        return stripped.endswith(":")

    def run(self, user_input: str):
        ctx = self._ext_ctx

        # Ephemeral skill rotation: skills declared last turn (e.g. at
        # turn_end) become active this turn. _next_skills is cleared for new
        # declarations collected during this turn.
        self._this_skills = set(self._next_skills)
        self._next_skills.clear()

        # input: intercept/transform/handle before the LLM sees it. add_skill
        # during input targets this turn.
        self._skill_target = "this"
        input_event = self.bus.emit(
            "input", {"text": user_input, "source": "user"}, ctx
        )
        self._skill_target = "next"
        action = input_event.get("action", "continue")
        if action == "handled":
            self.add_message("user", input_event.get("text", user_input))
            yield {"type": "assistant_start"}
            yield {"type": "assistant_done", "content": ""}
            self.bus.emit("agent_end", {"messages": list(self.messages)}, ctx)
            return
        if action == "transform" and input_event.get("text") is not None:
            user_input = input_event["text"]

        self.add_message("user", user_input)

        # Rebuild the base prompt from live state every turn (tools/context may
        # have changed since reset()). before_agent_start may rewrite it; its
        # result is stored as the base. Extension skills are appended AFTER
        # turn_start (below), so a skill declared at turn_start lands in this
        # turn's prompt rather than one turn late.
        self._system_prompt_base_value = self._system_prompt_base()
        self._skill_target = "this"
        before = self.bus.emit(
            "before_agent_start",
            {"prompt": user_input, "system_prompt": self._system_prompt_base_value},
            ctx,
        )
        self._skill_target = "next"
        if before.get("system_prompt"):
            self._system_prompt_base_value = before["system_prompt"]
        # Apply rotated skills (from last turn's turn_end) to messages[0].
        self._refresh_system_message()
        self.bus.emit("agent_start", {}, ctx)

        yield {"type": "assistant_start"}
        recent_tool_results: list[dict[str, Any]] = []
        iteration_tool_results: list[dict[str, Any]] = []

        try:
            iteration = 0
            while self.config.max_iterations < 0 or iteration < self.config.max_iterations:
                # turn_start: add_skill here targets this turn, then we
                # re-inject the skill suffix so it reaches this iteration's
                # LLM request (fixes the one-turn-late race).
                self._skill_target = "this"
                self.bus.emit(
                    "turn_start",
                    {"turn_index": iteration, "timestamp": datetime.now(
                        timezone.utc).isoformat()},
                    ctx,
                )
                self._skill_target = "next"
                self._refresh_system_message()
                full_response_parts: list[str] = []
                full_tool_calls: list[dict[str, Any]] = []
                iteration_tool_results = []
                yield {
                    "type": "debug",
                    "label": "iteration_start",
                    "data": {"iteration": iteration + 1, "message_count": len(self.messages)},
                }

                # context: prune/redact the request payload (does not mutate
                # stored history). Deepcopy only when an extension cares.
                if self.bus.handlers("context"):
                    context_payload = self.bus.emit(
                        "context", {"messages": copy.deepcopy(
                            self.messages)}, ctx
                    )
                    request_messages = context_payload.get(
                        "messages", self.messages)
                else:
                    request_messages = self.messages

                self.bus.emit("message_start", {
                              "message": {"role": "assistant"}}, ctx)
                for chunk in self.client.chat_stream(
                    request_messages,
                    tools=self.tools if self.config.tools_enabled else None,
                ):
                    yield {"type": "debug", "label": "llm_chunk", "data": chunk}
                    if "error" in chunk:
                        error_message = f"API Error: {chunk['error']}"
                        yield {"type": "error", "message": error_message}
                        return

                    content = chunk.get("content") or ""
                    if content:
                        full_response_parts.append(content)
                        yield {"type": "content_delta", "delta": content}

                    tool_calls = chunk.get("tool_calls") or []
                    if tool_calls:
                        full_tool_calls = self._merge_tool_calls(
                            full_tool_calls, tool_calls)
                        yield {
                            "type": "debug",
                            "label": "merged_tool_calls",
                            "data": full_tool_calls,
                        }

                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": "".join(full_response_parts),
                }
                if full_tool_calls:
                    assistant_message["tool_calls"] = full_tool_calls
                self.messages.append(assistant_message)
                self._trim_history()

                # message_end: allow replacing the finalized assistant message.
                end_event = self.bus.emit(
                    "message_end", {"message": assistant_message}, ctx)
                replaced = end_event.get("message")
                if isinstance(replaced, dict) and replaced.get("role") == "assistant":
                    assistant_message = replaced
                    self.messages[-1] = assistant_message

                if not full_tool_calls:
                    if self._should_append_tool_results(
                        assistant_message["content"], recent_tool_results
                    ):
                        suffix = self._format_tool_results_for_fallback(
                            recent_tool_results)
                        if suffix:
                            assistant_message["content"] += suffix
                            self.messages[-1]["content"] = assistant_message["content"]
                            yield {
                                "type": "debug",
                                "label": "assistant_tool_result_fallback",
                                "data": {"tool_count": len(recent_tool_results)},
                            }
                            yield {"type": "content_delta", "delta": suffix}
                    yield {
                        "type": "debug",
                        "label": "assistant_message_complete",
                        "data": assistant_message,
                    }
                    yield {
                        "type": "assistant_done",
                        "content": assistant_message["content"],
                    }
                    self.bus.emit(
                        "turn_end",
                        {
                            "turn_index": iteration,
                            "message": assistant_message,
                            "message_count": len(self.messages),
                            "tool_results": list(iteration_tool_results),
                        },
                        ctx,
                    )
                    return

                for index, tool_call in enumerate(full_tool_calls):
                    name, arguments, tool_call_id = self._normalize_tool_call(
                        tool_call,
                        iteration,
                        index,
                    )
                    if not name:
                        result = "Error: received malformed tool call from model."
                        self.messages.append(
                            {
                                "role": "tool",
                                "name": "<invalid>",
                                "content": result,
                                "tool_call_id": tool_call_id,
                            }
                        )
                        recent_tool_results.append(
                            {"name": "<invalid>", "arguments": {}, "result": result}
                        )
                        yield {"type": "tool_result", "name": "<invalid>", "result": result}
                        continue

                    yield {"type": "tool_call", "name": name, "arguments": arguments}

                    # tool_call: audit/gate/mutate before execution.
                    tc_event = self.bus.emit(
                        "tool_call",
                        {"tool_call_id": tool_call_id,
                            "name": name, "input": arguments},
                        ctx,
                    )
                    if tc_event.get("blocked"):
                        reason = tc_event.get(
                            "reason", "Blocked by an extension.")
                        result = f"Blocked: {reason}"
                        self.messages.append(
                            {
                                "role": "tool",
                                "name": name,
                                "content": result,
                                "tool_call_id": tool_call_id,
                            }
                        )
                        recent_tool_results.append(
                            {"name": name, "arguments": arguments, "result": result}
                        )
                        self._trim_history()
                        yield {"type": "tool_result", "name": name, "result": result}
                        continue
                    arguments = tc_event.get("input", arguments)

                    result = self.tool_registry.execute(name, arguments)

                    # tool_result: filter/redact/transform the result.
                    is_error = isinstance(
                        result, str) and result.startswith("Error:")
                    tr_event = self.bus.emit(
                        "tool_result",
                        {
                            "tool_call_id": tool_call_id,
                            "name": name,
                            "input": arguments,
                            "content": result,
                            "details": {},
                            "is_error": is_error,
                        },
                        ctx,
                    )
                    result = tr_event.get("content", result)
                    is_error = tr_event.get("is_error", is_error)

                    yield {
                        "type": "debug",
                        "label": "tool_execution",
                        "data": {
                            "name": name,
                            "arguments": arguments,
                            "result_preview": result[:1000],
                        },
                    }
                    self.messages.append(
                        {
                            "role": "tool",
                            "name": name,
                            "content": result,
                            "tool_call_id": tool_call_id,
                        }
                    )
                    recent_tool_results.append(
                        {"name": name, "arguments": arguments, "result": result}
                    )
                    iteration_tool_results.append(
                        {"name": name, "arguments": arguments, "result": result}
                    )
                    self._trim_history()
                    yield {"type": "tool_result", "name": name, "result": result}

                self.bus.emit(
                    "turn_end",
                    {
                        "turn_index": iteration,
                        "message": assistant_message,
                        "message_count": len(self.messages),
                        "tool_results": list(iteration_tool_results),
                    },
                    ctx,
                )
                iteration += 1

            warning = "Reached maximum iterations without a final answer."
            self.messages.append({"role": "assistant", "content": warning})
            yield {"type": "content_delta", "delta": f"\n\n{warning}"}
            yield {"type": "assistant_done", "content": warning}
        finally:
            self.bus.emit("agent_end", {"messages": list(self.messages)}, ctx)
