from __future__ import annotations
from typing import TYPE_CHECKING, Any
import os
from difflib import get_close_matches
from .model_profiles import ModelProfile, default_base_url_for_provider
from .external_tools import find_tool_script, move_tool_script
from .scaffold import ScaffoldError, create_user_tool
from .tools import BUILTIN_ORIGIN, EXTERNAL_ORIGIN
from .project_context import GLOBAL_SCOPE, PROJECT_SCOPE

if TYPE_CHECKING:
    from .ui import PyAgentApp


def _truncate(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."

# --- Helper Parsers ---


def _parse_bool_option(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_max_iterations(value: str) -> tuple[int | None, str | None]:
    try:
        iterations = int(value.strip())
    except ValueError:
        return None, "Max iterations must be an integer. Use `-1` for infinite."
    if iterations == 0 or iterations < -1:
        return None, "Max iterations must be a positive integer or `-1` for infinite."
    return iterations, None


def _parse_profile_add_options(app: PyAgentApp, args: list[str]) -> tuple[dict[str, Any], str | None]:
    if not args:
        return {}, "Usage: `/profile add <name> provider=<provider> model=<model> [base_url=<url>] [api_key_env=<ENV>] [api_key=<KEY>] [default=true|false] [switch=true|false] [header.<Name>=<Value>]`"
    name = args[0].strip()
    if not name:
        return {}, "Profile name must not be empty."
    options: dict[str, Any] = {"name": name, "headers": {}}
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
            options["headers"][header_name] = value
            continue
        if normalized_key in {"default", "switch"}:
            options[normalized_key] = _parse_bool_option(value)
            continue
        options[normalized_key] = value
    provider = str(options.get("provider", "")).strip()
    model = str(options.get("model", "")).strip()
    if not provider or not model:
        return {}, "`provider` and `model` are required."
    return options, None

# --- Command Handlers ---


def handle_help(app: PyAgentApp, args: list[str]) -> bool:
    app._add_system_note(app._command_help_text())
    return True


def handle_tools(app: PyAgentApp, args: list[str]) -> bool:
    if not args:
        app._add_system_note(app._format_tools_list())
        return True
    action = args[0].lower()
    rest = args[1:]
    if action in {"on", "off"} and not rest:
        enabled = action == "on"
        app.agent.set_tools_enabled(enabled)
        state = "enabled" if enabled else "disabled"
        app._add_system_note(
            f"Tools {state} for this session. Conversation reset so the updated system prompt takes effect.")
        app._set_status(app._ready_status())
        return True
    if action == "reload" and not rest:
        discovery = app.agent.reload_external_tools()
        registry = app.agent.tool_registry
        external_count = len(registry.names_by_origin(EXTERNAL_ORIGIN))
        details = [
            f"Reloaded external tools from `{app.agent.config.user_dir}/tools/`.",
            f"Built-in tools: `{len(registry.names_by_origin(BUILTIN_ORIGIN))}`",
            f"External tools registered: `{external_count}`",
        ]
        if discovery is not None:
            if discovery.broken:
                details.append(
                    f"Skipped due to errors: `{len(discovery.broken)}`")
            if discovery.disabled:
                details.append(
                    f"Disabled scripts: `{len(discovery.disabled)}`")
            if not discovery.runner_available and discovery.runner_message:
                details.append(
                    f"Runner unavailable: {discovery.runner_message}")
        app._add_system_note("\n".join(details))
        app._set_status(app._ready_status())
        return True
    if action == "new":
        if len(rest) != 1:
            app._add_system_note("Usage: `/tools new <name>`")
            return True
        try:
            created = create_user_tool(
                rest[0], user_dir=app.agent.config.user_dir)
        except ScaffoldError as exc:
            app._add_system_note(f"Could not create tool: {exc}")
            return True
        app._add_system_note(
            f"Created starter tool at `{created}`.\n\nEdit it, then run `/tools reload` to register it.")
        return True
    if action in {"enable", "disable"}:
        if len(rest) != 1:
            app._add_system_note(f"Usage: `/tools {action} <name>`")
            return True
        new_path, error = move_tool_script(
            rest[0], user_dir=app.agent.config.user_dir, enable=(action == "enable"))
        if error or new_path is None:
            app._add_system_note(error or "Unknown error.")
            return True
        app._add_system_note(
            f"Moved tool to `{new_path}`.\n\nRun `/tools reload` to apply the change.")
        return True
    if action == "open":
        if len(rest) != 1:
            app._add_system_note("Usage: `/tools open <name>`")
            return True
        located = find_tool_script(rest[0], user_dir=app.agent.config.user_dir)
        if located is None:
            app._add_system_note(
                f"No tool named `{rest[0]}` was found under `{app.agent.config.user_dir}/tools/`.")
            return True
        app._add_system_note(f"Tool script path: `{located}`")
        return True
    app._add_system_note(app._tools_usage_text())
    return True


def handle_profiles(app: PyAgentApp, args: list[str]) -> bool:
    if len(args) == 1 and args[0].lower() == "reload":
        try:
            app.agent.reload_profiles()
        except ValueError as exc:
            app._add_system_note(f"Could not reload profiles: `{exc}`")
            return True
        app._add_system_note(
            f"Reloaded profiles from `{app.agent.profile_store.path}`. Active profile: `{app.agent.current_profile().name}`.")
        app._set_status(app._ready_status())
        return True
    current = app.agent.current_profile().name
    lines = []
    default_name = app.agent.profile_store.default_profile
    for name in app.agent.profile_names():
        profile = app.agent.profile_store.get(name)
        markers = []
        if name == current:
            markers.append("current")
        if name == default_name:
            markers.append("default")
        marker_text = f" ({', '.join(markers)})" if markers else ""
        auth = f"api_key_env={profile.api_key_env}" if profile.api_key_env else (
            "inline api key" if profile.api_key else "no api key")
        lines.append(
            f"- `{name}`{marker_text} — `{profile.provider}` • `{profile.model}` • `{profile.base_url}` • {auth}")
    app._add_system_note("Saved profiles:\n" + ("\n".join(lines) if lines else "<no profiles>") +
                         f"\n\nProfile file: `{app.agent.profile_store.path}`")
    return True


def handle_profile(app: PyAgentApp, args: list[str]) -> bool:
    if not args:
        profile = app.agent.current_profile()
        app._add_system_note(
            f"Current profile:\n- Name: `{profile.name}`\n- Provider: `{profile.provider}`\n- Model: `{profile.model}`\n- Base URL: `{profile.base_url}`\n- API key env: `{profile.api_key_env or '<none>'}`")
        return True
    if args[0].lower() == "add":
        options, error = _parse_profile_add_options(app, args[1:])
        if error:
            app._add_system_note(error)
            return True
        provider = str(options.get("provider", "")).strip()
        profile_name = str(options.get("name", "")).strip()
        model = str(options.get("model", "")).strip()
        headers = options.get("headers") or {}
        make_default = bool(options.get("default", False))
        switch_to = bool(options.get("switch", False))
        try:
            base_url = str(options.get("base_url")
                           or default_base_url_for_provider(provider)).strip()
            profile = ModelProfile(
                name=profile_name, provider=provider, model=model, base_url=base_url,
                api_key=str(options.get("api_key", "")).strip() or None,
                api_key_env=str(options.get(
                    "api_key_env", "")).strip() or None,
                headers={str(k): str(v) for k, v in headers.items()},
            )
            app.agent.save_profile(profile, make_default=make_default)
            if switch_to:
                app.agent.set_profile(profile.name)
        except ValueError as exc:
            app._add_system_note(f"Could not save profile: `{exc}`")
            return True
        details = [f"Saved profile `{profile.name}`.", f"Provider: `{profile.provider}`", f"Model: `{profile.model}`",
                   f"Base URL: `{profile.base_url}`", f"Profile file: `{app.agent.profile_store.path}`"]
        if make_default:
            details.append("Set as default profile.")
        if switch_to:
            details.append("Switched to the new profile.")
        app._add_system_note("\n".join(details))
        app._set_status(app._ready_status())
        return True
    profile_name = " ".join(args).strip()
    old_profile = app.agent.current_profile().name
    try:
        app.agent.set_profile(profile_name)
    except ValueError as exc:
        app._add_system_note(str(exc))
        return True
    app._add_system_note(
        f"Switched profile from `{old_profile}` to `{app.agent.current_profile().name}`. Conversation preserved.")
    app._set_status(app._ready_status())
    return True


def handle_model(app: PyAgentApp, args: list[str]) -> bool:
    profile = app.agent.current_profile()
    if not args:
        app._add_system_note(
            f"Current model:\n- Profile: `{profile.name}`\n- Provider: `{profile.provider}`\n- Model: `{profile.model}`\nUsage:\n- `/model list` — list models from the current endpoint\n- `/model <name>` — override the active profile's model")
        return True
    if len(args) == 1 and args[0].lower() == "list":
        model_names, error = app.agent.available_models()
        if error:
            app._add_system_note(f"Could not list models: `{error}`")
            return True
        if not model_names:
            app._add_system_note(
                "This endpoint did not report any available models.")
            return True
        app._add_system_note(
            f"Available models for `{profile.name}`:\n" + "\n".join(f"- `{name}`" for name in model_names))
        return True
    new_model = " ".join(args).strip()
    if not new_model:
        app._add_system_note("Usage: `/model list` or `/model <name>`")
        return True
    old_model = profile.model
    if new_model == old_model:
        app._add_system_note(f"Already using model `{new_model}`.")
        return True
    app.agent.set_model(new_model)
    app._add_system_note(
        f"Switched model from `{old_model}` to `{new_model}` within profile `{profile.name}`.")
    app._set_status(app._ready_status())
    return True


def handle_max_iterations(app: PyAgentApp, args: list[str]) -> bool:
    if len(args) != 1:
        app._add_system_note("Usage: `/max_iterations <positive integer|-1>`")
        return True
    max_iterations, error = _parse_max_iterations(args[0])
    if error:
        app._add_system_note(error)
        return True
    app.agent.config.max_iterations = max_iterations
    app._add_system_note(
        f"Set agent tool-loop max iterations to {app._max_iterations_text()} for this session.")
    app._set_status(app._ready_status())
    return True


def handle_status(app: PyAgentApp, args: list[str]) -> bool:
    profile = app.agent.current_profile()
    app._add_system_note(
        f"Current status:\n- {app._status_summary()}\n- Base URL: `{profile.base_url}`\n- Profile file: `{app.agent.profile_store.path}`\n"
        f"- Agent tool-loop max iterations: {app._max_iterations_text()}\n- Bash enabled: `{app.agent.config.bash_enabled}`\n"
        f"- Bash read-only: `{app.agent.config.bash_readonly_mode}`\n- Project instruction files loaded: `{len(app.agent.project_context_files)}`\n"
        f"- Working directory: `{os.getcwd()}`"
    )
    return True


def handle_cwd(app: PyAgentApp, args: list[str]) -> bool:
    app._add_system_note(f"Current working directory: `{os.getcwd()}`")
    return True


def handle_history(app: PyAgentApp, args: list[str]) -> bool:
    if args and args[0].lower() == "search":
        query = " ".join(args[1:]).strip()
        if not query:
            app._add_system_note("Usage: `/history search <text>`")
            return True
        matches = [entry for entry in app.input_history if query.lower()
                   in entry.lower()]
        if not matches:
            app._add_system_note(
                f"No prompt history entries matched `{query}`.")
            return True
        history_lines = "\n".join(
            f"- {_truncate(entry.replace(chr(10), ' ⏎ '), 160)}" for entry in matches[-10:])
        app._add_system_note(
            f"Prompt history matches for `{query}`:\n{history_lines}")
        return True
    if not app.input_history:
        app._add_system_note("Prompt history is empty.")
    else:
        history_lines = "\n".join(
            f"{i + 1}. {_truncate(entry.replace(chr(10), ' ⏎ '), 120)}" for i, entry in enumerate(app.input_history[-10:]))
        app._add_system_note(
            f"Recent prompts:\n{history_lines}\n\nTip: use `/history search <text>` to find an older prompt.")
    return True


def handle_prompt(app: PyAgentApp, args: list[str]) -> bool:
    system_prompt = app.agent.messages[0]["content"] if app.agent.messages else "<missing>"
    app._add_system_note(
        f"Active system prompt:\n\n```text\n{system_prompt}\n```")
    return True


def handle_context(app: PyAgentApp, args: list[str]) -> bool:
    app._add_system_note(app._context_status_text())
    return True


def handle_reload_profiles(app: PyAgentApp, args: list[str]) -> bool:
    try:
        app.agent.reload_profiles()
    except ValueError as exc:
        app._add_system_note(f"Could not reload profiles: `{exc}`")
        return True
    app._add_system_note(
        f"Reloaded profiles from `{app.agent.profile_store.path}`. Active profile: `{app.agent.current_profile().name}`.")
    app._set_status(app._ready_status())
    return True


def handle_reload_context(app: PyAgentApp, args: list[str]) -> bool:
    previous_files = set(app.agent.project_context_files)
    app.project_context, app.context_sources = load_full_context(os.getcwd())
    app.project_context_files = [
        source.label for source in app.context_sources]
    app.agent.set_project_context(
        app.project_context, app.project_context_files)
    current_files = set(app.project_context_files)
    added = sorted(current_files - previous_files)
    removed = sorted(previous_files - current_files)
    details = [app._context_status_text()]
    if added:
        details.append("Added:\n" + "\n".join(f"- `{path}`" for path in added))
    if removed:
        details.append(
            "Removed:\n" + "\n".join(f"- `{path}`" for path in removed))
    if not app.project_context_files:
        details = [
            "Reloaded project instructions. No user-global or project instruction files were found."]
    app._add_system_note("\n\n".join(details))
    app._set_status(app._ready_status())
    return True


def handle_debug(app: PyAgentApp, args: list[str]) -> bool:
    if not args:
        status = "on" if app.debug_visible else "off"
        app._add_system_note(f"Debug pane is currently `{status}`.")
        return True
    arg = args[0].lower()
    if arg in {"on", "off"}:
        app.debug_visible = arg == "on"
        app._debug_log_widget().display = app.debug_visible
        app._add_system_note(
            f"Debug pane {'enabled' if app.debug_visible else 'disabled'}.")
        return True
    app._add_system_note("Usage: `/debug on` or `/debug off`")
    return True

# --- Registry ---


COMMAND_REGISTRY = {
    "/help": handle_help,
    "/tools": handle_tools,
    "/reload_tools": lambda app, args: handle_tools(app, ["reload"]),
    "/profiles": handle_profiles,
    "/profile": handle_profile,
    "/model": handle_model,
    "/max_iterations": handle_max_iterations,
    "/status": handle_status,
    "/cwd": handle_cwd,
    "/history": handle_history,
    "/prompt": handle_prompt,
    "/context": handle_context,
    "/reload_profiles": handle_reload_profiles,
    "/reload_context": handle_reload_context,
    "/debug": handle_debug,
}
