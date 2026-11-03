# Testing skill

Use this skill when changing logic in the agent loop, tools, config, model profiles, provider clients, project-context loading, or user-facing behavior.

## Baseline expectation

For non-trivial changes, run:

```bash
python -m py_compile pyagent/*.py test_agent.py test_run_logging.py test_sessions.py
python -m unittest -v
```

## What to test

### Provider/profile changes

Add or update tests for:

- profile file loading and `api_mode` default/round-trip validation
- provider and API-mode client selection
- API key env handling where relevant
- model listing behavior
- streamed content/tool-call/error normalization
- Responses history/tool-schema adaptation and reasoning-item preservation across tool continuations

### Tool changes

Add or update tests for:

- happy-path behavior
- error handling
- config-driven behavior

Examples:

- shell safety policy
- search result formatting
- multi-edit behavior
- append semantics

### Agent-loop changes

Test cases should cover:

- tool call execution
- malformed tool calls
- fallback behavior when the model stops after a tool result
- reset behavior if system prompt composition changes
- switching profiles or models if agent state changes
- model-client and extension-session cleanup after single-shot or API runs

### Run logging changes

`test_run_logging.py` covers shared logging across TUI, CLI, named agents, and all four API run routes. Use a temporary `PYAGENT_USER_DIR`, profiles file, and system prompt; mock model clients rather than contacting providers.

Check boolean parsing and default-off behavior, TUI overrides, one UUID file per run, exclusive creation/collision retries, matching API/SSE/log IDs, full message records despite history trimming, parent IDs/reset, sanitized credential metadata, and client compatibility with older responses. Test concurrent runs, early generator closure, model/extension errors, blocked-producer disconnects, and best-effort open/write failures. A missing terminal record means incomplete; never infer successful completion from a filename. Debug capture must remain independent of saved logs.

For a manual server smoke test, start with `PYAGENT_LOGGING=on pyagent serve`, issue simultaneous streaming requests, and match each SSE `run_id` to a separate file under the temporary user directory. Repeat with `PYAGENT_LOGGING=off` and verify no new files. In the TUI, test `/logging`, repeated on/off toggles, multiple prompts, and `/clear` without changing the logging preference.

### Saved-session and resume changes

`test_sessions.py` covers the read-only session store and CLI/TUI checkpoint restoration. Use temporary user/workspace/profile directories and mock model clients. Check schema/sequence/run-ID validation, bounded parsing, duplicate IDs, symlink/path rejection, incomplete and legacy classification, complete tool blocks, and native Ollama ID normalization. A failed or cancelled run is not automatically resumable.

Check explicit parent grouping, branching, missing ancestors, cycle detection, unchanged source bytes, current system instructions/skills/credentials, named revision and resolved-workspace handling, saved/current permission intersection, and history trimming. Opening a checkpoint must not call a model, replay a tool/hook, or create a run file. Cover CLI response stdout vs diagnostics stderr, TUI confirmation/busy guards, failed/cancelled candidate cleanup, preserved logging preference, and prompt/transcript restoration. Test continuation through Ollama, Chat Completions, and Responses message formats.

Manual smoke test: enable logging, submit a multi-tool prompt, exit, inspect it with `pyagent sessions show <id>`, then `pyagent resume <id>`. Verify restored transcript/scrolling, `/context`, `/sessions`, and `/resume <id> --confirm`. Submit a new prompt and check its file has a new UUID and the selected parent. Repeat with logging off; opening a checkpoint should never make a model request.

### Agent-definition and orchestration changes

Agent definitions are versioned resources with API, CLI, client, workspace, and tool-policy surfaces. Add focused tests for every layer touched.

Definition and storage tests should cover:

- schema-version validation and rejection of unknown fields
- safe agent names and path traversal rejection
- creation of revision 1 and immutable numbered revision snapshots
- updates preserving omitted fields and creating the next revision
- reading current and pinned historical revisions
- duplicate definitions, missing definitions, invalid iteration limits, and deletion of revision history
- atomic-write failure behavior where practical

Resource-resolution tests should cover:

- default and explicitly selected profiles, including model overrides and API-mode validation
- installed prompt references and missing prompts
- scoped and unscoped user/project skill references
- inherited tools (`tools: null`), explicit allowlists, empty tool lists, disabled tools, and extension-provided tools
- relative and absolute workspaces, missing workspaces, and effective iteration limits
- validation failures occurring before a model request is made

Workspace and tool-policy tests should cover:

- relative built-in tool paths resolving from the agent workspace rather than process-global cwd
- file tools rejecting `..`, absolute, and symlink paths that escape a restricted workspace
- bash and external-tool subprocesses starting in the configured workspace
- tool allowlists surviving extension loading and external-tool registry rebuilds
- an omitted `bash` entry making Bash unavailable even when it is globally enabled
- legacy TUI and `/run` behavior remaining compatible when workspace restriction is not requested

Interface tests should cover:

- REST create/list/show/revisions/update/validate/delete/run operations and status codes
- pinned-revision execution and explicit message-history loading
- Python client endpoint construction, query encoding, response validation, and typed `AgentRunResponse`
- CLI create/list/show/revisions/update/validate/remove/run behavior and exit codes
- partial update semantics: omitted fields are preserved, repeated list flags replace lists, and `--no-tools` produces an empty allowlist

Use temporary user directories, profile files, prompts, skills, tools, and workspaces in tests. Do not let automated tests create resources under the developer's real `~/.pyagent/`. Mock model streams unless a test is explicitly designated as an opt-in live integration test.

For a manual installed-package smoke test, run from outside the repository so the checkout cannot shadow site-packages:

```bash
cd /tmp
export PYAGENT_USER_DIR="$(mktemp -d)"
pyagent agents create smoke-reader --profile <profile> --tool read_file --workspace <project>
pyagent agents validate smoke-reader
pyagent agents run smoke-reader "Read README.md and return its first heading"
pyagent agents revisions smoke-reader
pyagent agents remove smoke-reader
rm -rf "$PYAGENT_USER_DIR"
```

Use a disposable user directory and read-only tools. Verify the resolved tool list before running the agent. `PYAGENT_USER_DIR` does not relocate model profiles or the active system prompt, so set their dedicated path variables too when the smoke test must isolate those files.

### Project-context changes

Test:

- instruction file discovery
- prompt/context inclusion
- reload behavior where practical
- context/status reporting where practical
- truncation or size-limit behavior if changed

## UI testing guidance

The current automated tests are stronger for logic than for full TUI behavior.
For `pyagent/ui.py` changes, do both:

1. automated validation via `py_compile` and `unittest`
2. manual checks in the running app

Manual checks should include, when relevant:

- transcript scrolling
- auto-follow while streaming
- slash commands
- profile/model switching
- profile reload and profile creation commands
- prompt history
- `/history search` behavior
- `/context` and `/reload_context` behavior
- unknown-command suggestion behavior
- multiline prompt behavior
- debug pane toggling

## Documentation alignment

If a user-visible behavior changes, update `README.md` in the same change.
If profile or provider behavior changes, update `AGENTS.md` and any relevant skills docs too.

## Rule of thumb

If you fix a bug, add or improve a test that would catch it again.
