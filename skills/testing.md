# Testing skill

Use this skill when changing logic in the agent loop, tools, config, model profiles, provider clients, project-context loading, or user-facing behavior.

## Baseline expectation

For non-trivial changes, run:

```bash
python -m py_compile pyagent/*.py test_agent.py
python -m unittest -v
```

## What to test

### Provider/profile changes

Add or update tests for:

- profile file loading
- provider selection
- API key env handling where relevant
- model listing behavior
- streamed content/tool-call normalization

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
