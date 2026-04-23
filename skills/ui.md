# UI skill

Use this skill when changing the Textual interface in `ui.py`.

## Goals

- Keep the interface compact and readable.
- Preserve the current keyboard-driven workflow.
- Keep transcript rendering stable during streaming.
- Prefer incremental UI improvements over large rewrites.

## Important current behaviors

- Prompt input is a `TextArea`-based multiline widget.
- `Enter` sends.
- `Shift+Enter` inserts a newline.
- `Ctrl+P` / `Ctrl+N` navigate prompt history.
- `↑` / `↓` and paging keys scroll the transcript.
- Tool messages are intentionally compact.
- Profile/model switching is exposed through slash commands.
- Auto-follow logic is sensitive to Textual layout timing.

## Be careful about

### Textual layout defaults

Textual containers often default to flexible heights.
If message widgets stop expanding correctly, transcript scrolling and auto-follow can break.
When editing transcript/message layout, preserve the current `height: auto` behavior for message content.

### Transcript scrolling

If you modify scroll logic:

- test while the assistant is actively streaming
- test after tool calls
- test manual scroll-up followed by returning to bottom

### Prompt box behavior

If you change the prompt input:

- preserve multiline editing
- preserve auto-grow behavior
- preserve prompt subtitle guidance
- avoid conflicts with transcript scrolling keys

### Slash commands

If you change slash commands or status text:

- keep `/help` accurate
- keep profile/model switching discoverable
- keep profile reload / creation syntax discoverable
- keep status bar text short

## Preferred approach

- Small CSS/layout changes first
- Small helper methods in `ui.py` over deeply nested logic
- Put discoverability in the prompt subtitle and `/help`, not in large persistent banners

## When to update docs/tests

Update `README.md` if you change:

- keybindings
- prompt behavior
- slash commands
- visible UI workflow
- profile/model switching behavior

Add tests when practical, but also manually test the TUI for layout/scroll changes.
