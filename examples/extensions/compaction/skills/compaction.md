# Compaction Extension Skill

You have access to a `compact_context` tool. Use it proactively when the
conversation is long and early detail is at risk of falling out of view.

## When to use it

- The conversation has many turns or large tool outputs, and
- earlier context (goals, constraints, decisions) is no longer clearly in
  your working window.

## How to use it

1. Call `compact_context` with the `text` argument set to the older
   conversation content you want condensed (paste the relevant prior
   messages — user requests, key decisions, tool results worth keeping).
2. The tool returns a condensed summary. Treat that summary as authoritative
   for the condensed span and continue from it.
3. Prefer keeping the most recent few exchanges verbatim; only condense the
   older middle of the conversation.

## Notes

- Compaction is lossy by design — only condense what you no longer need
  verbatim.
- If a specific earlier fact is critical, restate it explicitly after
  compacting rather than relying on the summary to preserve it.
