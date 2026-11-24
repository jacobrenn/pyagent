"""Built-in context compaction.

The agent keeps a bounded working set in ``agent.messages``. When the
estimated size of that history crosses a soft budget, older context is
compacted in place using a three-stage pipeline (cheapest first):

1. **Prune** — tool outputs older than the keep-recent window are replaced
   with a short stub. This reclaims the bulk of context cheaply without an
   LLM call. Tool-call structure is never touched.
2. **Summarize** — if history is still over the soft budget, everything
   except the pinned prefix (system message + first user message) and the
   keep-recent window is summarized by the active model into a single
   synthetic user message. Prior summaries are folded in, so repeated
   compactions degrade gracefully instead of amnesia.
3. **Backstop** — if history is still over the hard budget (summarizer
   failed or was unavailable), the oldest non-pinned blocks are dropped and an
   explicit marker message is left behind. Nothing is ever dropped silently.

History is rewritten in place; the run log and checkpoints see the compacted
form, so resume keeps working unchanged. The provider-reported prompt token
count (when available) calibrates the local char/token estimate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD_TOKENS = 4

# Marker prefixing the synthetic summary message. Also used to detect a prior
# summary so repeated compactions can fold it in instead of duplicating it.
COMPACTION_MARKER = "[Compacted context — earlier turns were summarized by PyAgent. Recent messages take precedence.]"
OVERFLOW_MARKER = "[Context overflow — {count} older message block(s) were dropped because the model's context window was exceeded and summarization was unavailable.]"
TOOL_PRUNE_PREFIX = "[tool output pruned"

PRUNE_MIN_CHARS = 500
PRUNE_KEEP_HEAD_CHARS = 200
MESSAGE_SNIPPET_CHARS = 1_500
TOOL_CALL_SNIPPET_CHARS = 500
SOURCE_CHUNK_CHARS = 48_000

# Fallback budgets when the model's context window is unknown.
UNKNOWN_WINDOW_SOFT_TOKENS = 24_000
UNKNOWN_WINDOW_HARD_TOKENS = 32_000
UNKNOWN_WINDOW_KEEP_TOKENS = 8_000
SOFT_RATIO = 0.65
HARD_RATIO = 0.85
KEEP_RATIO = 0.25

SummarizeFn = Callable[[list[dict[str, Any]], str | None], str]


@dataclass(frozen=True, slots=True)
class CompactionLimits:
    enabled: bool
    soft_tokens: int
    hard_tokens: int
    keep_recent_tokens: int
    max_summary_tokens: int
    model: str | None


def resolve_limits(config: Any) -> CompactionLimits:
    """Resolve effective limits from AppConfig (context window may be unknown)."""
    window = int(getattr(config, "context_window", 0) or 0)

    def _derive(explicit: int, ratio: float, unknown: int) -> int:
        if explicit and explicit > 0:
            return int(explicit)
        if window > 0:
            return max(1, int(window * ratio))
        return unknown

    soft = _derive(int(getattr(config, "compaction_soft_tokens", 0) or 0), SOFT_RATIO, UNKNOWN_WINDOW_SOFT_TOKENS)
    hard = _derive(int(getattr(config, "compaction_hard_tokens", 0) or 0), HARD_RATIO, UNKNOWN_WINDOW_HARD_TOKENS)
    keep = _derive(int(getattr(config, "compaction_keep_recent_tokens", 0) or 0), KEEP_RATIO, UNKNOWN_WINDOW_KEEP_TOKENS)
    hard = max(hard, soft + 1)
    model = (getattr(config, "compaction_model", "") or "").strip() or None
    return CompactionLimits(
        enabled=bool(getattr(config, "compaction_enabled", False)),
        soft_tokens=soft,
        hard_tokens=hard,
        keep_recent_tokens=keep,
        max_summary_tokens=int(getattr(config, "compaction_max_summary_tokens", 1024) or 1024),
        model=model,
    )


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """Rough local token estimate for one chat message (uncalibrated)."""
    chars = 0
    content = message.get("content")
    if isinstance(content, str):
        chars += len(content)
    elif content is not None:
        chars += len(json.dumps(content, default=str))
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        chars += len(str(function.get("name") or ""))
        chars += min(len(str(function.get("arguments") or "")), 10_000) + 16
    if message.get("name"):
        chars += len(str(message["name"]))
    return MESSAGE_OVERHEAD_TOKENS + max(0, chars // CHARS_PER_TOKEN)


def estimate_messages_tokens(messages: Iterable[dict[str, Any]], calibration: float = 1.0) -> int:
    total = sum(estimate_message_tokens(m) for m in messages)
    try:
        factor = float(calibration)
    except (TypeError, ValueError):
        factor = 1.0
    if not 0.25 <= factor <= 4.0:
        factor = 1.0
    return max(1, int(total * factor))


def _blocks(messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Group messages into atomic blocks (assistant tool_calls + following tool results)."""
    blocks: list[tuple[int, int]] = []
    index, n = 0, len(messages)
    while index < n:
        message = messages[index]
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            end = index + 1
            while end < n and messages[end].get("role") == "tool":
                end += 1
            blocks.append((index, end))
            index = end
            continue
        end = index + 1
        if role == "tool":
            while end < n and messages[end].get("role") == "tool":
                end += 1
        blocks.append((index, end))
        index = end
    return blocks


def _pinned_prefix_end(messages: list[dict[str, Any]]) -> int:
    """Index after the pinned prefix: system message + first user message."""
    pinned = 0
    if messages and messages[0].get("role") == "system":
        pinned = 1
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            pinned = max(pinned, index + 1)
            break
    return pinned


def _keep_recent_indexes(
    messages: list[dict[str, Any]],
    blocks: list[tuple[int, int]],
    keep_tokens: int,
) -> set[int]:
    keep: set[int] = set()
    used = 0
    for start, end in reversed(blocks):
        block_tokens = sum(estimate_message_tokens(messages[i]) for i in range(start, end))
        if keep and used + block_tokens > keep_tokens:
            break
        keep.update(range(start, end))
        used += block_tokens
    return keep


def _is_compaction_marker(message: dict[str, Any]) -> bool:
    return (
        message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and message["content"].startswith(COMPACTION_MARKER)
    )


def _serialize_messages(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "unknown")
        name = message.get("name")
        header = f"{role} ({name})" if name else role
        body = str(message.get("content") or "")
        if len(body) > MESSAGE_SNIPPET_CHARS:
            omitted = len(body) - MESSAGE_SNIPPET_CHARS
            body = f"{body[:MESSAGE_SNIPPET_CHARS]}\n…[{omitted} chars omitted]"
        chunk = [f"### {header}\n{body}"]
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            arguments = str(function.get("arguments") or "")
            if len(arguments) > TOOL_CALL_SNIPPET_CHARS:
                arguments = arguments[:TOOL_CALL_SNIPPET_CHARS] + "…"
            chunk.append(f"[tool call] {function.get('name') or '?'}({arguments})")
        parts.append("\n".join(chunk))
    return "\n\n".join(parts)


def _chunk_text(text: str, budget: int) -> list[str]:
    if len(text) <= budget:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in text.split("\n\n"):
        if current and current_len + len(paragraph) + 2 > budget:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(paragraph)
        current_len += len(paragraph) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


_SUMMARY_SYSTEM = (
    "You are PyAgent's context compaction module. You condense the earlier part of "
    "an agent conversation into a dense, factual brief that will replace the original. "
    "Rules:\n"
    "- Preserve exact identifiers verbatim: file paths, symbols, IDs, commands, URLs, "
    "error messages, and user-stated constraints.\n"
    "- Never invent facts. Mark anything uncertain as unknown.\n"
    "- Omit pleasantries, routine tool chatter, and reasoning that led nowhere.\n"
    "- Recent turns not included in the input take precedence over this brief.\n"
    "Use these sections:\n"
    "## Goal\n## Constraints and preferences\n## Decisions and rationale\n"
    "## Files, tools, and artifacts\n## Findings and data worth keeping\n## Open items"
)

_MERGE_INSTRUCTION = (
    "Merge the partial briefs below into a single brief following the same sections. "
    "Do not add new information; deduplicate and preserve exact identifiers."
)


def summarize_messages(
    middle: list[dict[str, Any]],
    previous_summary: str | None,
    complete: Callable[[str, str], str],
) -> str:
    """Summarize ``middle`` via ``complete(system, user) -> str``.

    Handles oversized transcripts by chunking and merging. Raises on failure;
    the caller is responsible for fallback handling.
    """
    transcript = _serialize_messages(middle)
    chunks = _chunk_text(transcript, SOURCE_CHUNK_CHARS)
    partials: list[str] = []
    for index, chunk in enumerate(chunks):
        header = f"Transcript segment {index + 1} of {len(chunks)}:\n\n" if len(chunks) > 1 else "Transcript:\n\n"
        prompt_parts = []
        if previous_summary:
            prompt_parts.append(
                "Previous compacted brief (update it; fold in the new information):\n"
                f"{previous_summary}\n"
            )
        prompt_parts.append(header + chunk)
        partials.append(complete(_SUMMARY_SYSTEM, "\n".join(prompt_parts)).strip())
    if len(partials) == 1:
        return partials[0]
    merged = "\n\n---\n\n".join(
        f"Partial brief {index + 1}:\n{part}" for index, part in enumerate(partials)
    )
    prefix = f"Previous compacted brief:\n{previous_summary}\n\n" if previous_summary else ""
    return complete(_SUMMARY_SYSTEM, f"{prefix}{_MERGE_INSTRUCTION}\n\n{merged}").strip()


@dataclass
class CompactionReport:
    estimated_tokens_before: int = 0
    estimated_tokens_after: int = 0
    pruned_tool_outputs: int = 0
    pruned_chars: int = 0
    summarized: bool = False
    summarized_messages: int = 0
    dropped_blocks: int = 0
    failure: str | None = None

    @property
    def changed(self) -> bool:
        return bool(self.pruned_tool_outputs or self.summarized or self.dropped_blocks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimated_tokens_before": self.estimated_tokens_before,
            "estimated_tokens_after": self.estimated_tokens_after,
            "pruned_tool_outputs": self.pruned_tool_outputs,
            "pruned_chars": self.pruned_chars,
            "summarized": self.summarized,
            "summarized_messages": self.summarized_messages,
            "dropped_blocks": self.dropped_blocks,
            "failure": self.failure,
        }


def _prune_old_tool_outputs(
    messages: list[dict[str, Any]],
    keep: set[int],
    pinned_end: int,
) -> tuple[int, int]:
    pruned = 0
    pruned_chars = 0
    for index, message in enumerate(messages):
        if index < pinned_end or index in keep:
            continue
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if len(content) < PRUNE_MIN_CHARS or content.startswith(TOOL_PRUNE_PREFIX):
            continue
        tool = message.get("name") or message.get("tool_call_id") or "tool"
        stub = (
            f"{TOOL_PRUNE_PREFIX}; {len(content)} chars total, tool={tool}; "
            f"showing first {PRUNE_KEEP_HEAD_CHARS} chars]\n"
            f"{content[:PRUNE_KEEP_HEAD_CHARS]}"
        )
        pruned_chars += len(content) - len(stub)
        message["content"] = stub
        pruned += 1
    return pruned, pruned_chars


def manage_context(
    messages: list[dict[str, Any]],
    *,
    limits: CompactionLimits,
    calibration: float = 1.0,
    complete: Callable[[str, str], str] | None = None,
    force: bool = False,
) -> tuple[list[dict[str, Any]], CompactionReport]:
    """Compact ``messages`` if needed. Returns (new_messages, report).

    Never raises for expected failures; summarization errors land in
    ``report.failure`` and the backstop guard kicks in instead.
    """
    report = CompactionReport()
    history = [dict(message) for message in messages]
    blocks = _blocks(history)
    report.estimated_tokens_before = estimate_messages_tokens(history, calibration)

    pin_end = _pinned_prefix_end(history)
    keep = _keep_recent_indexes(history, blocks, limits.keep_recent_tokens)
    pin_set = set(range(pin_end))
    keep |= pin_set

    # Stage 1: prune old tool outputs in place.
    report.pruned_tool_outputs, report.pruned_chars = _prune_old_tool_outputs(history, keep, pin_end)

    current_tokens = estimate_messages_tokens(history, calibration)

    if not force and current_tokens <= limits.soft_tokens:
        report.estimated_tokens_after = current_tokens
        return history, report

    # Stage 2: summarize the fold between the pinned prefix and the recent window.
    middle_indices = [i for i in range(pin_end, len(history)) if i not in keep]
    if middle_indices:
        middle = [history[i] for i in middle_indices]
        previous_summary = None
        remaining = []
        for message in middle:
            if _is_compaction_marker(message):
                if previous_summary is None:
                    previous_summary = str(message.get("content"))
                else:
                    previous_summary += f"\n\n{message['content']}"
            else:
                remaining.append(message)
        if complete is not None and (remaining or previous_summary):
            try:
                summary = summarize_messages(remaining, previous_summary, complete)
            except Exception as exc:  # noqa: BLE001 - compaction must never break a run
                report.failure = f"summarization failed: {exc}"
                summary = None
            if summary:
                summary_message = {
                    "role": "user",
                    "content": f"{COMPACTION_MARKER}\n\n{summary.strip()}",
                }
                middle_set = set(middle_indices)
                history = (
                    history[:pin_end]
                    + [summary_message]
                    + [m for i, m in enumerate(history) if i >= pin_end and i not in middle_set]
                )
                report.summarized = True
                report.summarized_messages = len(middle)
                current_tokens = estimate_messages_tokens(history, calibration)

    # Stage 3: hard backstop. Drop oldest non-pinned blocks until under budget.
    overflow_dropped = 0
    current_tokens = estimate_messages_tokens(history, calibration)
    while current_tokens > limits.hard_tokens:
        blocks = _blocks(history)
        target_block = None
        pin_end = _pinned_prefix_end(history)
        for start, end in blocks:
            if end <= pin_end:
                continue
            if any(_is_compaction_marker(history[i]) for i in range(start, end)):
                # Never drop the summary; drop the next block instead.
                continue
            target_block = (start, end)
            break
        if target_block is None:
            break
        start, end = target_block
        del history[start:end]
        overflow_dropped += 1
        current_tokens = estimate_messages_tokens(history, calibration)
    if overflow_dropped:
        pin_end = _pinned_prefix_end(history)
        history.insert(pin_end, {
            "role": "user",
            "content": OVERFLOW_MARKER.format(count=overflow_dropped),
        })
        report.dropped_blocks = overflow_dropped
        current_tokens = estimate_messages_tokens(history, calibration)

    report.estimated_tokens_after = current_tokens
    return history, report


def calibrate_estimate(current: float, prompt_tokens: int, raw_estimate: int) -> float:
    """Blend the local estimate with the provider-reported prompt size.

    Uses an exponential moving average anchored to the latest reading so the
    estimator tracks each model's true tokenizer instead of a fixed ratio.
    """
    if prompt_tokens <= 0 or raw_estimate <= 0:
        return current if current > 0 else 1.0
    ratio = min(2.0, max(0.5, prompt_tokens / raw_estimate))
    return 0.5 * (current if current > 0 else 1.0) + 0.5 * ratio
