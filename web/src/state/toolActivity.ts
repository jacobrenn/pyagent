import type { StreamEvent, TranscriptEntry } from "../types";

export function addToolCall(
  entries: TranscriptEntry[],
  event: StreamEvent,
  createEntry: (extra: Partial<TranscriptEntry>) => TranscriptEntry,
): TranscriptEntry[] {
  return [
    ...entries,
    createEntry({
      toolCallId: event.tool_call_id,
      toolName: event.name,
      arguments: event.arguments,
      toolStatus: "running",
    }),
  ];
}

export function correlateToolResult(
  entries: TranscriptEntry[],
  event: StreamEvent,
  createEntry: (extra: Partial<TranscriptEntry>) => TranscriptEntry,
): TranscriptEntry[] {
  let matched = false;
  const updated = entries.map((item) => {
    if (item.kind === "tool" && item.toolCallId === event.tool_call_id) {
      matched = true;
      return {
        ...item,
        result: event.result ?? "",
        toolStatus: event.is_error ? "error" : "complete",
      } as TranscriptEntry;
    }
    return item;
  });
  if (!matched) {
    updated.push(
      createEntry({
        toolCallId: event.tool_call_id,
        toolName: event.name,
        result: event.result ?? "",
        toolStatus: event.is_error ? "error" : "complete",
      }),
    );
  }
  return updated;
}
