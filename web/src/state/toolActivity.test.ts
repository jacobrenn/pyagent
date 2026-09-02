import { describe, expect, it } from "vitest";
import { addToolCall, correlateToolResult } from "./toolActivity";
import type { StreamEvent, TranscriptEntry } from "../types";

function makeTool(extra: Partial<TranscriptEntry>): TranscriptEntry {
  return {
    id: "tool-entry",
    kind: "tool",
    content: "",
    createdAt: "2025-01-01T00:00:00Z",
    ...extra,
  };
}

describe("tool activity correlation", () => {
  it("correlates results by tool call ID rather than order", () => {
    const first = addToolCall([], {
      type: "tool_call",
      tool_call_id: "call-a",
      name: "read",
      arguments: { path: "a.py" },
    }, makeTool);
    const both = addToolCall(first, {
      type: "tool_call",
      tool_call_id: "call-b",
      name: "read",
      arguments: { path: "b.py" },
    }, (extra) => ({ ...makeTool(extra), id: "tool-entry-b" }));

    const updated = correlateToolResult(both, {
      type: "tool_result",
      tool_call_id: "call-a",
      result: "contents-a",
      is_error: false,
    }, makeTool);

    expect(updated[0].result).toBe("contents-a");
    expect(updated[0].toolStatus).toBe("complete");
    expect(updated[1].toolStatus).toBe("running");
  });

  it("retains an orphaned error result as visible activity", () => {
    const updated = correlateToolResult([], {
      type: "tool_result",
      tool_call_id: "missing",
      name: "bash",
      result: "denied",
      is_error: true,
    } as StreamEvent, makeTool);

    expect(updated).toHaveLength(1);
    expect(updated[0]).toMatchObject({
      toolCallId: "missing",
      toolName: "bash",
      result: "denied",
      toolStatus: "error",
    });
  });
});
