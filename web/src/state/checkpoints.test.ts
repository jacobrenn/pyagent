import { describe, expect, it } from "vitest";
import type { Conversation, SavedRunDetail, TranscriptEntry } from "../types";
import { checkpointRequest, completedCheckpoint, forkCheckpointLocally, importCheckpoint } from "./checkpoints";

export const sourceId = "11111111-1111-4111-8111-111111111111";
export const childId = "22222222-2222-4222-8222-222222222222";
const base: Conversation = {
  id: "new-local-id", title: "New conversation", createdAt: "now", updatedAt: "now",
  profile: "current", model: "current-model", cwd: ".", skills: [], messages: [], entries: [],
};
const checkpoint: SavedRunDetail = {
  run_id: sourceId, parent_run_id: null, conversation_id: sourceId,
  timestamp: "2026-01-01T00:00:00Z", prompt_preview: "source task", outcome: "completed",
  resumable: true, reason: "Completed checkpoint.", profile: "saved", model: "saved-model",
  workspace: "/saved/workspace", agent: "reviewer", agent_revision: 1, warnings: [], skills: ["guide"],
  messages: [
    { role: "system", content: "OLD SYSTEM" },
    { role: "user", content: "source task" },
    { role: "assistant", content: "", tool_calls: [{ id: "call-1", function: { name: "read", arguments: "{\"path\":\"a.py\"}" } }] },
    { role: "tool", name: "read", content: "source contents", tool_call_id: "call-1" },
    { role: "assistant", content: "done" },
  ],
};
let sequence = 0;
const entry = (kind: TranscriptEntry["kind"], content: string, extra = {}): TranscriptEntry => ({
  id: String(++sequence), kind, content, createdAt: "now", ...extra,
});

describe("checkpoint conversation state", () => {
  it("imports canonical history into a fresh local conversation without saved system instructions", () => {
    const result = importCheckpoint(base, checkpoint, entry);
    expect(base.entries).toEqual([]);
    expect(result.id).toBe(base.id);
    expect(result.messages).toEqual(checkpoint.messages.slice(1));
    expect(result.entries.some((item) => item.content.includes("OLD SYSTEM"))).toBe(false);
    const tool = result.entries.find((item) => item.kind === "tool")!;
    expect(tool.toolStatus).toBe("complete");
    expect(tool.result).toBe("source contents");
    expect(tool.toolCallId).toBe("call-1");
    expect(result.agentRevision).toBe(1);
    expect(result.resumeFromRunId).toBe(sourceId);
    expect(checkpointRequest(result, "next")).toEqual({ message: "next", resume_from_run_id: sourceId });
    expect(() => importCheckpoint(base, { ...checkpoint, resumable: false }, entry)).toThrow();
  });

  it("requires positive persistence confirmation rather than assuming an ID denotes a saved log", () => {
    const imported = importCheckpoint(base, checkpoint, entry);
    expect(completedCheckpoint(imported, { type: "done", run_id: childId, log_saved: true }).resumeFromRunId).toBe(childId);
    for (const log_saved of [undefined, false]) {
      const result = completedCheckpoint(imported, { type: "done", run_id: childId, log_saved });
      expect(result.resumeFromRunId).toBeUndefined();
      expect(result.checkpointMode).toBe(true);
      expect(() => checkpointRequest(result, "next")).toThrow("No saved continuation");
    }
    expect(completedCheckpoint(base, { type: "done", run_id: childId, log_saved: true })).toBe(base);
  });

  it("forks explicitly with current ad hoc settings and warns about unknown side effects", () => {
    const imported = importCheckpoint(base, checkpoint, entry);
    const fork = forkCheckpointLocally({ ...base, id: "fork-id" }, imported, entry);
    expect(fork.agent).toBeUndefined();
    expect(fork.resumeFromRunId).toBeUndefined();
    expect(fork.checkpointMode).toBeUndefined();
    expect(fork.profile).toBe("current");
    expect(fork.cwd).toBe("/saved/workspace");
    expect(fork.messages.slice(0, -1)).toEqual(imported.messages);
    expect(fork.messages.at(-1)?.content).toContain("Interrupted tools");
    expect(checkpointRequest(fork, "next")).toBeNull();
    expect(imported.resumeFromRunId).toBe(sourceId);
  });
});
