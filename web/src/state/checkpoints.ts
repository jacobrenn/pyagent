import type { Conversation, RunRequest, SavedRunDetail, StreamEvent, TranscriptEntry } from "../types";
import { addToolCall, correlateToolResult } from "./toolActivity";

type EntryFactory = (
  kind: TranscriptEntry["kind"], content: string, extra?: Partial<TranscriptEntry>,
) => TranscriptEntry;

export function importCheckpoint(
  base: Conversation, checkpoint: SavedRunDetail, makeEntry: EntryFactory,
): Conversation {
  if (!checkpoint.resumable) throw new Error(checkpoint.reason);
  let entries: TranscriptEntry[] = [makeEntry("system",
    `Imported checkpoint ${checkpoint.run_id}. Current instructions and resources will be revalidated when you send a prompt. Historical tools are not replayed.`)];
  const messages = checkpoint.messages.filter((message) => message.role !== "system");
  for (const message of messages) {
    if (message.role === "tool") {
      entries = correlateToolResult(entries, {
        type: "tool_result", tool_call_id: message.tool_call_id,
        name: message.name ?? message.tool_name, result: message.content,
      }, (extra) => makeEntry("tool", "", extra));
      continue;
    }
    if (message.content) entries.push(makeEntry(message.role, message.content));
    for (const raw of message.tool_calls ?? []) {
      const call = raw as { id: string; function: { name: string; arguments?: unknown } };
      let args = call.function.arguments;
      if (typeof args === "string") {
        try { args = JSON.parse(args); } catch { args = { raw: args }; }
      }
      entries = addToolCall(entries, {
        type: "tool_call", tool_call_id: call.id, name: call.function.name,
        arguments: (args ?? {}) as Record<string, unknown>,
      }, (extra) => makeEntry("tool", "", extra));
    }
  }
  return {
    ...base,
    title: `Resumed · ${checkpoint.prompt_preview.slice(0, 44) || checkpoint.run_id.slice(0, 8)}`,
    profile: checkpoint.profile ?? "", model: checkpoint.model ?? "",
    cwd: checkpoint.workspace ?? ".", skills: checkpoint.skills,
    agent: checkpoint.agent ?? undefined, agentRevision: checkpoint.agent_revision ?? undefined,
    importedFromRunId: checkpoint.run_id, checkpointMode: true,
    resumeFromRunId: checkpoint.run_id, messages, entries,
  };
}

export function checkpointRequest(conversation: Conversation, message: string): RunRequest | null {
  if (!conversation.checkpointMode) return null;
  if (!conversation.resumeFromRunId) {
    throw new Error("No saved continuation is available. Review the transcript and explicitly fork locally.");
  }
  // Do not send browser history, skills, or runtime overrides alongside a checkpoint.
  return { message, resume_from_run_id: conversation.resumeFromRunId };
}

export function completedCheckpoint(conversation: Conversation, event: StreamEvent): Conversation {
  if (!conversation.checkpointMode) return conversation;
  return {
    ...conversation,
    resumeFromRunId: event.log_saved === true ? event.run_id : undefined,
  };
}

export function forkCheckpointLocally(
  base: Conversation, source: Conversation, makeEntry: EntryFactory,
): Conversation {
  const notice = "Explicit local fork: using the last confirmed model history and current ad hoc settings, without saved checkpoint restrictions or server ancestry. Interrupted tools may already have changed the workspace; inspect before repeating actions.";
  return {
    ...base,
    title: `Local fork · ${source.title.slice(0, 40)}`,
    cwd: source.cwd,
    importedFromRunId: source.importedFromRunId,
    messages: [...source.messages, { role: "user", content: notice }],
    entries: [...source.entries, makeEntry("system", notice)],
  };
}
