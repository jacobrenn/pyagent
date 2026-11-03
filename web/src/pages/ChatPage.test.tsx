// @vitest-environment jsdom
import { render } from "preact";
import { act } from "preact/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import { listConversations, saveConversation } from "../state/conversations";
import type { Conversation, ProfilesResponse, SavedRunDetail } from "../types";
import { ChatPage } from "./ChatPage";

vi.mock("../api/client", async (original) => ({
  ...await original<typeof import("../api/client")>(),
  api: {
    listSessions: vi.fn(), showSession: vi.fn(), sessionCapabilities: vi.fn(),
    streamRun: vi.fn(), streamAgentRun: vi.fn(), listSkills: vi.fn(), listProfileModels: vi.fn(),
  },
}));
vi.mock("../state/conversations", () => ({
  listConversations: vi.fn(), saveConversation: vi.fn(), deleteConversation: vi.fn(),
}));
vi.mock("../components/Transcript", () => ({
  Transcript: ({ entries }: { entries: { id: string; content: string }[] }) =>
    <div data-testid="transcript">{entries.map((item) => <p key={item.id}>{item.content}</p>)}</div>,
}));

const sourceId = "11111111-1111-4111-8111-111111111111";
const childId = "22222222-2222-4222-8222-222222222222";
const profiles: ProfilesResponse = {
  path: "/profiles.json", default_profile: "local", effective_default_profile: "local", default_overridden_by_env: false,
  profiles: [{ name: "local", provider: "ollama", api_mode: "chat_completions", model: "model",
    base_url: "http://localhost:11434", api_key_env: null, has_inline_api_key: false,
    headers: {}, redacted_headers: [], httpx_kwargs: {}, is_default: true }],
};
const existing: Conversation = {
  id: "existing", title: "Existing local chat", createdAt: "2026-01-01", updatedAt: "2026-01-01",
  profile: "local", model: "model", cwd: ".", skills: [], messages: [], entries: [],
};
const checkpoint: SavedRunDetail = {
  run_id: sourceId, parent_run_id: null, conversation_id: sourceId,
  timestamp: "2026-01-01T00:00:00Z", prompt_preview: "source task", outcome: "completed", resumable: true,
  reason: "Completed checkpoint.", profile: "local", model: "model", workspace: "/work/repo",
  agent: null, agent_revision: null, warnings: [], skills: [],
  messages: [{ role: "system", content: "saved system" }, { role: "user", content: "source task" }, { role: "assistant", content: "old answer" }],
};
let container: HTMLDivElement;

beforeEach(() => {
  vi.resetAllMocks();
  container = document.createElement("div");
  document.body.append(container);
  vi.mocked(listConversations).mockResolvedValue([existing]);
  vi.mocked(saveConversation).mockResolvedValue(undefined);
  vi.mocked(api.listSessions).mockResolvedValue({ runs: [checkpoint], total: 1, unavailable_count: 0, offset: 0, limit: 20 });
  vi.mocked(api.showSession).mockResolvedValue(checkpoint);
  vi.mocked(api.sessionCapabilities).mockResolvedValue({ enabled: true, resume_supported: true });
  vi.mocked(api.streamRun).mockImplementation(async (_payload, onEvent) => {
    onEvent({ type: "start", run_id: childId, parent_run_id: sourceId, warnings: ["Current resources apply"] });
    onEvent({ type: "done", run_id: childId, log_saved: true, response: "new answer", messages: [
      ...checkpoint.messages, { role: "user", content: "continue" }, { role: "assistant", content: "new answer" },
    ] });
  });
});

afterEach(() => {
  act(() => render(null, container));
  container.remove();
  vi.restoreAllMocks();
});

async function mount() {
  await act(async () => {
    render(<ChatPage profiles={profiles} agents={{ root: "/agents", agents: [] }} agentLaunch={null}
      onAgentLaunchHandled={() => {}} onOpenProfiles={() => {}} />, container);
  });
}

function button(text: string): HTMLButtonElement {
  const found = [...container.querySelectorAll("button")].find((item) => item.textContent?.trim() === text);
  if (!found) throw new Error(`No button ${text}: ${container.textContent}`);
  return found;
}

async function click(text: string) {
  await act(async () => { button(text).click(); });
}

async function importSaved() {
  await click("Saved checkpoints");
  await act(async () => { container.querySelector<HTMLButtonElement>(".saved-sessions__list button")!.click(); });
  await click("Import as new conversation");
}

async function submit(text = "continue") {
  await act(async () => {
    const textarea = container.querySelector("textarea")!;
    textarea.value = text;
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await act(async () => {
    container.querySelector("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
  });
}

async function flushSaves() {
  await act(async () => { await new Promise((resolve) => setTimeout(resolve, 160)); });
}

describe("browser checkpoint resumption", () => {
  it("imports into a new local conversation, without executing or replacing existing history", async () => {
    await mount();
    expect(api.listSessions).not.toHaveBeenCalled();
    await importSaved();
    expect(container.querySelectorAll(".conversation-item")).toHaveLength(2);
    expect(container.textContent).toContain("Existing local chat");
    expect(container.textContent).toContain("old answer");
    expect(container.querySelector('[data-testid="transcript"]')?.textContent).not.toContain("saved system");
    expect(api.streamRun).not.toHaveBeenCalled();
    expect(api.streamAgentRun).not.toHaveBeenCalled();
    const saved = vi.mocked(saveConversation).mock.calls.at(-1)![0];
    expect(saved.id).not.toBe(existing.id);
    expect(saved.resumeFromRunId).toBe(sourceId);
    expect(saved.checkpointMode).toBe(true);
    expect(container.querySelector<HTMLSelectElement>("#chat-runtime")!.disabled).toBe(true);
  });

  it("sends only the new prompt and checkpoint ID, then persists the explicitly saved child", async () => {
    await mount();
    await importSaved();
    await submit();
    expect(api.streamRun).toHaveBeenCalledWith(
      { message: "continue", resume_from_run_id: sourceId }, expect.any(Function), expect.any(Object));
    expect(container.textContent).toContain("Current resources apply");
    await flushSaves();
    const saved = vi.mocked(saveConversation).mock.calls.at(-1)![0];
    expect(saved.resumeFromRunId).toBe(childId);
    expect(saved.messages.filter((m) => m.role === "user").map((m) => m.content)).toEqual(["source task", "continue"]);
    await submit("next");
    expect(vi.mocked(api.streamRun).mock.calls.at(-1)![0]).toEqual({ message: "next", resume_from_run_id: childId });
  });

  it("uses a named checkpoint's route without reapplying its resolved directory as a base", async () => {
    vi.mocked(api.showSession).mockResolvedValue({ ...checkpoint, agent: "reviewer", agent_revision: 1 });
    vi.mocked(api.streamAgentRun).mockImplementation(async (_name, _payload, onEvent) => {
      onEvent({ type: "done", run_id: childId, log_saved: true, messages: checkpoint.messages });
    });
    await mount();
    await importSaved();
    await submit();
    expect(api.streamAgentRun).toHaveBeenCalledWith("reviewer",
      { message: "continue", resume_from_run_id: sourceId }, expect.any(Function), expect.any(Object));
    expect(api.streamRun).not.toHaveBeenCalled();
    expect(container.textContent).toContain("revision 1");
  });

  it("retains unsaved canonical history locally and requires an explicit current-settings fork", async () => {
    vi.mocked(api.streamRun).mockImplementationOnce(async (_payload, onEvent) => {
      onEvent({ type: "done", run_id: childId, log_saved: false, response: "unsaved answer",
        messages: [...checkpoint.messages, { role: "user", content: "continue" }, { role: "assistant", content: "unsaved answer" }] });
    });
    await mount();
    await importSaved();
    await submit();
    expect(container.textContent).toContain("No saved continuation");
    expect(container.querySelector("textarea")!.disabled).toBe(true);
    await flushSaves();
    const saved = vi.mocked(saveConversation).mock.calls.at(-1)![0];
    expect(saved.resumeFromRunId).toBeUndefined();
    expect(saved.messages.at(-1)?.content).toBe("unsaved answer");
    vi.spyOn(window, "confirm").mockReturnValue(true);
    await click("Fork locally with current settings");
    expect(container.querySelectorAll(".conversation-item")).toHaveLength(3);
    expect(container.querySelector("textarea")!.disabled).toBe(false);
    await submit("review side effects");
    const payload = vi.mocked(api.streamRun).mock.calls.at(-1)![0];
    expect(payload.resume_from_run_id).toBeUndefined();
    expect(payload.messages?.some((message) => message.content === "unsaved answer")).toBe(true);
    expect(payload.messages?.at(-1)?.content).toContain("Interrupted tools");
  });

  it("does not reuse the original checkpoint after a streamed error or stop", async () => {
    vi.mocked(api.streamRun).mockImplementationOnce(async (_payload, onEvent) => {
      onEvent({ type: "error", message: "tool failed" });
    });
    await mount();
    await importSaved();
    await submit();
    expect(container.textContent).toContain("tool failed");
    expect(container.querySelector("textarea")!.disabled).toBe(true);
    await submit("retry");
    expect(api.streamRun).toHaveBeenCalledTimes(1);
    await importSaved();
    vi.mocked(api.streamRun).mockImplementationOnce((_payload, _event, options) => new Promise((_resolve, reject) => {
      options?.signal?.addEventListener("abort", () => reject(new Error("aborted")));
    }));
    await submit();
    await click("Stop");
    expect(container.textContent).toContain("Run stopped by user");
    expect(container.querySelector("textarea")!.disabled).toBe(true);
    expect(api.streamRun).toHaveBeenCalledTimes(2);
  });

  it("never posts a resume request to an older or disabled server", async () => {
    vi.mocked(api.sessionCapabilities).mockResolvedValue({ resume_supported: false, enabled: false });
    await mount();
    await importSaved();
    await submit();
    expect(api.streamRun).not.toHaveBeenCalled();
    expect(container.textContent).toContain("does not support or enable");
  });

  it("can stop feature negotiation before any execution request is sent", async () => {
    vi.mocked(api.sessionCapabilities).mockImplementation((signal) => new Promise((_resolve, reject) => {
      signal?.addEventListener("abort", () => reject(new Error("aborted")));
    }));
    await mount();
    await importSaved();
    await submit();
    await click("Stop");
    expect(api.streamRun).not.toHaveBeenCalled();
    expect(container.textContent).toContain("Run stopped by user");
  });

  it("marks a truncated stream incomplete rather than inferring a saved checkpoint", async () => {
    vi.mocked(api.streamRun).mockImplementationOnce(async (_payload, event) => {
      event({ type: "start", run_id: childId });
      event({ type: "content_delta", delta: "partial answer" });
    });
    await mount();
    await importSaved();
    await submit();
    expect(container.textContent).toContain("without a terminal event");
    expect(container.textContent).toContain("partial answer");
    expect(container.querySelector("textarea")!.disabled).toBe(true);
  });

  it("restores persisted checkpoint mode and leaves legacy local conversations unchanged", async () => {
    vi.mocked(listConversations).mockResolvedValue([{ ...existing, checkpointMode: true,
      importedFromRunId: sourceId, resumeFromRunId: sourceId }]);
    await mount();
    await submit();
    expect(vi.mocked(api.streamRun).mock.calls[0][0]).toEqual({ message: "continue", resume_from_run_id: sourceId });
    act(() => render(null, container));
    vi.mocked(listConversations).mockResolvedValue([existing]);
    await mount();
    await submit();
    expect(vi.mocked(api.streamRun).mock.calls.at(-1)![0]).toMatchObject({ messages: [], profile: "local", skills: [] });
    expect(vi.mocked(api.streamRun).mock.calls.at(-1)![0].resume_from_run_id).toBeUndefined();
  });

  it("shows opt-in errors without disturbing the active local conversation", async () => {
    vi.mocked(api.listSessions).mockRejectedValue(new Error("Set PYAGENT_SESSION_API=on"));
    await mount();
    await click("Saved checkpoints");
    expect(container.querySelector('[role="alert"]')?.textContent).toContain("PYAGENT_SESSION_API");
    expect(container.querySelectorAll(".conversation-item")).toHaveLength(1);
    expect(api.streamRun).not.toHaveBeenCalled();
  });
});
