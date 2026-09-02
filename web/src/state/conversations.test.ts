import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import {
  deleteConversation,
  listConversations,
  saveConversation,
} from "./conversations";
import type { Conversation } from "../types";

const createdIds = new Set<string>();
const storage = new Map<string, string>();

beforeAll(() => {
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => storage.get(key) ?? null,
    setItem: (key: string, value: string) => storage.set(key, value),
  });
});

afterAll(() => vi.unstubAllGlobals());

function conversation(id: string, updatedAt: string): Conversation {
  return {
    id,
    title: id,
    createdAt: "2025-01-01T00:00:00Z",
    updatedAt,
    profile: "local",
    model: "test",
    cwd: ".",
    skills: [],
    messages: [],
    entries: [],
  };
}

afterEach(async () => {
  await Promise.all([...createdIds].map((id) => deleteConversation(id)));
  createdIds.clear();
});

describe("conversation persistence", () => {
  it("round-trips conversations and returns newest first", async () => {
    const first = conversation("test-old", "2025-01-01T00:00:00Z");
    const second = conversation("test-new", "2025-01-02T00:00:00Z");
    createdIds.add(first.id);
    createdIds.add(second.id);

    await saveConversation(first);
    await saveConversation(second);
    const stored = (await listConversations()).filter((item) =>
      createdIds.has(item.id),
    );

    expect(stored.map((item) => item.id)).toEqual([second.id, first.id]);
  });

  it("deletes a stored conversation", async () => {
    const value = conversation("test-delete", "2025-01-01T00:00:00Z");
    await saveConversation(value);
    await deleteConversation(value.id);

    expect((await listConversations()).some((item) => item.id === value.id)).toBe(false);
  });
});
