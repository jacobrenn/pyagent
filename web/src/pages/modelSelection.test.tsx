// @vitest-environment jsdom
import { render } from "preact";
import { act } from "preact/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type { AgentDefinition, Profile, ProfilesResponse } from "../types";
import { AgentsPage } from "./AgentsPage";
import { ProfilesPage } from "./ProfilesPage";

vi.mock("../api/client", () => ({
  api: {
    listSkills: vi.fn(),
    listTools: vi.fn(),
    listPrompts: vi.fn(),
    listAgentRevisions: vi.fn(),
    listProfileModels: vi.fn(),
    updateProfile: vi.fn(),
    updateAgent: vi.fn(),
  },
}));

const local: Profile = {
  name: "local",
  provider: "ollama",
  api_mode: "chat_completions",
  model: "local-default",
  base_url: "http://localhost:11434",
  api_key_env: null,
  has_inline_api_key: true,
  headers: {},
  redacted_headers: ["Authorization"],
  httpx_kwargs: {},
  is_default: true,
};
const profiles: ProfilesResponse = {
  path: "/profiles.json",
  default_profile: "local",
  effective_default_profile: "remote",
  default_overridden_by_env: true,
  profiles: [local, { ...local, name: "remote", model: "remote-default", is_default: false }],
};
const agent: AgentDefinition = {
  schema_version: 1,
  name: "reviewer",
  revision: 1,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  description: "Review code",
  profile: null,
  model: null,
  prompt: null,
  skills: [],
  tools: null,
  workspace: ".",
  max_iterations: null,
  labels: {},
  capabilities: [],
};
let container: HTMLDivElement;
let onRefresh: ReturnType<typeof vi.fn<() => Promise<void>>>;

beforeEach(() => {
  vi.resetAllMocks();
  container = document.createElement("div");
  document.body.append(container);
  onRefresh = vi.fn().mockResolvedValue(undefined);
  vi.mocked(api.listSkills).mockResolvedValue({ cwd: ".", user_dir: "/user", skills: [] });
  vi.mocked(api.listTools).mockResolvedValue({
    tools_enabled: true,
    builtin_tools_enabled: true,
    user_tools_enabled: true,
    user_dir: "/user",
    runner: { name: "uv", available: true, message: null },
    builtin: [], external: [], files: [], broken: [], disabled: [], collisions: [],
    discovery_error: null,
  });
  vi.mocked(api.listPrompts).mockResolvedValue({ kind: "prompts", root: "/prompts", items: [] });
  vi.mocked(api.listAgentRevisions).mockResolvedValue({ name: agent.name, revisions: [agent] });
  vi.mocked(api.listProfileModels).mockImplementation(async (name) => ({
    profile: name,
    models: [`${name}-a`, `${name}-b`],
  }));
  vi.mocked(api.updateAgent).mockResolvedValue({ ...agent, revision: 2 });
});

afterEach(() => {
  act(() => render(null, container));
  container.remove();
});

async function click(text: string) {
  const button = [...container.querySelectorAll("button")].find((item) => item.textContent === text);
  if (!button) throw new Error(`Missing button: ${text}`);
  await act(async () => button.click());
}

async function choose(select: HTMLSelectElement, value: string) {
  await act(async () => {
    select.value = value;
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

async function submit() {
  await act(async () => {
    container.querySelector("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
  });
}

async function mountAgents() {
  await act(async () => render(
    <AgentsPage
      agents={{ root: "/agents", agents: [agent] }}
      profiles={profiles}
      loading={false}
      onRefresh={onRefresh}
      onRun={vi.fn()}
    />,
    container,
  ));
}

describe("model selection in editors", () => {
  it("saves a discovered profile model without replacing credentials", async () => {
    await act(async () => render(
      <ProfilesPage profiles={profiles} loading={false} onRefresh={onRefresh} />,
      container,
    ));
    await click("Discover models");
    expect(api.listProfileModels).toHaveBeenCalledWith("local");
    await choose(container.querySelector<HTMLSelectElement>("select#profile-model")!, "local-b");
    await submit();

    expect(api.updateProfile).toHaveBeenCalledWith("local", expect.objectContaining({ model: "local-b" }));
    const payload = vi.mocked(api.updateProfile).mock.calls[0][1];
    expect(payload).not.toHaveProperty("api_key");
    expect(payload).not.toHaveProperty("headers");
    expect(payload).not.toHaveProperty("httpx_kwargs");
    expect(onRefresh).toHaveBeenCalled();
  });

  it("discovers using an agent's effective default profile and can save or clear an override", async () => {
    await mountAgents();
    await click("Discover models");
    expect(api.listProfileModels).toHaveBeenCalledWith("remote");
    const selector = container.querySelector<HTMLSelectElement>("select#agent-model")!;
    expect(selector.value).toBe("");
    expect(selector.options[0].textContent).toBe("Use profile model (remote-default)");

    await choose(selector, "remote-b");
    await submit();
    expect(api.updateAgent).toHaveBeenLastCalledWith("reviewer", expect.objectContaining({ model: "remote-b" }));
    await choose(selector, "");
    await submit();
    expect(api.updateAgent).toHaveBeenLastCalledWith("reviewer", expect.objectContaining({ model: null }));
  });

  it("refreshes the agent's model choices when selecting an explicit profile", async () => {
    await mountAgents();
    await click("Discover models");
    const profileLabel = [...container.querySelectorAll("label")].find((label) => label.textContent?.startsWith("Profile"))!;
    await choose(profileLabel.querySelector("select")!, "local");
    expect(container.querySelector("select#agent-model")).toBeNull();
    await click("Discover models");
    expect(api.listProfileModels).toHaveBeenLastCalledWith("local");
    const selector = container.querySelector<HTMLSelectElement>("select#agent-model")!;
    expect([...selector.options].map((option) => option.value)).toEqual(["", "local-a", "local-b"]);
    expect(selector.options[0].textContent).toBe("Use profile model (local-default)");
  });
});
