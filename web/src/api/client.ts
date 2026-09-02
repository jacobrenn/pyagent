import { consumeSse } from "./sse";
import type {
  AgentDefinition,
  AgentsResponse,
  AgentValidation,
  ExtensionsResponse,
  ProfilesResponse,
  ResourceInstallInput,
  ResourceListResponse,
  RunRequest,
  SkillsResponse,
  StreamEvent,
  ToolsResponse,
} from "../types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function errorMessage(response: Response): Promise<string> {
  const text = await response.text();
  try {
    const payload = JSON.parse(text) as { detail?: unknown };
    return payload.detail ? String(payload.detail) : text || response.statusText;
  } catch {
    return text || response.statusText || `HTTP ${response.status}`;
  }
}

async function requestJson<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  if (
    init.body &&
    !headers.has("Content-Type") &&
    !(init.body instanceof FormData)
  ) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, { ...init, headers });
  if (!response.ok) {
    throw new ApiError(await errorMessage(response), response.status);
  }
  return (await response.json()) as T;
}

function encodedPath(value: string): string {
  return value.split("/").map(encodeURIComponent).join("/");
}

function installResource(path: string, input: ResourceInstallInput) {
  if (input.file) {
    const form = new FormData();
    form.set("file", input.file);
    if (input.name?.trim()) form.set("name", input.name.trim());
    if (input.force) form.set("force", "true");
    return requestJson(path, { method: "POST", body: form });
  }
  return requestJson(path, {
    method: "POST",
    body: JSON.stringify({
      url: input.url?.trim(),
      ...(input.name?.trim() ? { name: input.name.trim() } : {}),
      ...(input.force ? { force: true } : {}),
    }),
  });
}

export const api = {
  health: () => requestJson<{ status: string }>("/health"),
  version: () => requestJson<{ version: string }>("/version"),

  listProfiles: () => requestJson<ProfilesResponse>("/profiles"),
  createProfile: (payload: Record<string, unknown>) =>
    requestJson("/profiles", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  updateProfile: (name: string, payload: Record<string, unknown>) =>
    requestJson(`/profiles/${encodeURIComponent(name)}`, {
      method: "PUT",
      body: JSON.stringify(payload),
    }),
  setDefaultProfile: (name: string) =>
    requestJson(`/profiles/${encodeURIComponent(name)}/default`, {
      method: "POST",
    }),
  deleteProfile: (name: string) =>
    requestJson(`/profiles/${encodeURIComponent(name)}`, {
      method: "DELETE",
    }),
  listProfileModels: (name: string) =>
    requestJson<{ profile: string; models: string[] }>(
      `/profiles/${encodeURIComponent(name)}/models`,
    ),

  listSkills: (cwd?: string) => {
    const query = cwd ? `?${new URLSearchParams({ cwd })}` : "";
    return requestJson<SkillsResponse>(`/skills${query}`);
  },
  installSkill: (input: ResourceInstallInput) =>
    installResource("/skills/install", input),
  deleteSkill: (name: string) =>
    requestJson(`/skills/${encodedPath(name)}`, { method: "DELETE" }),

  listTools: () => requestJson<ToolsResponse>("/tools"),
  installTool: (input: ResourceInstallInput) =>
    installResource("/tools/install", input),
  scaffoldTool: (name: string) =>
    requestJson("/tools/new", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),
  enableTool: (name: string) =>
    requestJson(`/tools/${encodedPath(name)}/enable`, { method: "POST" }),
  disableTool: (name: string) =>
    requestJson(`/tools/${encodedPath(name)}/disable`, { method: "POST" }),
  deleteTool: (name: string) =>
    requestJson(`/tools/${encodedPath(name)}`, { method: "DELETE" }),

  listExtensions: () => requestJson<ExtensionsResponse>("/extensions"),
  installExtension: (input: ResourceInstallInput) =>
    installResource("/extensions/install", input),
  scaffoldExtension: (name: string) =>
    requestJson("/extensions/new", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),
  enableExtension: (name: string) =>
    requestJson(`/extensions/${encodeURIComponent(name)}/enable`, {
      method: "POST",
    }),
  disableExtension: (name: string) =>
    requestJson(`/extensions/${encodeURIComponent(name)}/disable`, {
      method: "POST",
    }),
  deleteExtension: (name: string) =>
    requestJson(`/extensions/${encodeURIComponent(name)}`, { method: "DELETE" }),

  listPrompts: () => requestJson<ResourceListResponse>("/prompts"),

  listAgents: () => requestJson<AgentsResponse>("/agents"),
  createAgent: (payload: Record<string, unknown>) =>
    requestJson<AgentDefinition>("/agents", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  updateAgent: (name: string, payload: Record<string, unknown>) =>
    requestJson<AgentDefinition>(`/agents/${encodeURIComponent(name)}`, {
      method: "PUT",
      body: JSON.stringify(payload),
    }),
  deleteAgent: (name: string) =>
    requestJson(`/agents/${encodeURIComponent(name)}`, { method: "DELETE" }),
  validateAgent: (name: string, cwd?: string) => {
    const query = cwd ? `?${new URLSearchParams({ cwd })}` : "";
    return requestJson<AgentValidation>(
      `/agents/${encodeURIComponent(name)}/validate${query}`,
      { method: "POST" },
    );
  },
  listAgentRevisions: (name: string) =>
    requestJson<{ name: string; revisions: AgentDefinition[] }>(
      `/agents/${encodeURIComponent(name)}/revisions`,
    ),

  streamRun: async (
    payload: RunRequest,
    onEvent: (event: StreamEvent) => void,
    options: { signal?: AbortSignal; includeDebug?: boolean } = {},
  ): Promise<void> => {
    const query = options.includeDebug ? "?include_debug=true" : "";
    const response = await fetch(`/run/stream${query}`, {
      method: "POST",
      signal: options.signal,
      headers: {
        Accept: "text/event-stream",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      throw new ApiError(await errorMessage(response), response.status);
    }
    if (!response.body) {
      throw new ApiError("The server did not provide a response stream.", 502);
    }
    await consumeSse(response.body, onEvent);
  },

  streamAgentRun: async (
    name: string,
    payload: {
      message: string;
      messages: RunRequest["messages"];
      revision?: number;
      cwd?: string;
    },
    onEvent: (event: StreamEvent) => void,
    options: { signal?: AbortSignal; includeDebug?: boolean } = {},
  ): Promise<void> => {
    const query = options.includeDebug ? "?include_debug=true" : "";
    const response = await fetch(
      `/agents/${encodeURIComponent(name)}/run/stream${query}`,
      {
        method: "POST",
        signal: options.signal,
        headers: {
          Accept: "text/event-stream",
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      },
    );
    if (!response.ok) {
      throw new ApiError(await errorMessage(response), response.status);
    }
    if (!response.body) {
      throw new ApiError("The server did not provide a response stream.", 502);
    }
    await consumeSse(response.body, onEvent);
  },
};
