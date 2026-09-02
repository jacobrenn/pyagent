import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { api, ApiError } from "../api/client";
import { Transcript } from "../components/Transcript";
import {
  deleteConversation,
  listConversations,
  saveConversation,
} from "../state/conversations";
import { addToolCall, correlateToolResult } from "../state/toolActivity";
import type {
  AgentsResponse,
  Conversation,
  Profile,
  ProfilesResponse,
  Skill,
  StreamEvent,
  TranscriptEntry,
} from "../types";

interface ChatPageProps {
  profiles: ProfilesResponse | null;
  agents: AgentsResponse | null;
  agentLaunch: { name: string; revision: number; nonce: number } | null;
  onAgentLaunchHandled: () => void;
  onOpenProfiles: () => void;
}

function id(prefix: string): string {
  const value = globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2);
  return `${prefix}-${value}`;
}

function entry(
  kind: TranscriptEntry["kind"],
  content: string,
  extra: Partial<TranscriptEntry> = {},
): TranscriptEntry {
  return {
    id: id(kind),
    kind,
    content,
    createdAt: new Date().toISOString(),
    ...extra,
  };
}

function newConversation(
  profile?: Profile,
  agent?: { name: string; revision?: number },
): Conversation {
  const now = new Date().toISOString();
  return {
    id: id("conversation"),
    title: "New conversation",
    createdAt: now,
    updatedAt: now,
    profile: profile?.name ?? "",
    model: profile?.model ?? "",
    cwd: ".",
    skills: [],
    agent: agent?.name,
    agentRevision: agent?.revision,
    messages: [],
    entries: [],
  };
}

export function ChatPage({
  profiles,
  agents,
  agentLaunch,
  onAgentLaunchHandled,
  onOpenProfiles,
}: ChatPageProps) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [composer, setComposer] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [activeAssistantId, setActiveAssistantId] = useState<string | null>(null);
  const [runStatus, setRunStatus] = useState("Ready");
  const [includeDebug, setIncludeDebug] = useState(false);
  const [debugEvents, setDebugEvents] = useState<StreamEvent[]>([]);
  const [skills, setSkills] = useState<Skill[]>([]);
  const [skillsLoading, setSkillsLoading] = useState(false);
  const [models, setModels] = useState<string[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const saveTimersRef = useRef(new Map<string, number>());
  const pendingSavesRef = useRef(new Map<string, Conversation>());

  const scheduleConversationSave = (conversation: Conversation) => {
    const existing = saveTimersRef.current.get(conversation.id);
    if (existing !== undefined) window.clearTimeout(existing);
    pendingSavesRef.current.set(conversation.id, conversation);
    const timer = window.setTimeout(() => {
      saveTimersRef.current.delete(conversation.id);
      pendingSavesRef.current.delete(conversation.id);
      void saveConversation(conversation);
    }, 120);
    saveTimersRef.current.set(conversation.id, timer);
  };

  const profileList = profiles?.profiles ?? [];
  const effectiveProfile = profileList.find(
    (profile) => profile.name === profiles?.effective_default_profile,
  );
  const active = conversations.find((item) => item.id === activeId) ?? null;
  const selectedProfile = profileList.find(
    (profile) => profile.name === active?.profile,
  );
  const selectedAgent = agents?.agents.find(
    (agent) => agent.name === active?.agent,
  );

  useEffect(() => {
    return () => {
      for (const timer of saveTimersRef.current.values()) {
        window.clearTimeout(timer);
      }
      for (const conversation of pendingSavesRef.current.values()) {
        void saveConversation(conversation);
      }
    };
  }, []);

  useEffect(() => {
    void listConversations().then((stored) => {
      if (stored.length) {
        setConversations(stored);
        setActiveId(stored[0].id);
      } else {
        const created = newConversation(effectiveProfile);
        setConversations([created]);
        setActiveId(created.id);
        void saveConversation(created);
      }
      setLoaded(true);
    });
  }, []);

  useEffect(() => {
    if (!loaded || !profiles || !active || active.profile) {
      return;
    }
    updateConversation(active.id, (conversation) => ({
      ...conversation,
      profile: effectiveProfile?.name ?? "",
      model: effectiveProfile?.model ?? "",
    }));
  }, [profiles, loaded, active?.id]);

  useEffect(() => {
    if (!loaded || !agentLaunch) return;
    const definition = agents?.agents.find((item) => item.name === agentLaunch.name);
    const created = newConversation(effectiveProfile, {
      name: agentLaunch.name,
      revision: agentLaunch.revision,
    });
    created.title = `${agentLaunch.name} run`;
    if (definition?.profile) {
      const profile = profileList.find((item) => item.name === definition.profile);
      created.profile = profile?.name ?? definition.profile;
      created.model = definition.model ?? profile?.model ?? "";
    }
    setConversations((current) => [created, ...current]);
    setActiveId(created.id);
    setDebugEvents([]);
    setModels([]);
    void saveConversation(created);
    onAgentLaunchHandled();
  }, [agentLaunch?.nonce, loaded]);

  const updateConversation = (
    conversationId: string,
    updater: (conversation: Conversation) => Conversation,
  ) => {
    setConversations((current) =>
      current.map((conversation) => {
        if (conversation.id !== conversationId) {
          return conversation;
        }
        const updated = {
          ...updater(conversation),
          updatedAt: new Date().toISOString(),
        };
        scheduleConversationSave(updated);
        return updated;
      }),
    );
  };

  const createNewConversation = () => {
    if (streaming) {
      return;
    }
    const created = newConversation(effectiveProfile);
    setConversations((current) => [created, ...current]);
    setActiveId(created.id);
    setDebugEvents([]);
    setModels([]);
    void saveConversation(created);
  };

  const removeConversation = async (conversationId: string) => {
    if (streaming) {
      return;
    }
    const saveTimer = saveTimersRef.current.get(conversationId);
    if (saveTimer !== undefined) window.clearTimeout(saveTimer);
    saveTimersRef.current.delete(conversationId);
    pendingSavesRef.current.delete(conversationId);
    await deleteConversation(conversationId);
    const remaining = conversations.filter((item) => item.id !== conversationId);
    if (remaining.length) {
      setConversations(remaining);
      if (activeId === conversationId) {
        setActiveId(remaining[0].id);
      }
      return;
    }
    const created = newConversation(effectiveProfile);
    setConversations([created]);
    setActiveId(created.id);
    void saveConversation(created);
  };

  const loadSkills = async () => {
    if (!active) {
      return;
    }
    setSkillsLoading(true);
    try {
      const response = await api.listSkills(active.cwd || ".");
      setSkills(response.skills);
    } catch (error) {
      setRunStatus(error instanceof Error ? error.message : "Could not load skills");
    } finally {
      setSkillsLoading(false);
    }
  };

  const loadModels = async () => {
    if (!active?.profile) {
      return;
    }
    setModelsLoading(true);
    try {
      const response = await api.listProfileModels(active.profile);
      setModels(response.models);
      setRunStatus(
        response.models.length ? `Found ${response.models.length} models` : "No models reported",
      );
    } catch (error) {
      setRunStatus(error instanceof Error ? error.message : "Could not load models");
    } finally {
      setModelsLoading(false);
    }
  };

  const handleStreamEvent = (
    conversationId: string,
    eventData: StreamEvent,
    state: { assistantId: string | null; receivedContent: boolean },
  ) => {
    if (eventData.type === "start") {
      setRunStatus(`Connected · ${eventData.model ?? "model"}`);
      return;
    }
    if (eventData.type === "debug") {
      setDebugEvents((current) => [...current.slice(-199), eventData]);
      return;
    }
    if (eventData.type === "assistant_start") {
      if (!state.assistantId) {
        const assistant = entry("assistant", "");
        state.assistantId = assistant.id;
        setActiveAssistantId(assistant.id);
        updateConversation(conversationId, (conversation) => ({
          ...conversation,
          entries: [...conversation.entries, assistant],
        }));
      }
      setRunStatus("Thinking…");
      return;
    }
    if (eventData.type === "content_delta") {
      if (!state.assistantId) {
        const assistant = entry("assistant", "");
        state.assistantId = assistant.id;
        setActiveAssistantId(assistant.id);
        updateConversation(conversationId, (conversation) => ({
          ...conversation,
          entries: [...conversation.entries, assistant],
        }));
      }
      state.receivedContent = true;
      const assistantId = state.assistantId;
      updateConversation(conversationId, (conversation) => ({
        ...conversation,
        entries: conversation.entries.map((item) =>
          item.id === assistantId
            ? { ...item, content: item.content + (eventData.delta ?? "") }
            : item,
        ),
      }));
      return;
    }
    if (eventData.type === "tool_call") {
      state.assistantId = null;
      setActiveAssistantId(null);
      updateConversation(conversationId, (conversation) => ({
        ...conversation,
        entries: addToolCall(conversation.entries, eventData, (extra) =>
          entry("tool", "", extra),
        ),
      }));
      setRunStatus(`Running tool · ${eventData.name ?? "unknown"}`);
      return;
    }
    if (eventData.type === "tool_result") {
      updateConversation(conversationId, (conversation) => ({
        ...conversation,
        entries: correlateToolResult(conversation.entries, eventData, (extra) =>
          entry("tool", "", extra),
        ),
      }));
      setRunStatus("Thinking…");
      return;
    }
    if (eventData.type === "done") {
      if (!state.receivedContent) {
        if (state.assistantId) {
          const assistantId = state.assistantId;
          updateConversation(conversationId, (conversation) => ({
            ...conversation,
            entries: conversation.entries.map((item) =>
              item.id === assistantId
                ? { ...item, content: eventData.response ?? "" }
                : item,
            ),
          }));
        } else if (eventData.response) {
          const assistant = entry("assistant", eventData.response);
          updateConversation(conversationId, (conversation) => ({
            ...conversation,
            entries: [...conversation.entries, assistant],
          }));
        }
      }
      updateConversation(conversationId, (conversation) => ({
        ...conversation,
        profile: eventData.profile ?? conversation.profile,
        model: eventData.model ?? conversation.model,
        messages: (eventData.messages ?? []).filter(
          (message) => message.role !== "system",
        ),
      }));
      setRunStatus("Complete");
      return;
    }
    if (eventData.type === "error") {
      updateConversation(conversationId, (conversation) => ({
        ...conversation,
        entries: [
          ...conversation.entries,
          entry("error", eventData.message ?? "The agent run failed."),
        ],
      }));
      setRunStatus("Run failed");
    }
  };

  const submit = async () => {
    const prompt = composer.trim();
    if (!active || !prompt || streaming || (!active.agent && !active.profile)) {
      return;
    }

    const conversationId = active.id;
    const history = active.messages;
    const userEntry = entry("user", prompt);
    updateConversation(conversationId, (conversation) => ({
      ...conversation,
      title:
        conversation.entries.length === 0
          ? `${prompt.slice(0, 44)}${prompt.length > 44 ? "…" : ""}`
          : conversation.title,
      messages: [...conversation.messages, { role: "user", content: prompt }],
      entries: [...conversation.entries, userEntry],
    }));
    setComposer("");
    setStreaming(true);
    setDebugEvents([]);
    setRunStatus("Connecting…");
    const controller = new AbortController();
    abortRef.current = controller;
    const eventState = { assistantId: null as string | null, receivedContent: false };

    try {
      const onEvent = (eventData: StreamEvent) =>
        handleStreamEvent(conversationId, eventData, eventState);
      if (active.agent) {
        await api.streamAgentRun(
          active.agent,
          {
            message: prompt,
            messages: history,
            revision: active.agentRevision,
            cwd: active.cwd || ".",
          },
          onEvent,
          { signal: controller.signal, includeDebug },
        );
      } else {
        await api.streamRun(
          {
            message: prompt,
            messages: history,
            profile: active.profile,
            model: active.model || undefined,
            cwd: active.cwd || ".",
            skills: active.skills,
          },
          onEvent,
          { signal: controller.signal, includeDebug },
        );
      }
    } catch (error) {
      if (controller.signal.aborted) {
        updateConversation(conversationId, (conversation) => ({
          ...conversation,
          entries: [
            ...conversation.entries,
            entry("system", "Run stopped by user."),
          ],
        }));
        setRunStatus("Stopped");
      } else {
        const message =
          error instanceof ApiError || error instanceof Error
            ? error.message
            : "Could not connect to PyAgent.";
        updateConversation(conversationId, (conversation) => ({
          ...conversation,
          entries: [...conversation.entries, entry("error", message)],
        }));
        setRunStatus("Connection failed");
      }
    } finally {
      abortRef.current = null;
      setActiveAssistantId(null);
      setStreaming(false);
    }
  };

  const activeSkillSet = useMemo(() => new Set(active?.skills ?? []), [active?.skills]);

  if (!loaded || !active) {
    return <div class="page-loading">Loading conversations…</div>;
  }

  return (
    <div class="chat-layout">
      <aside class="conversation-sidebar">
        <div class="sidebar-heading">
          <span>Conversations</span>
          <button
            class="icon-button"
            onClick={createNewConversation}
            disabled={streaming}
            title="New conversation"
          >
            +
          </button>
        </div>
        <div class="conversation-list">
          {conversations.map((conversation) => (
            <button
              key={conversation.id}
              class={`conversation-item${conversation.id === activeId ? " is-active" : ""}`}
              onClick={() => !streaming && setActiveId(conversation.id)}
            >
              <span>{conversation.title}</span>
              <small>{new Date(conversation.updatedAt).toLocaleDateString()}</small>
            </button>
          ))}
        </div>
        <button
          class="sidebar-delete"
          onClick={() => void removeConversation(active.id)}
          disabled={streaming}
        >
          Delete conversation
        </button>
      </aside>

      <section class="chat-panel">
        <div class="run-bar">
          <div class="run-bar__select">
            <label htmlFor="chat-runtime">Runtime</label>
            <select
              id="chat-runtime"
              value={active.agent ?? ""}
              disabled={streaming}
              onChange={(event) => {
                const agentName = event.currentTarget.value;
                const definition = agents?.agents.find((item) => item.name === agentName);
                updateConversation(active.id, (conversation) => ({
                  ...conversation,
                  agent: agentName || undefined,
                  agentRevision: definition?.revision,
                }));
              }}
            >
              <option value="">Ad hoc chat</option>
              {(agents?.agents ?? []).map((agent) => (
                <option key={agent.name} value={agent.name}>
                  Agent · {agent.name} (r{agent.revision})
                </option>
              ))}
            </select>
          </div>
          {!active.agent && (
            <div class="run-bar__select run-bar__select--profile">
              <label htmlFor="chat-profile">Profile</label>
              <select
                id="chat-profile"
                value={active.profile}
                disabled={streaming}
                onChange={(event) => {
                  const profile = profileList.find(
                    (item) => item.name === event.currentTarget.value,
                  );
                  updateConversation(active.id, (conversation) => ({
                    ...conversation,
                    profile: profile?.name ?? "",
                    model: profile?.model ?? "",
                  }));
                  setModels([]);
                }}
              >
                {!profileList.length && <option value="">No profiles</option>}
                {profileList.map((profile) => (
                  <option key={profile.name} value={profile.name}>
                    {profile.name}{profile.is_default ? " · default" : ""}
                  </option>
                ))}
              </select>
            </div>
          )}
          <div class="run-bar__meta">
            <span>{active.agent ? selectedAgent?.name ?? active.agent : selectedProfile?.provider ?? "No provider"}</span>
            <i />
            <span>
              {active.agent
                ? `revision ${active.agentRevision ?? selectedAgent?.revision ?? "current"}`
                : active.model || selectedProfile?.model || "No model"}
            </span>
          </div>
          <div class={`run-status${streaming ? " is-active" : ""}`}>
            <i />
            {runStatus}
          </div>
        </div>

        <div class="transcript-shell">
          <Transcript
            key={active.id}
            entries={active.entries}
            activeAssistantId={activeAssistantId}
            streaming={streaming}
          />
        </div>

        <div class="composer-shell">
          {!profileList.length && !active.agent && (
            <button class="inline-alert" onClick={onOpenProfiles}>
              Create a model profile before starting a conversation →
            </button>
          )}
          <details class="run-settings">
            <summary>Run settings</summary>
            {active.agent && (
              <p class="agent-run-note">
                <strong>{selectedAgent?.name ?? active.agent}</strong> controls the profile, model, prompt, skills, and tool allowlist. The directory below is the base for a relative agent workspace.
              </p>
            )}
            <div class="run-settings__grid">
              {!active.agent && (
                <label>
                  <span>Model override</span>
                  <div class="field-action">
                    <input
                      value={active.model}
                      list="available-models"
                      disabled={streaming}
                      onInput={(event) =>
                        updateConversation(active.id, (conversation) => ({
                          ...conversation,
                          model: event.currentTarget.value,
                        }))
                      }
                    />
                    <button
                      type="button"
                      onClick={() => void loadModels()}
                      disabled={modelsLoading || streaming || !active.profile}
                    >
                      {modelsLoading ? "…" : "Find"}
                    </button>
                  </div>
                  <datalist id="available-models">
                    {models.map((model) => <option key={model} value={model} />)}
                  </datalist>
                </label>
              )}
              <label>
                <span>{active.agent ? "Workspace base directory" : "Working directory"}</span>
                <input
                  value={active.cwd}
                  disabled={streaming}
                  onInput={(event) =>
                    updateConversation(active.id, (conversation) => ({
                      ...conversation,
                      cwd: event.currentTarget.value,
                    }))
                  }
                />
              </label>
            </div>
            {!active.agent && <div class="skills-setting">
              <div>
                <span class="field-label">Skills</span>
                <small>Loaded explicitly for this conversation</small>
              </div>
              <button
                type="button"
                onClick={() => void loadSkills()}
                disabled={skillsLoading || streaming}
              >
                {skillsLoading ? "Loading…" : "Discover skills"}
              </button>
            </div>}
            {!active.agent && skills.length > 0 && (
              <div class="skill-options">
                {skills.map((skill) => (
                  <label key={skill.id} title={skill.preview}>
                    <input
                      type="checkbox"
                      checked={activeSkillSet.has(skill.id)}
                      disabled={streaming || Boolean(skill.error)}
                      onChange={(event) =>
                        updateConversation(active.id, (conversation) => ({
                          ...conversation,
                          skills: event.currentTarget.checked
                            ? [...conversation.skills, skill.id]
                            : conversation.skills.filter((item) => item !== skill.id),
                        }))
                      }
                    />
                    <span>{skill.title || skill.label}</span>
                    <small>{skill.scope}</small>
                  </label>
                ))}
              </div>
            )}
            <label class="debug-toggle">
              <input
                type="checkbox"
                checked={includeDebug}
                onChange={(event) => setIncludeDebug(event.currentTarget.checked)}
              />
              Capture debug events
            </label>
            {debugEvents.length > 0 && (
              <details class="debug-panel">
                <summary>{debugEvents.length} debug events</summary>
                <pre>{JSON.stringify(debugEvents, null, 2)}</pre>
              </details>
            )}
          </details>
          <form
            class="composer"
            onSubmit={(event) => {
              event.preventDefault();
              void submit();
            }}
          >
            <textarea
              value={composer}
              placeholder="Ask PyAgent about this workspace…"
              disabled={streaming || (!active.agent && !active.profile)}
              rows={3}
              onInput={(event) => setComposer(event.currentTarget.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void submit();
                }
              }}
            />
            {streaming ? (
              <button
                class="send-button send-button--stop"
                type="button"
                onClick={() => abortRef.current?.abort()}
              >
                Stop
              </button>
            ) : (
              <button
                class="send-button"
                type="submit"
                disabled={!composer.trim() || (!active.agent && !active.profile)}
              >
                Send <span>↗</span>
              </button>
            )}
          </form>
          <p class="composer-hint">Enter to send · Shift+Enter for a new line</p>
        </div>
      </section>
    </div>
  );
}
