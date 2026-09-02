import { useEffect, useMemo, useState } from "preact/hooks";
import { api } from "../api/client";
import type {
  AgentDefinition,
  AgentsResponse,
  AgentValidation,
  Profile,
  ProfilesResponse,
  ResourceItem,
  Skill,
  ToolInfo,
} from "../types";

interface AgentsPageProps {
  agents: AgentsResponse | null;
  profiles: ProfilesResponse | null;
  loading: boolean;
  onRefresh: () => Promise<void>;
  onRun: (agent: AgentDefinition) => void;
}

interface AgentDraft {
  name: string;
  description: string;
  profile: string;
  model: string;
  prompt: string;
  workspace: string;
  maxIterations: string;
  skills: string[];
  allTools: boolean;
  tools: string[];
  capabilities: string;
  labels: string;
}

function blankDraft(): AgentDraft {
  return {
    name: "",
    description: "",
    profile: "",
    model: "",
    prompt: "",
    workspace: ".",
    maxIterations: "",
    skills: [],
    allTools: true,
    tools: [],
    capabilities: "",
    labels: "{}",
  };
}

function draftForAgent(agent: AgentDefinition): AgentDraft {
  return {
    name: agent.name,
    description: agent.description,
    profile: agent.profile ?? "",
    model: agent.model ?? "",
    prompt: agent.prompt ?? "",
    workspace: agent.workspace ?? ".",
    maxIterations: agent.max_iterations?.toString() ?? "",
    skills: [...agent.skills],
    allTools: agent.tools === null,
    tools: [...(agent.tools ?? [])],
    capabilities: agent.capabilities.join(", "),
    labels: JSON.stringify(agent.labels, null, 2),
  };
}

function parseLabels(value: string): Record<string, string> {
  const parsed: unknown = JSON.parse(value || "{}");
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Labels must be a JSON object.");
  }
  if (Object.values(parsed).some((item) => typeof item !== "string")) {
    throw new Error("Every label value must be a string.");
  }
  return parsed as Record<string, string>;
}

function parseMaxIterations(value: string): number | null {
  if (!value.trim()) return null;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || (parsed !== -1 && parsed < 1)) {
    throw new Error("Max iterations must be blank, -1, or an integer greater than zero.");
  }
  return parsed;
}

function commaList(value: string): string[] {
  return [...new Set(value.split(",").map((item) => item.trim()).filter(Boolean))];
}

export function AgentsPage({
  agents,
  profiles,
  loading,
  onRefresh,
  onRun,
}: AgentsPageProps) {
  const agentList = agents?.agents ?? [];
  const [selectedName, setSelectedName] = useState("");
  const [creating, setCreating] = useState(false);
  const [draft, setDraft] = useState<AgentDraft>(blankDraft());
  const [availableSkills, setAvailableSkills] = useState<Skill[]>([]);
  const [availableTools, setAvailableTools] = useState<ToolInfo[]>([]);
  const [prompts, setPrompts] = useState<ResourceItem[]>([]);
  const [validation, setValidation] = useState<AgentValidation | null>(null);
  const [revisions, setRevisions] = useState<AgentDefinition[]>([]);
  const [resourcesLoading, setResourcesLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [validating, setValidating] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");

  const selected = useMemo(
    () => agentList.find((agent) => agent.name === selectedName) ?? null,
    [agentList, selectedName],
  );

  const loadResources = async (cwd = draft.workspace || ".") => {
    setResourcesLoading(true);
    try {
      const [skillsResponse, toolsResponse, promptsResponse] = await Promise.all([
        api.listSkills(cwd),
        api.listTools(),
        api.listPrompts(),
      ]);
      setAvailableSkills(skillsResponse.skills);
      setAvailableTools([...toolsResponse.builtin, ...toolsResponse.external]);
      setPrompts(promptsResponse.items);
      setError("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not discover agent resources.");
    } finally {
      setResourcesLoading(false);
    }
  };

  useEffect(() => {
    if (!creating && !selectedName && agentList.length) {
      setSelectedName(agentList[0].name);
    }
  }, [agentList, selectedName, creating]);

  useEffect(() => {
    if (!creating && selected) {
      const next = draftForAgent(selected);
      setDraft(next);
      setValidation(null);
      setNotice("");
      setError("");
      void loadResources(next.workspace);
      void api.listAgentRevisions(selected.name)
        .then((result) => setRevisions(result.revisions))
        .catch(() => setRevisions([]));
    }
  }, [selected?.name, selected?.revision, creating]);

  const beginCreate = () => {
    const next = blankDraft();
    setCreating(true);
    setSelectedName("");
    setDraft(next);
    setValidation(null);
    setRevisions([]);
    setNotice("");
    setError("");
    void loadResources(next.workspace);
  };

  const chooseAgent = (agent: AgentDefinition) => {
    setCreating(false);
    setSelectedName(agent.name);
  };

  const updateDraft = <K extends keyof AgentDraft>(field: K, value: AgentDraft[K]) =>
    setDraft((current) => ({ ...current, [field]: value }));

  const toggleSkill = (id: string, checked: boolean) => {
    updateDraft(
      "skills",
      checked
        ? [...new Set([...draft.skills, id])]
        : draft.skills.filter((item) => item !== id),
    );
  };

  const toggleTool = (name: string, checked: boolean) => {
    updateDraft(
      "tools",
      checked
        ? [...new Set([...draft.tools, name])]
        : draft.tools.filter((item) => item !== name),
    );
  };

  const payload = (): Record<string, unknown> => ({
    description: draft.description.trim(),
    profile: draft.profile || null,
    model: draft.model.trim() || null,
    prompt: draft.prompt || null,
    skills: draft.skills,
    tools: draft.allTools ? null : draft.tools,
    workspace: draft.workspace.trim() || null,
    max_iterations: parseMaxIterations(draft.maxIterations),
    labels: parseLabels(draft.labels),
    capabilities: commaList(draft.capabilities),
  });

  const save = async () => {
    setSaving(true);
    setError("");
    setNotice("");
    try {
      if (!draft.name.trim()) throw new Error("Agent name is required.");
      let saved: AgentDefinition;
      if (creating) {
        saved = await api.createAgent({ name: draft.name.trim(), ...payload() });
        setCreating(false);
        setSelectedName(saved.name);
        setNotice(`Created ${saved.name}.`);
      } else if (selected) {
        saved = await api.updateAgent(selected.name, payload());
        setNotice(`Saved revision ${saved.revision} of ${saved.name}.`);
      } else {
        return;
      }
      await onRefresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not save agent.");
    } finally {
      setSaving(false);
    }
  };

  const validate = async () => {
    if (!selected) return;
    setValidating(true);
    setError("");
    try {
      const result = await api.validateAgent(selected.name);
      setValidation(result);
      setNotice(result.valid ? "Agent definition is ready to run." : "Validation found blocking errors.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not validate agent.");
    } finally {
      setValidating(false);
    }
  };

  const remove = async () => {
    if (!selected || !confirm(`Delete agent “${selected.name}” and all of its revisions?`)) return;
    setError("");
    try {
      await api.deleteAgent(selected.name);
      setSelectedName("");
      setValidation(null);
      await onRefresh();
      setNotice(`Deleted ${selected.name}.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not delete agent.");
    }
  };

  const selectedProfile = profiles?.profiles.find((profile) => profile.name === draft.profile);

  return (
    <div class="management-page agents-page">
      <section class="management-hero">
        <div>
          <p class="eyebrow">Reusable runtimes</p>
          <h1>Agents</h1>
          <p>Compose a profile, prompt, workspace, skills, tools, and run limits into a versioned agent.</p>
        </div>
        <button class="primary-button" onClick={beginCreate}>+ New agent</button>
      </section>

      <div class="agents-layout">
        <aside class="agent-list-panel">
          <div class="profile-list-panel__header">
            <span>{agentList.length} defined</span>
            <button class="text-button" onClick={() => void onRefresh()}>Refresh</button>
          </div>
          {loading && <div class="mini-loading">Loading agents…</div>}
          {agentList.map((agent) => (
            <button
              key={agent.name}
              class={`agent-list-item${selected?.name === agent.name && !creating ? " is-active" : ""}`}
              onClick={() => chooseAgent(agent)}
            >
              <div>
                <strong>{agent.name}</strong>
                <span>r{agent.revision}</span>
              </div>
              <p>{agent.description || "No description"}</p>
              <small>{agent.profile || "default profile"} · {agent.tools === null ? "all tools" : `${agent.tools.length} tools`}</small>
            </button>
          ))}
          {!loading && !agentList.length && (
            <div class="profile-empty">
              <p>No agents defined.</p>
              <button onClick={beginCreate}>Create the first one</button>
            </div>
          )}
          {agents?.root && <p class="store-path">{agents.root}</p>}
        </aside>

        <section class="agent-editor">
          <header class="agent-editor__header">
            <div>
              <p class="eyebrow">{creating ? "New definition" : "Agent definition"}</p>
              <h2>{creating ? "Create an agent" : selected?.name ?? "Select an agent"}</h2>
              {selected && !creating && (
                <small>Revision {selected.revision} · updated {new Date(selected.updated_at).toLocaleString()}</small>
              )}
            </div>
            {!creating && selected && (
              <div class="profile-actions">
                <button onClick={() => void validate()} disabled={validating}>
                  {validating ? "Validating…" : "Validate"}
                </button>
                <button class="run-agent-button" onClick={() => onRun(selected)}>Run in chat</button>
                <button class="danger-button" onClick={() => void remove()}>Delete</button>
              </div>
            )}
          </header>

          {(creating || selected) && (
            <form
              onSubmit={(event) => {
                event.preventDefault();
                void save();
              }}
            >
              <div class="form-grid agent-form-grid">
                <label>
                  <span>Name</span>
                  <input
                    value={draft.name}
                    disabled={!creating}
                    required
                    pattern="[A-Za-z_][A-Za-z0-9_-]*"
                    onInput={(event) => updateDraft("name", event.currentTarget.value)}
                    placeholder="reviewer"
                  />
                </label>
                <label>
                  <span>Profile <small>blank uses default</small></span>
                  <select
                    value={draft.profile}
                    onChange={(event) => updateDraft("profile", event.currentTarget.value)}
                  >
                    <option value="">Default profile</option>
                    {(profiles?.profiles ?? []).map((profile: Profile) => (
                      <option key={profile.name} value={profile.name}>{profile.name}</option>
                    ))}
                  </select>
                </label>
                <label class="form-grid__wide">
                  <span>Description</span>
                  <textarea
                    rows={3}
                    value={draft.description}
                    onInput={(event) => updateDraft("description", event.currentTarget.value)}
                    placeholder="Reviews changes for correctness and maintainability."
                  />
                </label>
                <label>
                  <span>Model override <small>{selectedProfile?.model ? `profile: ${selectedProfile.model}` : "optional"}</small></span>
                  <input
                    value={draft.model}
                    onInput={(event) => updateDraft("model", event.currentTarget.value)}
                    placeholder="Use the profile model"
                  />
                </label>
                <label>
                  <span>System prompt resource <small>optional</small></span>
                  <select
                    value={draft.prompt}
                    onChange={(event) => updateDraft("prompt", event.currentTarget.value)}
                  >
                    <option value="">Active system prompt</option>
                    {prompts.map((prompt) => (
                      <option key={prompt.label} value={prompt.label}>{prompt.label}</option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Workspace</span>
                  <div class="field-action">
                    <input
                      value={draft.workspace}
                      onInput={(event) => updateDraft("workspace", event.currentTarget.value)}
                      placeholder="."
                    />
                    <button type="button" onClick={() => void loadResources(draft.workspace)} disabled={resourcesLoading}>
                      {resourcesLoading ? "…" : "Scan"}
                    </button>
                  </div>
                </label>
                <label>
                  <span>Max iterations <small>blank uses server default; -1 is unlimited</small></span>
                  <input
                    type="number"
                    value={draft.maxIterations}
                    onInput={(event) => updateDraft("maxIterations", event.currentTarget.value)}
                    placeholder="Server default"
                  />
                </label>
              </div>

              <section class="agent-resource-editor">
                <header>
                  <div>
                    <h3>Skills</h3>
                    <p>Loaded into this agent's context on every run.</p>
                  </div>
                  <span>{draft.skills.length} selected</span>
                </header>
                {availableSkills.length ? (
                  <div class="selection-grid">
                    {availableSkills.map((skill) => (
                      <label key={skill.id} title={skill.preview}>
                        <input
                          type="checkbox"
                          checked={draft.skills.includes(skill.id)}
                          disabled={Boolean(skill.error)}
                          onChange={(event) => toggleSkill(skill.id, event.currentTarget.checked)}
                        />
                        <span>{skill.title || skill.label}</span>
                        <small>{skill.scope}</small>
                      </label>
                    ))}
                  </div>
                ) : (
                  <EmptySelection text="No skills discovered for this workspace." />
                )}
              </section>

              <section class="agent-resource-editor">
                <header>
                  <div>
                    <h3>Tools</h3>
                    <p>Use all available tools, or pin an explicit allowlist.</p>
                  </div>
                  <label class="inline-switch">
                    <input
                      type="checkbox"
                      checked={draft.allTools}
                      onChange={(event) => updateDraft("allTools", event.currentTarget.checked)}
                    />
                    All available
                  </label>
                </header>
                {!draft.allTools && (
                  availableTools.length ? (
                    <div class="selection-grid selection-grid--tools">
                      {availableTools.map((tool) => (
                        <label key={`${tool.origin}:${tool.name}`} title={tool.description ?? ""}>
                          <input
                            type="checkbox"
                            checked={draft.tools.includes(tool.name)}
                            onChange={(event) => toggleTool(tool.name, event.currentTarget.checked)}
                          />
                          <span><code>{tool.name}</code></span>
                          <small>{tool.origin}</small>
                        </label>
                      ))}
                    </div>
                  ) : <EmptySelection text="No enabled tools are currently available." />
                )}
                {!draft.allTools && draft.tools.length === 0 && (
                  <p class="no-tools-note">This agent will run in text-only mode with no tools.</p>
                )}
              </section>

              <details class="agent-advanced">
                <summary>Metadata and advanced fields</summary>
                <div class="form-grid">
                  <label>
                    <span>Capabilities <small>comma-separated metadata</small></span>
                    <input
                      value={draft.capabilities}
                      onInput={(event) => updateDraft("capabilities", event.currentTarget.value)}
                      placeholder="review, python"
                    />
                  </label>
                  <label>
                    <span>Labels <small>JSON string mapping</small></span>
                    <textarea
                      rows={5}
                      value={draft.labels}
                      onInput={(event) => updateDraft("labels", event.currentTarget.value)}
                      spellcheck={false}
                    />
                  </label>
                </div>
              </details>

              {validation && (
                <section class={`validation-card validation-card--${validation.valid ? "valid" : "invalid"}`}>
                  <header>
                    <strong>{validation.valid ? "Ready to run" : "Definition is invalid"}</strong>
                    <span>revision {validation.revision}</span>
                  </header>
                  {validation.errors.map((item) => <p key={item} class="validation-error">{item}</p>)}
                  {validation.warnings.map((item) => <p key={item} class="validation-warning">{item}</p>)}
                  <details>
                    <summary>Resolved runtime</summary>
                    <pre>{JSON.stringify(validation.resolved, null, 2)}</pre>
                  </details>
                </section>
              )}

              {revisions.length > 1 && !creating && (
                <details class="revision-history">
                  <summary>{revisions.length} stored revisions</summary>
                  <div>
                    {[...revisions].reverse().map((revision) => (
                      <span key={revision.revision}>
                        <strong>r{revision.revision}</strong>
                        {new Date(revision.updated_at).toLocaleString()}
                      </span>
                    ))}
                  </div>
                </details>
              )}

              {error && <div class="form-message form-message--error">{error}</div>}
              {notice && <div class="form-message form-message--success">{notice}</div>}

              <footer class="form-footer">
                {creating && (
                  <button
                    type="button"
                    onClick={() => {
                      setCreating(false);
                      setSelectedName(agentList[0]?.name ?? "");
                    }}
                  >
                    Cancel
                  </button>
                )}
                <button class="primary-button" type="submit" disabled={saving}>
                  {saving ? "Saving…" : creating ? "Create agent" : "Save as new revision"}
                </button>
              </footer>
            </form>
          )}
        </section>
      </div>
    </div>
  );
}

function EmptySelection({ text }: { text: string }) {
  return <div class="empty-selection">{text}</div>;
}
