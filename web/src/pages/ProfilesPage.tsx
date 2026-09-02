import { useEffect, useMemo, useState } from "preact/hooks";
import { api } from "../api/client";
import {
  buildProfileCreatePayload,
  buildProfileUpdatePayload,
  type ProfileDraft,
} from "../api/profilePayload";
import type { Profile, ProfilesResponse } from "../types";

interface ProfilesPageProps {
  profiles: ProfilesResponse | null;
  loading: boolean;
  onRefresh: () => Promise<void>;
}

function blankDraft(): ProfileDraft {
  return {
    name: "",
    provider: "ollama",
    apiMode: "chat_completions",
    model: "",
    baseUrl: "",
    apiKey: "",
    apiKeyEnv: "",
    headers: {},
    httpxKwargs: {},
    makeDefault: false,
    clearInlineKey: false,
    replaceHeaders: true,
    replaceTransport: true,
  };
}

function draftForProfile(profile: Profile): ProfileDraft {
  return {
    name: profile.name,
    provider: profile.provider,
    apiMode: profile.api_mode,
    model: profile.model,
    baseUrl: profile.base_url,
    apiKey: "",
    apiKeyEnv: profile.api_key_env ?? "",
    headers: profile.headers,
    httpxKwargs: profile.httpx_kwargs,
    makeDefault: profile.is_default,
    clearInlineKey: false,
    replaceHeaders: false,
    replaceTransport: false,
  };
}

function parseObject(value: string, label: string): Record<string, unknown> {
  const parsed: unknown = JSON.parse(value || "{}");
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`${label} must be a JSON object.`);
  }
  return parsed as Record<string, unknown>;
}

export function ProfilesPage({
  profiles,
  loading,
  onRefresh,
}: ProfilesPageProps) {
  const profileList = profiles?.profiles ?? [];
  const [selectedName, setSelectedName] = useState("");
  const [creating, setCreating] = useState(false);
  const [draft, setDraft] = useState<ProfileDraft>(blankDraft());
  const [headersText, setHeadersText] = useState("{}");
  const [transportText, setTransportText] = useState("{}");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [models, setModels] = useState<string[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);

  const selected = useMemo(
    () => profileList.find((profile) => profile.name === selectedName) ?? null,
    [profileList, selectedName],
  );

  useEffect(() => {
    if (!creating && !selectedName && profileList.length) {
      setSelectedName(profileList[0].name);
    }
  }, [profileList, selectedName, creating]);

  useEffect(() => {
    if (!creating && selected) {
      const nextDraft = draftForProfile(selected);
      setDraft(nextDraft);
      setHeadersText(JSON.stringify(nextDraft.headers, null, 2));
      setTransportText(JSON.stringify(nextDraft.httpxKwargs, null, 2));
      setModels([]);
      setError("");
      setNotice("");
    }
  }, [selected?.name, creating]);

  const beginCreate = () => {
    const next = blankDraft();
    setCreating(true);
    setSelectedName("");
    setDraft(next);
    setHeadersText("{}");
    setTransportText("{}");
    setModels([]);
    setError("");
    setNotice("");
  };

  const chooseProfile = (profile: Profile) => {
    setCreating(false);
    setSelectedName(profile.name);
  };

  const updateDraft = <K extends keyof ProfileDraft>(
    field: K,
    value: ProfileDraft[K],
  ) => setDraft((current) => ({ ...current, [field]: value }));

  const save = async () => {
    setError("");
    setNotice("");
    setSaving(true);
    try {
      const headers = parseObject(headersText, "Headers");
      if (Object.values(headers).some((value) => typeof value !== "string")) {
        throw new Error("Every header value must be a string.");
      }
      const finalizedDraft: ProfileDraft = {
        ...draft,
        headers: headers as Record<string, string>,
        httpxKwargs: parseObject(transportText, "HTTP transport options"),
      };
      if (creating) {
        await api.createProfile(buildProfileCreatePayload(finalizedDraft));
        setSelectedName(finalizedDraft.name.trim());
        setCreating(false);
        setNotice(`Created ${finalizedDraft.name.trim()}.`);
      } else if (selected) {
        await api.updateProfile(
          selected.name,
          buildProfileUpdatePayload(finalizedDraft, selected),
        );
        setNotice(`Saved ${selected.name}.`);
      }
      await onRefresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not save profile.");
    } finally {
      setSaving(false);
    }
  };

  const setDefault = async () => {
    if (!selected) return;
    setError("");
    try {
      await api.setDefaultProfile(selected.name);
      await onRefresh();
      setNotice(`${selected.name} is now the stored default.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not set default.");
    }
  };

  const remove = async () => {
    if (!selected || !confirm(`Delete profile “${selected.name}”?`)) return;
    setError("");
    try {
      await api.deleteProfile(selected.name);
      setSelectedName("");
      await onRefresh();
      setNotice(`Deleted ${selected.name}.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not delete profile.");
    }
  };

  const discoverModels = async () => {
    if (!selected) return;
    setModelsLoading(true);
    setError("");
    try {
      const response = await api.listProfileModels(selected.name);
      setModels(response.models);
      setNotice(
        response.models.length
          ? `Found ${response.models.length} available models.`
          : "The endpoint did not report any models.",
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not list models.");
    } finally {
      setModelsLoading(false);
    }
  };

  return (
    <div class="profiles-page">
      <section class="profiles-hero">
        <div>
          <p class="eyebrow">Model routing</p>
          <h1>Profiles</h1>
          <p>Configure local and remote model endpoints without leaving PyAgent.</p>
        </div>
        <button class="primary-button" onClick={beginCreate}>+ New profile</button>
      </section>

      {profiles?.default_overridden_by_env && (
        <div class="env-banner">
          <span>Environment override</span>
          <p>
            <code>PYAGENT_PROFILE</code> currently selects <strong>{profiles.effective_default_profile}</strong>.
            Changing the stored default will not override it.
          </p>
        </div>
      )}

      <div class="profiles-layout">
        <aside class="profile-list-panel">
          <div class="profile-list-panel__header">
            <span>{profileList.length} configured</span>
            <button class="text-button" onClick={() => void onRefresh()}>Refresh</button>
          </div>
          {loading && <div class="mini-loading">Loading profiles…</div>}
          {profileList.map((profile) => (
            <button
              key={profile.name}
              class={`profile-list-item${selected?.name === profile.name && !creating ? " is-active" : ""}`}
              onClick={() => chooseProfile(profile)}
            >
              <div>
                <strong>{profile.name}</strong>
                {profile.is_default && <span class="badge badge--default">Default</span>}
              </div>
              <span>{profile.model}</span>
              <small>{profile.provider} · {profile.api_mode}</small>
            </button>
          ))}
          {!loading && !profileList.length && (
            <div class="profile-empty">
              <p>No profiles configured.</p>
              <button onClick={beginCreate}>Create the first one</button>
            </div>
          )}
        </aside>

        <section class="profile-editor">
          <header class="profile-editor__header">
            <div>
              <p class="eyebrow">{creating ? "New connection" : "Connection details"}</p>
              <h2>{creating ? "Create a profile" : selected?.name ?? "Select a profile"}</h2>
            </div>
            {!creating && selected && (
              <div class="profile-actions">
                <button onClick={() => void discoverModels()} disabled={modelsLoading}>
                  {modelsLoading ? "Discovering…" : "Discover models"}
                </button>
                {!selected.is_default && <button onClick={() => void setDefault()}>Set default</button>}
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
              <div class="form-grid">
                <label>
                  <span>Name</span>
                  <input
                    value={draft.name}
                    disabled={!creating}
                    required
                    onInput={(event) => updateDraft("name", event.currentTarget.value)}
                    placeholder="local-qwen"
                  />
                </label>
                <label>
                  <span>Provider</span>
                  <select
                    value={draft.provider}
                    onChange={(event) => {
                      const provider = event.currentTarget.value;
                      setDraft((current) => ({
                        ...current,
                        provider,
                        apiMode:
                          provider === "ollama"
                            ? "chat_completions"
                            : current.apiMode,
                      }));
                    }}
                  >
                    <option value="ollama">Ollama</option>
                    <option value="openai">OpenAI</option>
                    <option value="openai_compatible">OpenAI compatible</option>
                    <option value="vllm">vLLM</option>
                  </select>
                </label>
                <label>
                  <span>API mode</span>
                  <select
                    value={draft.apiMode}
                    onChange={(event) => updateDraft("apiMode", event.currentTarget.value)}
                  >
                    <option value="chat_completions">Chat Completions</option>
                    <option value="responses" disabled={draft.provider === "ollama"}>Responses</option>
                  </select>
                </label>
                <label>
                  <span>Model</span>
                  <input
                    value={draft.model}
                    list="profile-model-list"
                    required
                    onInput={(event) => updateDraft("model", event.currentTarget.value)}
                    placeholder="qwen2.5-coder:7b"
                  />
                  <datalist id="profile-model-list">
                    {models.map((model) => <option key={model} value={model} />)}
                  </datalist>
                </label>
                <label class="form-grid__wide">
                  <span>Base URL <small>{creating ? "optional" : ""}</small></span>
                  <input
                    value={draft.baseUrl}
                    onInput={(event) => updateDraft("baseUrl", event.currentTarget.value)}
                    placeholder={draft.provider === "ollama" ? "http://localhost:11434" : "https://api.openai.com/v1"}
                  />
                </label>
                <label>
                  <span>API key environment variable</span>
                  <input
                    value={draft.apiKeyEnv}
                    onInput={(event) => updateDraft("apiKeyEnv", event.currentTarget.value)}
                    placeholder="OPENAI_API_KEY"
                  />
                </label>
                <label>
                  <span>Inline API key {selected?.has_inline_api_key && <small>currently set</small>}</span>
                  <input
                    type="password"
                    value={draft.apiKey}
                    onInput={(event) => updateDraft("apiKey", event.currentTarget.value)}
                    placeholder={selected?.has_inline_api_key ? "Leave blank to preserve" : "Not recommended"}
                  />
                </label>
              </div>

              {!creating && selected?.has_inline_api_key && (
                <label class="check-row">
                  <input
                    type="checkbox"
                    checked={draft.clearInlineKey}
                    onChange={(event) => updateDraft("clearInlineKey", event.currentTarget.checked)}
                  />
                  Clear the stored inline API key
                </label>
              )}

              <div class="advanced-fields">
                <details>
                  <summary>Headers</summary>
                  {!creating && selected && (
                    <label class="check-row">
                      <input
                        type="checkbox"
                        checked={draft.replaceHeaders}
                        onChange={(event) => updateDraft("replaceHeaders", event.currentTarget.checked)}
                      />
                      Replace the complete header mapping
                    </label>
                  )}
                  {selected?.redacted_headers.length ? (
                    <p class="field-warning">
                      Hidden values: {selected.redacted_headers.join(", ")}. Replacing headers removes hidden values unless they are entered again.
                    </p>
                  ) : null}
                  <textarea
                    rows={7}
                    value={headersText}
                    disabled={!creating && !draft.replaceHeaders}
                    onInput={(event) => setHeadersText(event.currentTarget.value)}
                    spellcheck={false}
                  />
                </details>
                <details>
                  <summary>HTTP transport options</summary>
                  {!creating && selected && (
                    <label class="check-row">
                      <input
                        type="checkbox"
                        checked={draft.replaceTransport}
                        onChange={(event) => updateDraft("replaceTransport", event.currentTarget.checked)}
                      />
                      Replace the complete transport mapping
                    </label>
                  )}
                  {transportText.includes("<redacted>") && (
                    <p class="field-warning">
                      This mapping contains hidden values. Re-enter them before replacing the mapping.
                    </p>
                  )}
                  <textarea
                    rows={7}
                    value={transportText}
                    disabled={!creating && !draft.replaceTransport}
                    onInput={(event) => setTransportText(event.currentTarget.value)}
                    spellcheck={false}
                  />
                </details>
              </div>

              {creating && (
                <label class="check-row">
                  <input
                    type="checkbox"
                    checked={draft.makeDefault}
                    onChange={(event) => updateDraft("makeDefault", event.currentTarget.checked)}
                  />
                  Make this the stored default profile
                </label>
              )}

              {error && <div class="form-message form-message--error">{error}</div>}
              {notice && <div class="form-message form-message--success">{notice}</div>}

              <footer class="form-footer">
                {creating && (
                  <button
                    type="button"
                    onClick={() => {
                      setCreating(false);
                      setSelectedName(profileList[0]?.name ?? "");
                    }}
                  >
                    Cancel
                  </button>
                )}
                <button class="primary-button" type="submit" disabled={saving}>
                  {saving ? "Saving…" : creating ? "Create profile" : "Save changes"}
                </button>
              </footer>
            </form>
          )}
        </section>
      </div>
      {profiles?.path && <p class="profile-path">Profile store · {profiles.path}</p>}
    </div>
  );
}
