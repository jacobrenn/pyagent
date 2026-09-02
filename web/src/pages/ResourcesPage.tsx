import { useEffect, useMemo, useState } from "preact/hooks";
import { api } from "../api/client";
import type {
  ExtensionsResponse,
  ResourceInstallInput,
  Skill,
  SkillsResponse,
  ToolsResponse,
} from "../types";

type ResourceTab = "tools" | "skills" | "extensions";
type SourceMode = "url" | "file";

function resourceTitle(tab: ResourceTab): string {
  if (tab === "tools") return "Tool";
  if (tab === "skills") return "Skill";
  return "Extension";
}

export function ResourcesPage() {
  const [tab, setTab] = useState<ResourceTab>("tools");
  const [tools, setTools] = useState<ToolsResponse | null>(null);
  const [skills, setSkills] = useState<SkillsResponse | null>(null);
  const [extensions, setExtensions] = useState<ExtensionsResponse | null>(null);
  const [cwd, setCwd] = useState(".");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [sourceMode, setSourceMode] = useState<SourceMode>("url");
  const [sourceUrl, setSourceUrl] = useState("");
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [installName, setInstallName] = useState("");
  const [force, setForce] = useState(false);

  const refresh = async (skillCwd = cwd) => {
    setLoading(true);
    try {
      const [nextTools, nextSkills, nextExtensions] = await Promise.all([
        api.listTools(),
        api.listSkills(skillCwd || "."),
        api.listExtensions(),
      ]);
      setTools(nextTools);
      setSkills(nextSkills);
      setExtensions(nextExtensions);
      setError("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load resources.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  const runAction = async (label: string, action: () => Promise<unknown>) => {
    setBusy(label);
    setError("");
    setNotice("");
    try {
      const result = await action() as { message?: string };
      setNotice(result?.message ?? `${label} complete.`);
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : `${label} failed.`);
    } finally {
      setBusy("");
    }
  };

  const install = async () => {
    if (sourceMode === "url" && !sourceUrl.trim()) {
      setError("Enter an HTTP(S) URL to install.");
      return;
    }
    if (sourceMode === "file" && !sourceFile) {
      setError("Choose a file to upload.");
      return;
    }
    const input: ResourceInstallInput = {
      ...(sourceMode === "url" ? { url: sourceUrl } : { file: sourceFile ?? undefined }),
      name: installName,
      force,
    };
    await runAction(`Install ${resourceTitle(tab).toLowerCase()}`, async () => {
      if (tab === "tools") return api.installTool(input);
      if (tab === "skills") return api.installSkill(input);
      return api.installExtension(input);
    });
    setSourceUrl("");
    setSourceFile(null);
  };

  const scaffold = async () => {
    if (!installName.trim()) {
      setError(`Enter a name for the new ${resourceTitle(tab).toLowerCase()}.`);
      return;
    }
    await runAction(`Create ${resourceTitle(tab).toLowerCase()}`, () =>
      tab === "tools"
        ? api.scaffoldTool(installName.trim())
        : api.scaffoldExtension(installName.trim()),
    );
  };

  const installedSkillCount = useMemo(
    () => skills?.skills.filter((skill) => skill.scope === "user").length ?? 0,
    [skills],
  );

  return (
    <div class="management-page">
      <section class="management-hero">
        <div>
          <p class="eyebrow">Runtime library</p>
          <h1>Resources</h1>
          <p>Install and control the tools, skills, and extensions available to agents.</p>
        </div>
        <button class="secondary-button" onClick={() => void refresh()} disabled={loading}>
          {loading ? "Refreshing…" : "Refresh inventory"}
        </button>
      </section>

      <div class="resource-tabs" role="tablist" aria-label="Resource type">
        {(["tools", "skills", "extensions"] as ResourceTab[]).map((item) => (
          <button
            key={item}
            role="tab"
            aria-selected={tab === item}
            class={tab === item ? "is-active" : ""}
            onClick={() => {
              setTab(item);
              setError("");
              setNotice("");
            }}
          >
            <span>{item}</span>
            <small>
              {item === "tools"
                ? tools?.files.length ?? 0
                : item === "skills"
                  ? installedSkillCount
                  : (extensions?.enabled.length ?? 0) + (extensions?.disabled.length ?? 0)}
            </small>
          </button>
        ))}
      </div>

      <div class="resource-layout">
        <aside class="install-panel">
          <header>
            <p class="eyebrow">Add to library</p>
            <h2>Install {resourceTitle(tab).toLowerCase()}</h2>
          </header>
          <div class="source-toggle">
            <button
              class={sourceMode === "url" ? "is-active" : ""}
              onClick={() => setSourceMode("url")}
            >
              From URL
            </button>
            <button
              class={sourceMode === "file" ? "is-active" : ""}
              onClick={() => setSourceMode("file")}
            >
              Upload file
            </button>
          </div>

          {sourceMode === "url" ? (
            <label>
              <span>HTTP(S) URL</span>
              <input
                type="url"
                value={sourceUrl}
                onInput={(event) => setSourceUrl(event.currentTarget.value)}
                placeholder={
                  tab === "extensions"
                    ? "Git repository, .py, or .zip URL"
                    : `https://example.com/${tab === "tools" ? "tool.py" : "skill.md"}`
                }
              />
            </label>
          ) : (
            <label>
              <span>{tab === "extensions" ? "Python file or zip package" : "Local file"}</span>
              <input
                type="file"
                accept={
                  tab === "tools"
                    ? ".py,text/x-python"
                    : tab === "skills"
                      ? ".md,.skill,text/markdown,text/plain"
                      : ".py,.zip,text/x-python,application/zip"
                }
                onChange={(event) => setSourceFile(event.currentTarget.files?.[0] ?? null)}
              />
            </label>
          )}

          <label>
            <span>
              Installed name
              <small>{tab === "extensions" && sourceMode === "url" ? "required for Git URLs" : "optional"}</small>
            </span>
            <input
              value={installName}
              onInput={(event) => setInstallName(event.currentTarget.value)}
              placeholder={tab === "tools" ? "review_tool.py" : tab === "skills" ? "review.md" : "review_guard"}
            />
          </label>
          <label class="check-row">
            <input
              type="checkbox"
              checked={force}
              onChange={(event) => setForce(event.currentTarget.checked)}
            />
            Replace an existing installed resource
          </label>
          <button class="primary-button install-submit" onClick={() => void install()} disabled={Boolean(busy)}>
            {busy.startsWith("Install") ? "Installing…" : `Install ${resourceTitle(tab).toLowerCase()}`}
          </button>

          {tab !== "skills" && (
            <div class="scaffold-box">
              <span>Starting from scratch?</span>
              <p>Create a working starter {resourceTitle(tab).toLowerCase()} using the name above.</p>
              <button onClick={() => void scaffold()} disabled={Boolean(busy)}>
                Scaffold new {resourceTitle(tab).toLowerCase()}
              </button>
            </div>
          )}

          <p class="trust-note">
            Installed tools and extensions are executable code. Only install sources you trust.
          </p>
        </aside>

        <section class="inventory-panel">
          {error && <div class="form-message form-message--error">{error}</div>}
          {notice && <div class="form-message form-message--success">{notice}</div>}
          {loading && !tools && <div class="page-inline-loading">Loading resource inventory…</div>}
          {tab === "tools" && tools && (
            <ToolsInventory
              tools={tools}
              busy={busy}
              onAction={runAction}
            />
          )}
          {tab === "skills" && skills && (
            <SkillsInventory
              skills={skills.skills}
              cwd={cwd}
              busy={busy}
              onCwdChange={setCwd}
              onDiscover={() => void refresh(cwd)}
              onAction={runAction}
            />
          )}
          {tab === "extensions" && extensions && (
            <ExtensionsInventory
              extensions={extensions}
              busy={busy}
              onAction={runAction}
            />
          )}
        </section>
      </div>
    </div>
  );
}

interface ActionProps {
  busy: string;
  onAction: (label: string, action: () => Promise<unknown>) => Promise<void>;
}

function ToolsInventory({ tools, busy, onAction }: { tools: ToolsResponse } & ActionProps) {
  const externalByPath = new Map(
    tools.external
      .filter((tool) => tool.source)
      .map((tool) => [tool.source as string, tool]),
  );
  return (
    <>
      <InventoryHeader
        title="Tools"
        count={tools.files.length}
        meta={`${tools.builtin.length} built in · runner ${tools.runner.name} ${tools.runner.available === false ? "unavailable" : "ready"}`}
      />
      {tools.runner.message && <div class="inventory-warning">{tools.runner.message}</div>}
      {tools.discovery_error && <div class="inventory-warning">{tools.discovery_error}</div>}
      <div class="inventory-section">
        <h3>User-managed scripts</h3>
        {!tools.files.length && <EmptyInventory text="No user tools installed yet." />}
        {tools.files.map((file) => {
          const loaded = externalByPath.get(file.path);
          const problem = tools.broken.find((item) => item.script_path === file.path);
          return (
            <article class="resource-row" key={file.path}>
              <div class={`resource-icon resource-icon--${file.disabled ? "disabled" : problem ? "warning" : "tool"}`}>T</div>
              <div class="resource-row__body">
                <div>
                  <strong>{file.label.replace(/^disabled\//, "")}</strong>
                  <StatusBadge state={file.disabled ? "disabled" : problem ? "broken" : loaded ? "loaded" : "installed"} />
                </div>
                <p>{loaded?.description ?? problem?.error ?? file.path}</p>
              </div>
              <div class="row-actions">
                <button
                  onClick={() => void onAction(file.disabled ? "Enable tool" : "Disable tool", () =>
                    file.disabled ? api.enableTool(file.label) : api.disableTool(file.label),
                  )}
                  disabled={Boolean(busy)}
                >
                  {file.disabled ? "Enable" : "Disable"}
                </button>
                <button
                  class="danger-button"
                  onClick={() => {
                    if (confirm(`Delete tool “${file.label}”?`)) {
                      void onAction("Delete tool", () => api.deleteTool(file.label));
                    }
                  }}
                  disabled={Boolean(busy)}
                >
                  Delete
                </button>
              </div>
            </article>
          );
        })}
      </div>
      <details class="builtin-inventory">
        <summary>{tools.builtin.length} built-in tools</summary>
        <div class="compact-resource-grid">
          {tools.builtin.map((tool) => (
            <div key={tool.name} title={tool.description ?? ""}>
              <code>{tool.name}</code>
              <span>{tool.description}</span>
            </div>
          ))}
        </div>
      </details>
      {tools.collisions.length > 0 && (
        <div class="inventory-warning">
          Name collisions: {tools.collisions.map((item) => item.name).join(", ")}. Built-ins take precedence.
        </div>
      )}
    </>
  );
}

function SkillsInventory({
  skills,
  cwd,
  busy,
  onCwdChange,
  onDiscover,
  onAction,
}: {
  skills: Skill[];
  cwd: string;
  onCwdChange: (cwd: string) => void;
  onDiscover: () => void;
} & ActionProps) {
  return (
    <>
      <InventoryHeader
        title="Skills"
        count={skills.length}
        meta="User skills are manageable; project skills are read-only"
      />
      <div class="inventory-filter">
        <label>
          <span>Project directory</span>
          <input value={cwd} onInput={(event) => onCwdChange(event.currentTarget.value)} />
        </label>
        <button onClick={onDiscover}>Discover</button>
      </div>
      {!skills.length && <EmptyInventory text="No skills discovered for this directory." />}
      {skills.map((skill) => (
        <article class="resource-row" key={skill.id}>
          <div class={`resource-icon resource-icon--${skill.scope === "user" ? "skill" : "project"}`}>S</div>
          <div class="resource-row__body">
            <div>
              <strong>{skill.title || skill.label}</strong>
              <StatusBadge state={skill.scope} />
            </div>
            <p>{skill.error || skill.preview || skill.path}</p>
            <code>{skill.id}</code>
          </div>
          {skill.scope === "user" && (
            <div class="row-actions">
              <button
                class="danger-button"
                onClick={() => {
                  if (confirm(`Delete skill “${skill.label}”?`)) {
                    void onAction("Delete skill", () => api.deleteSkill(skill.label));
                  }
                }}
                disabled={Boolean(busy)}
              >
                Delete
              </button>
            </div>
          )}
        </article>
      ))}
    </>
  );
}

function ExtensionsInventory({
  extensions,
  busy,
  onAction,
}: { extensions: ExtensionsResponse } & ActionProps) {
  const items = [...extensions.enabled, ...extensions.disabled];
  return (
    <>
      <InventoryHeader
        title="Extensions"
        count={items.length}
        meta="Enabled extensions auto-load in each new agent run"
      />
      {!items.length && <EmptyInventory text="No extensions installed yet." />}
      {items.map((extension) => (
        <article class="resource-row" key={`${extension.state}:${extension.name}`}>
          <div class={`resource-icon resource-icon--${extension.state}`}>E</div>
          <div class="resource-row__body">
            <div>
              <strong>{extension.name}</strong>
              <StatusBadge state={extension.state} />
            </div>
            <p>{extension.path}</p>
          </div>
          <div class="row-actions">
            <button
              onClick={() => void onAction(
                extension.state === "enabled" ? "Disable extension" : "Enable extension",
                () => extension.state === "enabled"
                  ? api.disableExtension(extension.name)
                  : api.enableExtension(extension.name),
              )}
              disabled={Boolean(busy)}
            >
              {extension.state === "enabled" ? "Disable" : "Enable"}
            </button>
            <button
              class="danger-button"
              onClick={() => {
                if (confirm(`Delete extension “${extension.name}”?`)) {
                  void onAction("Delete extension", () => api.deleteExtension(extension.name));
                }
              }}
              disabled={Boolean(busy)}
            >
              Delete
            </button>
          </div>
        </article>
      ))}
      <p class="inventory-footnote">
        Enable/disable changes apply to new API runs. Active runs keep their already-loaded extension bus.
      </p>
    </>
  );
}

function InventoryHeader({ title, count, meta }: { title: string; count: number; meta: string }) {
  return (
    <header class="inventory-header">
      <div>
        <p class="eyebrow">Inventory</p>
        <h2>{title}</h2>
      </div>
      <div>
        <strong>{count}</strong>
        <span>{meta}</span>
      </div>
    </header>
  );
}

function StatusBadge({ state }: { state: string }) {
  return <span class={`status-badge status-badge--${state}`}>{state}</span>;
}

function EmptyInventory({ text }: { text: string }) {
  return <div class="empty-inventory">{text}</div>;
}
