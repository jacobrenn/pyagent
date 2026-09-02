import { useEffect, useState } from "preact/hooks";
import { api } from "./api/client";
import { AgentsPage } from "./pages/AgentsPage";
import { ChatPage } from "./pages/ChatPage";
import { ProfilesPage } from "./pages/ProfilesPage";
import { ResourcesPage } from "./pages/ResourcesPage";
import type {
  AgentDefinition,
  AgentsResponse,
  ProfilesResponse,
  RuntimeInfo,
} from "./types";

type Page = "chat" | "profiles" | "resources" | "agents";

function pageFromHash(): Page {
  const route = window.location.hash.split("?", 1)[0];
  if (route === "#/profiles") return "profiles";
  if (route === "#/resources") return "resources";
  if (route === "#/agents") return "agents";
  return "chat";
}

export function App() {
  const [page, setPage] = useState<Page>(pageFromHash());
  const [profiles, setProfiles] = useState<ProfilesResponse | null>(null);
  const [profilesLoading, setProfilesLoading] = useState(true);
  const [profilesError, setProfilesError] = useState("");
  const [agents, setAgents] = useState<AgentsResponse | null>(null);
  const [agentsLoading, setAgentsLoading] = useState(true);
  const [agentsError, setAgentsError] = useState("");
  const [agentLaunch, setAgentLaunch] = useState<{
    name: string;
    revision: number;
    nonce: number;
  } | null>(null);
  const [runtime, setRuntime] = useState<RuntimeInfo>({
    health: "loading",
    version: "…",
  });

  const refreshProfiles = async () => {
    setProfilesLoading(true);
    try {
      setProfiles(await api.listProfiles());
      setProfilesError("");
    } catch (error) {
      setProfilesError(
        error instanceof Error ? error.message : "Could not load profiles.",
      );
    } finally {
      setProfilesLoading(false);
    }
  };

  const refreshAgents = async () => {
    setAgentsLoading(true);
    try {
      setAgents(await api.listAgents());
      setAgentsError("");
    } catch (error) {
      setAgentsError(
        error instanceof Error ? error.message : "Could not load agents.",
      );
    } finally {
      setAgentsLoading(false);
    }
  };

  useEffect(() => {
    const handleHash = () => setPage(pageFromHash());
    window.addEventListener("hashchange", handleHash);
    void refreshProfiles();
    void refreshAgents();
    void Promise.all([api.health(), api.version()])
      .then(([health, version]) =>
        setRuntime({
          health: health.status === "ok" ? "online" : "offline",
          version: version.version,
        }),
      )
      .catch(() => setRuntime({ health: "offline", version: "unknown" }));
    return () => window.removeEventListener("hashchange", handleHash);
  }, []);

  const navigate = (nextPage: Page) => {
    window.location.hash = `#/${nextPage}`;
    setPage(nextPage);
  };

  const runAgent = (agent: AgentDefinition) => {
    setAgentLaunch({
      name: agent.name,
      revision: agent.revision,
      nonce: Date.now(),
    });
    navigate("chat");
  };

  return (
    <div class="app-shell">
      <header class="app-header">
        <button class="brand" onClick={() => navigate("chat")}>
          <span class="brand__mark">P</span>
          <span>
            <strong>PyAgent</strong>
            <small>coding workspace</small>
          </span>
        </button>
        <nav aria-label="Primary navigation">
          <button
            class={page === "chat" ? "is-active" : ""}
            onClick={() => navigate("chat")}
          >
            Chat
          </button>
          <button
            class={page === "agents" ? "is-active" : ""}
            onClick={() => navigate("agents")}
          >
            Agents
          </button>
          <button
            class={page === "resources" ? "is-active" : ""}
            onClick={() => navigate("resources")}
          >
            Resources
          </button>
          <button
            class={page === "profiles" ? "is-active" : ""}
            onClick={() => navigate("profiles")}
          >
            Profiles
          </button>
          <a href="/docs" target="_blank" rel="noreferrer">API</a>
        </nav>
        <div class="runtime-pill" title={`PyAgent ${runtime.version}`}>
          <i class={`runtime-pill__dot runtime-pill__dot--${runtime.health}`} />
          <span>{runtime.health === "online" ? "Server online" : runtime.health}</span>
          <small>v{runtime.version}</small>
        </div>
      </header>

      {(profilesError || agentsError) && (
        <div class="global-error">
          <span>Some server configuration could not be loaded.</span>
          <code>{profilesError || agentsError}</code>
          <button onClick={() => void Promise.all([refreshProfiles(), refreshAgents()])}>Retry</button>
        </div>
      )}

      <main class="app-main">
        {page === "chat" && (
          <ChatPage
            profiles={profiles}
            agents={agents}
            agentLaunch={agentLaunch}
            onAgentLaunchHandled={() => setAgentLaunch(null)}
            onOpenProfiles={() => navigate("profiles")}
          />
        )}
        {page === "agents" && (
          <AgentsPage
            agents={agents}
            profiles={profiles}
            loading={agentsLoading}
            onRefresh={refreshAgents}
            onRun={runAgent}
          />
        )}
        {page === "resources" && <ResourcesPage />}
        {page === "profiles" && (
          <ProfilesPage
            profiles={profiles}
            loading={profilesLoading}
            onRefresh={refreshProfiles}
          />
        )}
      </main>
    </div>
  );
}
