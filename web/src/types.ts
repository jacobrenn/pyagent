export interface ChatMessage {
  role: "user" | "assistant" | "tool" | "system";
  content: string;
  tool_calls?: unknown[];
  tool_call_id?: string;
  name?: string;
  tool_name?: string;
  [key: string]: unknown;
}

export interface Profile {
  name: string;
  provider: string;
  api_mode: string;
  model: string;
  base_url: string;
  api_key_env: string | null;
  has_inline_api_key: boolean;
  headers: Record<string, string>;
  redacted_headers: string[];
  httpx_kwargs: Record<string, unknown>;
  is_default: boolean;
}

export interface ProfilesResponse {
  path: string;
  default_profile: string;
  effective_default_profile: string;
  default_overridden_by_env: boolean;
  profiles: Profile[];
}

export interface Skill {
  id: string;
  scope: string;
  label: string;
  title: string;
  preview: string;
  path: string;
  error: string | null;
}

export interface SkillsResponse {
  cwd: string;
  user_dir: string;
  skills: Skill[];
}

export interface ToolInfo {
  name: string;
  origin: string;
  source: string | null;
  description: string | null;
  parameters: Record<string, unknown> | null;
}

export interface ToolFile {
  label: string;
  path: string;
  disabled: boolean;
}

export interface ToolsResponse {
  tools_enabled: boolean;
  builtin_tools_enabled: boolean;
  user_tools_enabled: boolean;
  user_dir: string;
  runner: {
    name: string;
    available: boolean | null;
    message: string | null;
  };
  builtin: ToolInfo[];
  external: ToolInfo[];
  files: ToolFile[];
  broken: { script_path: string; error: string | null }[];
  disabled: { script_path: string }[];
  collisions: { name: string; external_path: string | null }[];
  discovery_error: string | null;
}

export interface ExtensionItem {
  name: string;
  state: "enabled" | "disabled";
  path: string;
}

export interface ExtensionsResponse {
  user_dir: string;
  extensions_dir: string;
  enabled: ExtensionItem[];
  disabled: ExtensionItem[];
}

export interface ResourceItem {
  label: string;
  path: string;
}

export interface ResourceListResponse {
  kind: string;
  root: string;
  items: ResourceItem[];
}

export interface AgentDefinition {
  schema_version: number;
  name: string;
  revision: number;
  created_at: string;
  updated_at: string;
  description: string;
  profile: string | null;
  model: string | null;
  prompt: string | null;
  skills: string[];
  tools: string[] | null;
  workspace: string | null;
  max_iterations: number | null;
  labels: Record<string, string>;
  capabilities: string[];
}

export interface AgentsResponse {
  root: string;
  agents: AgentDefinition[];
}

export interface AgentValidation {
  name: string;
  revision: number;
  valid: boolean;
  errors: string[];
  warnings: string[];
  resolved: Record<string, unknown>;
}

export interface ResourceInstallInput {
  url?: string;
  file?: File;
  name?: string;
  force?: boolean;
}

export interface StreamEvent {
  schema_version?: number;
  run_id?: string;
  sequence?: number | string;
  timestamp?: string;
  type: string;
  delta?: string;
  response?: string;
  message?: string;
  code?: string;
  profile?: string;
  provider?: string;
  api_mode?: string;
  model?: string;
  context_files?: string[];
  messages?: ChatMessage[];
  tool_call_id?: string;
  name?: string;
  arguments?: Record<string, unknown>;
  result?: string;
  is_error?: boolean;
  label?: string;
  data?: unknown;
  agent?: string;
  revision?: number;
  [key: string]: unknown;
}

export interface RunRequest {
  message: string;
  messages: ChatMessage[];
  profile?: string;
  model?: string;
  cwd?: string;
  skills: string[];
}

export type TranscriptEntryKind =
  | "user"
  | "assistant"
  | "tool"
  | "system"
  | "error";

export interface TranscriptEntry {
  id: string;
  kind: TranscriptEntryKind;
  content: string;
  createdAt: string;
  toolCallId?: string;
  toolName?: string;
  arguments?: Record<string, unknown>;
  result?: string;
  toolStatus?: "running" | "complete" | "error";
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  profile: string;
  model: string;
  cwd: string;
  skills: string[];
  agent?: string;
  agentRevision?: number;
  messages: ChatMessage[];
  entries: TranscriptEntry[];
}

export interface RuntimeInfo {
  health: "loading" | "online" | "offline";
  version: string;
}
