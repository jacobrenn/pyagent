import type { Profile } from "../types";

export interface ProfileDraft {
  name: string;
  provider: string;
  apiMode: string;
  model: string;
  baseUrl: string;
  apiKey: string;
  apiKeyEnv: string;
  headers: Record<string, string>;
  httpxKwargs: Record<string, unknown>;
  makeDefault: boolean;
  clearInlineKey: boolean;
  replaceHeaders: boolean;
  replaceTransport: boolean;
}

export function buildProfileCreatePayload(
  draft: ProfileDraft,
): Record<string, unknown> {
  return {
    name: draft.name.trim(),
    provider: draft.provider,
    api_mode: draft.apiMode,
    model: draft.model.trim(),
    ...(draft.baseUrl.trim() ? { base_url: draft.baseUrl.trim() } : {}),
    ...(draft.apiKey.trim() ? { api_key: draft.apiKey.trim() } : {}),
    ...(draft.apiKeyEnv.trim()
      ? { api_key_env: draft.apiKeyEnv.trim() }
      : {}),
    headers: draft.headers,
    httpx_kwargs: draft.httpxKwargs,
    make_default: draft.makeDefault,
  };
}

export function buildProfileUpdatePayload(
  draft: ProfileDraft,
  existing: Profile,
): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    provider: draft.provider,
    api_mode: draft.apiMode,
    model: draft.model.trim(),
    base_url: draft.baseUrl.trim(),
  };

  if (draft.clearInlineKey) {
    payload.api_key = null;
  } else if (draft.apiKey.trim()) {
    payload.api_key = draft.apiKey.trim();
  }

  const apiKeyEnv = draft.apiKeyEnv.trim();
  if (apiKeyEnv !== (existing.api_key_env ?? "")) {
    payload.api_key_env = apiKeyEnv || null;
  }
  if (draft.replaceHeaders) {
    payload.headers = draft.headers;
  }
  if (draft.replaceTransport) {
    payload.httpx_kwargs = draft.httpxKwargs;
  }
  return payload;
}
