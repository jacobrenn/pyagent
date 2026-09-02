import { describe, expect, it } from "vitest";
import {
  buildProfileUpdatePayload,
  type ProfileDraft,
} from "./profilePayload";
import type { Profile } from "../types";

const profile: Profile = {
  name: "remote",
  provider: "openai_compatible",
  api_mode: "responses",
  model: "gpt-test",
  base_url: "https://example.test/v1",
  api_key_env: "OPENAI_API_KEY",
  has_inline_api_key: true,
  headers: { "X-Project": "PyAgent" },
  redacted_headers: ["Authorization"],
  httpx_kwargs: { verify: false },
  is_default: false,
};

const draft: ProfileDraft = {
  name: profile.name,
  provider: profile.provider,
  apiMode: profile.api_mode,
  model: profile.model,
  baseUrl: profile.base_url,
  apiKey: "",
  apiKeyEnv: profile.api_key_env ?? "",
  headers: profile.headers,
  httpxKwargs: profile.httpx_kwargs,
  makeDefault: false,
  clearInlineKey: false,
  replaceHeaders: false,
  replaceTransport: false,
};

describe("buildProfileUpdatePayload", () => {
  it("preserves hidden secrets and mappings by omitting them", () => {
    const payload = buildProfileUpdatePayload(draft, profile);
    expect(payload).not.toHaveProperty("api_key");
    expect(payload).not.toHaveProperty("api_key_env");
    expect(payload).not.toHaveProperty("headers");
    expect(payload).not.toHaveProperty("httpx_kwargs");
  });

  it("clears credentials only when requested", () => {
    const payload = buildProfileUpdatePayload(
      { ...draft, clearInlineKey: true, apiKeyEnv: "" },
      profile,
    );
    expect(payload.api_key).toBeNull();
    expect(payload.api_key_env).toBeNull();
  });

  it("replaces advanced mappings explicitly", () => {
    const payload = buildProfileUpdatePayload(
      {
        ...draft,
        replaceHeaders: true,
        replaceTransport: true,
        headers: { "X-New": "yes" },
        httpxKwargs: { verify: true },
      },
      profile,
    );
    expect(payload.headers).toEqual({ "X-New": "yes" });
    expect(payload.httpx_kwargs).toEqual({ verify: true });
  });
});
