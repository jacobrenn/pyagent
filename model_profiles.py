from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any

from config import AppConfig


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
PROVIDER_ALIASES = {
    "ollama": "ollama",
    "openai": "openai_compatible",
    "openai-compatible": "openai_compatible",
    "openai_compatible": "openai_compatible",
    "vllm": "openai_compatible",
}


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    provider: str
    model: str
    base_url: str
    api_key: str | None = None
    api_key_env: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def resolved_provider(self) -> str:
        return normalize_provider(self.provider)

    def resolved_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            value = os.getenv(self.api_key_env, "").strip()
            if not value:
                raise ValueError(
                    f"Profile '{self.name}' requires environment variable '{self.api_key_env}' for its API key."
                )
            return value
        return None


@dataclass(slots=True)
class ProfileStore:
    path: str
    default_profile: str
    profiles: dict[str, ModelProfile]

    def names(self) -> list[str]:
        return sorted(self.profiles)

    def get(self, name: str | None = None) -> ModelProfile:
        profile_name = name or self.default_profile
        if profile_name not in self.profiles:
            available = ", ".join(self.names()) or "<none>"
            raise ValueError(
                f"Unknown profile '{profile_name}'. Available profiles: {available}"
            )
        return self.profiles[profile_name]


def normalize_provider(provider: str) -> str:
    normalized = PROVIDER_ALIASES.get(provider.strip().lower())
    if normalized is None:
        supported = ", ".join(sorted(PROVIDER_ALIASES))
        raise ValueError(
            f"Unsupported provider '{provider}'. Supported values: {supported}"
        )
    return normalized


def default_base_url_for_provider(provider: str) -> str:
    normalized = normalize_provider(provider)
    return DEFAULT_OLLAMA_BASE_URL if normalized == "ollama" else DEFAULT_OPENAI_BASE_URL


def _profile_from_dict(name: str, data: dict[str, Any]) -> ModelProfile:
    provider = normalize_provider(str(data.get("provider", "ollama")))
    model = str(data.get("model", "")).strip()
    if not model:
        raise ValueError(f"Profile '{name}' must define a non-empty 'model'.")

    base_url = str(data.get("base_url") or default_base_url_for_provider(provider)).strip()
    headers = data.get("headers") or {}
    if not isinstance(headers, dict):
        raise ValueError(f"Profile '{name}' field 'headers' must be an object.")

    return ModelProfile(
        name=name,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=str(data.get("api_key", "")).strip() or None,
        api_key_env=str(data.get("api_key_env", "")).strip() or None,
        headers={str(key): str(value) for key, value in headers.items()},
    )


def _store_from_json(path: str, payload: dict[str, Any]) -> ProfileStore:
    profiles_payload = payload.get("profiles")
    if not isinstance(profiles_payload, dict) or not profiles_payload:
        raise ValueError("Model profile file must contain a non-empty 'profiles' object.")

    profiles = {
        str(name): _profile_from_dict(str(name), profile_data)
        for name, profile_data in profiles_payload.items()
        if isinstance(profile_data, dict)
    }
    if not profiles:
        raise ValueError("Model profile file did not contain any valid profiles.")

    default_profile = str(payload.get("default_profile") or "").strip() or next(iter(profiles))
    if default_profile not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(
            f"Default profile '{default_profile}' was not found. Available profiles: {available}"
        )

    return ProfileStore(path=path, default_profile=default_profile, profiles=profiles)


def _env_profile() -> ModelProfile:
    provider = normalize_provider(os.getenv("PYAGENT_PROVIDER", "ollama"))
    model = os.getenv("PYAGENT_MODEL", "gemma4:31b").strip() or "gemma4:31b"
    default_base_url = default_base_url_for_provider(provider)
    return ModelProfile(
        name="default",
        provider=provider,
        model=model,
        base_url=os.getenv("PYAGENT_BASE_URL", default_base_url).strip() or default_base_url,
        api_key=os.getenv("PYAGENT_API_KEY", "").strip() or None,
        api_key_env=os.getenv("PYAGENT_API_KEY_ENV", "").strip() or None,
        headers={},
    )


def save_profile_store(store: ProfileStore) -> None:
    payload = {
        "default_profile": store.default_profile,
        "profiles": {
            name: _profile_to_dict(store.profiles[name])
            for name in store.names()
        },
    }
    path = Path(os.path.expanduser(store.path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def update_profile_store(
    store: ProfileStore,
    profile: ModelProfile,
    make_default: bool = False,
) -> ProfileStore:
    store.profiles[profile.name] = profile
    if make_default or not store.default_profile:
        store.default_profile = profile.name
    elif store.default_profile not in store.profiles:
        store.default_profile = profile.name
    return store


def _profile_to_dict(profile: ModelProfile) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": profile.provider,
        "model": profile.model,
        "base_url": profile.base_url,
    }
    if profile.api_key:
        payload["api_key"] = profile.api_key
    if profile.api_key_env:
        payload["api_key_env"] = profile.api_key_env
    if profile.headers:
        payload["headers"] = dict(sorted(profile.headers.items()))
    return payload


def load_profile_store(path: str | None = None) -> ProfileStore:
    resolved_path = os.path.expanduser(path or AppConfig().model_profiles_path)
    profile_path = Path(resolved_path)
    if profile_path.is_file():
        try:
            payload = json.loads(profile_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse model profile file {profile_path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Model profile file must contain a JSON object at the top level.")
        return _store_from_json(str(profile_path), payload)

    fallback_profile = _env_profile()
    return ProfileStore(
        path=str(profile_path),
        default_profile=fallback_profile.name,
        profiles={fallback_profile.name: fallback_profile},
    )
