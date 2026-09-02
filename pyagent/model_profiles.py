from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .config import AppConfig


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_API_MODE = "chat_completions"
API_MODE_ALIASES = {
    "chat.completions": DEFAULT_API_MODE,
    "chat-completions": DEFAULT_API_MODE,
    "chat_completions": DEFAULT_API_MODE,
    "responses": "responses",
}
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
    httpx_kwargs: dict[str, Any] = field(default_factory=dict)
    api_mode: str = DEFAULT_API_MODE

    def resolved_provider(self) -> str:
        return normalize_provider(self.provider)

    def resolved_api_mode(self) -> str:
        api_mode = normalize_api_mode(self.api_mode)
        if self.resolved_provider() == "ollama" and api_mode != DEFAULT_API_MODE:
            raise ValueError(
                f"API mode '{api_mode}' cannot be used with provider 'ollama'."
            )
        return api_mode

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


def normalize_api_mode(api_mode: str) -> str:
    normalized = API_MODE_ALIASES.get(api_mode.strip().lower())
    if normalized is None:
        supported = ", ".join(sorted(set(API_MODE_ALIASES.values())))
        raise ValueError(
            f"Unsupported API mode '{api_mode}'. Supported values: {supported}"
        )
    return normalized


def default_base_url_for_provider(provider: str) -> str:
    normalized = normalize_provider(provider)
    return DEFAULT_OLLAMA_BASE_URL if normalized == "ollama" else DEFAULT_OPENAI_BASE_URL


def _read_optional_object_field(data: dict[str, Any], *names: str) -> dict[str, Any]:
    for field_name in names:
        value = data.get(field_name)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(
                f"Profile field '{field_name}' must be an object.")
        return value
    return {}


def _profile_from_dict(name: str, data: dict[str, Any]) -> ModelProfile:
    provider = normalize_provider(str(data.get("provider", "ollama")))
    model = str(data.get("model", "")).strip()
    if not model:
        raise ValueError(f"Profile '{name}' must define a non-empty 'model'.")

    base_url = str(data.get("base_url")
                   or default_base_url_for_provider(provider)).strip()
    api_mode = normalize_api_mode(
        str(data.get("api_mode", DEFAULT_API_MODE)))
    if provider == "ollama" and api_mode != DEFAULT_API_MODE:
        raise ValueError(
            f"Profile '{name}' cannot use API mode '{api_mode}' with provider 'ollama'."
        )
    headers = _read_optional_object_field(data, "headers")
    httpx_kwargs = _read_optional_object_field(
        data, "httpx_kwargs", "http_kwargs")

    return ModelProfile(
        name=name,
        provider=provider,
        model=model,
        base_url=base_url,
        api_mode=api_mode,
        api_key=str(data.get("api_key", "")).strip() or None,
        api_key_env=str(data.get("api_key_env", "")).strip() or None,
        headers={str(key): str(value) for key, value in headers.items()},
        httpx_kwargs=dict(httpx_kwargs),
    )


def _store_from_json(path: str, payload: dict[str, Any]) -> ProfileStore:
    profiles_payload = payload.get("profiles")
    if not isinstance(profiles_payload, dict) or not profiles_payload:
        raise ValueError(
            "Model profile file must contain a non-empty 'profiles' object.")

    profiles = {
        str(name): _profile_from_dict(str(name), profile_data)
        for name, profile_data in profiles_payload.items()
        if isinstance(profile_data, dict)
    }
    if not profiles:
        raise ValueError(
            "Model profile file did not contain any valid profiles.")

    default_profile = str(payload.get("default_profile")
                          or "").strip() or next(iter(profiles))
    if default_profile not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(
            f"Default profile '{default_profile}' was not found. Available profiles: {available}"
        )

    return ProfileStore(path=path, default_profile=default_profile, profiles=profiles)


def _env_profile() -> ModelProfile:
    provider = normalize_provider(os.getenv("PYAGENT_PROVIDER", "ollama"))
    api_mode = normalize_api_mode(
        os.getenv("PYAGENT_API_MODE", DEFAULT_API_MODE))
    if provider == "ollama" and api_mode != DEFAULT_API_MODE:
        raise ValueError(
            f"API mode '{api_mode}' cannot be used with provider 'ollama'."
        )
    model = os.getenv(
        "PYAGENT_MODEL", "gemma4:latest").strip() or "gemma4:latest"
    default_base_url = default_base_url_for_provider(provider)
    return ModelProfile(
        name="default",
        provider=provider,
        model=model,
        base_url=os.getenv("PYAGENT_BASE_URL",
                           default_base_url).strip() or default_base_url,
        api_mode=api_mode,
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
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            json.dump(payload, temporary_file, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, path)
    except OSError:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def validate_model_profile(profile: ModelProfile) -> None:
    if not profile.name.strip():
        raise ValueError("Profile name must not be empty.")
    if not profile.model.strip():
        raise ValueError(f"Profile '{profile.name}' must define a non-empty model.")
    if not profile.base_url.strip():
        raise ValueError(f"Profile '{profile.name}' must define a non-empty base URL.")
    profile.resolved_provider()
    profile.resolved_api_mode()


def normalize_model_profile(profile: ModelProfile) -> ModelProfile:
    validate_model_profile(profile)
    return replace(
        profile,
        name=profile.name.strip(),
        provider=profile.resolved_provider(),
        model=profile.model.strip(),
        base_url=profile.base_url.strip(),
        api_mode=profile.resolved_api_mode(),
        api_key=profile.api_key.strip() if profile.api_key else None,
        api_key_env=profile.api_key_env.strip() if profile.api_key_env else None,
        headers={str(key): str(value) for key, value in profile.headers.items()},
        httpx_kwargs=dict(profile.httpx_kwargs),
    )


def update_profile_store(
    store: ProfileStore,
    profile: ModelProfile,
    make_default: bool = False,
) -> ProfileStore:
    profile = normalize_model_profile(profile)
    store.profiles[profile.name] = profile
    if make_default or not store.default_profile:
        store.default_profile = profile.name
    elif store.default_profile not in store.profiles:
        store.default_profile = profile.name
    return store


def set_default_profile(store: ProfileStore, name: str) -> ProfileStore:
    if name not in store.profiles:
        available = ", ".join(store.names()) or "<none>"
        raise ValueError(
            f"Unknown profile '{name}'. Available profiles: {available}"
        )
    store.default_profile = name
    return store


def remove_profile(store: ProfileStore, name: str) -> ProfileStore:
    if name not in store.profiles:
        available = ", ".join(store.names()) or "<none>"
        raise ValueError(
            f"Unknown profile '{name}'. Available profiles: {available}"
        )
    if len(store.profiles) == 1:
        raise ValueError("Cannot remove the only model profile.")
    if name == store.default_profile:
        raise ValueError(
            "Cannot remove the default model profile. Set another default first."
        )
    del store.profiles[name]
    return store


def _profile_to_dict(profile: ModelProfile) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": profile.provider,
        "model": profile.model,
        "base_url": profile.base_url,
    }
    if profile.resolved_provider() == "openai_compatible":
        payload["api_mode"] = profile.resolved_api_mode()
    if profile.api_key:
        payload["api_key"] = profile.api_key
    if profile.api_key_env:
        payload["api_key_env"] = profile.api_key_env
    if profile.headers:
        payload["headers"] = dict(sorted(profile.headers.items()))
    if profile.httpx_kwargs:
        payload["httpx_kwargs"] = profile.httpx_kwargs
    return payload


def load_profile_store(path: str | None = None) -> ProfileStore:
    resolved_path = os.path.expanduser(path or AppConfig().model_profiles_path)
    profile_path = Path(resolved_path)
    if profile_path.is_file():
        try:
            payload = json.loads(profile_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Could not parse model profile file {profile_path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(
                "Model profile file must contain a JSON object at the top level.")
        return _store_from_json(str(profile_path), payload)

    fallback_profile = _env_profile()
    store = ProfileStore(
        path=str(profile_path),
        default_profile=fallback_profile.name,
        profiles={fallback_profile.name: fallback_profile},
    )
    save_profile_store(store)
    return store
