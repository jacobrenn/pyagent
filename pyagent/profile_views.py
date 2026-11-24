"""Shared, redacted view of a model profile.

Used by both the HTTP API (``pyagent/api.py``) and the CLI so both surfaces
present the same shape and apply the same secret redaction.
"""

from __future__ import annotations

from typing import Any


def is_sensitive_profile_key(name: str) -> bool:
    normalized = name.strip().lower().replace("_", "-")
    if normalized in {"auth", "authentication", "key"}:
        return True
    sensitive_markers = (
        "authorization",
        "api-key",
        "apikey",
        "token",
        "secret",
        "password",
        "credential",
        "cookie",
    )
    return normalized.endswith("-key") or any(
        marker in normalized for marker in sensitive_markers
    )


def redact_profile_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "<redacted>"
                if is_sensitive_profile_key(str(key))
                else redact_profile_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_profile_value(item) for item in value]
    return value


def profile_to_dict(profile: Any, default_profile: str) -> dict[str, Any]:
    """Return the redacted, JSON-serializable view of a profile."""
    visible_headers: dict[str, str] = {}
    redacted_headers: list[str] = []
    for name, value in profile.headers.items():
        if is_sensitive_profile_key(name):
            redacted_headers.append(name)
        else:
            visible_headers[name] = value
    return {
        "name": profile.name,
        "provider": profile.resolved_provider(),
        "api_mode": profile.resolved_api_mode(),
        "model": profile.model,
        "base_url": profile.base_url,
        "api_key_env": profile.api_key_env,
        "has_inline_api_key": bool(profile.api_key),
        "headers": visible_headers,
        "redacted_headers": sorted(redacted_headers),
        "httpx_kwargs": redact_profile_value(profile.httpx_kwargs),
        "is_default": profile.name == default_profile,
    }


def profile_list_payload(
    store: Any,
    *,
    effective_default: str,
    default_overridden_by_env: bool,
) -> dict[str, Any]:
    return {
        "path": store.path,
        "default_profile": store.default_profile,
        "effective_default_profile": effective_default,
        "default_overridden_by_env": default_overridden_by_env,
        "profiles": [
            profile_to_dict(store.get(name), store.default_profile)
            for name in store.names()
        ],
    }
