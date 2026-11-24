"""Shared saved-run (checkpoint) summaries for the API and the CLI.

Mirrors the ``/sessions`` response shapes so both surfaces present the same
fields. Plain dicts; no FastAPI imports.
"""

from __future__ import annotations

from typing import Any


def saved_run_summary(saved: Any) -> dict[str, Any]:
    def text(key: str) -> str | None:
        value = saved.metadata.get(key)
        return value[:1024] if isinstance(value, str) else None

    revision = saved.metadata.get("agent_revision")
    if isinstance(revision, str) and len(revision) <= 16 and revision.isdecimal():
        revision = int(revision)
    if type(revision) is not int or not 0 < revision <= 9_007_199_254_740_991:
        revision = None
    return {
        "run_id": saved.run_id,
        "parent_run_id": saved.parent_run_id,
        "conversation_id": saved.conversation_id,
        "timestamp": saved.timestamp,
        "prompt_preview": " ".join(saved.prompt[:1000].split())[:160],
        "outcome": saved.outcome,
        "resumable": saved.resumable,
        "reason": saved.reason,
        "profile": text("profile"),
        "model": text("model"),
        "workspace": text("workspace"),
        "agent": text("agent"),
        "agent_revision": revision,
        "warnings": [
            *saved.warnings,
            *([saved.lineage_warning] if saved.lineage_warning else []),
        ],
    }


def saved_run_detail(saved: Any) -> dict[str, Any]:
    skills = saved.metadata.get("skills", [])
    detail = saved_run_summary(saved)
    detail["messages"] = list(saved.messages)
    detail["skills"] = [
        key for key in skills if isinstance(key, str)
    ] if isinstance(skills, list) else []
    return detail
