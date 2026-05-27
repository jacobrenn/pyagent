from __future__ import annotations

from typing import Any

from .agent import Agent

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - exercised via CLI guard
    raise RuntimeError(
        "FastAPI support is not installed. Install `fastapi` and `uvicorn` to use `pyagent api`."
    ) from exc


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    profile: str | None = None
    model: str | None = None
    cwd: str | None = None
    skills: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    response: str
    profile: str
    provider: str
    model: str
    context_files: list[str] = Field(default_factory=list)


app = FastAPI(title="PyAgent API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_model=ChatResponse)
def run(request: ChatRequest) -> ChatResponse:
    from .main import build_agent_for_request

    try:
        agent = build_agent_for_request(
            profile=request.profile,
            model=request.model,
            cwd=request.cwd,
            skills=request.skills,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    final_response = ""
    for event in agent.run(request.message):
        if event.get("type") == "assistant_done":
            final_response = event.get("content", "")
            break

    profile = agent.current_profile()
    return ChatResponse(
        response=final_response,
        profile=profile.name,
        provider=profile.provider,
        model=profile.model,
        context_files=list(agent.project_context_files),
    )


def create_app() -> Any:
    return app
