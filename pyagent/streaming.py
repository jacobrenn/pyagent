from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import queue
import threading
from typing import Any, AsyncIterator, Callable
from uuid import uuid4


STREAM_SCHEMA_VERSION = 1
DEFAULT_HEARTBEAT_SECONDS = 15.0
DEFAULT_QUEUE_SIZE = 256
_TERMINAL_SENTINEL = object()
_RESERVED_EVENT_FIELDS = {
    "schema_version",
    "run_id",
    "sequence",
    "timestamp",
    "type",
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def encode_sse_event(event: dict[str, Any]) -> str:
    """Encode one JSON event using Server-Sent Events framing."""
    event_type = str(event.get("type") or "message").replace("\r", "").replace(
        "\n", ""
    )
    event_id = str(event.get("sequence", ""))
    data = json.dumps(
        event,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return f"id: {event_id}\nevent: {event_type}\ndata: {data}\n\n"


def encode_sse_comment(comment: str) -> str:
    """Encode an SSE comment, used as an idle connection heartbeat."""
    text = str(comment).replace("\r", " ").replace("\n", " ")
    return f": {text}\n\n"


def _event_envelope(
    *,
    run_id: str,
    sequence: int,
    event_type: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": STREAM_SCHEMA_VERSION,
        "run_id": run_id,
        "sequence": sequence,
        "timestamp": _timestamp(),
        "type": event_type,
    }
    for key, value in (data or {}).items():
        if key not in _RESERVED_EVENT_FIELDS:
            payload[key] = value
    return payload


async def stream_agent_sse(
    agent: Any,
    message: str,
    *,
    start_data: dict[str, Any],
    completion_data: Callable[[str], dict[str, Any]],
    include_debug: bool = False,
    heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
    queue_size: int = DEFAULT_QUEUE_SIZE,
) -> AsyncIterator[str]:
    """Run a synchronous Agent generator and expose its events as SSE.

    The producer owns the Agent lifecycle and always closes both the run
    generator and the Agent. The async consumer only handles framing,
    sequencing, heartbeats, and disconnect signalling.
    """
    run_id = str(uuid4())
    event_queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, queue_size))
    cancelled = threading.Event()

    def offer(item: Any) -> bool:
        while not cancelled.is_set():
            try:
                event_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        run_generator: Any | None = None
        terminal_event: dict[str, Any] | None = None
        try:
            run_generator = iter(agent.run(message))
            for raw_event in run_generator:
                if cancelled.is_set():
                    break
                if not isinstance(raw_event, dict):
                    raise TypeError("Agent emitted a non-object streaming event.")
                event_type = raw_event.get("type")
                if not isinstance(event_type, str) or not event_type:
                    raise ValueError("Agent emitted a streaming event without a type.")
                if terminal_event is not None:
                    continue
                if event_type == "debug" and not include_debug:
                    continue
                if event_type == "error":
                    terminal_event = {
                        "type": "error",
                        "code": "agent_error",
                        "message": str(
                            raw_event.get("message") or "Agent run failed"
                        ),
                    }
                    continue
                if event_type == "assistant_done":
                    final_response = str(raw_event.get("content", ""))
                    terminal_event = {
                        "type": "done",
                        **completion_data(final_response),
                    }
                    continue
                if not offer(dict(raw_event)):
                    break
            else:
                if not cancelled.is_set() and terminal_event is None:
                    terminal_event = {
                        "type": "error",
                        "code": "incomplete_run",
                        "message": (
                            "Agent run finished without a final assistant response."
                        ),
                    }
        except Exception as exc:
            if not cancelled.is_set() and terminal_event is None:
                terminal_event = {
                    "type": "error",
                    "code": "internal_error",
                    "message": str(exc) or "Agent streaming failed.",
                }
        finally:
            close_generator = getattr(run_generator, "close", None)
            if callable(close_generator):
                try:
                    close_generator()
                except Exception:
                    pass
            try:
                agent.close(reason="api_request_complete")
            except Exception:
                pass
            if not cancelled.is_set():
                if terminal_event is not None:
                    offer(terminal_event)
                offer(_TERMINAL_SENTINEL)

    producer = threading.Thread(
        target=produce,
        name=f"pyagent-stream-{run_id}",
        daemon=True,
    )
    producer.start()

    sequence = 1
    yield encode_sse_event(
        _event_envelope(
            run_id=run_id,
            sequence=sequence,
            event_type="start",
            data=start_data,
        )
    )

    timeout = max(0.1, float(heartbeat_seconds))
    try:
        while True:
            try:
                item = await asyncio.to_thread(
                    event_queue.get,
                    True,
                    timeout,
                )
            except queue.Empty:
                yield encode_sse_comment(f"heartbeat {_timestamp()}")
                continue
            if item is _TERMINAL_SENTINEL:
                break
            event_type = str(item.get("type") or "message")
            sequence += 1
            yield encode_sse_event(
                _event_envelope(
                    run_id=run_id,
                    sequence=sequence,
                    event_type=event_type,
                    data=item,
                )
            )
    finally:
        cancelled.set()
