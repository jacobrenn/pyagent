"""Synchronous event bus for extensions.

Extensions subscribe via ``register(bus, name)`` where ``bus`` is a scoped
view that auto-tags handlers with the extension name (so ``unload`` can remove
them by name). Emits are synchronous and deterministic — a handler returns a
dict to mutate the payload, or ``None`` to pass through. Veto keys
(``blocked``, ``cancel``) short-circuit on the first truthy value; other
fields are last-writer-wins. A crashing handler is logged and skipped.
"""
from __future__ import annotations

import inspect
import traceback
from typing import Any, Callable

Handler = Callable[[dict, "Ctx"], Any | None]

# First-truthy-wins short-circuit keys (a gate must not be overruled later).
_VETO_KEYS = frozenset({"blocked", "cancel"})


class NoOpLog:
    """Default logger: swallows everything until /logging attaches a real one."""

    def debug(self, *_a: Any, **_k: Any) -> None: ...
    def info(self, *_a: Any, **_k: Any) -> None: ...
    def warn(self, *_a: Any, **_k: Any) -> None: ...
    def error(self, *_a: Any, **_k: Any) -> None: ...
    def log_event(self, *_a: Any, **_k: Any) -> None: ...
    def log_extension_skills(self, *_a: Any, **_k: Any) -> None: ...


class Ctx:
    """Minimal per-handler context: attribution, skill intent, log.

    ``add_skill(key)`` declares a skill (by filename stem under the *declaring
    extension's* ``~/.pyagent/extensions/<ext>/skills/<key>.md``) to be injected
    into the system prompt *next* turn and auto-expunged after — lean by
    default, ephemeral by design. The extension name is taken from
    ``ctx.extension`` (set by the bus per handler) so each skill is resolved
    against the extension that declared it.
    """

    __slots__ = ("extension", "add_skill", "log")

    def __init__(self, add_skill: Callable[[str], None], log: Any) -> None:
        self.extension: str = ""
        self.add_skill = add_skill
        self.log = log


class _ScopedBus:
    """Auto-tags every ``on`` with the extension name so unload can target it."""

    def __init__(self, bus: "EventBus", name: str) -> None:
        self._bus = bus
        self._name = name

    def on(self, event: str, handler: Handler | None = None):
        def register(fn: Handler) -> Handler:
            self._bus.on(event, fn, extension=self._name)
            return fn

        return register(handler) if handler is not None else register

    def off(self, event: str, handler: Handler) -> None:
        self._bus.off(event, handler)


class EventBus:
    """Synchronous pub/sub router with fault isolation."""

    def __init__(self, log: Any) -> None:
        self._log = log
        self._handlers: dict[str, list[tuple[Handler, str]]] = {}

    def on(self, event: str, handler: Handler | None = None, *, extension: str = ""):
        if handler is not None:
            self._handlers.setdefault(event, []).append((handler, extension))
            return handler

        def register(fn: Handler) -> Handler:
            self._handlers.setdefault(event, []).append((fn, extension))
            return fn

        return register

    def off(self, event: str, handler: Handler) -> None:
        hs = self._handlers.get(event)
        if hs:
            self._handlers[event] = [(h, n)
                                     for (h, n) in hs if h is not handler]

    def off_extension(self, name: str) -> None:
        """Remove every handler tagged with ``name`` (``/extension unload``)."""
        for ev in list(self._handlers):
            kept = [(h, n) for (h, n) in self._handlers[ev] if n != name]
            if kept:
                self._handlers[ev] = kept
            else:
                del self._handlers[ev]

    def clear(self) -> None:
        """Drop all handlers (``/extension reload`` starts from a clean slate)."""
        self._handlers.clear()

    def scoped(self, name: str) -> _ScopedBus:
        return _ScopedBus(self, name)

    def handlers(self, event: str) -> list[Handler]:
        return [h for (h, _) in self._handlers.get(event, [])]

    @property
    def events(self) -> list[str]:
        return list(self._handlers)

    def loaded_extensions(self) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for hs in self._handlers.values():
            for _, n in hs:
                if n and n not in seen:
                    seen.add(n)
                    names.append(n)
        return sorted(names)

    def emit(self, event: str, payload: dict, ctx: Ctx) -> dict:
        handlers = list(self._handlers.get(event, []))
        if not handlers:
            self._log_event(event, "", payload)
            return payload
        for handler, ext_name in list(handlers):
            ctx.extension = ext_name
            result: Any = None
            try:
                result = handler(payload, ctx)
            except Exception:
                self._log_fault(event, handler, ctx)
                continue

            if inspect.iscoroutine(result):
                result.close()
                dbg = getattr(self._log, "debug", None)
                if callable(dbg):
                    dbg(
                        "extension handler returned a coroutine (sync-only); ignored",
                        {"event": event, "extension": ext_name},
                    )
                continue
            self._log_event(event, ext_name, payload, result)
            if isinstance(result, dict):
                if any(result.get(k) for k in _VETO_KEYS):
                    payload.update(result)
                    return payload
                payload.update(result)
        return payload

    def _log_event(
        self, event: str, extension: str, payload: dict, result: Any = None
    ) -> None:
        log_event = getattr(self._log, "log_event", None)
        if not callable(log_event):
            return
        try:
            log_event(event, extension or None, payload, result)
        except Exception:
            pass

    def set_log(self, log: Any) -> None:
        self._log = log

    def _log_fault(self, event: str, handler: Handler, ctx: Ctx) -> None:
        err = getattr(self._log, "error", None)
        if not callable(err):
            return
        err(
            "extension handler failed",
            {
                "event": event,
                "handler": getattr(handler, "__name__", repr(handler)),
                "extension": ctx.extension,
                "traceback": traceback.format_exc(),
            },
        )


if __name__ == "__main__":
    log = NoOpLog()
    bus = EventBus(log)
    ctx = Ctx(add_skill=lambda _k: None, log=log)

    @bus.on("tool_call", extension="sg")
    def gate(payload, ctx):
        if payload["name"] == "bash" and "rm -rf" in payload["input"].get("command", ""):
            return {"blocked": True, "reason": "destructive"}
        return None

    out = bus.emit("tool_call", {"name": "bash",
                   "input": {"command": "ls"}}, ctx)
    assert "blocked" not in out, out
    out = bus.emit("tool_call", {"name": "bash",
                   "input": {"command": "rm -rf /"}}, ctx)
    assert out.get("blocked") is True, out
    assert ctx.extension == "sg"

    sb = bus.scoped("ext2")

    @sb.on("turn_start")
    def boom(payload, ctx):
        raise RuntimeError("kaboom")

    out = bus.emit("turn_start", {"turn_index": 0}, ctx)
    assert out == {"turn_index": 0}
    assert bus.loaded_extensions() == ["ext2", "sg"]
    bus.off_extension("sg")
    assert bus.loaded_extensions() == ["ext2"]
    bus.clear()
    assert bus.loaded_extensions() == []
    print("bus self-check OK")
