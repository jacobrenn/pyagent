"""Template PyAgent extension — copy this file, rename, edit.

An extension is a single ``.py`` file (or ``<name>/__init__.py`` package) under
``~/.pyagent/extensions/`` that defines ``register(bus, name)``. Subscribe to
lifecycle events with ``@bus.on(...)`` (the scoped bus auto-tags your handlers
so ``/extension unload <name>`` can remove them). Return a dict to mutate the
event payload (veto keys like ``blocked`` short-circuit), or ``None`` to pass
through.

Skills are plain Markdown under
``~/.pyagent/extensions/<name>/skills/<key>.md``; inject one for the *next*
turn with ``ctx.add_skill("<key>")`` — it auto-expunges after. Tools live as
UV scripts under ``~/.pyagent/extensions/<name>/tools/`` and are discovered
only while this extension is loaded; an extension never declares tools itself.

See ``extensions_prd.md`` for the full event catalog.
"""
from __future__ import annotations


def register(bus, name):
    @bus.on("tool_call")
    def on_tool_call(payload, ctx):
        # Example safeguard: block destructive bash. ``blocked`` is a veto key
        # — the first truthy value short-circuits the event.
        if payload["name"] == "bash" and "rm -rf" in payload["input"].get("command", ""):
            return {"blocked": True, "reason": f"{name} blocks destructive commands"}
        return None

    @bus.on("turn_end")
    def on_turn_end(payload, ctx):
        # Inject a skill next turn once the conversation grows. The skill text
        # is read from ~/.pyagent/extensions/template/skills/<key>.md. It is
        # injected for one turn only and auto-expunged; re-declare each turn
        # to persist.
        if payload.get("message_count", 0) > 20:
            ctx.add_skill("template")


if __name__ == "__main__":  # ponytail: cheapest smoke test
    from pyagent.extensions import Ctx, EventBus, NoOpLog

    log = NoOpLog()
    bus = EventBus(log)
    register(bus.scoped("template"), "template")
    assert bus.loaded_extensions() == ["template"]
    ctx = Ctx(add_skill=lambda _k: None, log=log)

    out = bus.emit("tool_call", {"name": "bash", "input": {"command": "ls"}}, ctx)
    assert "blocked" not in out
    out = bus.emit("tool_call", {"name": "bash", "input": {"command": "rm -rf /"}}, ctx)
    assert out["blocked"] and "template" in out["reason"]

    added: list[str] = []
    ctx2 = Ctx(add_skill=added.append, log=log)
    bus.emit("turn_end", {"message_count": 25}, ctx2)
    assert added == ["template"]
    bus.emit("turn_end", {"message_count": 5}, ctx2)
    assert added == ["template"]  # not re-added below threshold
    print("template self-check OK")
