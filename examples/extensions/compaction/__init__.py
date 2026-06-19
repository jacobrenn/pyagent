"""Reference extension: context compaction via a surfaced skill + tool.

This extension is **deterministic**: it only decides *when* to surface
guidance. It listens for ``turn_end`` and, once the conversation grows past a
soft threshold, injects the ``compaction`` skill for the *next* turn (auto-
expunged after). That skill tells the LLM it has a ``compact_context`` tool
available and how to use it.

The ``compact_context`` tool itself is a standalone UV script the user places
in ``~/.pyagent/tools/`` (see ``examples/tools/compact_context.py``). The
extension never declares tools itself and never touches agent state — it only
injects guidance, keeping the loop lean by default.

Install:
  1. Copy this file to ``~/.pyagent/extensions/compaction/__init__.py``
     (or ``/extension new compaction`` then replace its body).
  2. Copy ``examples/tools/compact_context.py`` to ``~/.pyagent/tools/``.
  3. Copy ``compaction.md`` to ``~/.pyagent/skills/extensions/compaction.md``.
  4. ``/extension reload`` (or ``/extension load compaction``).
"""
from __future__ import annotations

# Soft threshold: once the stored message count crosses this, surface the
# compaction skill next turn. Sized at half the agent's hard history limit
# when available, else a sane default. Kept in-module so the rule is
# deterministic and self-contained (extensions_prd.md §2).
DEFAULT_SOFT_THRESHOLD = 12


def _soft_threshold(payload: dict) -> int:
    # The agent does not expose config to extensions (lean ctx). The turn_end
    # payload carries message_count; we use a fixed threshold. Users editing
    # this file can tune it.
    return DEFAULT_SOFT_THRESHOLD


def register(bus, name):
    @bus.on("turn_end")
    def on_turn_end(payload, ctx):
        count = payload.get("message_count", 0)
        if count > _soft_threshold(payload):
            ctx.add_skill("compaction")


if __name__ == "__main__":  # ponytail: cheapest smoke test
    from pyagent.extensions import Ctx, EventBus, NoOpLog

    log = NoOpLog()
    bus = EventBus(log)
    register(bus.scoped("compaction"), "compaction")
    added: list[str] = []
    ctx = Ctx(add_skill=added.append, log=log)

    bus.emit("turn_end", {"message_count": 5}, ctx)
    assert added == []  # under threshold: skill not injected
    bus.emit("turn_end", {"message_count": 20}, ctx)
    assert added == ["compaction"]  # over threshold: injected next turn
    print("compaction self-check OK")
