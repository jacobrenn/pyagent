"""Pyagent extension system — a synchronous event bus + a tiny loader.

The public surface for extension authors is the :class:`EventBus` (subscribe
via ``register(bus, name)``) and the :class:`Ctx` handed to every handler.
See ``extensions_prd.md`` for the full guide.
"""
from __future__ import annotations

from .bus import Ctx, EventBus, NoOpLog

__all__ = ["EventBus", "Ctx", "NoOpLog"]
