#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public LinkTools AI composition and runtime API."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .capability import AgentContext
    from .runtime import Agent, Execution, Runtime, RuntimeFeatures, Session
    from .workspace import Workspace

__all__ = [
    "Agent",
    "RuntimeFeatures",
    "Execution",
    "AgentContext",
    "Runtime",
    "Session",
    "Workspace",
]


def __getattr__(name: str) -> object:
    """Keep domain imports independent until a root export is requested."""
    if name in {"Agent", "Execution", "Runtime", "RuntimeFeatures", "Session"}:
        from .runtime import Agent, Execution, Runtime, RuntimeFeatures, Session

        globals().update(
            Agent=Agent, Execution=Execution, Runtime=Runtime,
            RuntimeFeatures=RuntimeFeatures, Session=Session,
        )
    elif name == "AgentContext":
        from .capability import AgentContext

        globals()[name] = AgentContext
    elif name == "Workspace":
        from .workspace import Workspace

        globals()[name] = Workspace
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return globals()[name]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
