#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CapabilityProvider Protocol + CapabilityContext.

A CapabilityProvider resolves one CapabilityRef into a CapabilityBundle. Each
concrete provider owns its own strong dependencies (a skill store, an MCP
connection manager, an entrypoint resolver, ...); the CapabilityContext carries
only the per-Run/per-spec dynamic state a provider may need (the execution
backend, the active exposure policy, identity for diagnostics)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .bundle import CapabilityBundle
from .policy import CapabilityToolExposurePolicy
from .ref import CapabilityRef

if TYPE_CHECKING:
    from ..execution.protocols import ExecutionBackend
    from ..events.store import EventStore


@dataclass(frozen=True)
class CapabilityContext:
    """Per-resolution dynamic state handed to every CapabilityProvider. Fields
    are added as providers need them; the exposure policy and execution backend
    cover the builtin path. Identity fields make resolution failures precise.
    ``event_store`` (optional) lets providers emit capability-lifecycle events."""

    agent_id: str
    exposure_policy: CapabilityToolExposurePolicy
    execution: "ExecutionBackend | None" = None
    run_id: "str | None" = None
    root_run_id: "str | None" = None
    session_id: "str | None" = None
    event_store: "EventStore | None" = None
    # Identity fields propagated to subagent child runs.
    user_id: "str | None" = None
    tenant_id: "str | None" = None
    workspace: Any = None


@runtime_checkable
class CapabilityProvider(Protocol):
    """Resolves one capability kind. ``kind`` is the provider's selector (the
    left side of a ``kind:name`` tool ref)."""

    kind: str

    async def resolve(
        self,
        ref: CapabilityRef,
        context: CapabilityContext,
    ) -> CapabilityBundle:
        ...


def _toolset_tool_names(toolset: Any) -> "tuple[str, ...]":
    """Best-effort tool-name extraction for conflict detection. pydantic-ai
    ``FunctionToolset`` exposes ``.tools`` (name -> tool); other toolset shapes
    return empty so unknown toolsets never produce false conflicts."""
    tools = getattr(toolset, "tools", None)
    if isinstance(tools, dict):
        return tuple(str(k) for k in tools.keys())
    return ()


def toolset_names(toolsets: "tuple[Any, ...]") -> "tuple[str, ...]":
    out: "list[str]" = []
    for ts in toolsets:
        out.extend(_toolset_tool_names(ts))
    return tuple(out)


async def _noop_emit(payload: Any) -> None:
    return None


def make_event_emitter(context: "CapabilityContext | None"):
    """Return an ``async emit(payload)`` bound to the context's EventStore + run
    ids, or a no-op when no store/run is wired. Capability toolset closures use
    this to fire per-operation events (skill.list, package.resource.read, ...)."""
    if context is None or context.event_store is None or context.run_id is None:
        return _noop_emit
    store = context.event_store
    run_id = context.run_id
    root_run_id = context.root_run_id or run_id
    session_id = context.session_id or run_id
    agent_id = context.agent_id

    async def emit(payload: Any) -> None:
        await store.append(
            stream_id=run_id, run_id=run_id, root_run_id=root_run_id,
            parent_run_id=None, session_id=session_id, runnable_id=agent_id,
            payload=payload,
        )

    return emit
