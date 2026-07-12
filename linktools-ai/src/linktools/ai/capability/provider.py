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

from .models import CapabilityBundle
from .exposure import CapabilityToolExposurePolicy
from .models import CapabilityRef

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
    parent_run_id: "str | None" = None
    session_id: "str | None" = None
    event_store: "EventStore | None" = None
    security_event_emitter: Any = None
    # Identity fields propagated to subagent child runs.
    user_id: "str | None" = None
    tenant_id: "str | None" = None
    workspace: Any = None


@runtime_checkable
class CapabilityProvider(Protocol):
    """Resolves one or more capability kinds. ``supported_kinds`` is the complete
    declaration of the kinds this provider
    handles. Providers owning related kinds declare all of them in one tuple.
    """

    supported_kinds: "tuple[str, ...]"

    async def resolve(
        self,
        ref: CapabilityRef,
        context: CapabilityContext,
    ) -> CapabilityBundle: ...


def provider_kinds(provider: "CapabilityProvider") -> "frozenset[str]":
    """Return the explicitly declared capability kinds."""
    kinds = provider.supported_kinds
    if not kinds:
        from ..errors import CapabilityResolutionError
        raise CapabilityResolutionError("CapabilityProvider.supported_kinds cannot be empty")
    if any(not isinstance(kind, str) or not kind for kind in kinds):
        from ..errors import CapabilityResolutionError
        raise CapabilityResolutionError("Capability provider kinds must be strings")
    return frozenset(kinds)


async def _noop_emit(payload: Any) -> None:
    return None


def make_event_emitter(context: "CapabilityContext | None"):
    """Return an ``async emit(payload)`` bound to the context's EventStore + run
    ids, or a no-op when no store/run is wired. Capability toolset closures use
    this to fire per-operation events (skill.list, package.resource.read, ...)."""
    if context is None or context.event_store is None or context.run_id is None:
        return _noop_emit
    if context.security_event_emitter is not None:
        return context.security_event_emitter.emit_observability
    store = context.event_store
    run_id = context.run_id
    from ..events.context import EventContext, append_event

    evt_ctx = EventContext(
        stream_id=run_id,
        run_id=run_id,
        root_run_id=context.root_run_id or run_id,
        parent_run_id=context.parent_run_id,
        session_id=context.session_id or run_id,
        runnable_id=context.agent_id,
    )

    async def emit(payload: Any) -> None:
        await append_event(store, evt_ctx, payload)

    return emit
