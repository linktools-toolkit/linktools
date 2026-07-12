#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Run lifecycle helpers. Centralizes the run-state
transitions both AgentRunner and SwarmRunner drive (mark_completed /
mark_failed / mark_cancelled) so the status enum + transition call live in one
place. ``prepare_run`` also owns session and context setup for Runtime entry
points."""

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .context import RunContext
from .models import RunStatus
from ..run.models import RunnableType

if TYPE_CHECKING:
    from ..storage.facade import Storage
    from .store import RunStore


@dataclass(frozen=True, slots=True)
class PreparedRun:
    """A run readied for execution: its ids + the contexts the runner drives."""

    run_id: str
    session_id: str
    context: RunContext


async def prepare_run(
    *,
    storage: "Storage",
    spec: Any,
    session_id: "str | None",
    run_id: "str | None",
    user_id: "str | None",
    tenant_id: "str | None",
) -> PreparedRun:
    """Resolve a session and mint the context shared by run and streaming."""
    from .._runtime.lifecycle import create_run_context, resolve_session
    from ..swarm.spec import SwarmSpec

    resolved_session = await resolve_session(storage, session_id)
    resolved_run = run_id or str(uuid.uuid4())
    run_context = create_run_context(
        run_id=resolved_run,
        session_id=resolved_session,
        runnable_id=spec.id,
        runnable_type=RunnableType.SWARM if isinstance(spec, SwarmSpec) else RunnableType.AGENT,
        user_id=user_id,
        tenant_id=tenant_id,
    )
    return PreparedRun(
        run_id=resolved_run,
        session_id=resolved_session,
        context=run_context,
    )


async def mark_completed(
    run_store: "RunStore",
    run_id: str,
    *,
    expected_version: int,
    result: Any = None,
) -> None:
    """Transition a run to SUCCEEDED with its result."""
    await run_store.transition(
        run_id,
        RunStatus.SUCCEEDED,
        expected_version=expected_version,
        result=result,
    )


async def mark_failed(
    run_store: "RunStore",
    run_id: str,
    *,
    expected_version: int,
    error: Any = None,
) -> None:
    """Transition a run to FAILED with error info."""
    await run_store.transition(
        run_id,
        RunStatus.FAILED,
        expected_version=expected_version,
        error=error,
    )


async def mark_cancelled(
    run_store: "RunStore",
    run_id: str,
    *,
    expected_version: int,
) -> None:
    """Transition a run to CANCELLED."""
    await run_store.transition(
        run_id,
        RunStatus.CANCELLED,
        expected_version=expected_version,
    )
