#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Run lifecycle helpers. Centralizes the run-state
transitions both AgentEngine and SwarmEngine drive (mark_completed /
mark_failed / mark_cancelled) so the status enum + transition call live in one
place. ``prepare_run`` also owns session and context setup for Runtime entry
points."""

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from .context import RunContext
from .models import RunInput, RunRecord, RunStatus
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
    context_metadata: "Mapping[str, Any] | None" = None,
) -> PreparedRun:
    """Resolve a session and mint the context shared by run and streaming.
    ``context_metadata`` is an optional caller-supplied mapping (e.g. task
    correlation ids) merged onto the RunContext so it flows to RunRecord /
    RunDefinitionSnapshot."""
    from ..runtime.assembly.lifecycle import create_run_context, resolve_session
    from ..swarm.spec import SwarmSpec

    resolved_session = await resolve_session(
        storage, session_id, user_id=user_id, tenant_id=tenant_id
    )
    resolved_run = run_id or str(uuid.uuid4())
    run_context = create_run_context(
        run_id=resolved_run,
        session_id=resolved_session,
        runnable_id=spec.id,
        runnable_type=RunnableType.SWARM
        if isinstance(spec, SwarmSpec)
        else RunnableType.AGENT,
        user_id=user_id,
        tenant_id=tenant_id,
        metadata=context_metadata,
    )
    return PreparedRun(
        run_id=resolved_run,
        session_id=resolved_session,
        context=run_context,
    )


async def create_and_start_run(
    run_store: "RunStore",
    *,
    context: RunContext,
    request: RunInput,
) -> RunRecord:
    """Create a new RunRecord and transition PENDING -> RUNNING. The single
    path both RunCoordinator (top-level run()/run_stream() entry points) and
    AgentEngine's execute() (direct-engine callers that bypass RunCoordinator
    entirely, e.g. most engine-level tests) use to establish a run's initial
    record -- AgentEngine only calls this itself when ``context.run_id`` has
    no record yet, so a caller that already created one via this function is
    never double-created."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    record = RunRecord(
        id=context.run_id,
        root_run_id=context.root_run_id,
        parent_run_id=context.parent_run_id,
        session_id=context.session_id,
        runnable_id=context.runnable_id,
        runnable_type=context.runnable_type,
        status=RunStatus.PENDING,
        input=request,
        result=None,
        error=None,
        version=1,
        created_at=now,
        started_at=None,
        finished_at=None,
    )
    created = await run_store.create(record)
    return await run_store.transition(
        context.run_id,
        RunStatus.RUNNING,
        expected_version=created.version,
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
