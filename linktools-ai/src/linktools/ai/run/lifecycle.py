#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Run lifecycle helpers (spec §15). Centralizes the run-state
transitions both AgentRunner and SwarmRunner drive (mark_completed /
mark_failed / mark_cancelled) so the status enum + transition call live in one
place. PreparedRun + prepare_run bundle the session/run/context setup the two
runners duplicate."""

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .context import RunContext
from .models import RunStatus

if TYPE_CHECKING:
    from ..storage.facade import Storage
    from .store import RunStore


@dataclass(frozen=True, slots=True)
class PreparedRun:
    """A run readied for execution: its ids + the contexts the runner drives."""

    run_id: str
    session_id: str
    run_context: RunContext


async def prepare_run(
    storage: "Storage",
    *,
    session_id: "str | None",
    run_id: "str | None" = None,
    runnable_id: str = "",
    runnable_type: Any = None,
    user_id: "str | None" = None,
    tenant_id: "str | None" = None,
    root_run_id: "str | None" = None,
    parent_run_id: "str | None" = None,
) -> PreparedRun:
    """Resolve (or create) a session and mint a RunContext for a new run.
    Returns a PreparedRun bundling the resolved ids + context. Shared by
    Runtime.run / run_stream / resume (§15.3)."""
    from .._runtime.lifecycle import create_run_context, resolve_session

    resolved_session = await resolve_session(storage, session_id)
    resolved_run = run_id or str(uuid.uuid4())
    run_context = create_run_context(
        run_id=resolved_run,
        session_id=resolved_session,
        runnable_id=runnable_id,
        runnable_type=runnable_type,
        user_id=user_id,
        tenant_id=tenant_id,
        root_run_id=root_run_id,
        parent_run_id=parent_run_id,
    )
    return PreparedRun(
        run_id=resolved_run,
        session_id=resolved_session,
        run_context=run_context,
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
