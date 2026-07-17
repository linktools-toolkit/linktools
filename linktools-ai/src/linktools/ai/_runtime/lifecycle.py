#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Run lifecycle helpers used by Runtime.run / run_stream / resume.
Centralizing session resolution + RunContext minting here keeps
the three entry points single-shape and ensures they resolve sessions and
lineage identically."""

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping

from ..errors import SessionAccessDeniedError, SessionError
from ..run.context import RunContext
from ..run.models import RunnableType
from ..session.models import SessionRecord, SessionStatus

if TYPE_CHECKING:
    from ..storage.facade import Storage


async def resolve_session(
    storage: "Storage",
    session_id: "str | None",
    *,
    user_id: "str | None",
    tenant_id: "str | None",
) -> str:
    """Return a usable session id, creating a fresh session when none is given.

    A caller-supplied session_id must already exist AND belong to the same
    (user_id, tenant_id) principal -- strict equality, so an unowned caller
    (None, None) can only open an unowned session, and a principal cannot claim
    a session owned by another. Ownership is never auto-filled. A mismatch or a
    missing session raises (SessionAccessDeniedError / SessionError); the
    denial message does not reveal whether the session belongs to someone
    else."""
    resolved = session_id or str(uuid.uuid4())
    if session_id is not None:
        existing = await storage.sessions.get(session_id)
        if existing is None:
            raise SessionError(f"session not found: {session_id}")
        if existing.user_id != user_id or existing.tenant_id != tenant_id:
            raise SessionAccessDeniedError(
                "session is not accessible to the current principal"
            )
    else:
        now = datetime.now(timezone.utc)
        await storage.sessions.create(
            SessionRecord(
                id=resolved,
                parent_id=None,
                user_id=user_id,
                tenant_id=tenant_id,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
    return resolved


def create_run_context(
    *,
    run_id: str,
    session_id: str,
    runnable_id: str,
    runnable_type: "RunnableType",
    user_id: "str | None" = None,
    tenant_id: "str | None" = None,
    workspace: "str | None" = None,
    root_run_id: "str | None" = None,
    parent_run_id: "str | None" = None,
    metadata: "Mapping[str, Any] | None" = None,
) -> RunContext:
    """Mint a RunContext for a top-level run. ``root_run_id`` defaults to the
    run itself; ``parent_run_id`` defaults to None (a resumed run passes the
    record's lineage through). ``workspace`` is restored from the snapshot on
    resume (None for fresh runs). ``metadata`` is an optional caller-supplied
    mapping merged onto the context (e.g. task correlation ids threaded from a
    TaskHandler) so it reaches RunRecord/RunDefinitionSnapshot downstream."""
    return RunContext(
        run_id=run_id,
        root_run_id=root_run_id or run_id,
        parent_run_id=parent_run_id,
        session_id=session_id,
        runnable_id=runnable_id,
        runnable_type=runnable_type,
        user_id=user_id,
        tenant_id=tenant_id,
        workspace=workspace,
        metadata=dict(metadata) if metadata else {},
    )
