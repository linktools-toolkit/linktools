#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Run lifecycle helpers used by Runtime.run / run_stream / resume.
Centralizing session resolution + RunContext minting here keeps
the three entry points single-shape and ensures they resolve sessions and
lineage identically."""

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..errors import SessionError
from ..run.context import RunContext
from ..run.models import RunnableType
from ..session.models import SessionRecord, SessionStatus

if TYPE_CHECKING:
    from ..storage.facade import Storage


async def resolve_session(storage: "Storage", session_id: "str | None") -> str:
    """Return a usable session id, creating a fresh session when none is given.
    A caller-supplied session_id must already exist (SessionError otherwise)."""
    resolved = session_id or str(uuid.uuid4())
    if session_id is not None:
        existing = await storage.sessions.get(session_id)
        if existing is None:
            raise SessionError(f"session not found: {session_id}")
    else:
        now = datetime.now(timezone.utc)
        await storage.sessions.create(
            SessionRecord(
                id=resolved,
                parent_id=None,
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
    root_run_id: "str | None" = None,
    parent_run_id: "str | None" = None,
) -> RunContext:
    """Mint a RunContext for a top-level run. ``root_run_id`` defaults to the
    run itself; ``parent_run_id`` defaults to None (a resumed run passes the
    record's lineage through)."""
    return RunContext(
        run_id=run_id,
        root_run_id=root_run_id or run_id,
        parent_run_id=parent_run_id,
        session_id=session_id,
        runnable_id=runnable_id,
        runnable_type=runnable_type,
        user_id=user_id,
        tenant_id=tenant_id,
        workspace=None,
    )
