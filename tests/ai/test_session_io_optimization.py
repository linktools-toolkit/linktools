#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session read-path I/O invariants."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from linktools.ai.core import (
    ExecutionStatus,
    HmacCursorSigner,
    Principal,
    ResourceKind,
    ResourceRef,
    SessionStatus,
)
from linktools.ai.runtime._session import DefaultSessionService
from linktools.ai.runtime.state._contracts import SessionRecord


class _Sessions:
    def __init__(self, record: SessionRecord) -> None:
        self._record = record

    async def get_header(
        self,
        session_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef | None:
        if (
            session_id != self._record.session_id
            or tenant_id != self._record.tenant_id
        ):
            return None
        return ResourceRef(
            ResourceKind.SESSION,
            session_id,
            tenant_id,
            self._record.owner_principal_id,
        )

    async def get(
        self,
        session_id: str,
        *,
        tenant_id: str,
    ) -> SessionRecord | None:
        if (
            session_id != self._record.session_id
            or tenant_id != self._record.tenant_id
        ):
            return None
        return self._record


class _Executions:
    def __init__(self) -> None:
        self.get_calls = 0

    async def get(self, execution_id: str, *, tenant_id: str) -> object:
        self.get_calls += 1
        assert execution_id == "execution"
        assert tenant_id == "tenant"
        return SimpleNamespace(
            execution_id="execution",
            tenant_id="tenant",
            session_id="session",
            parent_execution_id=None,
            status=ExecutionStatus.STARTED,
        )


class _Authorization:
    async def authorize(self, *args: object) -> None:
        del args


@pytest.mark.asyncio
async def test_session_load_reuses_reconciled_active_execution() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    record = SessionRecord(
        session_id="session",
        tenant_id="tenant",
        owner_principal_id="principal",
        status=SessionStatus.OPEN,
        revision=1,
        cwd=None,
        metadata={},
        created_at=now,
        updated_at=now,
        closed_at=None,
        active_execution_id="execution",
        agent_id="default",
    )
    executions = _Executions()
    service = DefaultSessionService(
        SimpleNamespace(sessions=_Sessions(record)),  # type: ignore[arg-type]
        executions,  # type: ignore[arg-type]
        _Authorization(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        HmacCursorSigner("session", b"session-key"),
        history_reader=object(),  # type: ignore[arg-type]
    )

    loaded = await service.load(
        "session",
        principal=Principal("principal", "tenant"),
    )

    assert loaded.active_execution_ids == ("execution",)
    assert loaded.view.active_execution_ids == ("execution",)
    assert executions.get_calls == 1
