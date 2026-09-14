#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session read-path I/O invariants."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from linktools.ai.core import (
    ExecutionStatus,
    HmacCursorSigner,
    Page,
    Principal,
    ResourceKind,
    ResourceRef,
    SessionStatus,
)
from linktools.ai.runtime._session import DefaultSessionService
from linktools.ai.runtime.service_api import ListSessionRequest
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


class _ListSessions:
    def __init__(self, records: tuple[SessionRecord, ...]) -> None:
        self._records = records

    async def list_page(
        self,
        *,
        tenant_id: str,
        owner_principal_id: str | None,
        cursor: str | None,
        limit: int,
        snapshot: int | None = None,
    ) -> tuple[int, Page[SessionRecord]]:
        assert tenant_id == "tenant"
        assert owner_principal_id == "principal"
        assert cursor is None
        assert limit == 10
        assert snapshot is None
        return 1, Page(self._records)


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


class _BatchExecutions:
    def __init__(self) -> None:
        self.get_calls = 0
        self.get_many_calls = 0

    async def get(self, execution_id: str, *, tenant_id: str) -> object:
        del execution_id, tenant_id
        self.get_calls += 1
        raise AssertionError("session list must not fetch active executions one by one")

    async def get_many(
        self,
        execution_ids: tuple[str, ...],
        *,
        tenant_id: str,
    ) -> dict[str, object]:
        self.get_many_calls += 1
        assert tenant_id == "tenant"
        assert execution_ids == ("execution-active", "execution-terminal")
        return {
            "execution-active": SimpleNamespace(
                execution_id="execution-active",
                tenant_id="tenant",
                session_id="session-active",
                parent_execution_id=None,
                status=ExecutionStatus.STARTED,
            ),
            "execution-terminal": SimpleNamespace(
                execution_id="execution-terminal",
                tenant_id="tenant",
                session_id="session-terminal",
                parent_execution_id=None,
                status=ExecutionStatus.SUCCEEDED,
            ),
        }


class _Authorization:
    async def authorize(self, *args: object) -> None:
        del args


def _session_record(
    session_id: str,
    *,
    active_execution_id: str | None,
) -> SessionRecord:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return SessionRecord(
        session_id=session_id,
        tenant_id="tenant",
        owner_principal_id="principal",
        status=SessionStatus.OPEN,
        revision=1,
        cwd=None,
        metadata={},
        created_at=now,
        updated_at=now,
        closed_at=None,
        active_execution_id=active_execution_id,
        agent_id="default",
    )


@pytest.mark.asyncio
async def test_session_load_reuses_reconciled_active_execution() -> None:
    record = _session_record("session", active_execution_id="execution")
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


@pytest.mark.asyncio
async def test_session_list_batches_active_execution_reads() -> None:
    records = (
        _session_record(
            "session-active",
            active_execution_id="execution-active",
        ),
        _session_record(
            "session-terminal",
            active_execution_id="execution-terminal",
        ),
        _session_record("session-idle", active_execution_id=None),
    )
    executions = _BatchExecutions()
    service = DefaultSessionService(
        SimpleNamespace(sessions=_ListSessions(records)),  # type: ignore[arg-type]
        executions,  # type: ignore[arg-type]
        _Authorization(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        HmacCursorSigner("session", b"session-key"),
        history_reader=object(),  # type: ignore[arg-type]
    )

    page = await service.list(
        ListSessionRequest(
            Principal("principal", "tenant"),
            limit=10,
        )
    )

    assert tuple(view.session_id for view in page.items) == (
        "session-active",
        "session-terminal",
        "session-idle",
    )
    assert tuple(view.active_execution_ids for view in page.items) == (
        ("execution-active",),
        (),
        (),
    )
    assert executions.get_many_calls == 1
    assert executions.get_calls == 0
