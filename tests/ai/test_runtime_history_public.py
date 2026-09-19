#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public read-only Runtime history composition coverage."""

from types import SimpleNamespace

import pytest

from linktools.ai.core import (
    HmacCursorSigner,
    Principal,
    PrincipalKind,
    ResourceKind,
    ResourceRef,
    SessionStatus,
    TenantAuthorizationPolicy,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import (
    ExecutionEvent,
    ExecutionHistoryItem,
    ExecutionTraceItem,
    Page,
    TranscriptItem,
    RuntimeState,
)
from linktools.ai.runtime._history_service import DefaultExecutionHistoryService
from linktools.ai.runtime._runtime_history import RuntimeHistory


class _Executions:
    async def get_header(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef | None:
        if execution_id == "execution" and tenant_id == "tenant":
            return ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id)
        return None

    async def get(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> object | None:
        if execution_id == "execution" and tenant_id == "tenant":
            return SimpleNamespace(execution_id=execution_id, tenant_id=tenant_id)
        return None


class _Reader:
    async def history(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: str | None,
        limit: int,
    ) -> Page[ExecutionHistoryItem]:
        assert tenant_id == "tenant"
        assert cursor is None
        assert limit == 100
        return Page((ExecutionHistoryItem(execution_id, 0, "user", "hello"),))

    async def trace(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: str | None,
        limit: int,
    ) -> Page[ExecutionTraceItem]:
        assert tenant_id == "tenant"
        return Page((ExecutionTraceItem(execution_id, 0, {"kind": "TEST"}),))

    async def transcript(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: str | None,
        limit: int,
    ) -> Page[TranscriptItem]:
        assert tenant_id == "tenant"
        return Page((TranscriptItem(execution_id, 0, "hello"),))


@pytest.mark.asyncio
async def test_execution_history_service_owns_authorization_boundary() -> None:
    service = DefaultExecutionHistoryService(
        _Executions(),
        TenantAuthorizationPolicy("tenant"),
        _Reader(),
    )
    principal = Principal("caller", "tenant", "service")

    history = await service.history("execution", principal=principal)
    trace = await service.trace("execution", principal=principal)
    transcript = await service.transcript("execution", principal=principal)

    assert history.items[0].content is None
    assert history.items[0].content_included is False
    assert trace.items[0].payload == {"kind": "TEST"}
    assert transcript.items[0].text is None
    assert transcript.items[0].content_included is False

    raw_history = await service.history(
        "execution",
        principal=principal,
        include_content=True,
    )
    raw_transcript = await service.transcript(
        "execution",
        principal=principal,
        include_content=True,
    )
    assert raw_history.items[0].content == "hello"
    assert raw_history.items[0].content_included is True
    assert raw_transcript.items[0].text == "hello"
    assert raw_transcript.items[0].content_included is True

    with pytest.raises(AIError) as error:
        await service.history(
            "execution",
            principal=Principal("caller", "other-tenant", "service"),
        )
    assert error.value.code is ErrorCode.AUTHORIZATION_DENIED


class _PagingReader(_Reader):
    async def history(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: str | None,
        limit: int,
    ) -> Page[ExecutionHistoryItem]:
        assert tenant_id == "tenant"
        if cursor is None:
            return Page(
                (ExecutionHistoryItem(execution_id, 0, "user", "hello"),),
                "inner-next",
            )
        assert cursor == "inner-next"
        return Page(
            (ExecutionHistoryItem(execution_id, 1, "assistant", "world"),),
            None,
        )


@pytest.mark.asyncio
async def test_history_cursor_binds_content_mode() -> None:
    service = DefaultExecutionHistoryService(
        _Executions(),
        TenantAuthorizationPolicy("tenant"),
        _PagingReader(),
        HmacCursorSigner("public-history", b"public-history-key"),
    )
    principal = Principal("caller", "tenant", "service")

    page = await service.history("execution", principal=principal)
    assert page.next_cursor is not None
    with pytest.raises(AIError) as raised:
        await service.history(
            "execution",
            principal=principal,
            cursor=page.next_cursor,
            include_content=True,
        )
    assert raised.value.code is ErrorCode.CURSOR_INVALID

    second = await service.history(
        "execution",
        principal=principal,
        cursor=page.next_cursor,
    )
    assert second.items[0].content is None
    assert second.items[0].content_included is False


class _EventExecutions:
    async def get_header(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef | None:
        if execution_id == "execution" and tenant_id == "tenant":
            return ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id)
        return None

    async def get(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> object | None:
        if execution_id == "execution" and tenant_id == "tenant":
            return SimpleNamespace(
                execution_id=execution_id,
                tenant_id=tenant_id,
                event_sequence=2,
            )
        return None


class _Events:
    def __init__(self) -> None:
        self.values = (
            ExecutionEvent("execution", 1, "STARTED", {"secret": "one"}),
            ExecutionEvent("execution", 2, "SUCCEEDED", {"secret": "two"}),
            ExecutionEvent("execution", 3, "LATE", {"secret": "late"}),
        )

    async def list(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        after_sequence: int,
        limit: int,
    ) -> Page[ExecutionEvent]:
        assert execution_id == "execution"
        assert tenant_id == "tenant"
        selected = tuple(
            value
            for value in self.values
            if value.sequence > after_sequence
        )[:limit]
        return Page(selected, None)


@pytest.mark.asyncio
async def test_runtime_history_execution_events_use_fixed_safe_cutoff() -> None:
    principal = Principal("caller", "tenant", "service")
    history = RuntimeHistory(
        DefaultExecutionHistoryService(
            _EventExecutions(),  # type: ignore[arg-type]
            TenantAuthorizationPolicy("tenant"),
            _Reader(),
        ),
        tenant_id="tenant",
        executions=_EventExecutions(),  # type: ignore[arg-type]
        events=_Events(),  # type: ignore[arg-type]
        authorization=TenantAuthorizationPolicy("tenant"),
        cursor_signer=HmacCursorSigner(
            "runtime-history",
            b"runtime-history-key",
        ),
    )

    first = await history.list_events(
        "execution",
        principal=principal,
        limit=1,
    )
    assert [event.sequence for event in first.items] == [1]
    assert first.items[0].payload == {}
    assert first.next_cursor is not None

    with pytest.raises(AIError) as raised:
        await history.list_events(
            "execution",
            principal=principal,
            cursor=first.next_cursor,
            include_content=True,
            limit=1,
        )
    assert raised.value.code is ErrorCode.CURSOR_INVALID

    second = await history.list_events(
        "execution",
        principal=principal,
        cursor=first.next_cursor,
        limit=1,
    )
    assert [event.sequence for event in second.items] == [2]
    assert second.items[0].payload == {}
    assert second.next_cursor is None


@pytest.mark.asyncio
async def test_runtime_history_opens_without_model_or_agent_composition() -> None:
    async with RuntimeHistory.open(
        "workspace",
        state=RuntimeState.in_memory(),
    ) as history:
        assert history.tenant_id == "default"
        with pytest.raises(AIError) as error:
            await history.history(
                "missing-execution",
                principal=Principal("caller", "default", "service"),
            )
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED


class _Sessions:
    def __init__(self) -> None:
        from datetime import datetime, timedelta, timezone

        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.records = (
            SimpleNamespace(
                session_id="older",
                agent_id="default",
                status=SessionStatus.OPEN,
                revision=1,
                cwd=".",
                active_execution_id=None,
                history_quality="complete",
                metadata={},
                created_at=base,
                updated_at=base,
                tenant_id="tenant",
                owner_principal_id="runtime",
            ),
            SimpleNamespace(
                session_id="newer",
                agent_id="auditor",
                status=SessionStatus.OPEN,
                revision=3,
                cwd=None,
                active_execution_id="execution",
                history_quality="complete",
                metadata={},
                created_at=base + timedelta(minutes=1),
                updated_at=base + timedelta(minutes=2),
                tenant_id="tenant",
                owner_principal_id="runtime",
            ),
        )

    async def list_page(
        self,
        *,
        tenant_id: str,
        owner_principal_id: str | None,
        cursor: str | None,
        limit: int,
        snapshot: int | None = None,
    ) -> tuple[int, Page[object]]:
        del limit
        values = tuple(
            record
            for record in self.records
            if record.tenant_id == tenant_id
            and (
                owner_principal_id is None
                or record.owner_principal_id == owner_principal_id
            )
        )
        start = 0 if cursor is None else int(cursor)
        end = min(len(values), start + 1)
        next_cursor = None if end == len(values) else str(end)
        return (
            1 if snapshot is None else snapshot,
            Page(values[start:end], next_cursor),
        )

    async def get_header(
        self,
        session_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef | None:
        record = next(
            (
                value
                for value in self.records
                if value.session_id == session_id and value.tenant_id == tenant_id
            ),
            None,
        )
        if record is None:
            return None
        return ResourceRef(
            ResourceKind.SESSION,
            record.session_id,
            record.tenant_id,
            record.owner_principal_id,
        )

    async def get(
        self,
        session_id: str,
        *,
        tenant_id: str,
    ) -> object | None:
        return next(
            (
                value
                for value in self.records
                if value.session_id == session_id and value.tenant_id == tenant_id
            ),
            None,
        )


@pytest.mark.asyncio
async def test_runtime_history_projects_owned_sessions_without_runtime_open() -> None:
    sessions = _Sessions()
    history = RuntimeHistory(
        SimpleNamespace(),
        tenant_id="tenant",
        sessions=sessions,  # type: ignore[arg-type]
        authorization=TenantAuthorizationPolicy("tenant"),
    )
    principal = Principal(
        "runtime",
        "tenant",
        PrincipalKind.LOCAL_TRUSTED.value,
    )

    recent = await history.recent_sessions(principal=principal, limit=20)
    selected = await history.inspect_session("newer", principal=principal)

    assert [item.session_id for item in recent] == ["newer", "older"]
    assert selected.agent_id == "auditor"
    assert selected.active_execution_ids == ("execution",)

    with pytest.raises(AIError) as denied:
        await history.inspect_session(
            "newer",
            principal=Principal("other", "tenant", PrincipalKind.LOCAL_TRUSTED.value),
        )
    assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
