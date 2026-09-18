#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public read-only Runtime history composition coverage."""

from types import SimpleNamespace

import pytest

from linktools.ai.core import (
    Principal,
    PrincipalKind,
    ResourceKind,
    ResourceRef,
    SessionStatus,
    TenantAuthorizationPolicy,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import (
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

    assert history.items[0].content == "hello"
    assert trace.items[0].payload == {"kind": "TEST"}
    assert transcript.items[0].text == "hello"

    with pytest.raises(AIError) as error:
        await service.history(
            "execution",
            principal=Principal("caller", "other-tenant", "service"),
        )
    assert error.value.code is ErrorCode.AUTHORIZATION_DENIED


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
                created_at=base + timedelta(minutes=1),
                updated_at=base + timedelta(minutes=2),
                tenant_id="tenant",
                owner_principal_id="runtime",
            ),
        )

    async def list(
        self,
        *,
        tenant_id: str,
        owner_principal_id: str | None = None,
    ) -> tuple[object, ...]:
        return tuple(
            record
            for record in self.records
            if record.tenant_id == tenant_id
            and (
                owner_principal_id is None
                or record.owner_principal_id == owner_principal_id
            )
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
    assert selected.active_execution_id == "execution"

    with pytest.raises(AIError) as denied:
        await history.inspect_session(
            "newer",
            principal=Principal("other", "tenant", PrincipalKind.LOCAL_TRUSTED.value),
        )
    assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
