#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public read-only Runtime history composition coverage."""

from types import SimpleNamespace

import pytest

from linktools.ai.core import (
    Principal,
    ResourceKind,
    ResourceRef,
    TenantAuthorizationPolicy,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import (
    DefaultExecutionHistoryService,
    ExecutionHistoryItem,
    ExecutionTraceItem,
    Page,
    RuntimeHistory,
    TranscriptItem,
)
from linktools.ai.workspace import Workspace


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
async def test_runtime_history_opens_without_model_or_agent_composition(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    workspace = Workspace.load(tmp_path)

    import linktools.ai.runtime._factory as runtime_factory

    def fail_model_build(_workspace: Workspace) -> object:
        raise AssertionError("RuntimeHistory must not build models")

    monkeypatch.setattr(runtime_factory, "_build_default_models", fail_model_build)

    async with RuntimeHistory.open(workspace) as history:
        assert history.tenant_id == "default"
        with pytest.raises(AIError) as error:
            await history.history(
                "missing-execution",
                principal=Principal("caller", "default", "service"),
            )
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
