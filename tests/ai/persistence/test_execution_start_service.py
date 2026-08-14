#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Execution service start-claim race coverage."""

import asyncio

import pytest
from linktools.ai.runtime import RuntimeState
from linktools.ai.core import Page, Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ExecutionRequest
from linktools.ai.runtime import RuntimeDomain
from linktools.ai.runtime.composition_api import DefaultExecutionService


class _History:
    async def trace(self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int) -> Page[object]:
        return Page((), None)

    async def transcript(self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int) -> Page[object]:
        return Page((), None)


class _Launcher:
    def __init__(self) -> None:
        self.calls = 0

    async def start(self, request: ExecutionRequest, execution: object) -> None:
        self.calls += 1
        await asyncio.sleep(0.01)

    async def cancel(self, execution: object) -> None:
        return None


@pytest.mark.asyncio
async def test_execution_start_claim_has_one_launcher_winner() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="service-start", tenant_id="tenant")
    launcher = _Launcher()
    service = DefaultExecutionService(
        state.execution,
        state._object_store(RuntimeDomain.EXECUTION),
        TenantAuthorizationPolicy(),
        sessions=state.conversation.sessions,
        backend=launcher,
        operation_ids=iter(("execution-a", "execution-b")).__next__,
        history_reader=_History(),
    )
    request = ExecutionRequest(prompt="hello", principal=Principal("owner", "tenant"), idempotency_key="same", memory_scope="test")
    first, second = await asyncio.gather(service.run("a" * 64, request), service.run("a" * 64, request))
    assert first.execution_id == second.execution_id
    assert launcher.calls == 1
    await state.close()


@pytest.mark.asyncio
async def test_execution_memory_scope_can_be_disabled_but_not_blank() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="memory-namespace-validation", tenant_id="tenant")
    try:
        service = DefaultExecutionService(
            state.execution,
            state._object_store(RuntimeDomain.EXECUTION),
            TenantAuthorizationPolicy(),
            sessions=state.conversation.sessions,
            backend=_Launcher(),
            history_reader=_History(),
        )
        principal = Principal("owner", "tenant")
        handle = await service.run(
            "a" * 64,
            ExecutionRequest("without memory", principal, "without-memory", memory_scope=None),
        )
        execution = await state.execution.executions.get(handle.execution_id, tenant_id=principal.tenant_id)
        assert execution is not None
        assert execution.memory_scope is None

        for value in ("", "  "):
            with pytest.raises(AIError) as error:
                await service.run(
                    "a" * 64,
                    ExecutionRequest("invalid memory", principal, f"invalid-{len(value)}", memory_scope=value),
                )
            assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID
    finally:
        await state.close()
