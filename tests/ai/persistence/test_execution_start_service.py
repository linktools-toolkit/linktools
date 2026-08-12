#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Execution service start-claim race coverage."""

import asyncio

import pytest
from linktools.ai.adapter import build_in_memory_runtime
from linktools.ai.core import Page, Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import DefaultExecutionService, ExecutionRequest


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
    runtime = build_in_memory_runtime(namespace="service-start")
    await runtime.initialize()
    launcher = _Launcher()
    service = DefaultExecutionService(
        runtime.persistence,
        TenantAuthorizationPolicy(),
        backend=launcher,
        operation_ids=iter(("execution-a", "execution-b")).__next__,
        history_reader=_History(),
    )
    request = ExecutionRequest(prompt="hello", principal=Principal("owner", "tenant"), idempotency_key="same", memory_namespace="test")
    first, second = await asyncio.gather(service.run("a" * 64, request), service.run("a" * 64, request))
    assert first.execution_id == second.execution_id
    assert launcher.calls == 1
    await runtime.close()


@pytest.mark.asyncio
async def test_execution_memory_namespace_can_be_disabled_but_not_blank() -> None:
    runtime = build_in_memory_runtime(namespace="memory-namespace-validation")
    await runtime.initialize()
    try:
        service = DefaultExecutionService(
            runtime.persistence,
            TenantAuthorizationPolicy(),
            backend=_Launcher(),
            history_reader=_History(),
        )
        principal = Principal("owner", "tenant")
        handle = await service.run(
            "a" * 64,
            ExecutionRequest("without memory", principal, "without-memory", memory_namespace=None),
        )
        execution = await runtime.persistence.executions.get(handle.execution_id, tenant_id=principal.tenant_id)
        assert execution is not None
        assert execution.memory_namespace is None

        for value in ("", "  "):
            with pytest.raises(AIError) as error:
                await service.run(
                    "a" * 64,
                    ExecutionRequest("invalid memory", principal, f"invalid-{len(value)}", memory_namespace=value),
                )
            assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID
    finally:
        await runtime.close()
