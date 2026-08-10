#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Execution service start-claim race coverage."""

import asyncio

import pytest
from linktools.ai.adapter import build_in_memory_runtime
from linktools.ai.core import Page, Principal, TenantAuthorizationPolicy
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
        launcher=launcher,
        operation_ids=iter(("execution-a", "execution-b")).__next__,
        history_reader=_History(),
    )
    request = ExecutionRequest(prompt="hello", principal=Principal("owner", "tenant"), idempotency_key="same", memory_namespace="test")
    first, second = await asyncio.gather(service.run("a" * 64, request), service.run("a" * 64, request))
    assert first.execution_id == second.execution_id
    assert launcher.calls == 1
    await runtime.close()
