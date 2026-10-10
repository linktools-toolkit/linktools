#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Startup recovery is optional; explicit recovery and owned close are not."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import ExecutionStatus, JsonValue, TaskStatus, canonical_sha256
from linktools.ai.errors import ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime.service_api import ExecutionRequest
from linktools.ai.runtime.state._contracts import ExecutionRecord
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext

from .test_live_history_readback_integration import _Models


_NAMESPACE = "startup-recovery"


def _capabilities() -> CapabilityGroup[None]:
    group: CapabilityGroup[None] = CapabilityGroup(_NAMESPACE)
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    return group


async def _admit_without_worker(
    path: Path, models: _Models, monkeypatch: pytest.MonkeyPatch,
) -> ExecutionRecord:
    storage = RuntimeStorage.filesystem(path)
    async with Runtime.open(
        _NAMESPACE, models=models, storage=storage, capabilities=(_capabilities(),),
    ) as runtime:
        async def interrupted_launch(
            request: ExecutionRequest, execution: ExecutionRecord, **kwargs: object,
        ) -> None:
            del request, execution, kwargs

        # The durable admission survives a worker that never reaches launch.
        with monkeypatch.context() as fault:
            fault.setattr(runtime._execution_service.runtime_backend(), "launch", interrupted_launch)
            run = await runtime.agents.get().start("recover this admission", idempotency_key="admission")
        record = await storage.execution.executions.get(run.execution_id, tenant_id=runtime.tenant_id)
        assert record is not None and record.status is ExecutionStatus.STARTED
        return record


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_default", (False, True))
async def test_startup_recovery_defaults_to_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_default: bool,
) -> None:
    calls: list[str] = []

    async def model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        del messages, info
        calls.append("recovered")
        yield "recovered admission"

    models = _Models(FunctionModel(stream_function=model))
    before = await _admit_without_worker(tmp_path / "state", models, monkeypatch)
    assert calls == []
    storage = RuntimeStorage.filesystem(tmp_path / "state")
    options = {"auto_recover": True} if explicit_default else {}
    async with Runtime.open(
        _NAMESPACE, models=models, storage=storage, capabilities=(_capabilities(),), **options,
    ) as runtime:
        result = (await runtime.executions.wait(
            before.execution_id, principal=runtime.default_principal, timeout_seconds=10,
        )).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert result.output == {"text": "recovered admission"}
    assert calls == ["recovered"]
    assert not storage.ready


@pytest.mark.asyncio
async def test_disabled_startup_preserves_admission_until_explicit_execution_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        del messages, info
        calls.append("recovered")
        yield "explicit recovery"

    models = _Models(FunctionModel(stream_function=model))
    before = await _admit_without_worker(tmp_path / "state", models, monkeypatch)
    storage = RuntimeStorage.filesystem(tmp_path / "state")
    async with Runtime.open(
        _NAMESPACE, models=models, storage=storage, capabilities=(_capabilities(),), auto_recover=False,
    ) as runtime:
        assert await storage.execution.executions.get(
            before.execution_id, tenant_id=runtime.tenant_id,
        ) == before
        assert calls == []
        await runtime.executions.recover(
            before.execution_id, principal=runtime.default_principal, idempotency_key="explicit-recovery",
        )
        result = (await runtime.executions.wait(
            before.execution_id, principal=runtime.default_principal, timeout_seconds=10,
        )).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert result.output == {"text": "explicit recovery"}
        await runtime.executions.recover(
            before.execution_id, principal=runtime.default_principal, idempotency_key="explicit-recovery",
        )
    assert calls == ["recovered"]
    assert not storage.ready


@pytest.mark.asyncio
async def test_disabled_startup_and_close_preserve_other_runtime_owned_execution(
    tmp_path: Path,
) -> None:
    owner_entered = asyncio.Event()
    owned_entered = asyncio.Event()
    release_owner = asyncio.Event()
    owned_stopped = asyncio.Event()
    owner_stopped = asyncio.Event()
    owner_calls: list[str] = []
    owned_calls: list[str] = []

    async def owner_model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        del messages, info
        owner_calls.append("owner")
        owner_entered.set()
        try:
            await release_owner.wait()
            yield "owner finished"
        finally:
            owner_stopped.set()

    async def owned_model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        del messages, info
        owned_calls.append("owned")
        owned_entered.set()
        try:
            await asyncio.Event().wait()
            yield "unreachable"
        finally:
            owned_stopped.set()

    path = tmp_path / "state"
    owner_storage = RuntimeStorage.sqlite(path)
    other_storage = RuntimeStorage.sqlite(path)
    assert owner_storage is not other_storage
    async with Runtime.open(
        _NAMESPACE, models=_Models(FunctionModel(stream_function=owner_model)),
        storage=owner_storage, capabilities=(_capabilities(),),
    ) as owner:
        try:
            owner_run = await owner.agents.get().start("keep running", idempotency_key="owner")
            await asyncio.wait_for(owner_entered.wait(), 10)
            before = await owner_storage.execution.executions.get(owner_run.execution_id, tenant_id=owner.tenant_id)
            head = await owner_storage.execution.executions.get_history_head(
                owner_run.execution_id, tenant_id=owner.tenant_id,
            )
            checkpoint = await owner_storage.recovery.checkpoints.get(
                owner_run.execution_id, tenant_id=owner.tenant_id,
            )
            async with Runtime.open(
                _NAMESPACE, models=_Models(FunctionModel(stream_function=owned_model)),
                storage=other_storage, capabilities=(_capabilities(),), auto_recover=False,
            ) as other:
                assert other.namespace == owner.namespace
                assert owned_calls == []
                assert await other_storage.execution.executions.get(
                    owner_run.execution_id, tenant_id=other.tenant_id,
                ) == before
                observed_head = await other_storage.execution.executions.get_history_head(
                    owner_run.execution_id, tenant_id=other.tenant_id,
                )
                assert head is not None and observed_head is not None
                assert observed_head.producer_generation == head.producer_generation
                assert observed_head.producer_claim_id == head.producer_claim_id
                assert await other_storage.recovery.checkpoints.get(
                    owner_run.execution_id, tenant_id=other.tenant_id,
                ) == checkpoint
                owned_run = await other.agents.get().start("close only this worker", idempotency_key="owned")
                await asyncio.wait_for(owned_entered.wait(), 10)
            assert not other_storage.ready
            assert owner_storage.ready
            assert owned_stopped.is_set()
            assert not owner_stopped.is_set()
            assert owned_calls == ["owned"]
            owned = await owner_storage.execution.executions.get(owned_run.execution_id, tenant_id=owner.tenant_id)
            assert owned is not None and owned.status is ExecutionStatus.CANCELLED
            assert await owner_storage.execution.executions.get(
                owner_run.execution_id, tenant_id=owner.tenant_id,
            ) == before
            after_close_head = await owner_storage.execution.executions.get_history_head(
                owner_run.execution_id, tenant_id=owner.tenant_id,
            )
            assert after_close_head is not None
            assert after_close_head.producer_generation == head.producer_generation
            assert after_close_head.producer_claim_id == head.producer_claim_id
            release_owner.set()
            assert (await owner_run.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
            assert owner_calls == ["owner"]
        finally:
            release_owner.set()
    assert not owner_storage.ready


@pytest.mark.asyncio
async def test_explicit_graph_recovery_remains_available_when_startup_disabled(tmp_path: Path) -> None:
    calls: list[str] = []

    async def execute(context: TaskNodeContext[None]) -> JsonValue:
        calls.append(context.graph_id)
        return "graph recovered"

    task = Task("test.startup-recovery", execute, effect_policy="none")
    graph = TaskGraph("recover-graph", (TaskNode("node", task=task),))
    storage = RuntimeStorage.filesystem(tmp_path / "state")
    async with Runtime.open(_NAMESPACE, models=ModelRegistry(), storage=storage) as runtime:
        submission = await runtime.tasks.bind(task).prepare_submission(graph, idempotency_key="graph-admission")
        await storage.task.admissions.admit_prepared(submission)
        lease = await storage.task.tasks.claim(
            graph.graph_id, "node", tenant_id=runtime.tenant_id, owner="interrupted-worker", lease_seconds=30,
        )
        await storage.task.tasks.mark_recovery_required(
            lease, tenant_id=runtime.tenant_id, error_code=ErrorCode.STORAGE_RECOVERY_REQUIRED.value,
            error_digest=canonical_sha256({"phase": "worker-interrupted"}),
        )
    storage = RuntimeStorage.filesystem(tmp_path / "state")
    async with Runtime.open(
        _NAMESPACE, models=ModelRegistry(), storage=storage, auto_recover=False,
    ) as runtime:
        run = await runtime.tasks.bind(task).get(graph.graph_id)
        assert (await run.state()).status is TaskStatus.RECOVERY_REQUIRED
        assert calls == []
        await run.recover(idempotency_key="explicit-graph-recovery")
        assert (await run.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
        assert calls == [graph.graph_id]
    assert not storage.ready


def test_startup_recovery_option_requires_a_boolean() -> None:
    with pytest.raises(TypeError, match="auto_recover must be bool"):
        Runtime.open(_NAMESPACE, models=ModelRegistry(), storage=RuntimeStorage.in_memory(), auto_recover="false")
