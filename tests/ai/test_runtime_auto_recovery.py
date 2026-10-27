#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from linktools.ai.task.engine import TaskEngine

from linktools.ai.persistence import Persistence
"""Startup recovery is optional; explicit recovery and owned close are not."""
from ._runtime_test_helpers import model_features

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus, JsonValue, TaskStatus, canonical_sha256
from linktools.ai.errors import ErrorCode

from linktools.ai.capability import CheckpointCapability
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.execution import ExecutionRequest
from linktools.ai.persistence.contracts import ExecutionRecord
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext

from .test_live_history_readback_integration import _Models


_NAMESPACE = "startup-recovery"


def _capabilities() -> RuntimeFeatures[None]:
    group: RuntimeFeatures[None] = RuntimeFeatures(_NAMESPACE)
    group.capability(CheckpointCapability(), scope="runtime")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    return group


async def _admit_without_worker(
    path: Path, models: _Models, monkeypatch: pytest.MonkeyPatch,
) -> ExecutionRecord:
    storage = PersistenceStorage.filesystem(path)
    async with Persistence.open(_NAMESPACE, storage=storage) as persistence_41, Runtime.attach(persistence=persistence_41, features=model_features(models, _capabilities())) as runtime:
        async def interrupted_launch(
            request: ExecutionRequest, execution: ExecutionRecord, **kwargs: object,
        ) -> None:
            del request, execution, kwargs

        # The durable admission survives a worker that never reaches launch.
        with monkeypatch.context() as fault:
            fault.setattr(runtime._execution_service.runtime_backend(), "launch", interrupted_launch)
            run = await runtime.agents.get().start("recover this admission", idempotency_key="admission")
        record = await runtime._state.execution.executions.get(run.execution_id, tenant_id=runtime.tenant_id)
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
    storage = PersistenceStorage.filesystem(tmp_path / "state")
    options = {"auto_recover": True} if explicit_default else {}
    async with Persistence.open(_NAMESPACE, storage=storage) as persistence_75, Runtime.attach(persistence=persistence_75, features=model_features(models, _capabilities()), **options) as runtime:
        result = (await runtime.executions.wait(
            before.execution_id, principal=runtime.default_principal, timeout_seconds=10,
        )).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert result.output == {"text": "recovered admission"}
    assert calls == ["recovered"]
    assert not runtime._state.ready


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
    storage = PersistenceStorage.filesystem(tmp_path / "state")
    async with Persistence.open(_NAMESPACE, storage=storage) as persistence_101, Runtime.attach(persistence=persistence_101, features=model_features(models, _capabilities()), auto_recover=False) as runtime:
        assert await runtime._state.execution.executions.get(
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
    assert not runtime._state.ready


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
    owner_storage = PersistenceStorage.sqlite(path)
    other_storage = PersistenceStorage.sqlite(path)
    assert owner_storage is not other_storage
    async with Persistence.open(_NAMESPACE, storage=owner_storage) as persistence_159, Runtime.attach(persistence=persistence_159, features=model_features(_Models(FunctionModel(stream_function=owner_model)), _capabilities())) as owner:
        try:
            owner_run = await owner.agents.get().start("keep running", idempotency_key="owner")
            await asyncio.wait_for(owner_entered.wait(), 10)
            before = await owner._state.execution.executions.get(owner_run.execution_id, tenant_id=owner.tenant_id)
            head = await owner._state.execution.executions.get_history_head(
                owner_run.execution_id, tenant_id=owner.tenant_id,
            )
            checkpoint = await owner._state.recovery.checkpoints.get(
                owner_run.execution_id, tenant_id=owner.tenant_id,
            )
            async with Persistence.open(_NAMESPACE, storage=other_storage) as persistence_173, Runtime.attach(persistence=persistence_173, features=model_features(_Models(FunctionModel(stream_function=owned_model)), _capabilities()), auto_recover=False) as other:
                assert other.namespace == owner.namespace
                assert owned_calls == []
                assert await other._state.execution.executions.get(
                    owner_run.execution_id, tenant_id=other.tenant_id,
                ) == before
                observed_head = await other._state.execution.executions.get_history_head(
                    owner_run.execution_id, tenant_id=other.tenant_id,
                )
                assert head is not None and observed_head is not None
                assert observed_head.producer_generation == head.producer_generation
                assert observed_head.producer_claim_id == head.producer_claim_id
                assert await other._state.recovery.checkpoints.get(
                    owner_run.execution_id, tenant_id=other.tenant_id,
                ) == checkpoint
                owned_run = await other.agents.get().start("close only this worker", idempotency_key="owned")
                await asyncio.wait_for(owned_entered.wait(), 10)
            assert not other._state.ready
            assert owner._state.ready
            assert owned_stopped.is_set()
            assert not owner_stopped.is_set()
            assert owned_calls == ["owned"]
            owned = await owner._state.execution.executions.get(owned_run.execution_id, tenant_id=owner.tenant_id)
            assert owned is not None and owned.status is ExecutionStatus.CANCELLED
            assert await owner._state.execution.executions.get(
                owner_run.execution_id, tenant_id=owner.tenant_id,
            ) == before
            after_close_head = await owner._state.execution.executions.get_history_head(
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
    assert not owner._state.ready


@pytest.mark.asyncio
async def test_explicit_graph_recovery_remains_available_when_startup_disabled(tmp_path: Path) -> None:
    calls: list[str] = []

    async def execute(context: TaskNodeContext[None]) -> JsonValue:
        calls.append(context.graph_id)
        return "graph recovered"

    task = Task("test.startup-recovery", execute, effect_policy="none")
    graph = TaskGraph("recover-graph", (TaskNode("node", task=task),))
    storage = PersistenceStorage.filesystem(tmp_path / "state")
    async with Persistence.open(_NAMESPACE, storage=storage) as persistence_228, Runtime.attach(features=RuntimeFeatures(), persistence=persistence_228) as runtime, TaskEngine.open(persistence=persistence_228, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        submission = await runtime_engine.with_definitions(task).prepare_submission(graph, idempotency_key="graph-admission")
        await runtime._state.task.admissions.admit_prepared(submission)
        lease = await runtime._state.task.tasks.claim(
            graph.graph_id, "node", tenant_id=runtime.tenant_id, owner="interrupted-worker", lease_seconds=30,
        )
        await runtime._state.task.tasks.mark_recovery_required(
            lease, tenant_id=runtime.tenant_id, error_code=ErrorCode.STORAGE_RECOVERY_REQUIRED.value,
            error_digest=canonical_sha256({"phase": "worker-interrupted"}),
        )
    storage = PersistenceStorage.filesystem(tmp_path / "state")
    async with Persistence.open(_NAMESPACE, storage=storage) as persistence_239, Runtime.attach(features=RuntimeFeatures(), persistence=persistence_239, auto_recover=False) as runtime, TaskEngine.open(persistence=persistence_239, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        run = await runtime_engine.with_definitions(task).get(graph.graph_id)
        assert (await run.state()).status is TaskStatus.RECOVERY_REQUIRED
        assert calls == []
        await run.recover(idempotency_key="explicit-graph-recovery")
        assert (await run.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
        assert calls == [graph.graph_id]
    assert not runtime._state.ready


def test_startup_recovery_option_requires_a_boolean() -> None:
    with pytest.raises(TypeError, match="auto_recover must be bool"):
        Runtime.open("invalid-option", storage=PersistenceStorage.in_memory(), auto_recover="false")
