#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable archives cannot silently omit caller-owned native checkpoints."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai_harness.step_persistence import StepPersistence
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import ExecutionStatus, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import Persistence, PersistenceStorage, SqlStepStore
from linktools.ai.persistence._snapshot import SnapshotLimits
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import InMemoryObjectStore
from linktools.ai.task import Task, TaskGraph, TaskNode

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpointed", (False, True))
@pytest.mark.parametrize("interrupted", (False, True))
async def test_portable_export_identifies_native_dependency_without_runtime_features(
    tmp_path: Path, checkpointed: bool, interrupted: bool,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'backup.db'}")
    await provision_database(engine)
    native = SqlStepStore(engine, namespace="backup", tenant_id="default")
    entered = asyncio.Event()

    async def model(_messages, _info):
        entered.set()
        if interrupted:
            await asyncio.Event().wait()
        return ModelResponse(parts=[TextPart("retained")])

    features = RuntimeFeatures("backup")
    features.capability(ExecutionCapability(retain_history=True))
    if checkpointed:
        features.capability(StepPersistence(store=native, capture_frontier=True))
    features.agent("default", model="default", allow_tools=())
    storage = PersistenceStorage.sql(engine)
    try:
        async with Persistence.open("backup", storage=storage) as state:
            async with Runtime.attach(persistence=state,
                features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
                session = await runtime.agents.get().create_session("session")
                execution = await session.start("question", idempotency_key="input")
                await asyncio.wait_for(entered.wait(), 5)
                if interrupted and checkpointed:
                    await asyncio.wait_for(runtime.handoff(), 5)
                elif interrupted:
                    await execution.cancel()
                else:
                    assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        # A read-only exporter has no capabilities or StepStore binding. The
        # durable execution guarantee, not current configuration, is the proof.
        async with Persistence.open("backup", storage=storage, read_only=True) as offline:
            archive = InMemoryObjectStore()
            limits = SnapshotLimits(max_entries=10_000, max_bytes=10_000_000)
            if checkpointed:
                with pytest.raises(AIError) as rejected:
                    await offline.export_snapshot(object_store=archive, limits=limits)
                assert rejected.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED
                assert "StepStore backup" in str(rejected.value)
            else:
                assert await offline.export_snapshot(object_store=archive, limits=limits)
    finally:
        await native.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_callable_task_without_native_store_keeps_portable_backup(tmp_path: Path) -> None:
    async def echo(context):
        return 'done'
    storage = PersistenceStorage.sqlite(tmp_path / 'task-only.db')
    task = Task('echo', echo, effect_policy='none')
    async with Persistence.open('task-only-backup', storage=storage) as state:
        async with Runtime.attach(persistence=state) as runtime:
            engine = await runtime.tasks.bind(task)
            run = await engine.start(TaskGraph('graph', (TaskNode('echo', task=task, input={}),)), idempotency_key='run')
            result = (await run.wait(timeout_seconds=5)).result
            assert result.status is TaskStatus.SUCCEEDED
            execution_id = result.node_states[0].execution_id
            record = await state.execution.executions.get(execution_id, tenant_id='default')
            assert record is not None
            assert record.restart_checkpoints and record.agent_run_seq == 0
    async with Persistence.open('task-only-backup', storage=storage, read_only=True) as offline:
        assert await offline.export_snapshot(object_store=InMemoryObjectStore(), limits=SnapshotLimits(max_entries=10000, max_bytes=10000000))
