#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Captured dependency selection reads its authorized contract once per operation."""

from pathlib import Path
import json

import pytest

from linktools.ai.core import JsonValue, Principal, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.inputs import AgentTaskInput, AgentTaskInputContext, CaptureInputRequest, InputCaptures
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext, TaskNodeResultRef
from linktools.ai.task.engine import TaskEngine
from ._runtime_test_helpers import model_features
from .test_task_mixed_node_reliability import _TaskTestModels


@pytest.mark.asyncio
@pytest.mark.parametrize('value', [False, 0, {'number': 17}])
async def test_captured_dependency_reads_one_contract(tmp_path: Path, monkeypatch, value: JsonValue) -> None:
    task_reads = []
    storage_reads = []
    original_read = InputCaptures.read_task
    original_get = InputCaptures._get

    async def counted_read(self, reference, *, principal):
        task_reads.append(reference)
        return await original_read(self, reference, principal=principal)

    async def counted_get(self, key, digest=None):
        storage_reads.append(key)
        return await original_get(self, key, digest)

    monkeypatch.setattr(InputCaptures, 'read_task', counted_read)
    monkeypatch.setattr(InputCaptures, '_get', counted_get)
    counts = []

    async def source(context: TaskNodeContext[None]) -> JsonValue:
        return value

    async def build(context: AgentTaskInputContext) -> str:
        before = len(task_reads), len(storage_reads)
        result = await context.result('alias')
        counts.append((len(task_reads)-before[0], len(storage_reads)-before[1]))
        if len(counts) == 2:
            assert await context.result('live') == value
        return json.dumps(result)

    producer = Task('reuse.source', source, effect_policy='none')
    async with Persistence.open('capture-reuse', storage=PersistenceStorage.from_root(tmp_path, exclusive_writer=True)) as persistence, Runtime.open(persistence=persistence, features=model_features(_TaskTestModels())) as runtime, TaskEngine.open(persistence=persistence, context=runtime.context, authorization=runtime._execution_service._authorization) as engine:
        principal = Principal('capture', runtime.tenant_id)
        consumer = Task.from_agent('reuse.target', runtime.agents.get(), build_input=build)
        engine = engine.with_definitions(producer, consumer)
        first = await engine.start(TaskGraph('source', (
            TaskNode('source', task=producer),
            TaskNode('target', ('source',), task=consumer, input=AgentTaskInput(), input_refs={'alias': TaskNodeResultRef('source')}),
        )), idempotency_key='capture-reuse-source', principal=principal)
        assert (await first.wait()).result.status is TaskStatus.SUCCEEDED
        execution = await first.execution('target')
        capture = await engine.inputs.capture_input(execution.execution_id,
            CaptureInputRequest(principal, 'capture-reuse-input', 'clean'))
        capture = await engine.inputs.task_input(capture, principal=principal, input_mode='reproject_input')
        second = await engine.start(TaskGraph('replay', (
            TaskNode('live', task=producer),
            TaskNode('target', ('live',), task=consumer, input_capture=capture),
        )), idempotency_key='capture-reuse-replay', principal=principal)
        assert (await second.wait()).result.status is TaskStatus.SUCCEEDED
        assert counts == [(0, 0), (2, 5)]
        assert await engine.inputs.find_dependency(capture, 'absent', principal=principal) == (False, None)
        with pytest.raises(AIError) as missing:
            await engine.inputs.read_dependency(capture, 'absent', principal=principal)
        assert missing.value.code is ErrorCode.REQUEST_FIELD_INVALID
