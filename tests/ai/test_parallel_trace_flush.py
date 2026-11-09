#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public-API reproduction of parallel tool results crossing a Trace flush."""
import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from linktools.ai.capability import AgentContext, ExecutionCapability, TraceCapability, StepPersistenceCapability
from linktools.ai.core import CommitPolicy, Principal, TaskStatus
from linktools.ai.inputs import AgentTaskInput
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.task import Task, TaskGraph, TaskNode
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

if TYPE_CHECKING:
    from linktools.ai.inputs import AgentTaskInputContext, UserPromptInput
    from linktools.ai.storage import WriteBufferEntry
    from pydantic_ai.capabilities import ValidatedToolArgs
    from pydantic_ai.messages import ModelMessage, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, StreamFunctionDef
    from pydantic_ai.tools import RunContext, ToolDefinition

class ModelBinding:
    provider = 'test'
    model_identity = 'test:parallel-trace-flush'
    contract = {'version': 3, 'provider': provider, 'model_identity': model_identity,
                'input_mime_types': [], 'settings': {}}

    def __init__(self, stream: "StreamFunctionDef") -> None:
        self.stream = stream

    def materialize(self) -> FunctionModel:
        return FunctionModel(stream_function=self.stream)


class LatencyBuffer(InMemoryCacheBackend):
    async def read_batch(self, scope: str, *, limit: int) -> "tuple[WriteBufferEntry, ...]":
        await asyncio.sleep(0.05)
        return await super().read_batch(scope, limit=limit)


class FlushBeforeTools(AbstractCapability):
    def __init__(self, first_ready: asyncio.Event, first_returned: asyncio.Event,
                 third_ready: asyncio.Event, second_started: asyncio.Event) -> None:
        self.first_ready, self.first_returned = first_ready, first_returned
        self.third_ready, self.second_started = third_ready, second_started

    async def before_tool_execute(
        self, ctx: "RunContext[None]", *, call: "ToolCallPart",
        tool_def: "ToolDefinition", args: "ValidatedToolArgs",
    ) -> "ValidatedToolArgs":
        if tool_def.name != 'read_synthetic':
            return args
        if args['label'] == 'second':
            await asyncio.wait_for(self.first_returned.wait(), 10)
            await asyncio.sleep(0.03)
            self.second_started.set()
        elif args['label'] == 'third':
            await asyncio.wait_for(self.first_ready.wait(), 10)
        await TraceCapability.for_context(ctx).flush()
        if args['label'] == 'first':
            self.first_ready.set()
        if args['label'] == 'third':
            self.third_ready.set()
        return args


@pytest.mark.asyncio
@pytest.mark.parametrize('buffered', [False, True])
async def test_parallel_tools_remain_writable_while_another_tool_flushes(tmp_path: Path, buffered: bool) -> None:
    first_ready, first_returned, third_ready, second_started = [asyncio.Event() for _ in range(4)]
    calls = []

    async def stream(messages: "list[ModelMessage]", info: "AgentInfo") -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        returned = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        if not returned:
            yield {i: DeltaToolCall(name='read_synthetic', json_args=json.dumps({'label': label}),
                tool_call_id='call-' + label) for i, label in enumerate(('first', 'second', 'third'))}
        else:
            assert len(returned) == 3
            yield 'All three synthetic pages read.'

    async def read_synthetic(ctx: AgentContext[None], label: str) -> dict[str, str]:
        """Read an independent synthetic page without external effects."""
        if label == 'first':
            await asyncio.wait_for(third_ready.wait(), 10)
            first_returned.set()
        elif label == 'third':
            await asyncio.wait_for(second_started.wait(), 10)
        calls.append(label)
        return {'label': label, 'text': 'Synthetic page'}

    features = RuntimeFeatures('parallel-trace-flush')
    features.model('offline', ModelBinding(stream))
    features.agent('worker', model='offline', allow_tools=('read_synthetic',))
    features.tool(read_synthetic, effect_policy='none')
    features.capability(ExecutionCapability(commit=CommitPolicy.sync(), retain_history=True))
    features.capability(StepPersistenceCapability())
    features.capability(TraceCapability(commit=CommitPolicy.batch(
        buffer=LatencyBuffer(max_bytes=2**22), batch_size=128, flush_interval=3)
        if buffered else CommitPolicy.sync()))
    features.capability(FlushBeforeTools(first_ready, first_returned, third_ready, second_started), id='public-flush-hook')
    async with Runtime.open('parallel-trace-flush', features=features,
            storage=PersistenceStorage.from_root(tmp_path / 'runtime')) as runtime:
        principal = Principal('analyst', runtime.tenant_id)

        async def build_input(context: "AgentTaskInputContext") -> "UserPromptInput":
            return context.prompt

        task = Task.from_agent('parallel.worker', runtime.agents.get('worker'), build_input=build_input)
        run = await (await runtime.tasks.bind(task)).start(TaskGraph('parallel', (
            TaskNode('worker', task=task, input=AgentTaskInput('Read three independent pages.')),
        )), principal=principal, idempotency_key='once')
        outcome = await run.wait(timeout_seconds=30)
        assert outcome.result.status is TaskStatus.SUCCEEDED, outcome.result
        assert sorted(calls) == ['first', 'second', 'third']
