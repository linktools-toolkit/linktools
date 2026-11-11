#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public SDK reproduction of deferred file-result transcript publication."""
import asyncio
import json
from pathlib import Path
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import pytest
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ApprovalRequired
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from linktools.ai.capability import AgentContext, ExecutionCapability, FileSystem, StepPersistenceCapability
from linktools.ai.core import ApprovalDecision, service_principal
from linktools.ai.execution import ApprovalDecisionRequest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.inputs import AgentTaskInput
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.task import Task, TaskGraph, TaskNode
from linktools.ai.workspace import Workspace


if TYPE_CHECKING:
    from pydantic_ai.capabilities import ValidatedToolArgs
    from pydantic_ai.messages import ModelMessage, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, StreamFunctionDef
    from pydantic_ai.tools import RunContext, ToolDefinition


class _Binding:
    provider = 'test'
    model_identity = 'test:deferred-file'
    contract = {'version': 3, 'provider': provider, 'model_identity': model_identity,
                'input_mime_types': ['application/json'], 'settings': {}}

    def __init__(self, stream: "StreamFunctionDef") -> None:
        self.stream = stream

    def materialize(self) -> FunctionModel:
        return FunctionModel(stream_function=self.stream)


class _Gate(AbstractCapability):
    async def before_tool_execute(
        self, ctx: "RunContext[AgentContext[None]]", *, call: "ToolCallPart",
        tool_def: "ToolDefinition", args: "ValidatedToolArgs",
    ) -> "ValidatedToolArgs":
        if call.tool_name == 'continue_reading' and not ctx.tool_call_approved:
            raise ApprovalRequired()
        return args


@pytest.mark.asyncio
@pytest.mark.parametrize('change_source', [False, True])
async def test_completed_file_return_survives_deferred_reopen(tmp_path: Path, change_source: bool) -> None:
    root = tmp_path / 'workspace'
    path = root / 'context/bundle.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'fact': 'fixed synthetic source'}), encoding='utf-8')
    calls, builds = [], []

    async def model(
        messages: "list[ModelMessage]", info: "AgentInfo",
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        calls.append(True)
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if not returns:
            yield {0: DeltaToolCall(name='continue_reading', json_args='{}', tool_call_id='pause'),
                   1: DeltaToolCall(name='read_file', json_args='{"path":"context/bundle.json"}', tool_call_id='source-read')}
        else:
            assert any(part.tool_call_id == 'source-read' for part in returns)
            assert 'fixed synthetic source' in str(returns)
            yield 'Read the fixed source.'

    async def continue_reading(ctx: AgentContext[None]) -> str:
        """Continue after explicit synthetic approval."""
        return 'Approved.'

    async def build_input(context: object) -> str:
        builds.append(True)
        return 'Read the fixed file after approval.'

    def features() -> RuntimeFeatures[None]:
        result = RuntimeFeatures('deferred-file')
        result.capability(ExecutionCapability(retain_history=True))
        result.capability(StepPersistenceCapability())
        result.capability(_Gate(), id='gate')
        result.capability(FileSystem(Workspace.load(root)))
        result.model('default', _Binding(model))
        result.tool(continue_reading, effect_policy='none')
        result.agent('reader', allow_tools=('read_file', 'continue_reading'))
        return result

    principal = service_principal('default', 'deferred-file')
    storage = PersistenceStorage.from_root(tmp_path / 'state')
    async with Runtime.open('deferred-file', features=features(), storage=storage) as runtime:
        task = Task.from_agent('reader', runtime.agents.get('reader'), build_input=build_input)
        engine = await runtime.tasks.bind(task)
        try:
            run = await engine.start(TaskGraph('graph', (TaskNode('reader', task=task, input=AgentTaskInput('fixed')),)),
                                     principal=principal, idempotency_key='fixed')
            async with asyncio.timeout(15):
                while True:
                    try:
                        execution = await run.execution('reader')
                        view = await runtime.executions.inspect(execution.execution_id, principal=principal)
                        if view.status.value == 'WAITING_DEFERRED':
                            break
                    except AIError as error:
                        if error.code is not ErrorCode.EXECUTION_NOT_READY:
                            raise
                    await asyncio.sleep(.01)
            approvals = await runtime.approvals.list(execution.execution_id, principal=principal)
            assert len(approvals) == len(calls) == len(builds) == 1
            execution_id = execution.execution_id
        finally:
            await engine.close()
    if change_source:
        path.write_text('{}', encoding='utf-8')
    async with Runtime.open('deferred-file', features=features(), storage=storage) as runtime:
        task = Task.from_agent('reader', runtime.agents.get('reader'), build_input=build_input)
        engine = await runtime.tasks.bind(task)
        try:
            await engine.recover_pending(principal=principal)
            await runtime.approvals.decide(execution_id, ApprovalDecisionRequest(
                principal, approvals[0].approval_id, 'approve-once', ApprovalDecision.APPROVE))
            execution = await runtime.executions.get(execution_id, principal=principal)
            result = (await execution.wait(timeout_seconds=15)).result
            assert result.status.value == 'SUCCEEDED'
            assert len(calls) == 2 and len(builds) == 1
        finally:
            await engine.close()
    async with Runtime.open('deferred-file', features=features(), storage=storage) as runtime:
        retained = await runtime.executions.result(execution_id, principal=principal)
        assert retained.status == result.status and retained.output == result.output
        history = await runtime.executions.history(execution_id, principal=principal, include_content=True)
        assert history.next_cursor is None
        assert any(item.item_kind == 'tool_result' and item.tool_call_id == 'source-read'
                   and 'fixed synthetic source' in str(item.content) for item in history.items)
        assert len(calls) == 2 and len(builds) == 1
