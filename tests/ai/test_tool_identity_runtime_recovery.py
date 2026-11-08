#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool identity and explicit Trace visibility survive real Runtime boundaries."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from linktools.ai.capability import AgentContext, ExecutionCapability, TraceCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus, Principal
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import InMemoryCacheBackend

from ._runtime_test_helpers import _Models, _UsageFunctionModel


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("batch", "terminal"))
async def test_nested_tool_identity_matches_durable_ancestry_and_independent_source(tmp_path: Path, mode: str) -> None:
    identities = []
    observations = []
    principal = Principal("original-actor", "tenant", "service")
    path = tmp_path / "state.sqlite"
    buffer = InMemoryCacheBackend(max_bytes=4 * 1024 * 1024)
    commit = (CommitPolicy.batch(buffer=buffer, batch_size=128, flush_interval=60)
              if mode == "batch" else CommitPolicy.terminal(buffer=buffer))

    async def inspect_call(ctx: AgentContext[None]) -> str:
        identity = ctx.tool_call
        assert identity is not None and ctx.trace is not None
        assert identity.principal == principal
        assert identity.agent_id == "child"
        assert identity.tool_call_id == "leaf-call"
        assert identity.model_request_seq == 1
        async with Persistence.open("nested-identity", storage=PersistenceStorage.sqlite(path), tenant_id="tenant") as reader:
            archive = reader.run_store.read_store(PersistenceDomain.TRACE)
            assert not await archive.list_events(agent_run_id=identity.agent_run_id)
            await ctx.trace.flush()
            events = await archive.list_events(agent_run_id=identity.agent_run_id)
            assert any(event.tool_call_id == identity.tool_call_id and event.metadata.get("linktools.ai.model_request_seq") == str(identity.model_request_seq) for event in events)
            record = await reader.execution.executions.get(identity.execution_id, tenant_id="tenant")
            assert record is not None
            assert (identity.root_execution_id, identity.parent_execution_id, identity.session_id) == (
                record.root_execution_id, record.parent_execution_id, record.session_id,
            )
        identities.append(identity)
        return "inspected"

    def delegate(target):
        async def respond(messages, info):
            observations.append((target, [(p.tool_call_id, p.content) for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]))
            if any(isinstance(part, ToolReturnPart) and part.tool_call_id == f"delegate-{target}" for message in messages for part in message.parts):
                return ModelResponse([TextPart("delegated")])
            return ModelResponse([ToolCallPart("delegate_task", {"subagent_id": target, "task": "inspect"}, f"delegate-{target}")])
        return respond

    async def leaf(messages, info):
        observations.append(("leaf-model", [(p.tool_call_id, p.content) for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]))
        if any(isinstance(part, ToolReturnPart) and part.tool_call_id == "leaf-call" for message in messages for part in message.parts):
            return ModelResponse([TextPart("complete")])
        return ModelResponse([ToolCallPart("inspect_call", {}, "leaf-call")])

    features = RuntimeFeatures("nested-identity")
    features.capability(TraceCapability(commit=commit), scope="runtime")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.model("root", _Models(_UsageFunctionModel(delegate("child"))))
    features.model("leaf-model", _Models(_UsageFunctionModel(leaf)))
    features.tool(inspect_call, effect_policy="none")
    features.agent("default", model="root", allow_tools=(), allow_subagents=("child",))
    features.agent("child", model="leaf-model", allow_tools=("inspect_call",), allow_subagents=())
    async with Persistence.open("nested-identity", storage=PersistenceStorage.sqlite(path), tenant_id="tenant") as state, Runtime.attach(persistence=state, features=features, metrics=None) as runtime:
        execution = await runtime.agents.get().start("delegate inspection", principal=principal)
        result = (await execution.wait(timeout_seconds=15)).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert len(identities) == 1, observations
        identity = identities[0]
        child, = await runtime.executions.list_children(execution.execution_id, principal=principal)
        assert child.status is ExecutionStatus.SUCCEEDED
        assert identity.execution_id == child.execution_id
        assert identity.root_execution_id == execution.execution_id
        assert identity.parent_execution_id == execution.execution_id
        assert identity.execution_id != execution.execution_id


_CRASH_SCRIPT = r'''
import asyncio, json, os, sys
from dataclasses import asdict
from pathlib import Path
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from linktools.ai.capability import AgentContext, ExecutionCapability, TraceCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus, Principal
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import InMemoryCacheBackend
from tests.ai._runtime_test_helpers import _Models, _UsageFunctionModel, model_features, native_checkpoint
root, stage, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
principal = Principal('original-actor', 'tenant', 'service')
async def inspect_call(ctx: AgentContext[None]) -> str:
    identity = ctx.tool_call
    assert identity is not None and ctx.trace is not None
    assert identity.principal == principal
    assert identity.tool_call_id == ('before-crash' if stage == 'create' else 'after-restart')
    assert identity.model_request_seq is not None
    async with Persistence.open('crash-identity', storage=PersistenceStorage.sqlite(root/'state.sqlite'), tenant_id='tenant') as reader:
        archive = reader.run_store.read_store(PersistenceDomain.TRACE)
        before = await archive.list_events(agent_run_id=identity.agent_run_id)
        assert not any(event.tool_call_id == identity.tool_call_id for event in before)
        await ctx.trace.flush()
        events = await archive.list_events(agent_run_id=identity.agent_run_id)
        assert any(event.tool_call_id == identity.tool_call_id and event.metadata.get('linktools.ai.model_request_seq') == str(identity.model_request_seq) for event in events)
        execution = await reader.execution.executions.get(identity.execution_id, tenant_id='tenant')
        assert execution is not None
        assert identity.root_execution_id == execution.root_execution_id
        assert identity.parent_execution_id == execution.parent_execution_id
        assert identity.session_id == execution.session_id == 'session'
    with (root/'identities').open('a') as stream:
        stream.write(json.dumps(asdict(identity))+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    return identity.tool_call_id
async def model(messages, info):
    returned = [p.tool_call_id for m in messages for p in m.parts if isinstance(p,ToolReturnPart)]
    if not returned:
        assert stage == 'create', 'recovery must not repeat the first model request'
        return ModelResponse([ToolCallPart('inspect_call', {}, 'before-crash')])
    if stage == 'create':
        os._exit(73)
    if 'after-restart' not in returned:
        return ModelResponse([ToolCallPart('inspect_call', {}, 'after-restart')])
    return ModelResponse([TextPart('complete')])
async def main():
    features=RuntimeFeatures('crash-identity')
    buffer=InMemoryCacheBackend(max_bytes=4*1024*1024)
    commit=(CommitPolicy.batch(buffer=buffer,batch_size=128,flush_interval=60) if mode=='batch' else CommitPolicy.terminal(buffer=buffer))
    features.capability(TraceCapability(commit=commit),scope='runtime')
    features.capability(ExecutionCapability(retain_history=True),scope='runtime')
    features.capability(native_checkpoint(),scope='runtime')
    features.tool(inspect_call,effect_policy='none')
    features.agent('default',allow_tools=('inspect_call',))
    async with Persistence.open('crash-identity',storage=PersistenceStorage.sqlite(root/'state.sqlite'),tenant_id='tenant') as state, Runtime.attach(persistence=state,features=model_features(_Models(_UsageFunctionModel(model)),features),metrics=None,auto_recover=False) as runtime:
        if stage=='create':
            session=await runtime.agents.get().create_session('session',principal=principal)
            execution=await session.start('inspect',idempotency_key='start')
            (root/'execution').write_text(execution.execution_id)
            await execution.wait(timeout_seconds=15)
            raise AssertionError('crash not reached')
        execution_id=(root/'execution').read_text()
        original=json.loads((root/'identities').read_text().splitlines()[0])
        events=await state.run_store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id=original['agent_run_id'])
        assert any(event.tool_call_id=='before-crash' for event in events)
        original_events = [asdict(event) for event in events]
        await runtime.executions.recover(execution_id,principal=principal,idempotency_key='recover')
        execution=await runtime.executions.get(execution_id,principal=principal)
        result=(await execution.wait(timeout_seconds=15)).result
        assert result.status is ExecutionStatus.SUCCEEDED,result
        values=[json.loads(line) for line in (root/'identities').read_text().splitlines()]
        assert [value['tool_call_id'] for value in values]==['before-crash','after-restart']
        assert all(value['execution_id']==execution_id for value in values)
        assert all(value['principal']==original['principal'] for value in values)
        assert values[1]['agent_run_seq'] > values[0]['agent_run_seq']
        assert values[1]['agent_run_id'] != values[0]['agent_run_id']
        assert all(value['model_request_seq'] == 1 for value in values)
        async with Persistence.open('crash-identity', storage=PersistenceStorage.sqlite(root/'state.sqlite'), tenant_id='tenant') as reader:
            archive = reader.run_store.read_store(PersistenceDomain.TRACE)
            final_original = [asdict(event) for event in await archive.list_events(agent_run_id=original['agent_run_id'])]
            assert all(event in final_original for event in original_events)
            for value in values:
                captured = await archive.list_events(agent_run_id=value['agent_run_id'])
                assert any(event.tool_call_id == value['tool_call_id'] and event.metadata.get('linktools.ai.model_request_seq') == str(value['model_request_seq']) for event in captured)
        print('CRASH_IDENTITY_RESTORED',mode,flush=True)
asyncio.run(main())
'''


@pytest.mark.parametrize("mode", ("batch", "terminal"))
def test_tool_identity_and_trace_source_survive_process_crash(tmp_path: Path, mode: str) -> None:
    for stage, expected in (("create", 73), ("resume", 0)):
        result = subprocess.run([sys.executable, "-c", _CRASH_SCRIPT, str(tmp_path), stage, mode],
                                env=dict(os.environ), capture_output=True, text=True, timeout=40)
        assert result.returncode == expected, result.stdout + result.stderr
