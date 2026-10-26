#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Independent regression for accepted-context archival at effect promotion."""
from ._runtime_test_helpers import model_features
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import AgentContext, ExecutionCapability
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import CommitPolicy, ExecutionStatus
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.storage._database import SqlStorageContext
from ._runtime_test_helpers import _Models, _UsageFunctionModel


@pytest.mark.asyncio
@pytest.mark.parametrize('effect_policy', ('none', 'replay_safe'))
async def test_promotion_archives_accepted_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, effect_policy: str,
) -> None:
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart
    calls = []
    state = PersistenceStorage.sqlite(tmp_path / 'state.db', cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024), coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    owner = {}
    async def lookup(context: AgentContext[None]) -> str:
        record = await runtime._state.execution.executions.get(context.execution_id, tenant_id=context.principal.tenant_id)
        if record is not None:
            assert record.input_context.payload.decode()['kind'] == 'execution-context'
        calls.append(record)
        return 'value'
    async def model(messages, info):
        if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            assert await runtime._state.execution.executions.get(owner['execution'], tenant_id=owner['tenant']) is None
            return ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id='lookup-call')])
        return ModelResponse(parts=[TextPart('answer')])
    group = RuntimeFeatures('tool-pending')
    group.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal()), scope="runtime")
    group.tool(lookup, effect_policy=effect_policy)
    group.agent('default', model='default', allow_tools=('lookup',))
    async with Persistence.open('pending-tool', storage=state) as persistence_43, Runtime.open(persistence=persistence_43, features=model_features(_Models(_UsageFunctionModel(model)), group)) as runtime:
        session = await runtime.agents.get().create_session('session')
        original = SqlStorageContext.run_mutation
        commits = []
        async def count(context, callback, **kwargs):
            result = await original(context, callback, **kwargs)
            commits.append(kwargs.get('domain'))
            return result
        monkeypatch.setattr(SqlStorageContext, 'run_mutation', count)
        execution = await session.start('question')
        owner.update(execution=execution.execution_id, tenant=runtime.tenant_id)
        assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        assert len(calls) == 1
        if effect_policy == 'none':
            assert calls == [None]
            assert len(commits) == 1
        else:
            assert calls[0] is not None and calls[0].status is ExecutionStatus.STARTED
            assert len(commits) > 1


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", (False, True))
async def test_session_admission_reuses_verified_history_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal: bool) -> None:
    from linktools.ai.inputs import ExecutionContextArchive
    from linktools.ai.persistence._archive import ContentArchive
    prepared = []
    active = []
    body_reads = []
    original = ExecutionContextArchive.prepare
    original_read = ContentArchive._read_content
    async def observe(self, context, **kwargs):
        active.append(True)
        try:
            result = await original(self, context, **kwargs)
        finally:
            active.pop()
        prepared.append((len(context.model_messages()), kwargs.get('history_root'), len(result.records.records)))
        return result
    async def read(self, content):
        if active:
            body_reads.append(content.codec)
        return await original_read(self, content)
    monkeypatch.setattr(ExecutionContextArchive, 'prepare', observe)
    monkeypatch.setattr(ContentArchive, '_read_content', read)
    async def model(messages, info):
        return ModelResponse(parts=[TextPart('answer')])
    features = RuntimeFeatures('review-context')
    features.capability(ExecutionCapability(retain_history=True,
        commit=CommitPolicy.terminal() if terminal else CommitPolicy.sync()))
    features.agent('default', model='default', allow_tools=())
    async with Persistence.open('review-context', storage=PersistenceStorage.sqlite(tmp_path / 'context.db', coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024) if terminal else None)) as persistence_95, Runtime.open(persistence=persistence_95, features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
        session = await runtime.agents.get().create_session('session')
        for value in ('first', 'second', 'third'):
            result = await session.run(value, timeout_seconds=5)
            assert result.result.status is ExecutionStatus.SUCCEEDED
    later = [item for item in prepared if item[0] > 0]
    assert len(later) == 2
    assert all(root is not None and count == 0 for _, root, count in later)
    assert body_reads == []


@pytest.mark.asyncio
@pytest.mark.parametrize('rounds', (2, 4))
async def test_projection_boundary_read_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rounds: int) -> None:
    import json
    from contextvars import ContextVar
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart
    from linktools.ai.capability import TraceCapability, CheckpointCapability
    from linktools.ai.persistence._archive import ContentArchive
    from linktools.ai.persistence._history import TranscriptRepository
    from linktools.ai.persistence._step_archive import StateStepArchive
    phase = ContextVar('review_phase', default='other')
    observed = []
    details = []
    original_read = ContentArchive._read_content
    async def read(self, content):
        observed.append((phase.get(), content.identity, content.raw_size, content.payload.kind))
        return await original_read(self, content)
    def wrap(method, label):
        async def invoke(self, *args, **kwargs):
            token = phase.set(label)
            before = len(observed)
            try:
                return await method(self, *args, **kwargs)
            finally:
                if label == 'projection':
                    baseline = kwargs.get('baseline')
                    root = getattr(baseline.items[0], 'root', None) if baseline is not None and baseline.items else None
                    details.append({'domain':self._runtime_domain.value, 'messages':len(args[0]), 'baseline':None if root is None else root.count, 'body_reads':len(observed)-before})
                phase.reset(token)
        return invoke
    monkeypatch.setattr(ContentArchive, '_read_content', read)
    monkeypatch.setattr(TranscriptRepository, 'prepare_model_projection', wrap(TranscriptRepository.prepare_model_projection, 'projection'))
    monkeypatch.setattr(StateStepArchive, '_relocate_run_checkpoint', wrap(StateStepArchive._relocate_run_checkpoint, 'relocation'))
    async def lookup(ctx: AgentContext[None]) -> str:
        return 'value'
    async def model(messages, info):
        count = sum(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)
        return ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id=f'call-{count}')]) if count < rounds else ModelResponse(parts=[TextPart('done')])
    features = RuntimeFeatures('review-boundaries')
    features.capability(ExecutionCapability(retain_history=True))
    features.capability(TraceCapability(commit=CommitPolicy.sync()))
    features.capability(CheckpointCapability())
    features.tool(lookup, effect_policy='none')
    features.agent('default', model='default', allow_tools=('lookup',))
    async with Persistence.open('review-boundaries', storage=PersistenceStorage.sqlite(tmp_path / 'boundaries.db')) as persistence_151, Runtime.open(persistence=persistence_151, features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
        result = await runtime.agents.get().run('question', timeout_seconds=10)
        assert result.result.status is ExecutionStatus.SUCCEEDED
    summary = {label: {'calls': sum(item[0] == label for item in observed),
                       'object_calls': sum(item[0] == label and item[3] == 'object' for item in observed),
                       'bytes': sum(item[2] for item in observed if item[0] == label),
                       'unique': len({item[1] for item in observed if item[0] == label})}
               for label in ('projection', 'relocation', 'other')}
    Path(f'/tmp/review-boundary-counts-{rounds}.json').write_text(json.dumps({'summary':summary,'projection_calls':details}, sort_keys=True))
    assert summary['relocation']['calls'] == 0
    assert summary['projection']['calls'] <= 6 * rounds + 5
    assert summary['other']['calls'] <= 3 * rounds + 2


@pytest.mark.asyncio
async def test_pending_large_history_is_process_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    from linktools.ai.storage._object_sql import SqlObjectStore
    from linktools.ai.persistence.coordination import PendingTurnCoordination
    from linktools.ai.inputs import ExecutionContextArchive
    from linktools.ai.core import PersistenceDomain
    entered, finish = asyncio.Event(), asyncio.Event()
    large = 'review-private-history-' * 10000
    calls = 0
    blocked = False
    published = []
    original_put = SqlObjectStore.put
    original_publish = PendingTurnCoordination.publish
    async def put(self, *args, **kwargs):
        assert not blocked, 'Pending admission wrote a temporary object'
        return await original_put(self, *args, **kwargs)
    async def publish(self, record, coordination):
        await original_publish(self, record, coordination)
        raw = await self._cache.get(self._record_key(coordination))
        assert raw is not None and b'review-private-history-' not in raw
        observed = await self.read_execution(record.execution_id, tenant_id=self._tenant_id)
        assert observed is not None and observed.input_context is None
        published.append(record.input_context)
    monkeypatch.setattr(SqlObjectStore, 'put', put)
    monkeypatch.setattr(PendingTurnCoordination, 'publish', publish)
    async def model(messages, info):
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            await finish.wait()
        return ModelResponse(parts=[TextPart(large if calls == 1 else 'done')])
    features = RuntimeFeatures('review-pending-large')
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal()))
    features.agent('default', model='default', allow_tools=())
    async with Persistence.open('review-pending-large', storage=PersistenceStorage.sqlite(tmp_path / 'pending.db', coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))) as persistence_203, Runtime.open(persistence=persistence_203, features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
        session = await runtime.agents.get().create_session('session')
        assert (await session.run('first')).result.status is ExecutionStatus.SUCCEEDED
        published.clear()
        blocked = True
        execution = await session.start('second')
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert published and any(item is not None for item in published)
        finally:
            blocked = False
            finish.set()
        assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        record = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert record is not None and record.input_context is not None
        assert record.input_context.payload.decode()['kind'] == 'execution-context'
        archive = ExecutionContextArchive(runtime._state.execution.executions.state_store, runtime._state.archive_object_store(PersistenceDomain.EXECUTION), namespace='review-pending-large', tenant_id=runtime.tenant_id)
        loaded = await archive.load(record.input_context)
        assert large in str(loaded.model_messages())
