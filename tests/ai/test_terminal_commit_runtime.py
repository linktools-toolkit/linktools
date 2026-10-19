#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pending turns keep one local owner until their canonical terminal commit."""

from ._runtime_test_helpers import workspace_features

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, UserPromptPart

from linktools.ai.capability import ExecutionCapability, AgentContext, RuntimeFeatures
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode, ObservationError
from linktools.ai.capability import ConversationCapability
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.core import CommitPolicy
from linktools.ai.storage._database import SqlStorageContext

from ._runtime_test_helpers import _Models, _UsageFunctionModel


@pytest.mark.asyncio
async def test_pending_session_commits_once_and_retains_canonical_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []
    async def model(messages, info):
        calls.append(messages)
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart('answer')])
    group = RuntimeFeatures('pending')
    group.capability(ExecutionCapability(commit=CommitPolicy.terminal(), coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024)), scope="runtime")
    group.agent('default', model='default', allow_tools=())
    group.capability(ConversationCapability(), scope='runtime')
    database = tmp_path / 'state.db'
    storage = RuntimeStorage.sqlite(database, cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    async with Runtime.open('pending', storage=storage, models=_Models(_UsageFunctionModel(model)), features=group) as runtime:
        session = await runtime.agents.get().create_session('session')
        original = SqlStorageContext.run_mutation
        commits = []
        async def count(context, callback, **kwargs):
            result = await original(context, callback, **kwargs)
            commits.append(kwargs.get('domain'))
            return result
        monkeypatch.setattr(SqlStorageContext, 'run_mutation', count)
        execution = await session.start('first', idempotency_key='first')
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert commits == []
            assert await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id) is None
            assert (await runtime.executions.inspect(execution.execution_id, principal=runtime.default_principal)).status is ExecutionStatus.STARTED
            same = await session.start('first', idempotency_key='first')
            assert same.execution_id == execution.execution_id
            assert len(calls) == 1
            with pytest.raises(AIError) as mismatch:
                await session.start('different', idempotency_key='first')
            assert mismatch.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
            with pytest.raises(AIError) as busy:
                await session.start('second', idempotency_key='second')
            assert busy.value.code is ErrorCode.SESSION_BUSY
            waiter = asyncio.create_task(execution.wait())
            await asyncio.sleep(0)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert len(calls) == 1
        finally:
            release.set()
        result = (await execution.wait(timeout_seconds=5)).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert result.output == {'text': 'answer'}
        assert len(commits) == 1
        assert set(commits[0].removeprefix('runtime.state.group[').removesuffix(']').split(',')) == {'execution', 'conversation', 'recovery'}
        stored = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert stored is not None and stored.retention_closed and stored.conversation
        replay = await session.start('first', idempotency_key='first')
        assert replay.execution_id == execution.execution_id
        assert (await replay.wait()).result.output == result.output
        assert len(calls) == 1
        second = await session.start('second', idempotency_key='second')
        assert (await second.wait()).result.status is ExecutionStatus.SUCCEEDED
        assert len(calls) == 2
        assert [part.content for message in calls[-1] for part in message.parts if isinstance(part, UserPromptPart)] == ['first', 'second']
        assert len(commits) == 2
        assert commits[0] == commits[1]
    async with Runtime.open('pending', storage=RuntimeStorage.sqlite(database, cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024)), models=_Models(_UsageFunctionModel(model)), features=group) as reopened:
        replay = await reopened.agents.get().session('session').start('first', idempotency_key='first')
        assert (await replay.wait()).result.output == {'text': 'answer'}
        assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('effect_policy', ('none', 'replay_safe'))
async def test_actual_effect_promotes_before_invocation_but_read_only_tool_stays_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, effect_policy: str,
) -> None:
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart
    calls = []
    state = RuntimeStorage.sqlite(tmp_path / 'state.db', cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    owner = {}
    async def lookup(context: AgentContext[None]) -> str:
        record = await runtime._state.execution.executions.get(context.execution_id, tenant_id=context.principal.tenant_id)
        calls.append(record)
        return 'value'
    async def model(messages, info):
        if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            assert await runtime._state.execution.executions.get(owner['execution'], tenant_id=owner['tenant']) is None
            return ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id='lookup-call')])
        return ModelResponse(parts=[TextPart('answer')])
    group = RuntimeFeatures('tool-pending')
    group.capability(ExecutionCapability(commit=CommitPolicy.terminal(), coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024)), scope="runtime")
    group.tool(lookup, effect_policy=effect_policy)
    group.agent('default', model='default', allow_tools=('lookup',))
    group.capability(ConversationCapability(), scope='runtime')
    async with Runtime.open('pending-tool', storage=state, models=_Models(_UsageFunctionModel(model)), features=group) as runtime:
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
async def test_pending_cancel_promotes_and_quiesces_before_cancelled(tmp_path: Path) -> None:
    entered = asyncio.Event()
    stopped = asyncio.Event()
    async def model(messages, info):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
    group = RuntimeFeatures('cancel-pending')
    group.capability(ExecutionCapability(commit=CommitPolicy.terminal(), coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024)), scope="runtime")
    group.agent('default', model='default', allow_tools=())
    group.capability(ConversationCapability(), scope='runtime')
    state = RuntimeStorage.sqlite(tmp_path / 'state.db', cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    async with Runtime.open('pending-cancel', storage=state, models=_Models(_UsageFunctionModel(model)), features=group) as runtime:
        session = await runtime.agents.get().create_session('session')
        execution = await session.start('question')
        await asyncio.wait_for(entered.wait(), 5)
        assert await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id) is None
        await execution.cancel()
        assert stopped.is_set()
        result = (await execution.wait(timeout_seconds=5)).result
        assert result.status is ExecutionStatus.CANCELLED
        record = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert record is not None and record.status is ExecutionStatus.CANCELLED


@pytest.mark.asyncio
@pytest.mark.parametrize('model_metadata', (False, True))
async def test_pending_watch_observes_progress_and_confirmed_terminal(tmp_path: Path, model_metadata: bool) -> None:
    from linktools.ai.core import ExecutionEventType
    entered = asyncio.Event()
    release = asyncio.Event()
    async def model(messages, info):
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart('answer')])
    group = RuntimeFeatures('watch-pending')
    group.capability(ExecutionCapability(commit=CommitPolicy.terminal(), coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024)), scope="runtime")
    group.agent('default', model='default', allow_tools=())
    group.capability(ConversationCapability(), scope='runtime')
    state = RuntimeStorage.sqlite(tmp_path / 'state.db', cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    async with Runtime.open('pending-watch', storage=state, models=_Models(_UsageFunctionModel(model)), features=group) as runtime:
        session = await runtime.agents.get().create_session('session')
        execution = await session.start('question')
        await asyncio.wait_for(entered.wait(), 5)
        seen = []
        ready = asyncio.Event()
        async def observe():
            async for item in execution.watch(include_model_interactions=model_metadata):
                tree = item.event if model_metadata else item
                if not hasattr(tree, 'event'):
                    continue
                seen.append(tree.event.event_type)
                if tree.event.event_type == ExecutionEventType.MODEL_REQUEST_STARTED:
                    ready.set()
        observer = asyncio.create_task(observe())
        try:
            await asyncio.wait_for(ready.wait(), 5)
        finally:
            release.set()
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        if model_metadata:
            with pytest.raises(ObservationError) as unavailable:
                await asyncio.wait_for(observer, 5)
            assert unavailable.value.safe_details["reason"] == "model_state_coverage_unavailable"
        else:
            await asyncio.wait_for(observer, 5)
            assert ExecutionEventType.EXECUTION_SUCCEEDED in seen
        assert ExecutionEventType.EXECUTION_STARTED in seen


@pytest.mark.asyncio
async def test_pending_approval_promotes_before_persisting_frontier(tmp_path: Path) -> None:
    from linktools.ai.workspace import Workspace, WorkspacePolicy, ToolPermissionPolicy
    from .test_runtime_approval_composition import _ToolModels
    workspace = Workspace.load(tmp_path, policy=WorkspacePolicy(tool_permissions=ToolPermissionPolicy(default='ask')))
    group = workspace_features('approval-pending', workspace=workspace)
    group.capability(ExecutionCapability(commit=CommitPolicy.terminal(), coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024)), scope="runtime")
    group.agent('default', model='default', allow_tools=('read_file',))
    group.capability(ConversationCapability(), scope='runtime')
    state = RuntimeStorage.sqlite(tmp_path / 'state.db', cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    async with Runtime.open('pending-approval', storage=state, models=_ToolModels(), features=group) as runtime:
        session = await runtime.agents.get().create_session('session')
        execution = await session.start('read a file')
        record = None
        for _ in range(200):
            record = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
            if record is not None and record.status is ExecutionStatus.WAITING_DEFERRED:
                break
            await asyncio.sleep(.01)
        assert record is not None and record.status is ExecutionStatus.WAITING_DEFERRED
        approvals = await runtime._state.recovery.approvals.list_pending(execution.execution_id, tenant_id=runtime.tenant_id)
        assert len(approvals) == 1


@pytest.mark.asyncio
async def test_pending_sessionless_agent_commits_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    group = RuntimeFeatures('standalone-pending')
    group.capability(ExecutionCapability(commit=CommitPolicy.terminal(), coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024)), scope="runtime")
    group.agent('default', model='default', allow_tools=())
    group.capability(ConversationCapability(), scope='runtime')
    commits = []
    async def model(messages, info):
        assert commits == []
        return ModelResponse(parts=[TextPart('answer')])
    state = RuntimeStorage.sqlite(tmp_path / 'state.db', cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    async with Runtime.open('standalone-pending', storage=state, models=_Models(_UsageFunctionModel(model)), features=group) as runtime:
        original = SqlStorageContext.run_mutation
        async def count(context, callback, **kwargs):
            result = await original(context, callback, **kwargs)
            commits.append(kwargs.get('domain'))
            return result
        monkeypatch.setattr(SqlStorageContext, 'run_mutation', count)
        execution = await runtime.agents.get().start('question', idempotency_key='request')
        assert (await execution.wait(timeout_seconds=5)).result.output == {'text': 'answer'}
        assert len(commits) == 1
        replay = await runtime.agents.get().start('question', idempotency_key='request')
        assert replay.execution_id == execution.execution_id
        assert len(commits) == 1


@pytest.mark.asyncio
async def test_pending_session_force_close_promotes_before_closing(tmp_path: Path) -> None:
    from linktools.ai.core import SessionStatus
    from linktools.ai.runtime.service_api import CloseSessionRequest
    entered = asyncio.Event()
    async def model(messages, info):
        entered.set()
        await asyncio.Event().wait()
    group = RuntimeFeatures('close-pending')
    group.capability(ExecutionCapability(commit=CommitPolicy.terminal(), coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024)), scope="runtime")
    group.agent('default', model='default', allow_tools=())
    group.capability(ConversationCapability(), scope='runtime')
    async with Runtime.open('close-pending', storage=RuntimeStorage.sqlite(tmp_path / 'state.db', cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024)), models=_Models(_UsageFunctionModel(model)), features=group) as runtime:
        session = await runtime.agents.get().create_session('session')
        execution = await session.start('question')
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(AIError) as active:
            await runtime.sessions.close('session', CloseSessionRequest(runtime.default_principal, 'normal-close'))
        assert active.value.code is ErrorCode.SESSION_ACTIVE_EXECUTIONS
        closed = await runtime.sessions.close('session', CloseSessionRequest(runtime.default_principal, 'force-close', True))
        assert closed.status is SessionStatus.CLOSED
        assert (await runtime.executions.inspect(execution.execution_id, principal=runtime.default_principal)).status is ExecutionStatus.CANCELLED


@pytest.mark.asyncio
async def test_pending_unknown_commit_reconciles_without_replaying_model_or_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = RuntimeFeatures('unknown-pending')
    group.capability(ExecutionCapability(commit=CommitPolicy.terminal(), coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024)), scope="runtime")
    group.agent('default', model='default', allow_tools=())
    group.capability(ConversationCapability(), scope='runtime')
    calls = []
    async def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart('answer')])
    state = RuntimeStorage.sqlite(tmp_path / 'state.db', cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    async with Runtime.open('unknown-pending', storage=state, models=_Models(_UsageFunctionModel(model)), features=group) as runtime:
        session = await runtime.agents.get().create_session('session')
        storage_group = runtime._state.execution.executions.state_store.storage_group
        original_mutate = type(storage_group).mutate
        original_read = type(storage_group).read_many
        committed = asyncio.Event()
        unavailable = True
        writes = []
        async def lose_ack(owner, stores, callback):
            result = await original_mutate(owner, stores, callback)
            if owner is storage_group:
                writes.append(1)
                if len(writes) == 1:
                    committed.set()
                    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return result
        async def fail_confirmation(owner, stores, callback):
            if owner is storage_group and committed.is_set() and unavailable:
                raise OSError('source confirmation temporarily unavailable')
            return await original_read(owner, stores, callback)
        monkeypatch.setattr(type(storage_group), 'mutate', lose_ack)
        monkeypatch.setattr(type(storage_group), 'read_many', fail_confirmation)
        execution = await session.start('question', idempotency_key='request')
        await asyncio.wait_for(committed.wait(), 5)
        with pytest.raises(AIError) as unknown:
            await execution.wait(timeout_seconds=5)
        assert unknown.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        assert len(writes) == 1 and len(calls) == 1
        unavailable = False
        assert (await execution.wait(timeout_seconds=5)).result.output == {'text': 'answer'}
        assert await runtime._state.retention.release_execution_handoff(execution.execution_id, tenant_id=runtime.tenant_id, terminal_commit=True)
        replay = await session.start('question', idempotency_key='request')
        assert replay.execution_id == execution.execution_id
        assert (await replay.wait()).result.status is ExecutionStatus.SUCCEEDED
        assert len(writes) == 1 and len(calls) == 1
        following = await session.start('next', idempotency_key='next')
        assert (await following.wait()).result.status is ExecutionStatus.SUCCEEDED
        assert len(calls) == 2 and len(writes) == 2


@pytest.mark.asyncio
async def test_pending_confirmed_rollback_releases_session_without_fabricating_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = RuntimeFeatures('rollback-pending')
    group.capability(ExecutionCapability(commit=CommitPolicy.terminal(), coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024)), scope="runtime")
    group.agent('default', model='default', allow_tools=())
    group.capability(ConversationCapability(), scope='runtime')
    async def model(messages, info):
        return ModelResponse(parts=[TextPart('answer')])
    state = RuntimeStorage.sqlite(tmp_path / 'state.db', cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    context = Runtime.open('rollback-pending', storage=state, models=_Models(_UsageFunctionModel(model)), features=group)
    runtime = await context.__aenter__()
    try:
        session = await runtime.agents.get().create_session('session')
        storage_group = runtime._state.execution.executions.state_store.storage_group
        original = type(storage_group).mutate
        failed = False
        async def rollback(owner, stores, callback):
            nonlocal failed
            if owner is storage_group and not failed:
                failed = True
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return await original(owner, stores, callback)
        monkeypatch.setattr(type(storage_group), 'mutate', rollback)
        execution = await session.start('question', idempotency_key='request')
        with pytest.raises(AIError) as conflict:
            await execution.wait(timeout_seconds=5)
        assert conflict.value.code is ErrorCode.STORAGE_CONFLICT
        following = await session.start('next', idempotency_key='next')
        assert (await following.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
    finally:
        with pytest.raises(AIError) as failure:
            await context.__aexit__(None, None, None)
        assert failure.value.code is ErrorCode.STORAGE_CONFLICT
