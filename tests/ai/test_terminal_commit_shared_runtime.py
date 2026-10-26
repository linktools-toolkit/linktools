#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Cross-owner observation and cancellation of coordinated pending turns."""
from ._runtime_test_helpers import model_features
import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import ExecutionCapability
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionCancelRequestCommit, ExecutionRecord
from linktools.ai.capability import CheckpointCapability
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.execution import CancelExecutionRequest
from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.core import CommitPolicy
from linktools.ai.storage._database import SqlStorageContext

from ._runtime_test_helpers import _Models, _UsageFunctionModel


def _group(terminal: bool = True):
    group = RuntimeFeatures('shared-pending')
    group.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal() if terminal else CommitPolicy.sync()), scope='runtime')
    group.agent('default', model='default', allow_tools=())
    if not terminal:
        # Checkpoint guarantees force source admission even with shared cache.
        group.capability(CheckpointCapability(), scope='runtime')
    return group


@pytest.mark.asyncio
async def test_shared_pending_duplicate_observe_wait_and_one_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = InMemoryCacheBackend(max_bytes=2**20)
    entered, finish = asyncio.Event(), asyncio.Event()
    calls = []
    async def model(messages, info):
        calls.append(messages)
        entered.set()
        await finish.wait()
        return ModelResponse(parts=[TextPart('answer')])
    models = _Models(_UsageFunctionModel(model))
    database = tmp_path / 'state.db'
    async with Persistence.open('shared-pending', storage=PersistenceStorage.sqlite(database, cache=cache, coordination=cache)) as persistence_48, Runtime.attach(persistence=persistence_48, features=model_features(models, _group())) as owner, Persistence.open('shared-pending', storage=PersistenceStorage.sqlite(database, cache=cache, coordination=cache)) as persistence_48, Runtime.attach(persistence=persistence_48, features=model_features(models, _group())) as reader:
        session = await owner.agents.get().create_session('session')
        original = SqlStorageContext.run_mutation
        commits = []
        async def count(context, callback, **kwargs):
            result = await original(context, callback, **kwargs)
            commits.append(kwargs.get('domain'))
            return result
        monkeypatch.setattr(SqlStorageContext, 'run_mutation', count)
        execution = await session.start('question', idempotency_key='request')
        await asyncio.wait_for(entered.wait(), 5)
        remote = reader.agents.get().session('session')
        try:
            same = await remote.start('question', idempotency_key='request')
            assert same.execution_id == execution.execution_id
            assert len(calls) == 1 and commits == []
            view = await reader.executions.inspect(execution.execution_id, principal=reader.default_principal)
            assert view.status is ExecutionStatus.STARTED
            with pytest.raises(AIError) as not_ready:
                await reader.executions.result(execution.execution_id, principal=reader.default_principal)
            assert not_ready.value.code is ErrorCode.EXECUTION_NOT_READY
            with pytest.raises(AIError) as conflict:
                await remote.start('different', idempotency_key='request')
            assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
            with pytest.raises(AIError) as busy:
                await remote.start('next', idempotency_key='next')
            assert busy.value.code is ErrorCode.SESSION_BUSY
        finally:
            finish.set()
        assert (await same.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        assert len(commits) == 1
        again = await remote.start('question', idempotency_key='request')
        assert again.execution_id == execution.execution_id and len(calls) == 1
        assert (await again.wait()).result.output == {'text': 'answer'}


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent_owner_checkpoint", (False, True))
async def test_remote_cancel_signals_owner_then_confirms_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, concurrent_owner_checkpoint: bool,
) -> None:
    from linktools.ai.runtime._local import LocalExecutionBackend

    owner_ready, release_owner = asyncio.Event(), asyncio.Event()
    cancel = LocalExecutionBackend.cancel

    async def held_owner_cancel(self, execution, *, force=False):
        if execution.execution_id in self._tasks:
            owner_ready.set()
            await release_owner.wait()
        return await cancel(self, execution, force=force)

    monkeypatch.setattr(LocalExecutionBackend, 'cancel', held_owner_cancel)
    if concurrent_owner_checkpoint:
        reader_checkpoint_ready = asyncio.Event()
        commit_cancel_checkpoint = LocalExecutionBackend.commit_cancel_checkpoint

        async def ordered_cancel_checkpoint(
            self: LocalExecutionBackend, commit: ExecutionCancelRequestCommit,
            *, expected_status: ExecutionStatus,
        ) -> ExecutionRecord:
            if commit.execution_id in self._tasks:
                await asyncio.wait_for(reader_checkpoint_ready.wait(), 5)
            else:
                assert expected_status is ExecutionStatus.STARTED
                reader_checkpoint_ready.set()
                await asyncio.wait_for(owner_ready.wait(), 5)
            return await commit_cancel_checkpoint(self, commit, expected_status=expected_status)

        monkeypatch.setattr(LocalExecutionBackend, 'commit_cancel_checkpoint', ordered_cancel_checkpoint)
    cache = InMemoryCacheBackend(max_bytes=2**20)
    entered, stopped = asyncio.Event(), asyncio.Event()
    async def model(messages, info):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
    models = _Models(_UsageFunctionModel(model))
    database = tmp_path / 'state.db'
    reader_storage = PersistenceStorage.sqlite(database, cache=cache, coordination=cache)
    async with Persistence.open('shared-cancel', storage=PersistenceStorage.sqlite(database, cache=cache, coordination=cache)) as persistence_130, Runtime.attach(persistence=persistence_130, features=model_features(models, _group())) as owner, Persistence.open('shared-cancel', storage=reader_storage) as persistence_130, Runtime.attach(persistence=persistence_130, features=model_features(models, _group())) as reader:
        session = await owner.agents.get().create_session('session')
        execution = await session.start('question', idempotency_key='request')
        await asyncio.wait_for(entered.wait(), 5)
        request = CancelExecutionRequest(reader.default_principal, 'remote-cancel')
        try:
            result = await asyncio.wait_for(reader.executions.cancel(execution.execution_id, request), 8)
            await asyncio.wait_for(owner_ready.wait(), 5)
            assert not stopped.is_set()
            assert result.cancelled is False
            pending = await reader._state.execution.executions.get(execution.execution_id, tenant_id=reader.tenant_id)
            assert pending.status is ExecutionStatus.CANCELLING
        finally:
            release_owner.set()
        await asyncio.wait_for(stopped.wait(), 5)
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.CANCELLED
        result = await reader.executions.cancel(execution.execution_id, request)
        assert result.cancelled is True
        record = await reader._state.execution.executions.get(execution.execution_id, tenant_id=reader.tenant_id)
        assert record.status is ExecutionStatus.CANCELLED


@pytest.mark.asyncio
async def test_shared_pending_early_preview_and_canonical_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_ai.models.function import FunctionModel
    from linktools.ai.storage import InMemoryStreamBackend
    class SharedCache(InMemoryCacheBackend):
        def __init__(self):
            super().__init__(max_bytes=2**20)
            streams = InMemoryStreamBackend(max_bytes=2**20)
            self.open_stream = streams.open_stream
            self.read_stream = streams.read_stream
    cache = SharedCache()
    entered, finish = asyncio.Event(), asyncio.Event()
    async def model(messages, info):
        entered.set()
        yield 'hello '
        await finish.wait()
        yield 'world'
    models = _Models(FunctionModel(stream_function=model))
    database = tmp_path / 'state.db'
    async with Persistence.open('shared-stream', storage=PersistenceStorage.sqlite(database, cache=cache, coordination=cache)) as persistence_171, Runtime.attach(persistence=persistence_171, features=model_features(models, _group())) as owner, Persistence.open('shared-stream', storage=PersistenceStorage.sqlite(database, cache=cache, coordination=cache)) as persistence_171, Runtime.attach(persistence=persistence_171, features=model_features(models, _group())) as reader:
        execution = await owner.agents.get().start('question')
        remote = await reader.executions.get(execution.execution_id)
        observed = remote.watch(include_content=True)
        async def first_text():
            async for item in observed:
                if item.event.event_type == 'ASSISTANT_TEXT_DELTA':
                    return item
            raise AssertionError('missing preview')
        try:
            first = await asyncio.wait_for(first_text(), 5)
            assert first.event.payload['text'] == 'hello '
            assert not finish.is_set()
        finally:
            finish.set()
        remaining = [item async for item in observed]
        assert any(item.event.event_type == 'EXECUTION_SUCCEEDED' for item in remaining)
        assert (await remote.wait(timeout_seconds=5)).result.output == {'text': 'hello world'}
        assert (await execution.wait()).result.output == {'text': 'hello world'}


@pytest.mark.asyncio
async def test_lost_pending_lease_never_executes_effect(tmp_path: Path) -> None:
    from pydantic_ai.messages import ToolCallPart
    from linktools.ai.capability import AgentContext
    class LostLeaseCache(InMemoryCacheBackend):
        lost = False
        async def renew_turn(self, lease, *, lease_seconds):
            return None if self.lost else await super().renew_turn(lease, lease_seconds=lease_seconds)
    cache = LostLeaseCache(max_bytes=2**20)
    entered, finish = asyncio.Event(), asyncio.Event()
    effects = []
    async def effect(context: AgentContext[None]) -> str:
        effects.append(context.execution_id)
        return 'effect'
    async def model(messages, info):
        entered.set()
        await finish.wait()
        return ModelResponse(parts=[ToolCallPart('effect', {}, tool_call_id='call')])
    group = RuntimeFeatures('lost-pending')
    group.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal()), scope="runtime")
    group.tool(effect, effect_policy='replay_safe')
    group.agent('default', model='default', allow_tools=('effect',))
    storage = PersistenceStorage.sqlite(tmp_path / 'state.db', cache=cache, coordination=cache)
    async with Persistence.open("lost-pending", storage=storage) as persistence:
        opened = Runtime.attach(persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), group))
        runtime = await opened.__aenter__()
        try:
            execution = await runtime.agents.get().start('question')
            await asyncio.wait_for(entered.wait(), 5)
            cache.lost = True
            finish.set()
            with pytest.raises(AIError) as failed:
                await execution.wait(timeout_seconds=5)
            assert failed.value.code is ErrorCode.STORAGE_CONFLICT
            assert effects == []
            assert await runtime._state.execution.executions.state_store.read(lambda _: runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)) is None
        finally:
            with pytest.raises(AIError) as close_error:
                await opened.__aexit__(None, None, None)
            assert close_error.value.code is ErrorCode.STORAGE_CONFLICT


@pytest.mark.asyncio
async def test_release_failure_does_not_undo_confirmed_success(tmp_path: Path) -> None:
    class ReleaseFailure(InMemoryCacheBackend):
        async def release_turn(self, lease):
            raise OSError('release unavailable')
    cache = ReleaseFailure(max_bytes=2**20)
    async def model(messages, info):
        return ModelResponse(parts=[TextPart('answer')])
    async with Persistence.open('release-failure', storage=PersistenceStorage.sqlite(tmp_path / 'state.db', cache=cache, coordination=cache)) as persistence_241, Runtime.attach(persistence=persistence_241, features=model_features(_Models(_UsageFunctionModel(model)), _group())) as runtime:
        execution = await runtime.agents.get().start('question')
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        assert (await runtime.executions.result(execution.execution_id, principal=runtime.default_principal)).output == {'text': 'answer'}


@pytest.mark.asyncio
@pytest.mark.parametrize('remote', (False, True))
async def test_live_dependency_hold_promotes_before_acquisition(tmp_path: Path, remote: bool) -> None:
    cache = InMemoryCacheBackend(max_bytes=2**20)
    entered, finish = asyncio.Event(), asyncio.Event()
    async def model(messages, info):
        entered.set()
        await finish.wait()
        return ModelResponse(parts=[TextPart('answer')])
    models = _Models(_UsageFunctionModel(model))
    database = tmp_path / 'state.db'
    reader_storage = PersistenceStorage.sqlite(database, cache=cache, coordination=cache)
    async with Persistence.open('shared-hold', storage=PersistenceStorage.sqlite(database, cache=cache, coordination=cache)) as persistence_259, Runtime.attach(persistence=persistence_259, features=model_features(models, _group())) as owner, Persistence.open('shared-hold', storage=reader_storage) as persistence_259, Runtime.attach(persistence=persistence_259, features=model_features(models, _group())) as reader:
        execution = await owner.agents.get().start('question')
        await asyncio.wait_for(entered.wait(), 5)
        service = reader.executions._service if remote else owner.executions._service
        try:
            acquired = await asyncio.wait_for(service.acquire_dependency_hold(execution.execution_id, tenant_id=owner.tenant_id, hold_id='hold'), 8)
            assert acquired
            stored = await reader._state.execution.executions.state_store.read(lambda _: reader._state.execution.executions.get(execution.execution_id, tenant_id=owner.tenant_id))
            assert stored is not None and stored.status is ExecutionStatus.STARTED
            assert stored.retention_closed is False
        finally:
            finish.set()
        assert (await execution.wait()).result.status is ExecutionStatus.SUCCEEDED
        await service.release_dependency_hold(execution.execution_id, tenant_id=owner.tenant_id, hold_id='hold')


@pytest.mark.asyncio
async def test_remote_public_capture_promotes_before_source_input_read(tmp_path: Path) -> None:
    from linktools.ai.inputs import CaptureInputRequest
    from linktools.ai.core import Principal
    cache = InMemoryCacheBackend(max_bytes=2**20)
    entered, finish = asyncio.Event(), asyncio.Event()
    async def model(messages, info):
        entered.set()
        await finish.wait()
        return ModelResponse(parts=[TextPart('answer')])
    models = _Models(_UsageFunctionModel(model))
    database = tmp_path / 'state.db'
    reader_storage = PersistenceStorage.sqlite(database, cache=cache, coordination=cache)
    async with Persistence.open('shared-capture', storage=PersistenceStorage.sqlite(database, cache=cache, coordination=cache)) as persistence_288, Runtime.attach(persistence=persistence_288, features=model_features(models, _group())) as owner, Persistence.open('shared-capture', storage=reader_storage) as persistence_288, Runtime.attach(persistence=persistence_288, features=model_features(models, _group())) as reader:
        principal = Principal('user', owner.tenant_id)
        execution = await owner.agents.get().start('question', principal=principal)
        await asyncio.wait_for(entered.wait(), 5)
        try:
            captured = await asyncio.wait_for(reader.executions.capture_input(execution.execution_id, CaptureInputRequest(principal, 'capture')), 8)
            assert captured.source_execution_id == execution.execution_id
            record = await reader._state.execution.executions.state_store.read(lambda _: reader._state.execution.executions.get(execution.execution_id, tenant_id=reader.tenant_id))
            assert record is not None and record.input_context is not None
        finally:
            finish.set()
        assert (await execution.wait()).result.status is ExecutionStatus.SUCCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize('first_terminal', (False, True))
async def test_mixed_writers_share_live_session_admission(tmp_path: Path, first_terminal: bool) -> None:
    cache = InMemoryCacheBackend(max_bytes=2**20)
    entered, finish = asyncio.Event(), asyncio.Event()
    calls = []
    async def model(messages, info):
        calls.append(messages)
        entered.set()
        await finish.wait()
        return ModelResponse(parts=[TextPart('answer')])
    models = _Models(_UsageFunctionModel(model))
    database = tmp_path / 'state.db'
    async with Persistence.open('mixed-admission', storage=PersistenceStorage.sqlite(database, cache=cache, coordination=cache)) as persistence_315, Runtime.attach(persistence=persistence_315, features=model_features(models, _group(first_terminal))) as first, Persistence.open('mixed-admission', storage=PersistenceStorage.sqlite(database, cache=cache, coordination=cache)) as persistence_315, Runtime.attach(persistence=persistence_315, features=model_features(models, _group(not first_terminal))) as second:
        session = await first.agents.get().create_session('session')
        execution = await session.start('question', idempotency_key='request')
        await asyncio.wait_for(entered.wait(), 5)
        remote = second.agents.get().session('session')
        try:
            duplicate = await remote.start('question', idempotency_key='request')
            assert duplicate.execution_id == execution.execution_id
            with pytest.raises(AIError) as busy:
                await remote.start('another', idempotency_key='other')
            assert busy.value.code is ErrorCode.SESSION_BUSY
            assert len(calls) == 1
        finally:
            finish.set()
        assert (await duplicate.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_lost_promotion_hint_does_not_claim_dependency(tmp_path: Path) -> None:
    class DropPromotion(InMemoryCacheBackend):
        async def put(self, key, value):
            if value != b'promote':
                await super().put(key, value)
    cache = DropPromotion(max_bytes=2**20)
    entered, finish = asyncio.Event(), asyncio.Event()
    async def model(messages, info):
        entered.set()
        await finish.wait()
        return ModelResponse(parts=[TextPart('answer')])
    models = _Models(_UsageFunctionModel(model))
    database = tmp_path / 'state.db'
    reader_storage = PersistenceStorage.sqlite(database, cache=cache, coordination=cache)
    async with Persistence.open('lost-hint', storage=PersistenceStorage.sqlite(database, cache=cache, coordination=cache)) as persistence_349, Runtime.attach(persistence=persistence_349, features=model_features(models, _group())) as owner, Persistence.open('lost-hint', storage=reader_storage) as persistence_349, Runtime.attach(persistence=persistence_349, features=model_features(models, _group())) as reader:
        execution = await owner.agents.get().start('question')
        await asyncio.wait_for(entered.wait(), 5)
        try:
            with pytest.raises(AIError) as unavailable:
                await reader.executions._service.acquire_dependency_hold(execution.execution_id, tenant_id=owner.tenant_id, hold_id='hold')
            assert unavailable.value.code is ErrorCode.EXECUTION_NOT_READY
            assert await reader._state.execution.executions.state_store.read(lambda _: reader._state.execution.executions.get(execution.execution_id, tenant_id=owner.tenant_id)) is None
        finally:
            finish.set()
        assert (await execution.wait()).result.status is ExecutionStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_remote_promotion_wait_has_a_bounded_confirmation_window(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    from linktools.ai.runtime import _local
    source_reads = []
    class Coordinator:
        async def request_promotion(self, execution_id, *, tenant_id):
            return 'owner'
        async def remote_promotion_pending(self, execution_id, owner_id, *, tenant_id):
            raise AssertionError('deadline must end the request before polling forever')
    backend = object.__new__(_local.LocalExecutionBackend)
    backend._tenant_id = 'tenant'
    backend._pending_turns = SimpleNamespace(coordination=Coordinator())
    async def read(execution_id, *, tenant_id):
        source_reads.append(execution_id)
        return None
    backend.load_execution = read
    clock = iter((0, 31_000_000_000))
    monkeypatch.setattr(_local, 'monotonic_ns', lambda: next(clock))
    with pytest.raises(AIError) as unavailable:
        await backend._promote_remote_pending('execution')
    assert unavailable.value.code is ErrorCode.EXECUTION_NOT_READY
    assert source_reads == ['execution', 'execution']


@pytest.mark.asyncio
async def test_source_replay_release_failure_keeps_confirmed_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import uuid
    from dataclasses import replace
    from linktools.ai.runtime._local import LocalExecutionBackend
    class ReleaseFailure(InMemoryCacheBackend):
        fail_release = False
        async def release_turn(self, lease):
            if self.fail_release:
                raise OSError('release unavailable')
            return await super().release_turn(lease)
    cache = ReleaseFailure(max_bytes=2**20)
    captured = []
    original = LocalExecutionBackend.start_pending
    async def capture(backend, request, reservation, session, coordination):
        captured.append((backend, request, reservation))
        return await original(backend, request, reservation, session, coordination)
    monkeypatch.setattr(LocalExecutionBackend, 'start_pending', capture)
    calls = []
    async def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart('answer')])
    async with Persistence.open('replay-release', storage=PersistenceStorage.sqlite(tmp_path / 'state.db', cache=cache, coordination=cache)) as persistence_409, Runtime.attach(persistence=persistence_409, features=model_features(_Models(_UsageFunctionModel(model)), _group())) as runtime:
        execution = await runtime.agents.get().start('question', idempotency_key='request')
        assert (await execution.wait()).result.status is ExecutionStatus.SUCCEEDED
        backend, request, reservation = captured[0]
        identity = reservation.idempotency
        candidate = 'candidate-' + uuid.uuid4().hex
        proposal = replace(reservation, execution=replace(reservation.execution, execution_id=candidate, root_execution_id=candidate), idempotency=replace(identity, resource_id=candidate))
        coordination = await backend._pending_turns.coordination.claim(
            session_key=f'request:{identity.scope}:{identity.idempotency_key_digest}',
            request_key=f'{identity.scope}:{identity.idempotency_key_digest}',
            request_digest=identity.request_digest, execution_id=candidate,
            owner_id=uuid.uuid4().hex, lease_seconds=30,
        )
        assert coordination is not None and coordination.lease is not None
        cache.fail_release = True
        assert await original(backend, request, proposal, None, coordination) == execution.execution_id
        assert len(calls) == 1
