#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Cross-owner observation and cancellation of coordinated pending turns."""
from ._runtime_test_helpers import model_features
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import ExecutionCapability, SessionCapability
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionCancelRequestCommit, ExecutionRecord
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.execution import CancelExecutionRequest
from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.core import CommitPolicy

from ._runtime_test_helpers import _Models, _UsageFunctionModel
from .test_runtime_turn_lifecycle import _ObservedCache


class _HostAuthorization(TenantAuthorizationPolicy):
    """Explicit host authorization for canonical SQL sessions."""


class _TurnNoticeChannel:
    """In-process carrier for the same host bridge used across owners."""

    def __init__(self):
        self.subscribers = []

    @asynccontextmanager
    async def subscribe(self, scope, *, on_notice, on_gap):
        subscriber = (scope, on_notice)
        self.subscribers.append(subscriber)
        try:
            yield
        finally:
            self.subscribers.remove(subscriber)

    def offer(self, notice):
        for scope, receive in tuple(self.subscribers):
            if scope == notice.scope:
                asyncio.get_running_loop().call_soon(receive, notice)


def _group(terminal: bool = True, *, auto_recover: bool = False):
    group = RuntimeFeatures('shared-pending', defaults={"execution": {"auto_recover": auto_recover}})
    group.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal() if terminal else CommitPolicy.sync()), scope='runtime')
    group.agent('default', model='default', allow_tools=())
    if not terminal:
        # The synchronous canonical writer also exercises durable native Step.
        group.capability(SessionCapability(checkpoints=True), scope='runtime')
    return group


@pytest.mark.asyncio
async def test_shared_pending_duplicate_wait_publishes_one_turn(tmp_path: Path) -> None:
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
    storage = PersistenceStorage.sqlite(database, cache=cache, coordination=cache)
    channel = _TurnNoticeChannel()
    async with (
        Persistence.open("shared-pending", storage=storage) as owner_store,
        Runtime.open(persistence=owner_store, features=model_features(models, _group()),
                     authorization=_HostAuthorization()) as owner,
        Persistence.open("shared-pending", storage=storage) as reader_store,
        Runtime.open(persistence=reader_store, features=model_features(models, _group()),
                     authorization=_HostAuthorization()) as reader,
        owner.sessions.bridge_turn_notices(channel),
        reader.sessions.bridge_turn_notices(channel),
    ):
        session = await owner.agents.get().create_session("session")
        principal = owner.default_principal

        async def start(runtime, prompt="question", key="request"):
            return await runtime.sessions.start(session.session_id, prompt, agent="default",
                                                principal=principal, idempotency_key=key)

        turn = await start(owner)
        await asyncio.wait_for(entered.wait(), 5)
        try:
            same = await start(reader)
            assert same.key.turn_id == turn.key.turn_id and len(calls) == 1
            view = await reader.history.turn_result(session.session_id, turn.key.turn_id, principal=principal)
            assert view["status"] == ExecutionStatus.PENDING_START.value
            assert view["conversation_committed"] is False
            assert (await reader.history.turn_session(session.session_id, principal=principal))["timeline"]["items"] == []
            with pytest.raises(AIError) as conflict:
                await start(reader, prompt="different")
            assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
            with pytest.raises(AIError) as busy:
                await start(reader, prompt="next", key="next")
            assert busy.value.code is ErrorCode.SESSION_BUSY
        finally:
            finish.set()
        assert (await same.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
        assert (await turn.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
        again = await start(reader)
        assert again.key.turn_id == turn.key.turn_id and len(calls) == 1
        assert (await again.wait_final(timeout_seconds=5)).view["output"] == {"text": "answer"}
        (published,) = (await reader.history.turn_session(session.session_id, principal=principal))["timeline"]["items"]
        assert published["turn_id"] == turn.key.turn_id and published["conversation_committed"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent_owner_checkpoint", (False, True))
async def test_remote_cancel_signals_owner_then_confirms_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, concurrent_owner_checkpoint: bool,
) -> None:
    from linktools.ai.runtime._local import LocalExecutionBackend

    owner_ready, release_owner = asyncio.Event(), asyncio.Event()
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
    cache = _ObservedCache()
    entered, stopped = asyncio.Event(), asyncio.Event()
    async def model(messages, info):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            owner_ready.set()
            while not release_owner.is_set():
                try:
                    await release_owner.wait()
                except asyncio.CancelledError:
                    pass
            stopped.set()
    models = _Models(_UsageFunctionModel(model))
    database = tmp_path / 'state.db'
    reader_storage = PersistenceStorage.sqlite(database, cache=cache, coordination=cache)
    async with Persistence.open('shared-cancel', storage=PersistenceStorage.sqlite(database, cache=cache, coordination=cache)) as persistence_130, Runtime.open(persistence=persistence_130, features=model_features(models, _group())) as owner, Persistence.open('shared-cancel', storage=reader_storage) as persistence_130, Runtime.open(persistence=persistence_130, features=model_features(models, _group())) as reader:
        execution = await owner.agents.get().start('question', idempotency_key='request')
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
    async with Persistence.open('shared-stream', storage=PersistenceStorage.sqlite(database, cache=cache, coordination=cache)) as persistence_171, Runtime.open(persistence=persistence_171, features=model_features(models, _group())) as owner, Persistence.open('shared-stream', storage=PersistenceStorage.sqlite(database, cache=cache, coordination=cache)) as persistence_171, Runtime.open(persistence=persistence_171, features=model_features(models, _group())) as reader:
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
    class LostLeaseCache(_ObservedCache):
        lost = False
        async def renew_turn(self, lease, *, lease_seconds):
            return None if self.lost else await super().renew_turn(lease, lease_seconds=lease_seconds)
    cache = LostLeaseCache()
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
    persistence = Persistence(storage)
    await persistence.initialize(namespace="lost-pending", tenant_id="default")
    source_group = persistence.execution.executions.state_store.storage_group
    try:
        opened = Runtime.open(persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), group))
        runtime = await opened.__aenter__()
        try:
            execution = await runtime.agents.get().start('question')
            await asyncio.wait_for(entered.wait(), 5)
            backend = runtime._execution_service.runtime_backend()
            owner = backend._turn_owners[execution.execution_id]
            cache.lost = True

            async def wait_for_owner_loss():
                while not owner.stopping:
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_for_owner_loss(), 5)
            finish.set()
            with pytest.raises(AIError) as failed:
                await execution.wait(timeout_seconds=5)
            assert failed.value.code is ErrorCode.STORAGE_CONFLICT
            assert effects == []
            assert await runtime._state.execution.executions.state_store.read(lambda _: runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)) is None
        finally:
            finish.set()
            await opened.__aexit__(None, None, None)
        assert effects == []
        assert not persistence._runtime_owners
        assert await persistence.execution.executions.get(
            execution.execution_id, tenant_id=runtime.tenant_id,
        ) is None
        assert await persistence.execution.executions.get(
            execution.execution_id, tenant_id=runtime.tenant_id,
        ) is None
        assert not cache._turns
    finally:
        try:
            await persistence.close()
        finally:
            await source_group.close()
            await source_group.context.close()
            assert source_group.context.closed


@pytest.mark.asyncio
async def test_release_failure_does_not_undo_confirmed_success(tmp_path: Path) -> None:
    class ReleaseFailure(InMemoryCacheBackend):
        async def release_turn(self, lease):
            raise OSError('release unavailable')
    cache = ReleaseFailure(max_bytes=2**20)
    async def model(messages, info):
        return ModelResponse(parts=[TextPart('answer')])
    async with Persistence.open('release-failure', storage=PersistenceStorage.sqlite(tmp_path / 'state.db', cache=cache, coordination=cache)) as persistence_241, Runtime.open(persistence=persistence_241, features=model_features(_Models(_UsageFunctionModel(model)), _group())) as runtime:
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
    async with Persistence.open('shared-hold', storage=PersistenceStorage.sqlite(database, cache=cache, coordination=cache)) as persistence_259, Runtime.open(persistence=persistence_259, features=model_features(models, _group())) as owner, Persistence.open('shared-hold', storage=reader_storage) as persistence_259, Runtime.open(persistence=persistence_259, features=model_features(models, _group())) as reader:
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
    async with Persistence.open('shared-capture', storage=PersistenceStorage.sqlite(database, cache=cache, coordination=cache)) as persistence_288, Runtime.open(persistence=persistence_288, features=model_features(models, _group())) as owner, Persistence.open('shared-capture', storage=reader_storage) as persistence_288, Runtime.open(persistence=persistence_288, features=model_features(models, _group())) as reader:
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
    storage = PersistenceStorage.sqlite(database, cache=cache, coordination=cache)
    channel = _TurnNoticeChannel()
    async with (
        Persistence.open("mixed-admission", storage=storage) as first_store,
        Runtime.open(persistence=first_store, features=model_features(models, _group(first_terminal)),
                     authorization=_HostAuthorization()) as first,
        Persistence.open("mixed-admission", storage=storage) as second_store,
        Runtime.open(persistence=second_store, features=model_features(models, _group(not first_terminal)),
                     authorization=_HostAuthorization()) as second,
        first.sessions.bridge_turn_notices(channel),
        second.sessions.bridge_turn_notices(channel),
    ):
        session = await first.agents.get().create_session('session')
        principal = first.default_principal
        turn = await first.sessions.start(session.session_id, "question", agent="default",
                                          principal=principal, idempotency_key="request")
        await asyncio.wait_for(entered.wait(), 5)
        try:
            duplicate = await second.sessions.start(session.session_id, "question", agent="default",
                                                    principal=principal, idempotency_key="request")
            assert duplicate.key.turn_id == turn.key.turn_id
            with pytest.raises(AIError) as busy:
                await second.sessions.start(session.session_id, "another", agent="default",
                                            principal=principal, idempotency_key="other")
            assert busy.value.code is ErrorCode.SESSION_BUSY
            assert len(calls) == 1
        finally:
            finish.set()
        assert (await duplicate.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
        assert (await turn.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
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
    async with Persistence.open('lost-hint', storage=PersistenceStorage.sqlite(database, cache=cache, coordination=cache)) as persistence_349, Runtime.open(persistence=persistence_349, features=model_features(models, _group())) as owner, Persistence.open('lost-hint', storage=reader_storage) as persistence_349, Runtime.open(persistence=persistence_349, features=model_features(models, _group())) as reader:
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
    failed_releases = []
    class ReleaseFailure(InMemoryCacheBackend):
        fail_release = False
        async def release_turn(self, lease):
            if self.fail_release:
                failed_releases.append(lease)
                raise OSError('release unavailable')
            return await super().release_turn(lease)
    cache = ReleaseFailure(max_bytes=2**20)
    captured = []
    replay_owners = []
    original = LocalExecutionBackend.start_pending
    async def capture(backend, request, reservation, session, coordination):
        captured.append((backend, request, reservation))
        if cache.fail_release:
            assert coordination is not None and coordination.lease is not None
            replay_owners.append(backend._turn_owners[reservation.execution.execution_id])
        return await original(backend, request, reservation, session, coordination)
    monkeypatch.setattr(LocalExecutionBackend, 'start_pending', capture)
    calls = []
    async def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart('answer')])
    async with Persistence.open('replay-release', storage=PersistenceStorage.sqlite(tmp_path / 'state.db', cache=cache, coordination=cache)) as persistence_409, Runtime.open(persistence=persistence_409, features=model_features(_Models(_UsageFunctionModel(model)), _group())) as runtime:
        admissions = []
        original_admit = runtime._execution_service._admit_execution_start

        async def capture_admission(*args, **kwargs):
            admissions.append((args, kwargs))
            return await original_admit(*args, **kwargs)

        monkeypatch.setattr(runtime._execution_service, '_admit_execution_start', capture_admission)
        execution = await runtime.agents.get().start('question', idempotency_key='request')
        assert (await execution.wait()).result.status is ExecutionStatus.SUCCEEDED
        backend, request, reservation = captured[0]
        identity = reservation.idempotency
        candidate = 'candidate-' + uuid.uuid4().hex
        proposal = replace(reservation, execution=replace(reservation.execution, execution_id=candidate, root_execution_id=candidate), idempotency=replace(identity, resource_id=candidate))
        previous_owner = backend._turn_owners.get(execution.execution_id)
        if previous_owner is not None:
            await asyncio.wait_for(previous_owner.closed.wait(), 5)
        args, kwargs = admissions[0]
        cache.fail_release = True
        # Replay through the admission owner so pending fallback validates the receipt.
        replayed = await original_admit(args[0], proposal, *args[2:], **kwargs)
        assert replayed.execution_id == execution.execution_id
        await asyncio.wait_for(replay_owners[0].closed.wait(), 5)
        assert failed_releases
        assert len(calls) == 1
