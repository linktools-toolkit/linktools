#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Isolated failure experiment; not a change to production durability policy."""

import asyncio
import json
import multiprocessing
import os
import sqlite3
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, UserPromptPart

from linktools.ai.capability import AgentContext, ExecutionCapability, TraceCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus
from linktools.ai.errors import ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._runtime_commands import StateCommands
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import InMemoryCacheBackend, InMemoryStreamBackend
from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_execution_terminal_payload_runtime import _PayloadBuffer


def _effect(path: Path, business_key: str) -> None:
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE IF NOT EXISTS effects (business_key TEXT PRIMARY KEY)')
        db.execute('CREATE TABLE IF NOT EXISTS attempts (worker INTEGER)')
        db.execute('INSERT INTO attempts VALUES (?)', (os.getpid(),))
        db.execute('INSERT OR IGNORE INTO effects VALUES (?)', (business_key,))


def _features(effect: Path, calls: list[str], phase: str = '',
              entered: asyncio.Event | None = None, release: asyncio.Event | None = None,
              buffer: _PayloadBuffer | None = None, trace: bool = False) -> RuntimeFeatures:
    async def action(ctx: AgentContext[None], business_key: str) -> str:
        _effect(effect, business_key)
        if phase == 'after_effect':
            os._exit(91)
        return 'applied'

    request_count = 0

    async def model(messages: list[object], info: object) -> ModelResponse:
        nonlocal request_count
        calls.append('model')
        last_user = next(part.content for message in reversed(messages)
                         for part in reversed(message.parts) if isinstance(part, UserPromptPart))
        if last_user == 'baseline':
            return ModelResponse(parts=[TextPart('baseline completed')])
        request_count += 1
        if request_count == 1:
            return ModelResponse(parts=[ToolCallPart(info.function_tools[0].name,
                {'business_key': 'business-request-1'}, tool_call_id=f'call-{os.getpid()}')])
        if entered is not None:
            entered.set()
            await release.wait()
        return ModelResponse(parts=[TextPart('work completed')])

    features = RuntimeFeatures('volatile-probe')
    features.capability(ExecutionCapability(commit=CommitPolicy.terminal(buffer=buffer), retain_history=True))
    if trace:
        features.capability(TraceCapability(commit=CommitPolicy.terminal()))
    features.tool(action, effect_policy='replay_safe')
    features.agent('default', model='default', allow_tools=('action',))
    return model_features(_Models(_UsageFunctionModel(model)), features)


def _storage(path: Path) -> PersistenceStorage:
    cache = InMemoryCacheBackend(max_bytes=16 * 1024 * 1024)
    return PersistenceStorage.sqlite(path, cache=cache, coordination=cache)


def _crash_worker(database: str, effects: str, phase: str) -> None:
    original = StateCommands.commit_pending_turn

    async def commit(self: StateCommands, *args: object, **kwargs: object) -> object:
        if phase == 'before_terminal':
            os._exit(91)
        result = await original(self, *args, **kwargs)
        if phase == 'after_terminal':
            os._exit(91)
        return result

    StateCommands.commit_pending_turn = commit

    async def run() -> None:
        async with Persistence.open('volatile-probe', storage=_storage(Path(database))) as persistence:
            async with Runtime.attach(persistence=persistence,
                                      features=_features(Path(effects), [], phase)) as runtime:
                execution = await runtime.agents.get().session('session').start('work', idempotency_key='work')
                Path(database + '.execution').write_text(execution.execution_id)
                await execution.wait(timeout_seconds=10)
        raise AssertionError('selected crash boundary was not reached')

    asyncio.run(run())


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ('after_effect', 'before_terminal', 'after_terminal'))
async def test_volatile_turn_process_crash_keeps_prior_turn_and_business_idempotency(tmp_path: Path, phase: str) -> None:
    database, effects = tmp_path / 'state.db', tmp_path / 'effects.db'
    async with Persistence.open('volatile-probe', storage=_storage(database)) as persistence:
        async with Runtime.attach(persistence=persistence, features=_features(effects, [])) as runtime:
            session = await runtime.agents.get().create_session('session')
            prior = await session.start('baseline', idempotency_key='baseline')
            assert (await prior.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
            prior_id = prior.execution_id

    process = multiprocessing.get_context('spawn').Process(
        target=_crash_worker, args=(str(database), str(effects), phase))
    process.start()
    try:
        await asyncio.to_thread(process.join, 30)
        assert process.exitcode == 91, process.exitcode
    finally:
        if process.is_alive():
            process.kill()
            await asyncio.to_thread(process.join, 5)
        process.close()

    lost_id = Path(str(database) + '.execution').read_text()
    calls = []
    async with Persistence.open('volatile-probe', storage=_storage(database)) as persistence:
        async with Runtime.attach(persistence=persistence, features=_features(effects, calls)) as runtime:
            before = await persistence.execution.executions.get(lost_id, tenant_id=persistence.tenant_id)
            assert (before is not None) is (phase == 'after_terminal')
            assert await persistence.execution.executions.get(prior_id, tenant_id=persistence.tenant_id) is not None
            previous = await runtime.agents.get().session('session').start('baseline', idempotency_key='baseline')
            assert previous.execution_id == prior_id
            assert (await previous.wait(timeout_seconds=10)).result.output == {'text': 'baseline completed'}
            assert calls == []
            execution = await runtime.agents.get().session('session').start('work', idempotency_key='work')
            assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
            assert (execution.execution_id == lost_id) is (phase == 'after_terminal')
            assert len(calls) == (0 if phase == 'after_terminal' else 2)
    with sqlite3.connect(effects) as db:
        effect_count = db.execute('SELECT COUNT(*) FROM effects').fetchone()[0]
        attempts = db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
    assert effect_count == 1
    assert attempts == (1 if phase == 'after_terminal' else 2)
    print(json.dumps({'phase': phase, 'model_replays': len(calls), 'tool_attempts': attempts,
                      'business_effects': effect_count, 'prior_result_replayed_without_model': True}))


@pytest.mark.asyncio
@pytest.mark.parametrize('interference', ('lost_lease', 'backend_unavailable'))
async def test_volatile_turn_replacement_does_not_publish_the_stale_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interference: str,
) -> None:
    from linktools.ai.errors import AIError, ErrorCode

    database, effects = tmp_path / 'state.db', tmp_path / 'effects.db'
    cache = InMemoryCacheBackend(max_bytes=16 * 1024 * 1024)
    storage = PersistenceStorage.sqlite(database, cache=cache, coordination=cache)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async with Persistence.open('volatile-probe', storage=storage) as first:
        async with Runtime.attach(persistence=first,
                                  features=_features(effects, calls, entered=entered, release=release)) as old:
            session = await old.agents.get().create_session('session')
            lost = await session.start('work', idempotency_key='work')
            try:
                await asyncio.wait_for(entered.wait(), 5)
                backend = old._execution_service.runtime_backend()
                pending = backend._pending_turns.get(lost.execution_id, tenant_id=old.tenant_id)
                coordinator = backend._pending_turns.coordination
                lease = pending.coordination.lease
                if interference == 'backend_unavailable':
                    original = cache.renew_turn

                    async def unavailable(*args: object, **kwargs: object) -> object:
                        raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

                    monkeypatch.setattr(cache, 'renew_turn', unavailable)
                    release.set()
                    with pytest.raises(AIError) as failed:
                        await lost.wait(timeout_seconds=10)
                    assert failed.value.code is ErrorCode.STORAGE_UNAVAILABLE
                    assert await first.execution.executions.get(lost.execution_id, tenant_id=first.tenant_id) is None
                    monkeypatch.setattr(cache, 'renew_turn', original)
                released = await coordinator.release(lease)
                assert released is (interference == 'lost_lease')
                replacement_calls = []
                async with Persistence.open('volatile-probe', storage=storage) as second:
                    async with Runtime.attach(persistence=second,
                                              features=_features(effects, replacement_calls)) as new:
                        replacement = await new.agents.get().session('session').start('work', idempotency_key='work')
                        assert replacement.execution_id != lost.execution_id
                        assert (await replacement.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
                        assert len(replacement_calls) == 2
                        release.set()
                        if interference == 'lost_lease':
                            with pytest.raises(AIError) as failed:
                                await lost.wait(timeout_seconds=10)
                            assert failed.value.code is ErrorCode.STORAGE_CONFLICT
                        assert await second.execution.executions.get(lost.execution_id, tenant_id=second.tenant_id) is None
                        stored = await second.execution.executions.get(replacement.execution_id, tenant_id=second.tenant_id)
                        assert stored.status is ExecutionStatus.SUCCEEDED
                        replay = await new.agents.get().session('session').start('work', idempotency_key='work')
                        assert replay.execution_id == replacement.execution_id
            finally:
                release.set()
    with sqlite3.connect(effects) as db:
        effects_count = db.execute('SELECT COUNT(*) FROM effects').fetchone()[0]
        attempts = db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
    assert effects_count == 1 and attempts == 2
    print(json.dumps({'fault': interference, 'tool_attempts': attempts, 'business_effects': effects_count,
                      'stale_execution_published': False, 'replacement_succeeded': True}))


@pytest.mark.asyncio
@pytest.mark.parametrize('buffered_trace', (False, True))
@pytest.mark.parametrize('receipt', ('direct', 'unknown', 'unknown_close'))
async def test_probe_lease_loss_after_last_check_before_sql_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, receipt: str, buffered_trace: bool,
) -> None:
    from linktools.ai.errors import AIError

    database, effects = tmp_path / 'state.db', tmp_path / 'effects.db'
    cache = InMemoryCacheBackend(max_bytes=16 * 1024 * 1024)
    streams = InMemoryStreamBackend(max_bytes=16 * 1024 * 1024)
    monkeypatch.setattr(cache, 'open_stream', streams.open_stream, raising=False)
    monkeypatch.setattr(cache, 'read_stream', streams.read_stream, raising=False)
    storage = PersistenceStorage.sqlite(database, cache=cache, coordination=cache)
    commit_entered, commit_release = asyncio.Event(), asyncio.Event()
    new_entered, new_release = asyncio.Event(), asyncio.Event()
    original = StateCommands.commit_pending_turn
    gated_execution = None

    async def gate(self: StateCommands, reservation: object, commit: object, **kwargs: object) -> object:
        nonlocal gated_execution
        if gated_execution is None:
            gated_execution = reservation.execution.execution_id
            commit_entered.set()
            await commit_release.wait()
        result = await original(self, reservation, commit, **kwargs)
        if receipt != 'direct' and reservation.execution.execution_id != gated_execution:
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        return result

    monkeypatch.setattr(StateCommands, 'commit_pending_turn', gate)
    outcomes = {}
    old_calls, new_calls = [], []
    old_buffer = _PayloadBuffer() if buffered_trace else None
    new_buffer = _PayloadBuffer() if buffered_trace else None
    async with Persistence.open('volatile-probe', storage=storage) as first:
        async with Runtime.attach(persistence=first, features=_features(effects, old_calls, buffer=old_buffer, trace=buffered_trace)) as old:
            session = await old.agents.get().create_session('session')
            stale = await session.start('work', idempotency_key='work')
            try:
                await asyncio.wait_for(commit_entered.wait(), 5)
                backend = old._execution_service.runtime_backend()
                pending = backend._pending_turns.get(stale.execution_id, tenant_id=old.tenant_id)
                assert await backend._pending_turns.coordination.release(pending.coordination.lease)
                async with Persistence.open('volatile-probe', storage=storage) as second:
                    async with Runtime.attach(persistence=second,
                                              features=_features(effects, new_calls, entered=new_entered, release=new_release, buffer=new_buffer, trace=buffered_trace)) as new:
                        fresh = await new.agents.get().session('session').start('work', idempotency_key='work')
                        await asyncio.wait_for(new_entered.wait(), 5)
                        assert fresh.execution_id != stale.execution_id
                        commit_release.set()
                        try:
                            outcomes['old'] = (await stale.wait(timeout_seconds=10)).result.status.value
                        except AIError as error:
                            outcomes['old'] = error.code.value
                        new_release.set()
                        if receipt == 'unknown_close':
                            new_backend = new._execution_service.runtime_backend()
                            await asyncio.gather(new_backend._tasks[fresh.execution_id], return_exceptions=True)
                            outcomes['new'] = new_backend.worker_failure(fresh.execution_id, tenant_id=new.tenant_id).code.value
                        else:
                            try:
                                outcomes['new'] = (await fresh.wait(timeout_seconds=10)).result.status.value
                            except AIError as error:
                                outcomes['new'] = error.code.value
                        records = [await second.execution.executions.get(value, tenant_id=second.tenant_id)
                                   for value in (stale.execution_id, fresh.execution_id)]
                        outcomes['durable_successes'] = sum(record is not None and record.status is ExecutionStatus.SUCCEEDED for record in records)
                        outcomes['old_record_present'] = records[0] is not None
                        outcomes['new_record_present'] = records[1] is not None
                        assert outcomes['durable_successes'] == 1
                        assert records[0] is not None and records[0].status is ExecutionStatus.SUCCEEDED
                        assert records[1] is None
                        assert outcomes['old'] == 'SUCCEEDED'
                        if receipt == 'unknown_close':
                            assert outcomes['new'] == ErrorCode.STORAGE_COMMIT_UNKNOWN.value
                            cleanup_entered, cleanup_release, shared_closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
                            cleanup = new_backend._cleanup_failed_pending_turn
                            close_shared = new_backend._live_broker.close_shared
                            assert new_backend._live_broker.shared is not None

                            async def delayed_cleanup(turn: object, *, forget: bool = False) -> None:
                                if not forget:
                                    cleanup_entered.set()
                                    await cleanup_release.wait()
                                await cleanup(turn, forget=forget)

                            async def observe_shared_close() -> None:
                                shared_closed.set()
                                await close_shared()

                            monkeypatch.setattr(new_backend, '_cleanup_failed_pending_turn', delayed_cleanup)
                            monkeypatch.setattr(new_backend._live_broker, 'close_shared', observe_shared_close)
                            closing = asyncio.create_task(new.close())
                            try:
                                await asyncio.wait_for(cleanup_entered.wait(), 5)
                                try:
                                    await asyncio.wait_for(shared_closed.wait(), 0.1)
                                    premature_shared_close = True
                                except asyncio.TimeoutError:
                                    premature_shared_close = False
                            finally:
                                cleanup_release.set()
                                await asyncio.wait_for(closing, 10)
                            assert not premature_shared_close
                            outcomes['new'] = 'closed_after_readback'
                        elif outcomes['new'] == ErrorCode.STORAGE_COMMIT_UNKNOWN.value:
                            with pytest.raises(AIError) as reconciled:
                                await fresh.wait(timeout_seconds=10)
                            outcomes['new'] = reconciled.value.code.value
                        assert outcomes['new'] == ('closed_after_readback' if receipt == 'unknown_close' else ErrorCode.IDEMPOTENCY_CONFLICT.value)
                        replay_calls = old_calls if receipt == 'unknown_close' else new_calls
                        before = len(replay_calls)
                        replay_runtime = old if receipt == 'unknown_close' else new
                        replay = await replay_runtime.agents.get().session('session').start('work', idempotency_key='work')
                        assert replay.execution_id == stale.execution_id
                        assert (await replay.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
                        assert len(replay_calls) == before
                        print(json.dumps({'fault': 'after_last_lease_check', 'receipt': receipt, 'buffered_trace': buffered_trace, **outcomes}))
            finally:
                commit_release.set()
                new_release.set()

    if buffered_trace:
        assert old_buffer.accepted and new_buffer.accepted
        assert set(old_buffer.accepted) <= set(old_buffer.acknowledged)
        assert set(new_buffer.accepted) <= set(new_buffer.acknowledged)
