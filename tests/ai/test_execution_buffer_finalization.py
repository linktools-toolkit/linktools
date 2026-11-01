#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical handoff releases settled buffer metadata without weakening source guards."""

import asyncio
from dataclasses import replace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._retention import RuntimeRetentionController
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._execution_buffer import _ExecutionPayloadBuffer
from linktools.ai.storage import InMemoryCacheBackend

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_execution_payload_buffer import _Buffer, _capture


@pytest.mark.asyncio
async def test_terminal_handoffs_do_not_accumulate_execution_buffer_metadata() -> None:
    transport = InMemoryCacheBackend(max_bytes=1024 * 1024)
    calls = 0

    async def model(_messages, _info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("buffer-finalization")
    features.capability(ExecutionCapability(retain_history=True,
        commit=CommitPolicy.terminal(buffer=transport)), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    async with Persistence.open("buffer-finalization", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence,
            features=model_features(_Models(_UsageFunctionModel(model)), features), metrics=None) as runtime:
            backend = runtime._execution_service.runtime_backend()
            owner = backend._execution_buffer
            assert owner is not None
            for number in range(100):
                handle = await runtime.agents.get().start("question", idempotency_key=f"turn-{number}")
                result = (await handle.wait(timeout_seconds=5)).result
                assert result.status is ExecutionStatus.SUCCEEDED and result.output == {"text": "answer"}
                source = await persistence.execution.executions.get(handle.execution_id, tenant_id=runtime.tenant_id)
                assert source is not None and source.status is ExecutionStatus.SUCCEEDED
                assert await runtime._execution_service._release_terminal(handle.execution_id, tenant_id=runtime.tenant_id)
            print({"executions": calls, "backend_pending": transport._pending_count,
                   "active_scopes": len(owner._scopes), "retired_scopes": len(owner._retired_scopes),
                   "confirmed_ids": sum(len(scope.confirmed) for scope in owner._scopes.values()),
                   "completed_cleanup_refs": sum(scope.cleanup is not None for scope in owner._scopes.values())})
            assert calls == 100 and transport._pending_count == 0
            assert owner._scopes == {} and owner._retired_scopes == []
            assert not owner._tasks


@pytest.mark.asyncio
async def test_final_handoff_retries_ack_without_replaying_source_and_requires_quiescence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _Buffer(max_bytes=1024 * 1024)
    transport.fail_ack = True
    entered, release = asyncio.Event(), asyncio.Event()
    allow_handoff = False
    original = RuntimeRetentionController.release_execution_handoff
    calls = 0

    async def canonical(self, *args, **kwargs):
        if not allow_handoff:
            return False
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(RuntimeRetentionController, "release_execution_handoff", canonical)

    async def model(_messages, _info):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("confirmed")])

    features = RuntimeFeatures("buffer-handoff-ack")
    features.capability(ExecutionCapability(retain_history=True,
        commit=CommitPolicy.terminal(buffer=transport)), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    async with Persistence.open("buffer-handoff-ack", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence,
            features=model_features(_Models(_UsageFunctionModel(model)), features), metrics=None) as runtime:
            backend = runtime._execution_service.runtime_backend()
            owner = backend._execution_buffer
            handle = await runtime.agents.get().start("question", idempotency_key="one")
            try:
                await asyncio.wait_for(entered.wait(), 2)
                with pytest.raises(AIError) as running:
                    await backend.forget_runtime_execution_payloads(handle.execution_id, tenant_id=runtime.tenant_id)
                assert running.value.code is ErrorCode.STORAGE_CONFLICT
            finally:
                release.set()
            assert (await handle.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
            scope = owner._scopes[handle.execution_id]
            assert not await runtime._execution_service._release_terminal(handle.execution_id, tenant_id=runtime.tenant_id)
            assert owner._scopes[handle.execution_id] is scope
            allow_handoff = True
            with pytest.raises(AIError) as ack_failure:
                await runtime._execution_service._release_terminal(handle.execution_id, tenant_id=runtime.tenant_id)
            assert ack_failure.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
            assert owner._scopes[handle.execution_id] is scope and scope.confirmed
            source = await persistence.execution.executions.get(handle.execution_id, tenant_id=runtime.tenant_id)
            assert source is not None and source.status is ExecutionStatus.SUCCEEDED
            transport.fail_ack = False
            assert await runtime._execution_service._release_terminal(handle.execution_id, tenant_id=runtime.tenant_id)
            assert await runtime._execution_service._release_terminal(handle.execution_id, tenant_id=runtime.tenant_id)
            assert not owner._scopes and not owner._retired_scopes
            assert not scope.confirmed and scope.cleanup is None
            assert transport._pending_count == 0 and calls == 1
            assert len(transport.admissions) == len({(item[0], item[1]) for item in transport.admissions})


@pytest.mark.asyncio
async def test_cancelled_finalizer_keeps_exact_ack_owned_until_retry() -> None:
    transport = _Buffer(max_bytes=65536)
    transport.ack_release.clear()
    owner = _ExecutionPayloadBuffer(transport)
    capture = _capture()
    identifier = capture.execution.execution_id
    await owner.append(capture)
    _value, ids = await owner.read(identifier, seal=True)
    scope = owner._scopes[identifier]
    owner.confirm(identifier, ids)
    await owner.release(identifier)
    await asyncio.wait_for(transport.ack_started.wait(), 2)
    finalizer = asyncio.create_task(owner.forget(identifier))
    try:
        await asyncio.sleep(0)
        finalizer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await finalizer
        assert owner._scopes[identifier] is scope
        assert scope.cleanup is not None and not scope.cleanup.done()
        assert scope.confirmed == set(ids) and transport._pending_count == 1
    finally:
        transport.ack_release.set()
        await asyncio.gather(finalizer, return_exceptions=True)
    await owner.forget(identifier)
    await owner.forget(identifier)
    assert not owner._scopes and scope.cleanup is None and not scope.confirmed
    assert transport._pending_count == 0 and len(transport.admissions) == 1
    assert {item for _scope, values in transport.acks for item in values} == set(ids)
    await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", ("unconfirmed", "sealed", "uncertain"))
async def test_finalizer_cannot_ack_unconfirmed_or_unknown_payloads(pending: str) -> None:
    transport = _Buffer(max_bytes=65536)
    owner = _ExecutionPayloadBuffer(transport)
    capture = _capture()
    identifier = capture.execution.execution_id
    if pending == "uncertain":
        transport.fail_append = transport.fail_read = True
        with pytest.raises(AIError):
            await owner.append(capture)
        ids = (transport.admissions[0][1],)
    else:
        await owner.append(capture)
        _value, ids = await owner.read(identifier, seal=pending == "sealed")
    scope = owner._scopes[identifier]
    with pytest.raises(AIError):
        await owner.forget(identifier)
    assert owner._scopes[identifier] is scope
    assert transport._pending_count == 1 and transport.acks == []
    assert tuple(entry.id for entry in scope.entries) == ids
    if pending == "uncertain":
        assert scope.uncertain is not None
        transport.fail_read = False
        owner.discard(identifier)
        await owner.flush_cleanup(identifier)
    else:
        with pytest.raises(AIError):
            await owner.close()
        owner.confirm(identifier, ids)
        owner.confirm(identifier, ids)
        await owner.release(identifier)
    await owner.forget(identifier)
    assert not owner._scopes and not scope.confirmed and scope.cleanup is None
    await owner.close()


@pytest.mark.asyncio
async def test_old_finalizer_and_ack_cannot_remove_new_generation_or_peer() -> None:
    transport = _Buffer(max_bytes=65536)
    transport.ack_release.clear()
    owner = _ExecutionPayloadBuffer(transport)
    capture = _capture()
    identifier = capture.execution.execution_id
    await owner.append(capture)
    owner.bind_producer(identifier, 7)
    _value, old_ids = await owner.read(identifier)
    old = owner._scopes[identifier]
    owner.confirm(identifier, old_ids)
    owner.retire(identifier, 7)
    for generation in (6, 7):
        with pytest.raises(AIError):
            owner.bind_producer(identifier, generation)
    await asyncio.wait_for(transport.ack_started.wait(), 2)
    finalizer = asyncio.create_task(owner.forget(identifier))
    await asyncio.sleep(0)
    owner.bind_producer(identifier, 11)
    current = owner._scopes[identifier]
    await owner.append(capture)
    peer = replace(capture, execution=replace(capture.execution, execution_id="peer", root_execution_id="peer"))
    await owner.append(peer)
    transport.ack_release.set()
    await finalizer
    assert owner._scopes[identifier] is current
    owner.require_producer(identifier, 11)
    with pytest.raises(AIError):
        owner.require_producer(identifier, 7)
    assert not old.confirmed and old.cleanup is None
    for value in (identifier, "peer"):
        _payload, ids = await owner.read(value, seal=True)
        owner.confirm(value, ids)
        await owner.release(value)
        await owner.forget(value)
    assert not owner._scopes and not owner._retired_scopes and transport._pending_count == 0
    await owner.close()


@pytest.mark.asyncio
async def test_scheduled_old_append_cannot_recreate_or_write_successor_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _Buffer(max_bytes=65536)
    owner = _ExecutionPayloadBuffer(transport)
    capture = _capture()
    identifier = capture.execution.execution_id
    owner.bind_producer(identifier, 7)
    owner.retire(identifier, 7)
    entered, release = asyncio.Event(), asyncio.Event()
    original = owner._append

    async def held(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(owner, "_append", held)
    admission = asyncio.create_task(owner.append(capture))
    await asyncio.wait_for(entered.wait(), 2)
    await owner.forget(identifier)
    owner.bind_producer(identifier, 11)
    successor = owner._scopes[identifier]
    release.set()
    with pytest.raises(AIError) as stale:
        await admission
    assert stale.value.code is ErrorCode.STORAGE_CONFLICT
    assert owner._scopes[identifier] is successor and successor.entries == []
    owner.require_producer(identifier, 11)
    assert transport._pending_count == 0 and transport.admissions == []
    await owner.release(identifier)
    await owner.forget(identifier)
    await owner.close()


@pytest.mark.asyncio
async def test_source_absence_does_not_authorize_forgetting_a_sealed_prefix() -> None:
    transport = _Buffer(max_bytes=65536)
    features = RuntimeFeatures("buffer-absent-source")
    features.capability(ExecutionCapability(retain_history=True,
        commit=CommitPolicy.terminal(buffer=transport)), scope="runtime")
    async with Persistence.open("buffer-absent-source", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence, features=features, metrics=None,
                                  auto_recover=False) as runtime:
            owner = runtime._execution_service.runtime_backend()._execution_buffer
            capture = _capture()
            identifier = capture.execution.execution_id
            await owner.append(capture)
            _value, ids = await owner.read(identifier, seal=True)
            scope = owner._scopes[identifier]
            try:
                assert await persistence.execution.executions.get(identifier, tenant_id=runtime.tenant_id) is None
                with pytest.raises(AIError):
                    await runtime._execution_service._release_terminal(identifier, tenant_id=runtime.tenant_id)
                assert owner._scopes[identifier] is scope
                assert transport._pending_count == 1 and transport.acks == []
                assert (await owner.source_prefix(identifier))[1] == ids
            finally:
                # This fixture never dispatched its sealed bytes to the source.
                owner.reject_source(identifier, ids)
                owner.discard(identifier)
                await owner.forget(identifier)
