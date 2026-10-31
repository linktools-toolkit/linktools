#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution terminal policies admit real bytes before source materialization."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._execution_buffer import _ExecutionPayloadCapture
from linktools.ai.storage import InMemoryCacheBackend

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_execution_payload_materialization import _Objects


class _PayloadBuffer(InMemoryCacheBackend):
    def __init__(self):
        super().__init__(max_bytes=64 * 1024 * 1024)
        self.captures = []
        self.accepted = []
        self.acknowledged = []

    async def append(self, scope, id, key, content, *, wait):
        assert not wait and isinstance(content, bytes)
        capture = _ExecutionPayloadCapture.decode(content)
        result = await super().append(scope, id, key, content, wait=wait)
        if result:
            self.captures.append(capture)
            self.accepted.append((scope, id))
        return result

    async def ack(self, scope, ids):
        self.acknowledged.extend((scope, id) for id in ids)
        await super().ack(scope, ids)


@pytest.mark.asyncio
@pytest.mark.parametrize(("outcome", "sessionless"), (("success", False), ("failure", False),
                                                        ("cancelled", False), ("success", True)))
async def test_pending_payloads_and_checkpoint_are_read_from_buffer_before_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str, sessionless: bool,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    buffer = _PayloadBuffer()
    prompt = ("input" * 16000,)
    answer = "answer" * 13000
    calls = 0

    class Objects(_Objects):
        async def put(self, *args, **kwargs):
            assert any(capture.terminal is not None for capture in buffer.captures)
            return await super().put(*args, **kwargs)

    objects = Objects()

    async def model(messages, info):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        if outcome == "failure":
            raise ValueError("provider failed")
        if outcome == "cancelled":
            raise asyncio.CancelledError
        return ModelResponse(parts=[TextPart(answer)])

    features = RuntimeFeatures("execution-payload-buffer")
    features.capability(ExecutionCapability(retain_history=True,
        commit=CommitPolicy.terminal(buffer=buffer)), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    storage = PersistenceStorage.sqlite(tmp_path / "state.db", object_store=objects,
        coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    async with Persistence.open("execution-payload-buffer", storage=storage) as persistence:
        async with Runtime.attach(persistence=persistence,
            features=model_features(_Models(_UsageFunctionModel(model)), features), metrics=None) as runtime:
            backend = runtime._execution_service.runtime_backend()
            source_preparing = False
            read = backend._execution_buffer.read
            commit = backend._runtime_commands.commit_pending_turn
            get_run, get_checkpoint = runtime._run_store.get_agent_run, runtime._run_store.latest_checkpoint

            async def observe_read(*args, **kwargs):
                nonlocal source_preparing
                result = await read(*args, **kwargs)
                source_preparing = result[0].terminal is not None
                return result

            async def source_commit(*args, **kwargs):
                nonlocal source_preparing
                try:
                    return await commit(*args, **kwargs)
                finally:
                    source_preparing = False

            async def guarded_run(*args, **kwargs):
                assert not source_preparing, "source publication must decode its buffered run"
                return await get_run(*args, **kwargs)

            async def guarded_checkpoint(*args, **kwargs):
                assert not source_preparing, "source publication must decode its buffered checkpoint"
                return await get_checkpoint(*args, **kwargs)

            monkeypatch.setattr(backend._execution_buffer, "read", observe_read)
            monkeypatch.setattr(backend._runtime_commands, "commit_pending_turn", source_commit)
            monkeypatch.setattr(runtime._run_store, "get_agent_run", guarded_run)
            monkeypatch.setattr(runtime._run_store, "latest_checkpoint", guarded_checkpoint)
            target = runtime.agents.get() if sessionless else await runtime.agents.get().create_session("session")
            execution = await target.start(prompt, idempotency_key="request")
            try:
                await asyncio.wait_for(entered.wait(), 5)
                assert buffer.captures and all(capture.reservation is not None for capture in buffer.captures)
                assert buffer.captures[-1].execution.input_context is not None
                assert await persistence.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id) is None
                assert objects.writes == []
            finally:
                release.set()
            result = (await execution.wait(timeout_seconds=10)).result
            assert result.status is {"success": ExecutionStatus.SUCCEEDED, "failure": ExecutionStatus.FAILED,
                                     "cancelled": ExecutionStatus.CANCELLED}[outcome]
            terminal = [capture for capture in buffer.captures if capture.terminal is not None][-1]
            assert terminal.execution.status is result.status
            assert terminal.checkpoint_payload is not None if outcome == "success" else terminal.checkpoint_payload is None
            if outcome == "success":
                assert terminal.terminal.result.output.kind == "inline"
                assert terminal.terminal.result.output.decode() == {"text": answer}
                assert result.output == {"text": answer}
            assert calls == 1
            assert objects.writes
        assert set(buffer.acknowledged) == set(buffer.accepted)


@pytest.mark.asyncio
@pytest.mark.parametrize("interference", ("late_audit", "partial_source"))
async def test_pending_payload_dispatch_retains_exact_source_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interference: str,
) -> None:
    from linktools.ai.core import ExecutionEventType
    from linktools.ai.errors import AIError, ErrorCode

    buffer = _PayloadBuffer()
    prepare_entered, prepare_release = asyncio.Event(), asyncio.Event()
    calls = source_calls = 0

    async def model(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("execution-payload-interference")
    features.capability(ExecutionCapability(retain_history=True,
        commit=CommitPolicy.terminal(buffer=buffer)), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    storage = PersistenceStorage.sqlite(tmp_path / "state.db",
        coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    persistence = Persistence(storage)
    await persistence.initialize(namespace="execution-payload-interference", tenant_id="default")
    source_group = persistence.execution.executions.state_store.storage_group
    try:
        context = Runtime.attach(persistence=persistence,
            features=model_features(_Models(_UsageFunctionModel(model)), features), metrics=None)
        runtime = await context.__aenter__()
        owners = tuple(persistence._runtime_owners)
        backend = runtime._execution_service.runtime_backend()
        source_commit = backend._runtime_commands.commit_pending_turn
        prepare = backend._context_archive.prepare

        async def gated_prepare(*args, **kwargs):
            prepare_entered.set()
            await prepare_release.wait()
            return await prepare(*args, **kwargs)

        async def intercepted_commit(*args, **kwargs):
            nonlocal source_calls
            source_calls += 1
            value = await source_commit(*args, **kwargs)
            if interference == "partial_source":
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR,
                    safe_details={"phase": "partial_publication_readback"})
            return value

        monkeypatch.setattr(backend._runtime_commands, "commit_pending_turn", intercepted_commit)
        if interference == "late_audit":
            monkeypatch.setattr(backend._context_archive, "prepare", gated_prepare)
        try:
            session = await runtime.agents.get().create_session("session")
            execution = await session.start("question", idempotency_key="request")
            if interference == "late_audit":
                await asyncio.wait_for(prepare_entered.wait(), 5)
                pending = backend._pending_turns.get(execution.execution_id, tenant_id=runtime.tenant_id)
                await backend._append_event(pending.execution, ExecutionEventType.MODEL_REQUEST_STARTED,
                    {"late": "must not be silently omitted"})
                prepare_release.set()
            with pytest.raises(AIError) as failed:
                await execution.wait(timeout_seconds=10)
            assert failed.value.code is (ErrorCode.STORAGE_CONFLICT if interference == "late_audit"
                                         else ErrorCode.STORAGE_INTEGRITY_ERROR)
            source = await persistence.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
            if interference == "late_audit":
                assert source is None and source_calls == 0
            else:
                assert source.status is ExecutionStatus.SUCCEEDED and source_calls == 1
                assert not buffer.acknowledged
                assert all([await buffer.read_batch(scope, limit=100) for scope, _ in buffer.accepted])
            assert calls == 1
        finally:
            prepare_release.set()
            with pytest.raises(AIError) as close_error:
                await context.__aexit__(None, None, None)
            assert close_error.value.code is (ErrorCode.STORAGE_CONFLICT if interference == "late_audit"
                                              else ErrorCode.STORAGE_RECOVERY_REQUIRED)
        if interference == "late_audit":
            assert set(buffer.acknowledged) == set(buffer.accepted)
        else:
            assert not buffer.acknowledged
    finally:
        try:
            with pytest.raises(AIError) as persistence_error:
                await persistence.close()
            assert persistence_error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
            assert persistence.ready and tuple(persistence._runtime_owners) == owners
        finally:
            # Preserve unresolved producer ownership; release only the fixture source.
            await source_group.close()
            await source_group.context.close()
            assert source_group.context.closed
