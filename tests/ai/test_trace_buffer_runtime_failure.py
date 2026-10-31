#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rejected ordinary Trace payloads still settle admitted Execution control."""

import asyncio
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import CheckpointCapability, TraceCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus, OperationStatus, PersistenceDomain, ResourceKind, agent_run_id
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import CancelExecutionRequest
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._step_contracts import StepEvent
from linktools.ai.runtime import Runtime
from linktools.ai.storage import InMemoryCacheBackend, StorageRoute

from ._runtime_test_helpers import model_features
from .test_explicit_runtime_features import _FeatureFunctionModel, _FeatureModels, _features


@pytest.mark.asyncio
@pytest.mark.parametrize("split,checkpoint,rejection,ack_failure,source_rejection", (
    (False, False, "before_provider", False, False),
    (False, False, "response", False, False),
    (False, True, "response", False, False),
    (True, True, "response", False, False),
    (False, False, "response", True, False),
    (False, False, "response", False, True),
    (False, False, "terminal", False, False),
))
async def test_rejected_trace_payload_finishes_source_control_without_replaying_or_saving_rejected_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split: bool, checkpoint: bool,
    rejection: str, ack_failure: bool, source_rejection: bool,
) -> None:
    calls = 0
    rejected = []

    class Buffer(InMemoryCacheBackend):
        rejecting = True
        ack_failure = False
        ack_calls = 0

        async def append(self, scope, id, key, content, *, wait):
            if self.rejecting and (rejection == "before_provider" or calls):
                rejected.append((scope, id))
                return False
            return await super().append(scope, id, key, content, wait=wait)

        async def ack(self, scope, ids):
            self.ack_calls += 1
            if self.ack_failure:
                raise OSError("ack transport failed")
            await super().ack(scope, ids)

    async def model(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart("rejected response body")])

    buffer = Buffer(max_bytes=4 * 1024 * 1024)
    buffer.ack_failure = ack_failure
    buffer.rejecting = rejection != "terminal"
    storage = PersistenceStorage.sqlite(tmp_path / "state.db")
    if split:
        storage = PersistenceStorage(replace(storage.plan, conversation=StorageRoute.sqlite(tmp_path / "conversation.db")),
                                     object_store=storage.object_store)
    policy = CommitPolicy.batch(buffer=buffer, batch_size=128, flush_interval=60)
    capabilities = [TraceCapability(commit=policy)]
    if checkpoint:
        capabilities.append(CheckpointCapability())
    features = model_features(_FeatureModels(_FeatureFunctionModel(model)), _features(*capabilities))
    async with Persistence.open("trace-buffer-failure", storage=storage) as persistence:
        async with Runtime.attach(persistence=persistence, features=features, metrics=None) as runtime:
            backend = runtime._execution_service.runtime_backend()
            commit_terminal = backend._runtime_commands.commit_terminal_checkpoint
            if rejection == "terminal":
                commit_success = backend._commit_success
                async def refuse_final_payload(*args, **kwargs):
                    if calls == 1:
                        run_id = args[4]
                        events = await runtime._run_store.list_events(agent_run_id=run_id)
                        await runtime._run_store.append_event(StepEvent(
                            run_id, "MODEL_REQUEST_STARTED", max((event.step_index for event in events), default=0) + 1,
                            metadata={"boundary": "rejected terminal marker"},
                        ))
                        buffer.rejecting = True
                    return await commit_success(*args, **kwargs)
                monkeypatch.setattr(backend, "_commit_success", refuse_final_payload)
            if source_rejection:
                async def reject_terminal(*args, **kwargs):
                    raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
                monkeypatch.setattr(backend._runtime_commands, "commit_terminal_checkpoint", reject_terminal)
            session = await runtime.agents.get().create_session("session")
            execution = await session.start("question", idempotency_key="request")
            if source_rejection:
                with pytest.raises(AIError):
                    await execution.wait(timeout_seconds=5)
                current = await persistence.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
                assert current.status is ExecutionStatus.STARTED
                run_id = agent_run_id(namespace=runtime.namespace, tenant_id=runtime.tenant_id,
                                      execution_id=execution.execution_id, agent_run_seq=current.agent_run_seq)
                generation = runtime._run_store.execution_producer_generation(run_id, execution_id=execution.execution_id)
                assert generation is not None
                assert runtime._run_store.trace_buffer_failure_settled(run_id,
                    execution_id=execution.execution_id, producer_generation=generation)
                with pytest.raises(AIError):
                    await runtime._run_store.preflight_close()
                assert runtime._run_store.trace_buffer_failure_settled(run_id,
                    execution_id=execution.execution_id, producer_generation=generation)
                monkeypatch.setattr(backend._runtime_commands, "commit_terminal_checkpoint", commit_terminal)
                committed = await backend._commit_failure(current, AIError(ErrorCode.STORAGE_UNAVAILABLE),
                    agent_run_id=run_id, producer_generation=generation)
                assert committed.execution_id == execution.execution_id
                assert committed.status is ExecutionStatus.FAILED
                await runtime._run_store.preflight_close()
                await backend.release_runtime_execution(execution.execution_id, tenant_id=runtime.tenant_id)
            outcome = (await execution.wait(timeout_seconds=5)).result
            assert outcome.status is ExecutionStatus.FAILED
            assert outcome.error_code == ErrorCode.STORAGE_UNAVAILABLE.value
            stored = await persistence.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
            assert stored is not None and stored.status is ExecutionStatus.FAILED
            assert stored.result is not None and stored.result.output is None
            assert calls == (0 if rejection == "before_provider" else 1)
            assert rejected
            archive = runtime._run_store.read_store(PersistenceDomain.TRACE)
            runs = await archive.list_agent_runs()
            resolved_prefix = []
            events_prefix = []
            for run in runs:
                interactions = await archive.list_model_interactions(agent_run_id=run.agent_run_id)
                resolved_prefix.extend(await archive.resolve_model_interactions(interactions))
                events_prefix.extend(await archive.list_events(agent_run_id=run.agent_run_id))
            if rejection == "terminal":
                assert "rejected response body" in repr(resolved_prefix)
                assert "rejected terminal marker" not in repr(events_prefix)
            else:
                assert "rejected response body" not in repr(resolved_prefix)
            session_record = await persistence.conversation.sessions.get("session", tenant_id=runtime.tenant_id)
            assert session_record.active_execution_id is None
            if ack_failure:
                assert buffer.ack_calls > 0
                buffer.ack_failure = False
            if not source_rejection:
                buffer.rejecting = False
                following = await session.start("next question", idempotency_key="following")
                assert (await following.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
                assert calls == (1 if rejection == "before_provider" else 2)


def _track_source_transactions(monkeypatch, archive, repository):
    group = repository.state_store.storage_group
    frame = ContextVar("terminal-frame", default=None)
    trace_frames, terminal_frames = [], []
    mutate, sync, terminal = group.mutate, archive._sync_projection_in_transaction, repository.commit_terminal_in_transaction

    async def framed(stores, callback):
        async def wrapped(transaction):
            if frame.get() is not None:
                return await callback(transaction)
            token = frame.set(object())
            try:
                return await callback(transaction)
            finally:
                frame.reset(token)
        return await mutate(stores, wrapped)

    async def record_trace(*args, **kwargs):
        trace_frames.append(frame.get())
        return await sync(*args, **kwargs)

    async def record_terminal(*args, **kwargs):
        terminal_frames.append(frame.get())
        return await terminal(*args, **kwargs)

    monkeypatch.setattr(group, "mutate", framed)
    monkeypatch.setattr(archive, "_sync_projection_in_transaction", record_trace)
    monkeypatch.setattr(repository, "commit_terminal_in_transaction", record_terminal)
    return trace_frames, terminal_frames


@pytest.mark.asyncio
@pytest.mark.parametrize("reject_cancel_tail", [False, True])
async def test_trace_buffer_cancellation_keeps_control_settled(tmp_path, monkeypatch, reject_cancel_tail):
    entered = asyncio.Event()
    calls = 0

    class Buffer(InMemoryCacheBackend):
        rejecting = False

        async def append(self, *args, **kwargs):
            if self.rejecting:
                return False
            return await super().append(*args, **kwargs)

    async def model(messages, info):
        nonlocal calls
        calls += 1
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("cancelled provider continued")

    buffer = Buffer(max_bytes=4 * 1024 * 1024)
    policy = CommitPolicy.batch(buffer=buffer, batch_size=128, flush_interval=60)
    features = model_features(_FeatureModels(_FeatureFunctionModel(model)), _features(
        TraceCapability(commit=policy),
    ))
    async with Persistence.open("terminal-trace-cancel", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as persistence:
        async with Runtime.attach(persistence=persistence, features=features, metrics=None) as runtime:
            store = runtime._run_store
            archive = store.read_store(PersistenceDomain.TRACE)
            repository = persistence.execution.executions
            trace_frames, terminal_frames = _track_source_transactions(monkeypatch, archive, repository)
            session = await runtime.agents.get().create_session("session")
            execution = await session.start("question")
            await asyncio.wait_for(entered.wait(), 5)
            queued = tuple(store.trace_persistence._buffers.values())
            assert len(queued) == 1
            accepted = await buffer.read_batch(queued[0].id, limit=100)
            assert accepted and not trace_frames
            buffer.rejecting = reject_cancel_tail
            await runtime.executions.cancel(execution.execution_id, CancelExecutionRequest(runtime.default_principal, "cancel"))
            try:
                result = (await execution.wait(timeout_seconds=5)).result
            except AIError as error:
                source = await persistence.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
                raise AssertionError(f"cancel wait failed with source status {source.status}") from error
            assert result.status is ExecutionStatus.CANCELLED
            assert calls == 1
            source = await persistence.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
            assert source.status is ExecutionStatus.CANCELLED
            head = await repository.get_history_head(execution.execution_id, tenant_id=runtime.tenant_id)
            assert head.state.value == "sealed"
            repeated = await runtime.executions.cancel(
                execution.execution_id, CancelExecutionRequest(runtime.default_principal, "cancel"),
            )
            assert repeated.cancelled and calls == 1
            session_record = await persistence.conversation.sessions.get("session", tenant_id=runtime.tenant_id)
            assert session_record.active_execution_id is None
            assert len(terminal_frames) == 1 and trace_frames
            operations = await persistence.execution.operations.list_pending(
                ResourceKind.EXECUTION, execution.execution_id, tenant_id=runtime.tenant_id,
                limit=10, states=frozenset(OperationStatus),
            )
            assert len(operations) == 1 and operations[0].status is OperationStatus.SUCCEEDED
            runs = await archive.list_agent_runs()
            assert len(runs) == 1
            assert await archive.list_model_interactions(agent_run_id=runs[0].agent_run_id)
            await store.preflight_close()
            assert await buffer.read_batch(queued[0].id, limit=100) == ()
