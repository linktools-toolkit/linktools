#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability-owned persistence preserves isolated reads and committed context."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from linktools.ai.persistence import Persistence
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence._checkpoint_persistence import CheckpointPersistence
from linktools.ai.persistence._conversation_persistence import ConversationPersistence
from linktools.ai.persistence._execution_repositories import ExecutionRepositoryImpl
from linktools.ai.persistence._memory import InMemoryStateStorageGroup, InMemoryStateStore
from linktools.ai.persistence._model_interaction_store import ModelInteractionStateStepArchive
from linktools.ai.persistence._step_archive import (
    InMemoryStepArchive,
    StagingAgentRunStore,
    StateStepArchive,
    _AgentRunHistoryLock,
)
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint, AgentRunRecord, StepEvent
from linktools.ai.persistence.state import StateTransactionNestingError
from linktools.ai.persistence.runs import RuntimeAgentRunStore
from linktools.ai.persistence import _trace_persistence
from linktools.ai.persistence._trace_persistence import TracePersistence

from .test_pending_turn_commit import _commit, _pending_turn, _turn_state
from .test_incremental_projection_boundaries import _storage
from .test_step_projection import _execution

pytestmark = pytest.mark.asyncio


async def test_conversation_owners_isolate_identical_run_ids_and_preserve_context() -> None:
    archives = [InMemoryStepArchive(PersistenceDomain.CONVERSATION) for _ in range(2)]
    for archive in archives:
        await archive.initialize()
    try:
        for archive, answer in zip(archives, ("first", "second")):
            messages = [ModelRequest(parts=[UserPromptPart("question")]),
                        ModelResponse(parts=[TextPart(answer)])]
            await archive.materialize_checkpoint(
                AgentRunRecord("same-run"), AgentRunCheckpoint("same-run", 1, messages),
            )
        for archive, answer in zip(archives, ("first", "second")):
            owner = ConversationPersistence(archive)
            request = dict(history_id=None, agent_run_id="same-run", tenant_id="tenant")
            messages = [message async for message in owner.iter_conversation_messages(**request)]
            assert len(messages) == await owner.conversation_message_count(**request) == 2
            assert messages[-1].parts == [TextPart(answer)]
            assert [message async for message in owner.iter_conversation_message_range(
                **request, start=1, end=2,
            )] == messages[1:]
            assert await owner.load_conversation_model_context(**request) == tuple(messages)
    finally:
        for archive in archives:
            await archive.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_conversation_owner_reads_committed_state_without_changing_cursor(
    tmp_path: Path, backend: str,
) -> None:
    async with _turn_state(backend, tmp_path) as state:
        turn = _pending_turn(state)
        await _commit(state, turn)
        owner = ConversationPersistence(state.storage.run_store.read_store(PersistenceDomain.CONVERSATION))
        request = dict(history_id="history", agent_run_id=turn.run.agent_run_id, tenant_id="tenant")
        before = await state.storage.conversation.sessions.get("session", tenant_id="tenant")
        assert [message async for message in owner.iter_conversation_messages(**request)] == turn.checkpoint.messages
        assert await owner.conversation_message_count(**request) == 2
        assert [message async for message in owner.iter_conversation_message_range(
            **request, start=1, end=2,
        )] == turn.checkpoint.messages[1:]
        assert await owner.load_conversation_model_context(
            **request, message_count=turn.cursor.message_count,
        ) == tuple(turn.checkpoint.messages)
        assert await state.storage.conversation.sessions.get("session", tenant_id="tenant") == before


async def test_state_conversation_reads_require_history_identity(tmp_path: Path) -> None:
    async with _turn_state("memory", tmp_path) as state:
        owner = ConversationPersistence(state.storage.run_store.read_store(PersistenceDomain.CONVERSATION))
        request = dict(history_id=None, agent_run_id="run", tenant_id="tenant")
        with pytest.raises(AIError) as missing:
            await owner.load_conversation_model_context(**request)
        assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        with pytest.raises(AIError) as missing:
            await owner.conversation_message_count(**request)
        assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        with pytest.raises(AIError) as missing:
            _ = [message async for message in owner.iter_conversation_messages(**request)]
        assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("complete", (False, True))
async def test_checkpoint_owner_requires_complete_recovery_before_materializing(complete: bool) -> None:
    staging = StagingAgentRunStore()
    archive = InMemoryStepArchive(PersistenceDomain.CHECKPOINT)
    await staging.initialize()
    await archive.initialize()
    try:
        owner = CheckpointPersistence(staging, archive, None)
        run = AgentRunRecord("recovery")
        checkpoint = AgentRunCheckpoint(
            "recovery", 1, [ModelRequest(parts=[UserPromptPart("question")])],
            state="complete" if complete else "interrupted",
        )
        await staging.register_agent_run(run)
        await staging.save_checkpoint(checkpoint)
        if complete:
            await owner.materialize_recovery_checkpoint(agent_run_id="recovery", require_complete=True)
            assert await archive.latest_checkpoint(agent_run_id="recovery") == checkpoint
        else:
            with pytest.raises(AIError) as interrupted:
                await owner.materialize_recovery_checkpoint(agent_run_id="recovery", require_complete=True)
            assert interrupted.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            assert await archive.get_agent_run(agent_run_id="recovery") is None
        assert await staging.latest_checkpoint(agent_run_id="recovery", include_interrupted=True) == checkpoint
    finally:
        await staging.close()
        await archive.close()


async def test_trace_pressure_and_timer_lifetimes_are_isolated_between_owners() -> None:
    writes: list[tuple[str, str]] = []
    published = asyncio.Event()
    owners: dict[str, TracePersistence] = {}
    tasks: set[asyncio.Task[object]] = set()

    async def ready() -> None:
        pass

    def make_owner(name: str) -> TracePersistence:
        async def flush(
            agent_run_id: str, *, execution_id: str,
            producer_generation: int | None = None, deferred: bool = False,
        ) -> None:
            writes.append((name, execution_id))
            owners[name].dirty.discard(agent_run_id)
            published.set()

        return TracePersistence(
            None, history_lock=_AgentRunHistoryLock(), ensure_ready=ready,
            require_mutable=lambda _run: None, has_flight=lambda _run: False,
            flush=flush, flush_boundary=flush, background_tasks=tasks,
        )

    owners.update((name, make_owner(name)) for name in ("first", "second"))
    try:
        for name, owner in owners.items():
            owner.configure("same-run", buffer=InMemoryCacheBackend(max_bytes=65536), flush_interval=60.0)
            owner.dirty.add("same-run")
            await owner.defer("same-run", execution_id=name, producer_generation=1)
        for _ in range(_trace_persistence.DEFAULT_TRACE_BATCH_SIZE - 1):
            await owners["first"].defer("same-run", execution_id="first", producer_generation=1)
        await asyncio.wait_for(published.wait(), 2)
        assert writes == [("first", "first")]
        assert owners["second"].dirty == {"same-run"}
        assert owners["second"]._buffers["same-run"].scope.status().pending == 1
        await asyncio.wait_for(owners["first"].settle_buffers(), 1)
        assert owners["second"]._buffers["same-run"].scope.status().accepting
        assert writes == [("first", "first")]
    finally:
        for owner in owners.values():
            owner.dirty.clear()
            await asyncio.wait_for(owner.settle_buffers(), 1)
        await asyncio.sleep(0)
        assert not tasks


@asynccontextmanager
async def _separate_trace_stores() -> AsyncIterator[tuple[ExecutionRepositoryImpl, StateStepArchive]]:
    group = InMemoryStateStorageGroup()
    core, evidence = InMemoryStateStore(group), InMemoryStateStore(group)
    await core.initialize()
    await evidence.initialize()
    repository = ExecutionRepositoryImpl(core, namespace="trace-owner", tenant_id="tenant")
    archive = ModelInteractionStateStepArchive(
        evidence, object_store=None, namespace="trace-owner", tenant_id="tenant",
        runtime_domain=PersistenceDomain.TRACE, execution_repository=repository,
    )
    await archive.initialize()
    try:
        execution = replace(_execution(), revision=1)
        await repository.create_with_history_head(execution)
        await core.mutate(lambda transaction: repository.admit_history_producer_in_transaction(
            transaction, execution,
        ))
        yield repository, archive
    finally:
        await archive.close()
        await evidence.close()
        await core.close()


async def test_separate_trace_store_fences_against_core_producer_generation() -> None:
    async with _separate_trace_stores() as (repository, archive):
        run = AgentRunRecord("run")
        await archive.register_agent_run(run, execution_id="execution")
        before = await repository.get_history_head("execution", tenant_id="tenant")
        event = StepEvent("run", "TOOL_CALL_STARTED", 1, tool_call_id="tool")
        await archive.sync_prepared_projection(
            run, events=(event,), checkpoints=(), execution_id="execution", producer_generation=1,
        )
        after = await repository.get_history_head("execution", tenant_id="tenant")
        assert after.revision == before.revision + 1
        assert await archive.list_events(agent_run_id="run") == [event]
        await repository.state_store.mutate(lambda transaction: repository.admit_history_producer_in_transaction(
            transaction, replace(_execution(), revision=2),
        ))
        with pytest.raises(AIError) as stale:
            await archive.sync_prepared_projection(
                run, events=(replace(event, step_index=2),), checkpoints=(),
                execution_id="execution", producer_generation=1,
            )
        assert stale.value.code is ErrorCode.STORAGE_CONFLICT
        assert await archive.list_events(agent_run_id="run") == [event]


@pytest.mark.parametrize("committed", (False, True))
async def test_trace_and_core_head_share_rollback_and_unknown_commit(
    monkeypatch: pytest.MonkeyPatch, committed: bool,
) -> None:
    async with _separate_trace_stores() as (repository, archive):
        run = AgentRunRecord("run")
        await archive.register_agent_run(run, execution_id="execution")
        before = await repository.get_history_head("execution", tenant_id="tenant")
        event = StepEvent("run", "TOOL_CALL_STARTED", 1, tool_call_id="tool")
        group = archive.state_store.storage_group
        mutate = group.mutate

        async def fail(stores, callback):
            async def operation(transaction):
                result = await callback(transaction)
                if not committed:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                return result
            await mutate(stores, operation)
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        with monkeypatch.context() as patch:
            patch.setattr(group, "mutate", fail)
            with pytest.raises(AIError) as error:
                await archive.append_event(event, execution_id="execution")
            assert error.value.code is (
                ErrorCode.STORAGE_COMMIT_UNKNOWN if committed else ErrorCode.STORAGE_CONFLICT
            )
        after = await repository.get_history_head("execution", tenant_id="tenant")
        assert after.revision == before.revision + int(committed)
        assert await archive.list_events(agent_run_id="run") == ([event] if committed else [])


async def test_trace_cannot_expand_an_existing_transaction_without_core_enlistment() -> None:
    async with _separate_trace_stores() as (repository, archive):
        before = await repository.get_history_head("execution", tenant_id="tenant")

        async def incomplete(_transaction) -> None:
            await archive.register_agent_run(AgentRunRecord("run"), execution_id="execution")

        with pytest.raises(StateTransactionNestingError):
            await archive.state_store.mutate(incomplete)
        assert await archive.get_agent_run(agent_run_id="run") is None
        assert await repository.get_history_head("execution", tenant_id="tenant") == before


@pytest.mark.parametrize("checkpoint_enabled", (False, True))
async def test_core_producer_fencing_does_not_require_trace_archive(checkpoint_enabled: bool) -> None:
    async with _separate_trace_stores() as (repository, unused_trace):
        recovery = ModelInteractionStateStepArchive(
            unused_trace.state_store, object_store=None, namespace="trace-owner", tenant_id="tenant",
            runtime_domain=PersistenceDomain.CHECKPOINT,
        )
        store = RuntimeAgentRunStore(
            StagingAgentRunStore(),
            conversation_archive=InMemoryStepArchive(PersistenceDomain.CONVERSATION),
            execution_archive=None, recovery_archive=recovery if checkpoint_enabled else None,
            execution_repository=repository, tenant_id="tenant",
        )
        await store.initialize()
        try:
            store.bind_execution_producer(
                "run", execution_id="execution", producer_generation=1,
                conversation=False, restart_checkpoints=checkpoint_enabled, durable_trace=False,
            )
            await store.register_agent_run(AgentRunRecord("run"), execution_id="execution")
            await repository.state_store.mutate(lambda transaction: repository.admit_history_producer_in_transaction(
                transaction, replace(_execution(), revision=2),
            ))
            with pytest.raises(AIError) as stale:
                await store.save_checkpoint(
                    AgentRunCheckpoint("run", 1, [ModelRequest(parts=[UserPromptPart("question")])]),
                    execution_id="execution", producer_generation=1,
                )
            assert stale.value.code is ErrorCode.STORAGE_CONFLICT
            assert await recovery.latest_checkpoint(agent_run_id="run", include_interrupted=True) is None
        finally:
            await store.preflight_close()
            await store.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_materialized_trace_partition_is_atomic_with_core_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    storage = Persistence(_storage(backend, tmp_path))
    await storage.initialize(namespace="trace-partition", tenant_id="tenant")
    try:
        repository = storage.execution.executions
        archive = storage.run_store.read_store(PersistenceDomain.TRACE)
        assert isinstance(archive, StateStepArchive)
        assert archive.state_store is not repository.state_store
        assert archive.state_store.storage_group is repository.state_store.storage_group
        execution = replace(_execution(), revision=1)
        await repository.create_with_history_head(execution)
        await repository.state_store.mutate(lambda transaction: repository.admit_history_producer_in_transaction(
            transaction, execution,
        ))
        run = AgentRunRecord("run")
        await archive.register_agent_run(run, execution_id="execution")
        event = StepEvent("run", "TOOL_CALL_STARTED", 1, tool_call_id="tool")
        before = await repository.get_history_head("execution", tenant_id="tenant")
        group = archive.state_store.storage_group
        mutate = group.mutate

        async def rollback(stores, callback):
            async def fail(transaction):
                await callback(transaction)
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return await mutate(stores, fail)

        with monkeypatch.context() as patch:
            patch.setattr(group, "mutate", rollback)
            with pytest.raises(AIError) as failure:
                await archive.append_event(event, execution_id="execution")
            assert failure.value.code is ErrorCode.STORAGE_CONFLICT
        assert await archive.list_events(agent_run_id="run") == []
        assert await repository.get_history_head("execution", tenant_id="tenant") == before
        await archive.append_event(event, execution_id="execution")
        assert await archive.list_events(agent_run_id="run") == [event]
        assert (await repository.get_history_head("execution", tenant_id="tenant")).revision == before.revision + 1
    finally:
        await storage.close()
