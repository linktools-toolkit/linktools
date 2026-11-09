#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tests.ai._runtime_test_helpers import native_checkpoint, seed_native_snapshot
"""Public observations advance independently from recoverable checkpoints."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart

from linktools.ai.persistence import Persistence
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceDomain, PersistenceStorage
from linktools.ai.persistence._model_interaction import StagedModelInteraction
from linktools.ai.persistence._transcript_staging import StagedTranscript
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint, AgentRunRecord

from .test_step_projection import _execution


def _storage(backend: str, path: Path) -> PersistenceStorage:
    if backend == "memory":
        return PersistenceStorage.in_memory()
    if backend == "local_sqlite":
        return PersistenceStorage.from_root(path / "state", exclusive_writer=True)
    return PersistenceStorage.sqlite(path / "state.db")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "local_sqlite", "sqlite"))
async def test_observed_transcript_does_not_extend_recoverable_checkpoint(
    tmp_path: Path, backend: str,
) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="independent-observation", tenant_id="tenant")
    state.run_store.bind_step_store(native_checkpoint().store)
    try:
        await state.execution.executions.create_execution(_execution())
        store = state.run_store
        run = AgentRunRecord("run")
        await store.register_agent_run(run)
        first = ModelRequest(parts=[UserPromptPart("recoverable")], run_id="run")
        second = ModelRequest(parts=[UserPromptPart("observed later")])
        await seed_native_snapshot(store.checkpoint_persistence.step_store, run_id="run", messages=[first])
        await store.save_checkpoint(AgentRunCheckpoint(
            "run", 1, [first], transcript_message_count_before=0,
        ))
        await store.flush_execution_projection("run", execution_id="execution")
        store.stage_transcript("run", StagedTranscript((first, second)))
        await store.flush_execution_projection("run", execution_id="execution")

        archive = store.read_store(PersistenceDomain.TRACE)
        assert [message async for message in archive.iter_messages(agent_run_id="run")] == [first, second]
        checkpoint = await archive.latest_checkpoint(agent_run_id="run")
        assert checkpoint is not None
        assert checkpoint.messages == [first]
        recovery = await store.read_store(PersistenceDomain.CHECKPOINT).latest_checkpoint(agent_run_id="run")
        assert recovery is not None
        assert recovery.messages == [first]
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "local_sqlite", "sqlite"))
async def test_overlapping_request_updates_keep_their_admitted_identities(
    tmp_path: Path, backend: str,
) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="overlapping-observation", tenant_id="tenant")
    state.run_store.bind_step_store(native_checkpoint().store)
    try:
        await state.execution.executions.create_execution(_execution())
        store = state.run_store
        await store.register_agent_run(AgentRunRecord("run"))
        now = datetime.now(timezone.utc)
        first = StagedModelInteraction(
            "run", 1, 1, "agent", None, {}, None, None, None,
            "RUNNING", None, None, None, now, None,
        )
        second = replace(first, model_request_seq=2, purpose="compaction")
        store.stage_model_interaction(first)
        store.stage_model_interaction(second)
        await store.flush_execution_projection("run", execution_id="execution")
        archive = store.read_store(PersistenceDomain.TRACE)
        assert await archive.model_interaction_count(agent_run_id="run") == 2
        for running in (second, first):
            finished = replace(
                running, status="FAILED", error_code="MODEL_API_ERROR",
                duration_ns=1, finished_at=now,
            )
            store.stage_model_interaction(finished)
            await store.flush_execution_projection("run", execution_id="execution")
            observed = await archive.list_model_interactions(
                agent_run_id="run", after_model_request_seq=running.model_request_seq - 1,
                limit=1,
            )
            assert observed[0].status == "FAILED"
            assert observed[0].model_request_seq == running.model_request_seq
        assert await archive.model_interaction_count(agent_run_id="run") == 2
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "local_sqlite", "sqlite"))
@pytest.mark.parametrize("fault", ("before_trace", "after_trace", "before_commit", "after_commit"))
async def test_trace_projection_commits_atomically_without_changing_native_recovery(
    tmp_path: Path, backend: str, fault: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.persistence._step_contracts import StepEvent

    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="atomic-checkpoint", tenant_id="tenant")
    state.run_store.bind_step_store(native_checkpoint().store)
    try:
        execution = replace(_execution(), revision=1, producer_generation=1, producer_claim_id="owner-claim")
        repository = state.execution.executions
        await repository.create_execution(execution)
        store = state.run_store
        store.bind_execution_producer("run", execution_id="execution", producer_generation=1)
        await store.register_agent_run(
            AgentRunRecord("run", agent_conversation_id="conversation"), execution_id="execution",
        )
        subscription = store.subscribe_model_interactions("conversation")
        public = store.read_store(PersistenceDomain.TRACE)
        recovery = store.read_store(PersistenceDomain.CHECKPOINT)
        message = ModelRequest(parts=[UserPromptPart("atomic message")], run_id="run")
        store.stage_transcript("run", StagedTranscript((message,)))
        await store.append_event(StepEvent("run", "MODEL_REQUEST_STARTED", 1))
        now = datetime.now(timezone.utc)
        store.stage_model_interaction(StagedModelInteraction(
            "run", 1, 1, "agent", None, {}, None, None, None,
            "RUNNING", None, None, None, now, None,
        ))
        checkpoint = AgentRunCheckpoint(
            "run", 1, [message], transcript_message_count_before=0,
        )
        await seed_native_snapshot(store.checkpoint_persistence.step_store, run_id="run", messages=[message])
        native_before = await recovery.latest_snapshot(run_id="run")
        await store.save_checkpoint(checkpoint, execution_id="execution", producer_generation=1)
        injected = False
        materialize = type(public)._sync_projection_in_transaction
        group = public.state_store.storage_group
        mutate = type(group).mutate

        async def materialize_with_fault(self, *args, **kwargs):
            nonlocal injected
            if self is public and fault == "before_trace" and not injected:
                injected = True
                raise OSError("trace participant failed before its write")
            result = await materialize(self, *args, **kwargs)
            if self is public and fault == "after_trace" and not injected:
                injected = True
                raise OSError("trace participant failed after its write")
            return result

        depth = 0

        async def mutate_with_lost_ack(self, stores, callback):
            nonlocal injected, depth
            outer = self is group and depth == 0
            depth += 1
            try:
                if outer and fault == "before_commit" and not injected:
                    async def fail_before_commit(transaction):
                        nonlocal injected
                        await callback(transaction)
                        injected = True
                        raise OSError("prepared transaction failed before commit")
                    return await mutate(self, stores, fail_before_commit)
                result = await mutate(self, stores, callback)
                if outer and fault == "after_commit" and not injected:
                    injected = True
                    raise OSError("commit acknowledgement was lost")
                return result
            finally:
                depth -= 1

        monkeypatch.setattr(type(public), "_sync_projection_in_transaction", materialize_with_fault)
        monkeypatch.setattr(type(group), "mutate", mutate_with_lost_ack)
        if fault == "after_commit":
            await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        else:
            with pytest.raises((OSError, AIError)) as raised:
                await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
            if isinstance(raised.value, AIError):
                assert raised.value.code is ErrorCode.INTERNAL_ERROR
            assert subscription.generation == 0
            assert await public.checkpoint_count(agent_run_id="run") == 0
            assert await recovery.latest_snapshot(run_id="run") == native_before
            assert await public.model_interaction_count(agent_run_id="run") == 0
            assert await public.list_events(agent_run_id="run") == []
            assert await public.transcript_repository.get_head("run") is None
            await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert injected
        assert await recovery.latest_snapshot(run_id="run") == native_before
        assert subscription.generation > 0
        assert [item async for item in public.iter_messages(agent_run_id="run")] == [message]
        assert await public.model_interaction_count(agent_run_id="run") == 1
        assert len(await public.list_events(agent_run_id="run")) == 1
        assert await public.checkpoint_count(agent_run_id="run") == 1
        saved = await public.latest_checkpoint(agent_run_id="run")
        assert saved is not None and saved.messages == [message]
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert subscription.generation > 0
        assert [item async for item in public.iter_messages(agent_run_id="run")] == [message]
        assert await public.model_interaction_count(agent_run_id="run") == 1
        assert len(await public.list_events(agent_run_id="run")) == 1
        assert await public.checkpoint_count(agent_run_id="run") == 1
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "local_sqlite", "sqlite"))
async def test_caller_cancellation_waits_for_atomic_trace_truth(
    tmp_path: Path, backend: str, monkeypatch: pytest.MonkeyPatch,
) -> None:

    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="cancel-checkpoint", tenant_id="tenant")
    state.run_store.bind_step_store(native_checkpoint().store)
    entered = asyncio.Event()
    release = asyncio.Event()
    task = None
    try:
        execution = replace(_execution(), revision=1, producer_generation=1, producer_claim_id="owner-claim")
        repository = state.execution.executions
        await repository.create_execution(execution)
        store = state.run_store
        store.bind_execution_producer("run", execution_id="execution", producer_generation=1)
        await store.register_agent_run(AgentRunRecord("run"), execution_id="execution")
        recovery = store.read_store(PersistenceDomain.CHECKPOINT)
        public = store.read_store(PersistenceDomain.TRACE)
        original = type(public)._sync_projection_in_transaction

        async def hold_before_commit(self, *args, **kwargs):
            result = await original(self, *args, **kwargs)
            if self is public:
                entered.set()
                await release.wait()
            return result

        monkeypatch.setattr(type(public), "_sync_projection_in_transaction", hold_before_commit)
        message = ModelRequest(parts=[UserPromptPart("survives caller cancellation")], run_id="run")
        checkpoint = AgentRunCheckpoint("run", 1, [message], transcript_message_count_before=0)
        await seed_native_snapshot(store.checkpoint_persistence.step_store, run_id="run", messages=[message])
        native_before = await recovery.latest_snapshot(run_id="run")
        await store.save_checkpoint(checkpoint, execution_id="execution", producer_generation=1)
        task = asyncio.create_task(store.flush_execution_projection(
            "run", execution_id="execution", producer_generation=1,
        ))
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await recovery.latest_snapshot(run_id="run") == native_before
        assert await public.checkpoint_count(agent_run_id="run") == 1
        assert [item async for item in public.iter_messages(agent_run_id="run")] == [message]
        await store.preflight_close()
        saved = await public.latest_checkpoint(agent_run_id="run")
        assert saved is not None and saved.messages == [message]
    finally:
        release.set()
        if task is not None and not task.done():
            await asyncio.gather(task, return_exceptions=True)
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "local_sqlite", "sqlite"))
@pytest.mark.parametrize("context_published", (False, True))
async def test_native_context_recovery_does_not_invent_unpublished_trace_history(
    tmp_path: Path, backend: str, context_published: bool,
) -> None:
    from pydantic_ai.messages import ModelResponse, TextPart

    from linktools.ai.core import HmacCursorSigner, agent_conversation_id, agent_run_id
    from linktools.ai.history._projection import StepExecutionHistoryReader
    from linktools.ai.persistence._step_archive import StagingAgentRunStore
    from linktools.ai.persistence._step_contracts import StepEvent
    from linktools.ai.persistence.runs import RuntimeAgentRunStore

    namespace = "deferred-context-restore"
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace=namespace, tenant_id="tenant")
    state.run_store.bind_step_store(native_checkpoint().store)
    run_id = agent_run_id(namespace=namespace, tenant_id="tenant", execution_id="execution", agent_run_seq=1)
    conversation_id = agent_conversation_id(namespace=namespace, tenant_id="tenant", execution_id="execution")
    try:
        repository = state.execution.executions
        execution = replace(_execution(), revision=1, producer_generation=1, producer_claim_id="owner-claim")
        await repository.create_execution(execution)
        restored = state.run_store
        public = restored.read_store(PersistenceDomain.TRACE)
        old = RuntimeAgentRunStore(
            StagingAgentRunStore(), execution_repository=repository, tenant_id="tenant",
            conversation_archive=restored.read_store(PersistenceDomain.CONVERSATION),
            execution_archive=public,
        )
        old.bind_step_store(native_checkpoint().store)
        await old.initialize()
        old.bind_execution_producer(run_id, execution_id="execution", producer_generation=1)
        run = AgentRunRecord(run_id, agent_conversation_id=conversation_id, metadata={"agent_run_seq": "1"})
        await old.register_agent_run(run, execution_id="execution")
        messages = [ModelRequest(parts=[UserPromptPart("raw question")]), ModelResponse(parts=[TextPart("raw answer")])]
        context = [ModelRequest(parts=[UserPromptPart("compacted context")], run_id=run_id)]
        checkpoint = AgentRunCheckpoint(
            run_id, 1, messages, context_messages=context, transcript_message_count_before=0,
        )
        await old.append_event(StepEvent(run_id, "AGENT_RUN_STARTED", 1, agent_conversation_id=conversation_id))
        await seed_native_snapshot(
            old.checkpoint_persistence.step_store, run_id=run_id, messages=context,
            conversation_id=conversation_id, metadata={"agent_run_seq": "1"},
        )
        await old.save_checkpoint(checkpoint, execution_id="execution", producer_generation=1)
        assert await public.latest_checkpoint(agent_run_id=run_id) is None
        reader = StepExecutionHistoryReader(
            namespace=namespace, executions=repository, store=public,
            cursor_signer=HmacCursorSigner(namespace, b"deferred-context"),
        )
        page = await reader.history("execution", tenant_id="tenant", cursor=None, limit=10)
        assert page.items == ()
        if context_published:
            await old.flush_execution_projection(run_id, execution_id="execution", producer_generation=1)
            page = await reader.history("execution", tenant_id="tenant", cursor=None, limit=10)
            assert [item.content for item in page.items] == ["raw question", "raw answer"]
        await repository.compare_and_swap(
            "execution", tenant_id="tenant", expected_revision=1, next_record=replace(execution, revision=2, producer_generation=2, producer_claim_id="replacement-claim"),
        )
        if not context_published:
            with pytest.raises(AIError) as stale:
                await old.flush_execution_projection(run_id, execution_id="execution", producer_generation=1)
            assert stale.value.code is ErrorCode.STORAGE_CONFLICT
            assert await public.latest_checkpoint(agent_run_id=run_id) is None
        restored.bind_execution_producer(run_id, execution_id="execution", producer_generation=2)
        await restored.register_agent_run(run, execution_id="execution")
        recovered = await restored.latest_checkpoint(agent_run_id=run_id)
        assert recovered is not None and recovered.context_messages == context
        loaded = await restored.load_loaded_model_context(PersistenceDomain.CHECKPOINT, run_id)
        assert loaded.model_messages() == tuple(context)
        await restored.preflight_close()
        await restored.flush_execution_projection(run_id, execution_id="execution", producer_generation=2)
        page = await reader.history("execution", tenant_id="tenant", cursor=None, limit=10)
        if context_published:
            assert await public.checkpoint_count(agent_run_id=run_id) == 1
            projected = await public.latest_checkpoint(agent_run_id=run_id)
            assert projected is not None and projected.messages == messages and projected.context_messages == context
            assert [item.content for item in page.items] == ["raw question", "raw answer"]
        else:
            assert await public.checkpoint_count(agent_run_id=run_id) == 0
            assert page.items == ()
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "local_sqlite", "sqlite", "transient"))
async def test_model_subscription_ignores_non_model_projection_flushes(
    tmp_path: Path, backend: str,
) -> None:
    from linktools.ai.persistence import PersistencePlan
    from linktools.ai.persistence._step_contracts import StepEvent
    from linktools.ai.storage import StorageRoute

    storage = (PersistenceStorage(PersistencePlan(execution=StorageRoute.transient()))
               if backend == "transient" else _storage(backend, tmp_path))
    state = Persistence(storage)
    await state.initialize(namespace="model-notifications", tenant_id="tenant")
    try:
        await state.execution.executions.create_execution(_execution())
        store = state.run_store
        await store.register_agent_run(AgentRunRecord("run", agent_conversation_id="conversation"))
        subscription = store.subscribe_model_interactions("conversation")
        try:
            now = datetime.now(timezone.utc)
            running = StagedModelInteraction(
                "run", 1, 1, "agent", None, {}, None, None, None,
                "RUNNING", None, None, None, now, None,
            )
            store.stage_model_interaction(running)
            await store.flush_execution_projection("run", execution_id="execution")
            generation = await asyncio.wait_for(subscription.wait(0), 1)
            await store.append_event(StepEvent("run", "TOOL_CALL_SUCCEEDED", 1))
            store.stage_transcript("run", StagedTranscript((ModelRequest(parts=[UserPromptPart("tool result")]),)))
            await store.flush_execution_projection("run", execution_id="execution")
            await store.flush_execution_projection("run", execution_id="execution")
            assert subscription.generation == generation
            finished = replace(running, status="FAILED", error_code="MODEL_API_ERROR", duration_ns=1, finished_at=now)
            store.stage_model_interaction(finished)
            await store.flush_execution_projection("run", execution_id="execution")
            assert await asyncio.wait_for(subscription.wait(generation), 1) > generation
            reader = store if backend == "transient" else store.read_store(PersistenceDomain.TRACE)
            values = await reader.list_model_interactions(agent_run_id="run")
            assert [(value.model_request_seq, value.status) for value in values] == [(1, "FAILED")]
            assert len(await reader.list_events(agent_run_id="run")) == 1
        finally:
            await subscription.close()
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "local_sqlite", "sqlite"))
async def test_model_subscription_retains_changes_staged_after_projection_capture(
    tmp_path: Path, backend: str,
) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="model-capture-notifications", tenant_id="tenant")
    try:
        await state.execution.executions.create_execution(_execution())
        store = state.run_store
        await store.register_agent_run(AgentRunRecord("run", agent_conversation_id="conversation"))
        subscription = store.subscribe_model_interactions("conversation")
        try:
            now = datetime.now(timezone.utc)
            running = StagedModelInteraction(
                "run", 1, 1, "agent", None, {}, None, None, None,
                "RUNNING", None, None, None, now, None,
            )
            store.stage_model_interaction(running)
            captured, flight = await store.capture_execution_projection("run")
            store.stage_model_interaction(replace(
                running, status="FAILED", error_code="MODEL_API_ERROR", duration_ns=1, finished_at=now,
            ))
            await store.commit_captured_execution_projection(captured, flight, execution_id="execution")
            generation = await asyncio.wait_for(subscription.wait(0), 1)
            archive = store.read_store(PersistenceDomain.TRACE)
            assert (await archive.list_model_interactions(agent_run_id="run"))[0].status == "RUNNING"
            await store.flush_execution_projection("run", execution_id="execution")
            assert await asyncio.wait_for(subscription.wait(generation), 1) > generation
            assert (await archive.list_model_interactions(agent_run_id="run"))[0].status == "FAILED"
        finally:
            await subscription.close()
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "local_sqlite", "sqlite"))
async def test_model_subscription_announces_first_empty_run_publication(
    tmp_path: Path, backend: str,
) -> None:
    from linktools.ai.core import HmacCursorSigner, agent_conversation_id, agent_run_id
    from linktools.ai.execution import UsageReadCutoff
    from linktools.ai.history._projection import StepExecutionHistoryReader
    from linktools.ai.persistence._step_contracts import StepEvent

    namespace = "empty-run-notification"
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace=namespace, tenant_id="tenant")
    try:
        await state.execution.executions.create_execution(_execution())
        store = state.run_store
        conversation = agent_conversation_id(namespace=namespace, tenant_id="tenant", execution_id="execution")
        run = agent_run_id(namespace=namespace, tenant_id="tenant", execution_id="execution", agent_run_seq=1)
        subscription = store.subscribe_model_interactions(conversation)
        try:
            await store.register_agent_run(AgentRunRecord(
                run, agent_conversation_id=conversation, metadata={"agent_run_seq": "1"},
            ))
            generation = await asyncio.wait_for(subscription.wait(0), 1)
            reader = StepExecutionHistoryReader(
                namespace=namespace, executions=state.execution.executions,
                store=store.read_store(PersistenceDomain.TRACE),
                cursor_signer=HmacCursorSigner(namespace, b"model-notification-test"),
            )
            before = await reader.capture_model_interaction_cutoffs("execution", tenant_id="tenant")
            assert not before.durable_history_available and before.cutoffs == ()
            await store.append_event(StepEvent(run, "AGENT_RUN_STARTED", 1))
            await store.flush_execution_projection(run, execution_id="execution")
            generation = await asyncio.wait_for(subscription.wait(generation), 1)
            after = await reader.capture_model_interaction_cutoffs("execution", tenant_id="tenant")
            assert after.durable_history_available
            assert after.cutoffs == (UsageReadCutoff("execution", 1, 0),)
            await store.append_event(StepEvent(run, "TOOL_CALL_SUCCEEDED", 2))
            await store.flush_execution_projection(run, execution_id="execution")
            assert subscription.generation == generation
            assert await reader.capture_model_interaction_cutoffs("execution", tenant_id="tenant") == after
        finally:
            await subscription.close()
    finally:
        await state.close()
