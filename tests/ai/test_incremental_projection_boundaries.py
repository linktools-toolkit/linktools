#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public observations advance independently from recoverable checkpoints."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeDomain, RuntimeStorage
from linktools.ai.runtime._model_interaction import StagedModelInteraction
from linktools.ai.runtime._transcript_staging import StagedTranscript
from linktools.ai.runtime.state._step_contracts import AgentRunCheckpoint, AgentRunRecord

from .test_step_projection import _execution


def _storage(backend: str, path: Path) -> RuntimeStorage:
    if backend == "memory":
        return RuntimeStorage.in_memory()
    if backend == "filesystem":
        return RuntimeStorage.filesystem(path / "state")
    return RuntimeStorage.sqlite(path / "state.db")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_observed_transcript_does_not_extend_recoverable_checkpoint(
    tmp_path: Path, backend: str,
) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace="independent-observation", tenant_id="tenant")
    try:
        await state.execution.executions.create_with_history_head(_execution())
        store = state.run_store
        run = AgentRunRecord("run")
        await store.register_agent_run(run)
        first = ModelRequest(parts=[UserPromptPart("recoverable")])
        second = ModelRequest(parts=[UserPromptPart("observed later")])
        await store.save_checkpoint(AgentRunCheckpoint(
            "run", 1, [first], transcript_message_count_before=0,
        ))
        await store.flush_execution_projection("run", execution_id="execution")
        store.stage_transcript("run", StagedTranscript((first, second)))
        await store.flush_execution_projection("run", execution_id="execution")

        archive = store.read_store(RuntimeDomain.EXECUTION)
        assert [message async for message in archive.iter_messages(agent_run_id="run")] == [first, second]
        checkpoint = await archive.latest_checkpoint(agent_run_id="run")
        assert checkpoint is not None
        assert checkpoint.messages == [first]
        recovery = await store.read_store(RuntimeDomain.RECOVERY).latest_checkpoint(agent_run_id="run")
        assert recovery is not None
        assert recovery.messages == [first]
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_overlapping_request_updates_keep_their_admitted_identities(
    tmp_path: Path, backend: str,
) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace="overlapping-observation", tenant_id="tenant")
    try:
        await state.execution.executions.create_with_history_head(_execution())
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
        archive = store.read_store(RuntimeDomain.EXECUTION)
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
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("fault", ("before_recovery", "after_recovery", "before_commit", "after_commit"))
async def test_recovery_checkpoint_commits_guarded_truth_before_public_projection(
    tmp_path: Path, backend: str, fault: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.runtime.state._step_archive import StateStepArchive
    from linktools.ai.runtime.state._step_contracts import StepEvent

    state = _storage(backend, tmp_path)
    await state.initialize(namespace="atomic-checkpoint", tenant_id="tenant")
    try:
        execution = replace(_execution(), revision=1)
        repository = state.execution.executions
        await repository.create_with_history_head(execution)
        await repository.state_store.mutate(
            lambda transaction: repository.admit_history_producer_in_transaction(transaction, execution)
        )
        store = state.run_store
        store.bind_execution_producer("run", execution_id="execution", producer_generation=1)
        await store.register_agent_run(
            AgentRunRecord("run", agent_conversation_id="conversation"), execution_id="execution",
        )
        subscription = store.subscribe_model_interactions("conversation")
        public = store.read_store(RuntimeDomain.EXECUTION)
        recovery = store.read_store(RuntimeDomain.RECOVERY)
        assert public.state_store.storage_group is recovery.state_store.storage_group
        message = ModelRequest(parts=[UserPromptPart("atomic message")])
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
        injected = False
        materialize = StateStepArchive.materialize_checkpoint_in_transaction
        group = public.state_store.storage_group
        mutate = type(group).mutate

        async def materialize_with_fault(self, *args, **kwargs):
            nonlocal injected
            if self is recovery and fault == "before_recovery" and not injected:
                injected = True
                raise OSError("recovery participant failed before its write")
            await materialize(self, *args, **kwargs)
            if self is recovery and fault == "after_recovery" and not injected:
                injected = True
                raise OSError("recovery participant failed after its write")

        async def mutate_with_lost_ack(self, stores, callback):
            nonlocal injected
            if self is group and fault == "before_commit" and not injected:
                async def fail_before_commit(transaction):
                    nonlocal injected
                    await callback(transaction)
                    injected = True
                    raise OSError("prepared transaction failed before commit")
                return await mutate(self, stores, fail_before_commit)
            result = await mutate(self, stores, callback)
            if self is group and fault == "after_commit" and not injected:
                injected = True
                raise OSError("commit acknowledgement was lost")
            return result

        monkeypatch.setattr(StateStepArchive, "materialize_checkpoint_in_transaction", materialize_with_fault)
        monkeypatch.setattr(type(group), "mutate", mutate_with_lost_ack)
        if fault == "after_commit":
            await store.save_checkpoint(checkpoint, execution_id="execution", producer_generation=1)
        else:
            with pytest.raises((OSError, AIError)) as raised:
                await store.save_checkpoint(checkpoint, execution_id="execution", producer_generation=1)
            if isinstance(raised.value, AIError):
                assert raised.value.code is ErrorCode.INTERNAL_ERROR
            assert subscription.generation == 0
            assert await public.checkpoint_count(agent_run_id="run") == 0
            assert await recovery.checkpoint_count(agent_run_id="run") == 0
            assert await public.model_interaction_count(agent_run_id="run") == 0
            assert await public.list_events(agent_run_id="run") == []
            assert await public.transcript_repository.get_head("run") is None
            assert await recovery.transcript_repository.get_head("run") is None
            await store.save_checkpoint(checkpoint, execution_id="execution", producer_generation=1)
        assert injected
        assert subscription.generation == 0
        assert await public.checkpoint_count(agent_run_id="run") == 0
        assert await public.latest_checkpoint(agent_run_id="run") is None
        saved = await recovery.latest_checkpoint(agent_run_id="run")
        assert saved is not None and saved.messages == [message]
        assert [item async for item in public.iter_messages(agent_run_id="run")] == []
        assert await public.model_interaction_count(agent_run_id="run") == 0
        assert await public.list_events(agent_run_id="run") == []
        await store.save_checkpoint(checkpoint, execution_id="execution", producer_generation=1)
        assert await public.checkpoint_count(agent_run_id="run") == 0
        assert await recovery.checkpoint_count(agent_run_id="run") == 1
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
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
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_caller_cancellation_waits_for_atomic_checkpoint_truth(
    tmp_path: Path, backend: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.runtime.state._step_archive import StateStepArchive

    state = _storage(backend, tmp_path)
    await state.initialize(namespace="cancel-checkpoint", tenant_id="tenant")
    entered = asyncio.Event()
    release = asyncio.Event()
    task = None
    try:
        execution = replace(_execution(), revision=1)
        repository = state.execution.executions
        await repository.create_with_history_head(execution)
        await repository.state_store.mutate(
            lambda transaction: repository.admit_history_producer_in_transaction(transaction, execution)
        )
        store = state.run_store
        store.bind_execution_producer("run", execution_id="execution", producer_generation=1)
        await store.register_agent_run(AgentRunRecord("run"), execution_id="execution")
        recovery = store.read_store(RuntimeDomain.RECOVERY)
        original = StateStepArchive.materialize_checkpoint_in_transaction

        async def hold_before_commit(self, *args, **kwargs):
            await original(self, *args, **kwargs)
            if self is recovery:
                entered.set()
                await release.wait()

        monkeypatch.setattr(StateStepArchive, "materialize_checkpoint_in_transaction", hold_before_commit)
        message = ModelRequest(parts=[UserPromptPart("survives caller cancellation")])
        checkpoint = AgentRunCheckpoint("run", 1, [message], transcript_message_count_before=0)
        task = asyncio.create_task(store.save_checkpoint(
            checkpoint, execution_id="execution", producer_generation=1,
        ))
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        saved = await recovery.latest_checkpoint(agent_run_id="run")
        assert saved is not None and saved.messages == [message]
        public = store.read_store(RuntimeDomain.EXECUTION)
        assert await public.checkpoint_count(agent_run_id="run") == 0
        assert [item async for item in public.iter_messages(agent_run_id="run")] == []
        await store.preflight_close()
        saved = await public.latest_checkpoint(agent_run_id="run")
        assert saved is not None and saved.messages == [message]
    finally:
        release.set()
        if task is not None and not task.done():
            await asyncio.gather(task, return_exceptions=True)
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("context_published", (False, True))
async def test_recovery_restores_context_without_assuming_public_checkpoint_acknowledgment(
    tmp_path: Path, backend: str, context_published: bool,
) -> None:
    from pydantic_ai.messages import ModelResponse, TextPart

    from linktools.ai.core import HmacCursorSigner, agent_conversation_id, agent_run_id
    from linktools.ai.runtime._history_projection import StepExecutionHistoryReader
    from linktools.ai.runtime.state._plan import RuntimeRetentionMode
    from linktools.ai.runtime.state._step_archive import StagingAgentRunStore
    from linktools.ai.runtime.state._step_contracts import StepEvent
    from linktools.ai.runtime.state._steps import RuntimeAgentRunStore

    namespace = "deferred-context-restore"
    state = _storage(backend, tmp_path)
    await state.initialize(namespace=namespace, tenant_id="tenant")
    run_id = agent_run_id(namespace=namespace, tenant_id="tenant", execution_id="execution", agent_run_seq=1)
    conversation_id = agent_conversation_id(namespace=namespace, tenant_id="tenant", execution_id="execution")
    try:
        repository = state.execution.executions
        execution = replace(_execution(), revision=1)
        await repository.create_with_history_head(execution)
        await repository.state_store.mutate(
            lambda transaction: repository.admit_history_producer_in_transaction(transaction, execution)
        )
        restored = state.run_store
        public = restored.read_store(RuntimeDomain.EXECUTION)
        old = RuntimeAgentRunStore(
            StagingAgentRunStore(), conversation_archive=restored.read_store(RuntimeDomain.CONVERSATION),
            execution_archive=public, recovery_archive=restored.read_store(RuntimeDomain.RECOVERY),
            conversation_retention=RuntimeRetentionMode.VOLATILE,
            execution_retention=RuntimeRetentionMode.VOLATILE,
            recovery_retention=RuntimeRetentionMode.VOLATILE,
        )
        await old.initialize()
        old.bind_execution_producer(run_id, execution_id="execution", producer_generation=1)
        run = AgentRunRecord(run_id, agent_conversation_id=conversation_id, metadata={"agent_run_seq": "1"})
        await old.register_agent_run(run, execution_id="execution")
        messages = [ModelRequest(parts=[UserPromptPart("raw question")]), ModelResponse(parts=[TextPart("raw answer")])]
        context = [ModelRequest(parts=[UserPromptPart("compacted context")])]
        checkpoint = AgentRunCheckpoint(
            run_id, 1, messages, context_messages=context, transcript_message_count_before=0,
        )
        await old.append_event(StepEvent(run_id, "AGENT_RUN_STARTED", 1, agent_conversation_id=conversation_id))
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
        updated = await repository.compare_and_swap(
            "execution", tenant_id="tenant", expected_revision=1, next_record=replace(execution, revision=2),
        )
        await repository.state_store.mutate(
            lambda transaction: repository.admit_history_producer_in_transaction(transaction, updated)
        )
        if not context_published:
            with pytest.raises(AIError) as stale:
                await old.preflight_close()
            assert stale.value.code is ErrorCode.STORAGE_CONFLICT
            assert await public.latest_checkpoint(agent_run_id=run_id) is None
        restored.bind_execution_producer(run_id, execution_id="execution", producer_generation=2)
        await restored.register_agent_run(run, execution_id="execution")
        recovered = await restored.latest_checkpoint(agent_run_id=run_id)
        assert recovered is not None and recovered.messages == messages and recovered.context_messages == context
        await restored.preflight_close()
        assert await public.checkpoint_count(agent_run_id=run_id) == 1
        projected = await public.latest_checkpoint(agent_run_id=run_id)
        assert projected is not None and projected.messages == messages and projected.context_messages == context
        await restored.flush_execution_projection(run_id, execution_id="execution", producer_generation=2)
        assert await public.checkpoint_count(agent_run_id=run_id) == 1
    finally:
        await state.close()
