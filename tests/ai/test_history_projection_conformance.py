#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution history projection for claimed and materialized attempts."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest
from linktools.ai.adapter import StepExecutionHistoryReader
from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    HmacCursorSigner,
    step_conversation_id,
    step_run_id,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime.state import (
    ExecutionHistorySealRecord,
    ExecutionReadModelBuild,
    ExecutionReadModelRepository,
    ExecutionRecord,
    ExecutionRunSealHead,
    ExecutionTerminalSealPlan,
    RuntimeDomain,
    StateLockOrderError,
    StateStepArchive,
    StateTransactionNestingError,
    StoredRecord,
)
from linktools.ai.runtime.state._history import _TranscriptAccumulator
from linktools.ai.runtime.state._steps import LockOrderError, _RunHistoryLock
from linktools.ai.runtime.state._store import (
    partition_digest,
    record_key_digest,
    sortable_identity,
)
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    RunRecord,
    StepEvent,
)


class _LockedProjectionCheckpoint:
    def __init__(self, history_lock: _RunHistoryLock) -> None:
        self._history_lock = history_lock
        self.acknowledged = False

    async def acknowledge(self) -> None:
        async with self._history_lock.hold("run"):
            self.acknowledged = True


def _record(status: ExecutionStatus, sequence: int) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    return ExecutionRecord(
        execution_id="execution",
        tenant_id="tenant",
        session_id=None,
        binding_digest="binding",
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=status,
        revision=0,
        event_sequence=0,
        agent_run_sequence=sequence,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_terminal_projection_acknowledgement_reenters_current_history_lock() -> None:
    history_lock = _RunHistoryLock()
    checkpoint = _LockedProjectionCheckpoint(history_lock)
    backend = object.__new__(LocalExecutionBackend)

    async with history_lock.hold("run"):
        await asyncio.wait_for(
            backend._acknowledge_projection_after_commit(checkpoint),
            timeout=0.1,
        )

    assert checkpoint.acknowledged


@pytest.mark.asyncio
async def test_terminal_commit_cancellation_still_finalizes_after_durable_commit() -> None:
    class Lifecycle:
        def __init__(self) -> None:
            self.finalized = asyncio.Event()

        async def finalize_execution_terminal_seal(self, plan: object) -> None:
            del plan
            self.finalized.set()

    lifecycle = Lifecycle()
    backend = object.__new__(LocalExecutionBackend)
    backend._step_reads = {
        RuntimeDomain.EXECUTION: object.__new__(StateStepArchive),
    }
    backend._step_lifecycle = lifecycle
    started = asyncio.Event()

    async def commit(*args: object, **kwargs: object) -> object:
        del args, kwargs
        started.set()
        await asyncio.sleep(0.01)
        return object()

    backend._commit_execution_terminal_checkpoint_locked = commit
    current = _record(ExecutionStatus.SUCCEEDED, 1)
    plan = ExecutionTerminalSealPlan("execution", "binding", (), ())
    task = asyncio.create_task(
        backend._commit_execution_terminal_checkpoint(
            current,
            object(),
            run_id=None,
            terminal_plan=plan,
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert lifecycle.finalized.is_set()


@pytest.mark.asyncio
async def test_run_history_lock_rejects_cross_run_nesting() -> None:
    history_lock = _RunHistoryLock()

    async with history_lock.hold("run-a"):
        with pytest.raises(LockOrderError):
            async with history_lock.hold("run-b"):
                pass


@pytest.mark.asyncio
async def test_state_callback_cannot_acquire_run_history_lock() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="lock-order", tenant_id="tenant")
    history_lock = _RunHistoryLock()
    try:
        async def callback(_transaction: object) -> None:
            async with history_lock.hold("run"):
                pass

        with pytest.raises(StateLockOrderError):
            await state.execution.executions.state_store.mutate(callback)
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_read_only_state_callback_rejects_mutation() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="read-only", tenant_id="tenant")
    try:
        async def callback(transaction: object) -> None:
            await transaction.next_sequence(b"x" * 32)

        with pytest.raises(StateTransactionNestingError):
            await state.execution.executions.state_store.read(callback)
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_read_only_state_callback_rejects_record_guard() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="read-only-guard", tenant_id="tenant")
    try:
        async def callback(transaction: object) -> None:
            await transaction.guard_record(
                b"x" * 32,
                expected_storage_version=0,
            )

        with pytest.raises(StateTransactionNestingError):
            await state.execution.executions.state_store.read(callback)
    finally:
        await state.close()


def test_transcript_accumulator_plans_then_applies_idempotently() -> None:
    accumulator = _TranscriptAccumulator("run")
    message = ModelRequest(parts=[UserPromptPart(content="hello")])

    advance = accumulator.plan((message,))

    assert accumulator.messages == ()
    accumulator.apply(advance)
    accumulator.apply(advance)
    assert accumulator.messages == (message,)


async def _materialize_attempt(state: RuntimeState, sequence: int, prompt: str) -> None:
    run_id = step_run_id(
        namespace="history",
        tenant_id="tenant",
        execution_id="execution",
        segment_sequence=sequence,
    )
    conversation_id = step_conversation_id(
        namespace="history",
        tenant_id="tenant",
        execution_id="execution",
    )
    now = datetime.now(timezone.utc)
    await state.steps.register_run(
        RunRecord(
            run_id=run_id,
            conversation_id=conversation_id,
            parent_run_id=None,
            agent_name="default",
            metadata={
                "segment_sequence": str(sequence),
                "agent_name": "default",
            },
            started_at=now,
        )
    )
    await state.steps.append_event(
        StepEvent(
            run_id=run_id,
            kind="model_request_started",
            step_index=1,
            timestamp=now,
            conversation_id=conversation_id,
            agent_name="default",
        )
    )
    await state.steps.append_event(
        StepEvent(
            run_id=run_id,
            kind="model_request_completed",
            step_index=2,
            timestamp=now,
            conversation_id=conversation_id,
            agent_name="default",
        )
    )
    await state.steps.save_snapshot(
        ContinuableSnapshot(
            run_id=run_id,
            step_index=2,
            messages=[
                ModelRequest(
                    parts=[UserPromptPart(content=prompt)],
                    conversation_id=conversation_id,
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(content="plan"),
                        TextPart(content="response"),
                    ],
                    conversation_id=conversation_id,
                ),
            ],
            conversation_id=conversation_id,
            parent_run_id=None,
            agent_name="default",
            timestamp=now,
        )
    )
    await state.steps.flush_execution_projection(run_id)
    seal = ExecutionHistorySealRecord(
        "execution",
        "tenant",
        1,
        (
            ExecutionRunSealHead(
                run_id,
                2,
                1,
                2,
                "projection",
            ),
        ),
        0,
        f"seal-{sequence}",
    )
    await state.execution.executions.state_store.mutate(
        lambda transaction: state.execution.executions.put_history_seal_in_transaction(
            transaction,
            seal,
        )
    )


def _reader(state: RuntimeState) -> StepExecutionHistoryReader:
    return StepExecutionHistoryReader(
        namespace="history",
        executions=state.execution.executions,
        store=state.steps.read_store(RuntimeDomain.EXECUTION),
        cursor_signer=HmacCursorSigner("history", b"history-key"),
    )


@pytest.mark.asyncio
async def test_terminal_reader_pages_from_execution_read_model() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="history", tenant_id="tenant")
    try:
        await state.execution.executions.create(_record(ExecutionStatus.SUCCEEDED, 1))
        await _materialize_attempt(state, 1, "read-model")
        await state.retention.release_execution_handoff("execution", tenant_id="tenant")
        read_model = ExecutionReadModelRepository(
            state.execution.executions.state_store,
            namespace="history",
            tenant_id="tenant",
        )
        reader = StepExecutionHistoryReader(
            namespace="history",
            executions=state.execution.executions,
            store=state.steps.read_store(RuntimeDomain.EXECUTION),
            cursor_signer=HmacCursorSigner("history", b"history-key"),
            read_model=read_model,
        )

        trace = await reader.trace("execution", tenant_id="tenant", cursor=None, limit=1)
        history = await reader.history("execution", tenant_id="tenant", cursor=None, limit=1)
        transcript = await reader.transcript(
            "execution",
            tenant_id="tenant",
            cursor=None,
            limit=1,
        )

        assert trace.next_cursor == "1"
        assert history.next_cursor is not None
        assert transcript.next_cursor == "1"
        model = await read_model.get_complete("execution", tenant_id="tenant")
        assert model is not None
        assert model.trace_count == 2
        assert model.history_count == 3
        assert model.transcript_count == 2
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_terminal_seal_reuses_durable_projection_after_staging_release(
    tmp_path: Path,
) -> None:
    state = RuntimeState.filesystem(tmp_path / "runtime")
    await state.initialize(namespace="history", tenant_id="tenant")
    try:
        await state.execution.executions.create(_record(ExecutionStatus.SUCCEEDED, 1))
        await _materialize_attempt(state, 1, "durable-only")
        await state.retention.release_execution_handoff("execution", tenant_id="tenant")
        run_id = step_run_id(
            namespace="history",
            tenant_id="tenant",
            execution_id="execution",
            segment_sequence=1,
        )
        terminal_plan = await state.steps.prepare_execution_terminal_seal(
            execution_id="execution",
            run_ids=(run_id,),
            binding_digest="binding",
        )
        assert terminal_plan.projections[0].projection_digest != "empty"
        archive = state.steps.read_store(RuntimeDomain.EXECUTION)
        assert isinstance(archive, StateStepArchive)
        head = await archive.execution_history_head(run_id)
        projection = terminal_plan.projections[0]
        assert head == (
            projection.target_event_offset,
            projection.target_snapshot_offset,
            projection.target_transcript_message_count,
            projection.projection_digest,
        ), (head, projection)
        assert await archive.verify_execution_projection_head(
            terminal_plan.projections[0]
        )
        await state.steps.finalize_execution_terminal_seal(terminal_plan)
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_read_model_rejects_a_different_complete_source() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="read-model-digest", tenant_id="tenant")
    try:
        await state.execution.executions.create(_record(ExecutionStatus.FAILED, 0))
        seal = ExecutionHistorySealRecord(
            "execution",
            "tenant",
            1,
            (),
            1,
            "seal",
        )
        await state.execution.executions.state_store.mutate(
            lambda transaction: state.execution.executions.put_history_seal_in_transaction(
                transaction,
                seal,
            )
        )
        repository_a = ExecutionReadModelRepository(
            state.execution.executions.state_store,
            namespace="read-model-digest",
            tenant_id="tenant",
        )
        repository_b = ExecutionReadModelRepository(
            state.execution.executions.state_store,
            namespace="read-model-digest",
            tenant_id="tenant",
        )
        builders_ready = asyncio.Event()
        builder_count = 0

        async def build(source_digest: str) -> ExecutionReadModelBuild:
            nonlocal builder_count
            builder_count += 1
            if builder_count == 2:
                builders_ready.set()
            await builders_ready.wait()
            return ExecutionReadModelBuild(
                "execution",
                "tenant",
                source_digest,
                (),
                (),
                (),
            )

        results = await asyncio.gather(
            repository_a.ensure(
                "execution",
                tenant_id="tenant",
                builder=lambda: build("source-a"),
            ),
            repository_b.ensure(
                "execution",
                tenant_id="tenant",
                builder=lambda: build("source-b"),
            ),
            return_exceptions=True,
        )
        assert sum(isinstance(value, AIError) for value in results) == 1
        error = next(value for value in results if isinstance(value, AIError))
        assert error.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_read_model_rebuilds_a_v1_record() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="read-model-v1", tenant_id="tenant")
    try:
        await state.execution.executions.create(_record(ExecutionStatus.FAILED, 0))
        seal = ExecutionHistorySealRecord(
            "execution",
            "tenant",
            1,
            (),
            1,
            "seal",
        )
        store = state.execution.executions.state_store
        await store.mutate(
            lambda transaction: state.execution.executions.put_history_seal_in_transaction(
                transaction,
                seal,
            )
        )
        key = record_key_digest(
            "read-model-v1",
            "tenant",
            "execution",
            "execution_read_model",
            "execution",
        )
        await store.mutate(
            lambda transaction: transaction.insert_record(
                StoredRecord(
                    key,
                    partition_digest(
                        "read-model-v1",
                        "tenant",
                        "execution",
                        "execution_read_model",
                    ),
                    None,
                    None,
                    "execution_read_model",
                    sortable_identity("execution"),
                    "COMPLETE",
                    0,
                    None,
                    0,
                    None,
                    {
                        "execution_id": "execution",
                        "tenant_id": "tenant",
                        "source_digest": "legacy-source",
                        "model_version": 1,
                        "status": "COMPLETE",
                        "trace_count": 0,
                        "history_count": 0,
                        "transcript_count": 0,
                        "revision": 1,
                    },
                )
            )
        )
        repository = ExecutionReadModelRepository(
            store,
            namespace="read-model-v1",
            tenant_id="tenant",
        )

        assert await repository.get_complete("execution", tenant_id="tenant") is None

        async def build() -> ExecutionReadModelBuild:
            return ExecutionReadModelBuild(
                "execution",
                "tenant",
                "v2-source",
                (),
                (),
                (),
            )

        result = await repository.ensure(
            "execution",
            tenant_id="tenant",
            builder=build,
        )
        assert result.model_version == 2
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_failed_claimed_attempt_without_run_is_skipped() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="history", tenant_id="tenant")
    try:
        await state.execution.executions.create(_record(ExecutionStatus.FAILED, 1))
        reader = _reader(state)

        history = await reader.history("execution", tenant_id="tenant", cursor=None, limit=200)
        trace = await reader.trace("execution", tenant_id="tenant", cursor=None, limit=200)
        transcript = await reader.transcript("execution", tenant_id="tenant", cursor=None, limit=200)

        assert history.items == ()
        assert trace.items == ()
        assert transcript.items == ()
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_history_skips_missing_non_final_attempt() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="history", tenant_id="tenant")
    try:
        await state.execution.executions.create(_record(ExecutionStatus.FAILED, 2))
        await _materialize_attempt(state, 2, "attempt-2")
        await state.retention.release_execution_handoff("execution", tenant_id="tenant")
        reader = _reader(state)

        history = await reader.history("execution", tenant_id="tenant", cursor=None, limit=200)
        trace = await reader.trace("execution", tenant_id="tenant", cursor=None, limit=200)
        transcript = await reader.transcript("execution", tenant_id="tenant", cursor=None, limit=200)

        assert [item.content for item in history.items] == ["attempt-2", "plan", "response"]
        assert [item.payload["segment_sequence"] for item in trace.items] == [2, 2]
        assert [item.text for item in transcript.items] == ["attempt-2", "response"]
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ("history", "trace", "transcript"))
async def test_successful_execution_requires_final_history_evidence(method_name: str) -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="history", tenant_id="tenant")
    try:
        await state.execution.executions.create(_record(ExecutionStatus.SUCCEEDED, 1))
        reader = _reader(state)
        method = getattr(reader, method_name)

        with pytest.raises(AIError) as error:
            await method("execution", tenant_id="tenant", cursor=None, limit=200)

        assert error.value.code is ErrorCode.EXECUTION_HISTORY_UNAVAILABLE
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_successful_history_preserves_user_prompt_and_projects_all_views() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="history", tenant_id="tenant")
    try:
        prompt = '  {"question":"你好\\nworld","x":1}  '
        await state.execution.executions.create(_record(ExecutionStatus.SUCCEEDED, 1))
        await _materialize_attempt(state, 1, prompt)
        await state.retention.release_execution_handoff("execution", tenant_id="tenant")
        reader = _reader(state)

        history = await reader.history("execution", tenant_id="tenant", cursor=None, limit=200)
        trace = await reader.trace("execution", tenant_id="tenant", cursor=None, limit=200)
        transcript = await reader.transcript("execution", tenant_id="tenant", cursor=None, limit=200)

        assert [item.content for item in history.items] == [prompt, "plan", "response"]
        assert [item.payload["segment_sequence"] for item in trace.items] == [1, 1]
        assert [item.text for item in transcript.items] == [prompt, "response"]
    finally:
        await state.close()
