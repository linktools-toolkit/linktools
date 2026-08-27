#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution history projection for claimed and materialized attempts."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._output import bind_output
from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    HmacCursorSigner,
    step_conversation_id,
    step_run_id,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime._history import StepExecutionHistoryReader
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime.state import (
    ConversationHistoryRecord,
    ExecutionHistoryHeadRecord,
    ExecutionHistorySealRecord,
    ExecutionHistoryState,
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
    TranscriptMessageRef,
)
from linktools.ai.runtime.state._history import (
    _conversation_overlap_signature,
    _overlap_signature,
)
from linktools.ai.runtime.state._steps import LockOrderError, _RunHistoryLock
from linktools.ai.runtime.state._store import (
    partition_digest,
    record_key_digest,
    sortable_identity,
)
from linktools.ai.spec import AgentSpec
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    RunRecord,
    StepEvent,
)
from sqlalchemy.ext.asyncio import create_async_engine


def _binding() -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("default", model="default"),
        model={"route_id": "default", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
        binding_digest="a" * 64,
    )


def _record(status: ExecutionStatus, sequence: int) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    return ExecutionRecord(
        execution_id="execution",
        tenant_id="tenant",
        session_id=None,
        binding_digest="a" * 64,
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
        mode="run",
        planning=False,
        thinking=False,
        binding=_binding(),
    )


def test_conversation_overlap_ignores_only_standing_system_prompt() -> None:
    old = ModelRequest(
        parts=(
            SystemPromptPart(content="old"),
            UserPromptPart(content="hello"),
        )
    )
    new = ModelRequest(
        parts=(
            SystemPromptPart(content="new"),
            UserPromptPart(content="hello"),
        )
    )

    assert _conversation_overlap_signature(old) == _conversation_overlap_signature(new)
    assert _overlap_signature(old) != _overlap_signature(new)
    assert len(old.parts) == 2
    assert len(new.parts) == 2


@pytest.mark.asyncio
async def test_history_head_requires_open_for_mutations() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="history-head", tenant_id="tenant")
    repository = state.execution.executions
    store = repository.state_store
    try:
        await repository.create_with_history_head(_record(ExecutionStatus.STARTED, 1))
        head = await repository.get_history_head("execution", tenant_id="tenant")
        assert head is not None
        assert head.state is ExecutionHistoryState.OPEN
        assert head.revision == 0
        assert head.seal_digest is None

        async def require_open(transaction: object) -> None:
            open_head, record = await repository.require_open_history_head_in_transaction(
                transaction,
                "execution",
            )
            sealed = ExecutionHistoryHeadRecord(
                "execution",
                "tenant",
                ExecutionHistoryState.SEALED,
                open_head.revision + 1,
                "d" * 64,
            )
            await repository.replace_history_head_in_transaction(
                transaction,
                record,
                sealed,
            )

        await store.mutate(require_open)
        head = await repository.get_history_head("execution", tenant_id="tenant")
        assert head is not None
        assert head.state is ExecutionHistoryState.SEALED
        assert head.seal_digest == "d" * 64

        async def mutate_sealed(transaction: object) -> None:
            await repository.require_open_history_head_in_transaction(
                transaction,
                "execution",
            )

        with pytest.raises(AIError) as raised:
            await store.mutate(mutate_sealed)
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_execution_projection_paths_reject_a_sealed_history_head(
    tmp_path: Path,
) -> None:
    state = RuntimeState.filesystem(tmp_path / "runtime")
    await state.initialize(namespace="history-fence", tenant_id="tenant")
    try:
        await state.execution.executions.create(_record(ExecutionStatus.STARTED, 1))
        await _materialize_attempt(state, 1, "before-seal")
        repository = state.execution.executions
        open_head = await repository.get_history_head("execution", tenant_id="tenant")
        assert open_head is not None
        assert open_head.state is ExecutionHistoryState.OPEN
        assert open_head.revision == 1

        async def seal(transaction: object) -> None:
            head, record = await repository.require_open_history_head_in_transaction(
                transaction,
                "execution",
            )
            await repository.replace_history_head_in_transaction(
                transaction,
                record,
                ExecutionHistoryHeadRecord(
                    head.execution_id,
                    head.tenant_id,
                    ExecutionHistoryState.SEALED,
                    head.revision + 1,
                    "s" * 64,
                ),
            )

        await repository.state_store.mutate(seal)
        archive = state.steps.read_store(RuntimeDomain.EXECUTION)
        assert isinstance(archive, StateStepArchive)
        run_id = step_run_id(
            namespace="history",
            tenant_id="tenant",
            execution_id="execution",
            segment_sequence=1,
        )
        run = await archive.get_run(run_id=run_id)
        assert run is not None
        before_events = await archive.list_events(run_id=run_id)
        before_head = await repository.get_history_head("execution", tenant_id="tenant")
        assert before_head is not None
        now = datetime.now(timezone.utc)

        with pytest.raises(AIError) as sync_error:
            await archive.sync_projection(
                run,
                events=(
                    StepEvent(
                        run_id=run_id,
                        kind="after-seal",
                        step_index=3,
                        timestamp=now,
                        conversation_id=run.conversation_id,
                        agent_name=run.agent_name,
                    ),
                ),
                snapshots=(),
                execution_id="execution",
            )
        assert sync_error.value.code is ErrorCode.STORAGE_CONFLICT

        with pytest.raises(AIError) as empty_sync_error:
            await archive.sync_projection(
                run,
                events=(),
                snapshots=(),
                execution_id="execution",
            )
        assert empty_sync_error.value.code is ErrorCode.STORAGE_CONFLICT

        with pytest.raises(AIError) as snapshot_error:
            await archive.materialize_snapshot(
                run,
                ContinuableSnapshot(
                    run_id=run_id,
                    step_index=4,
                    messages=[ModelRequest(parts=[UserPromptPart(content="after-seal")])],
                    conversation_id=run.conversation_id,
                    parent_run_id=run.parent_run_id,
                    agent_name=run.agent_name,
                    timestamp=now,
                ),
                execution_id="execution",
            )
        assert snapshot_error.value.code is ErrorCode.STORAGE_CONFLICT
        assert await archive.list_events(run_id=run_id) == before_events
        assert await repository.get_history_head("execution", tenant_id="tenant") == before_head
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_terminal_prepare_accepts_an_unprojected_execution_run(
    tmp_path: Path,
) -> None:
    state = RuntimeState.filesystem(tmp_path / "runtime")
    await state.initialize(namespace="history-unprojected", tenant_id="tenant")
    terminal_plan = None
    try:
        await state.execution.executions.create(_record(ExecutionStatus.STARTED, 1))
        run_id = step_run_id(
            namespace="history-unprojected",
            tenant_id="tenant",
            execution_id="execution",
            segment_sequence=1,
        )
        now = datetime.now(timezone.utc)
        await state.steps.register_run(
            RunRecord(
                run_id=run_id,
                conversation_id=step_conversation_id(
                    namespace="history-unprojected",
                    tenant_id="tenant",
                    execution_id="execution",
                ),
                parent_run_id=None,
                agent_name="default",
                metadata={"segment_sequence": "1"},
                started_at=now,
            )
        )

        archive = state.steps.read_store(RuntimeDomain.EXECUTION)
        assert isinstance(archive, StateStepArchive)
        assert await archive.get_run(run_id=run_id) is None
        terminal_plan = await state.steps.prepare_execution_terminal_seal(
            execution_id="execution",
            run_ids=(run_id,),
            binding_digest="a" * 64,
        )

        projection = terminal_plan.projections[0]
        assert projection.events == ()
        assert projection.snapshots == ()
        assert projection.target_event_offset == 0
        assert projection.target_snapshot_offset == 0
        assert projection.target_transcript_message_count == 0
        assert projection.projection_digest == "empty"
    finally:
        if terminal_plan is not None:
            await state.steps.discard_execution_terminal_seal(terminal_plan)
        await state.close()


@pytest.mark.asyncio
async def test_conversation_head_replacement_preserves_physical_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.db"
    provisioning_engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await provision_database(provisioning_engine)
    await provisioning_engine.dispose()
    state = RuntimeState.sqlite(path)
    await state.initialize(namespace="conversation-head", tenant_id="tenant")
    try:
        await state.conversation.histories.create(
            ConversationHistoryRecord(
                history_id="history",
                session_id="session",
                tenant_id="tenant",
                parent_history_id=None,
                prefix_index_head_id=None,
                inherited_message_count=0,
                inherited_history_item_count=0,
            )
        )
        now = datetime.now(timezone.utc)
        run = RunRecord(
            run_id="run",
            conversation_id="conversation",
            parent_run_id=None,
            agent_name="default",
            metadata={"history_id": "history"},
            started_at=now,
        )
        await state.steps.read_store(RuntimeDomain.CONVERSATION).materialize_snapshot(
            run,
            ContinuableSnapshot(
                run_id="run",
                step_index=1,
                messages=[ModelRequest(parts=[UserPromptPart(content="hello")])],
                conversation_id="conversation",
                parent_run_id=None,
                agent_name="default",
                timestamp=now,
            ),
        )
        head = await state.steps.read_store(
            RuntimeDomain.CONVERSATION
        ).transcript_repository.get_head("history")
        assert head is not None
        assert head.message_count == 1
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_projection_flight_resolves_waiters_after_abandon() -> None:
    steps = RuntimeState.in_memory()
    await steps.initialize(namespace="flight-abandon", tenant_id="tenant")
    try:
        store = steps.steps
        captured = await store.capture_execution_projection("missing-run")
        assert captured is None
        await store.wait_projection_flight("missing-run")
    finally:
        await steps.close()


@pytest.mark.asyncio
async def test_terminal_commit_cancellation_still_finalizes_after_durable_commit() -> None:
    class Lifecycle:
        def __init__(self) -> None:
            self.finalized = asyncio.Event()

        async def finalize_execution_terminal_seal(self, plan: object) -> None:
            del plan
            self.finalized.set()

        async def discard_execution_terminal_seal(self, plan: object) -> None:
            del plan

        async def prepare_execution_terminal_seal(self, **kwargs: object) -> ExecutionTerminalSealPlan:
            del kwargs
            return plan

    lifecycle = Lifecycle()
    backend = object.__new__(LocalExecutionBackend)
    backend._step_reads = {
        RuntimeDomain.EXECUTION: object.__new__(StateStepArchive),
    }
    backend._step_lifecycle = lifecycle
    backend._checkpoint_tasks = set()
    backend._pending_audit_events = {}
    backend._pending_audit_locks = {}
    backend._execution_durable_tasks = {}
    started = asyncio.Event()

    async def commit(*args: object, **kwargs: object) -> object:
        del args, kwargs
        started.set()
        await asyncio.sleep(0.01)
        return SimpleNamespace(execution=SimpleNamespace(event_sequence=0))

    backend._commit_execution_terminal_checkpoint_locked_body = commit
    current = _record(ExecutionStatus.SUCCEEDED, 0)
    plan = ExecutionTerminalSealPlan("execution", "a" * 64, (), ())
    task = asyncio.create_task(
        backend._commit_execution_terminal_checkpoint(
            current,
            object(),
            run_id=None,
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(AIError) as error:
        await task
    assert error.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
    await lifecycle.finalized.wait()
    await asyncio.sleep(0)
    assert not backend._checkpoint_tasks


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


async def _materialize_attempt(state: RuntimeState, sequence: int, prompt: str) -> None:
    repository = state.execution.executions
    if await repository.get_history_head("execution", tenant_id="tenant") is None:
        await repository.state_store.mutate(
            lambda transaction: repository.insert_history_head_in_transaction(
                transaction,
                ExecutionHistoryHeadRecord(
                    "execution",
                    "tenant",
                    ExecutionHistoryState.OPEN,
                    0,
                    None,
                ),
            )
        )
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
    await state.steps.flush_execution_projection(
        run_id,
        execution_id="execution",
    )
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
async def test_in_memory_raw_refs_fail_fast_even_when_snapshots_exist() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="history-in-memory-refs", tenant_id="tenant")
    try:
        await state.execution.executions.create(_record(ExecutionStatus.STARTED, 1))
        await _materialize_attempt(state, 1, "in-memory-ref")
        archive = state.steps.read_store(RuntimeDomain.EXECUTION)
        run_id = step_run_id(
            namespace="history",
            tenant_id="tenant",
            execution_id="execution",
            segment_sequence=1,
        )
        refs = (
            TranscriptMessageRef(RuntimeDomain.EXECUTION, run_id, 0),
            TranscriptMessageRef(RuntimeDomain.EXECUTION, "missing-run", 0),
            TranscriptMessageRef(RuntimeDomain.EXECUTION, run_id, 999),
        )
        for ref in refs:
            with pytest.raises(AIError) as raised:
                await archive.resolve_transcript_message_refs((ref,))
            assert raised.value.code is ErrorCode.STORAGE_DEPENDENCY_NOT_READY
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_terminal_reader_pages_from_execution_read_model(tmp_path: Path) -> None:
    state = RuntimeState.filesystem(tmp_path / "runtime")
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
            binding_digest="a" * 64,
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
async def test_read_model_accepts_current_v1_record() -> None:
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
                        "future_metadata": {"$future_v2": {"ignored": True}},
                    },
                )
            )
        )
        repository = ExecutionReadModelRepository(
            store,
            namespace="read-model-v1",
            tenant_id="tenant",
        )

        current = await repository.get_complete("execution", tenant_id="tenant")
        assert current is not None
        assert current.model_version == 1
        assert current.source_digest == "legacy-source"

        build_called = False

        async def build() -> ExecutionReadModelBuild:
            nonlocal build_called
            build_called = True
            return ExecutionReadModelBuild(
                "execution",
                "tenant",
                "unexpected-source",
                (),
                (),
                (),
            )

        reused = await repository.ensure(
            "execution",
            tenant_id="tenant",
            builder=build,
        )
        assert reused == current
        assert not build_called
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
