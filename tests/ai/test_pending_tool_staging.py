#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pending tools retain effect accountability without eager source promotion."""

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo
from sqlalchemy import event
from sqlalchemy.engine import Connection

from linktools.ai.capability import AgentContext, ExecutionCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus, TaskStatus, ToolOperationStatus, agent_run_id, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import Execution, ExecutionRecord
from linktools.ai.inputs import AgentTaskInput
from linktools.ai.model.messages import decode_model_messages, encode_model_messages
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.contracts import ToolOperationRecord
from linktools.ai.runtime import Runtime, RuntimeFeatures, Session
from linktools.ai.runtime._tool import RuntimeToolOperationBridge
from linktools.ai.storage import InMemoryCacheBackend, StoredPayload
from linktools.ai.task import Task, TaskGraph, TaskNode
from linktools.ai.task.engine import TaskEngine

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_pending_turn_commit import _pending_turn, _snapshot, _turn_state


def _assert_known_result(operation: ToolOperationRecord) -> None:
    assert operation.result_payload is not None
    payload = operation.result_payload.decode()
    assert isinstance(payload, bytes)
    messages = decode_model_messages(payload)
    assert len(messages) == 1 and len(messages[0].parts) == 1
    result = messages[0].parts[0]
    assert isinstance(result, ToolReturnPart)
    assert result.tool_call_id == operation.tool_operation_id and result.content == "known result"


@dataclass
class _ToolGate:
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    stopped: asyncio.Event = field(default_factory=asyncio.Event)
    cancellation_seen: asyncio.Event = field(default_factory=asyncio.Event)
    returned: asyncio.Event = field(default_factory=asyncio.Event)
    model_resumed: asyncio.Event = field(default_factory=asyncio.Event)
    finish: asyncio.Event = field(default_factory=asyncio.Event)
    effects: list[str] = field(default_factory=list)
    model_calls: int = 0


@dataclass
class _ToolRuntime:
    persistence: Persistence
    runtime: Runtime[None]
    session: Session[None]
    gate: _ToolGate

    async def source(self, execution_id: str) -> ExecutionRecord | None:
        repository = self.persistence.execution.executions
        return await repository.state_store.read(lambda _tx: repository.get(
            execution_id, tenant_id=self.runtime.tenant_id,
        ))

    async def operations(self, execution_id: str) -> tuple[ToolOperationRecord, ...]:
        repository = self.persistence.checkpoint.tools
        return await repository.state_store.read(lambda _tx: repository.list_by_execution(
            execution_id, tenant_id=self.runtime.tenant_id,
        ))

    async def start(self) -> Execution[None]:
        execution = await self.session.start("work", idempotency_key="request")
        await asyncio.wait_for(self.gate.entered.wait(), 5)
        return execution

    async def finish(self, execution: Execution[None]) -> None:
        self.gate.release.set()
        await asyncio.wait_for(self.gate.model_resumed.wait(), 5)
        self.gate.finish.set()
        assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED


@asynccontextmanager
async def _tool_runtime(
    tmp_path: Path, *, effect_policy: Literal["replay_safe", "non_replay_safe"] = "replay_safe",
    terminal: bool = True, fail_after_entry: bool = False, return_after_cancel: bool = False,
) -> AsyncIterator[_ToolRuntime]:
    gate = _ToolGate()

    async def work(context: AgentContext[None]) -> str:
        gate.effects.append(context.execution_id)
        gate.entered.set()
        try:
            while not gate.release.is_set():
                try:
                    await gate.release.wait()
                except asyncio.CancelledError:
                    if not return_after_cancel:
                        raise
                    gate.cancellation_seen.set()
            if fail_after_entry:
                raise RuntimeError("effect response was lost")
            gate.returned.set()
            return "known result"
        finally:
            gate.stopped.set()

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        gate.model_calls += 1
        if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[ToolCallPart("work", {}, tool_call_id="call")])
        gate.model_resumed.set()
        await gate.finish.wait()
        return ModelResponse(parts=[TextPart("done")])

    features = RuntimeFeatures("pending-tool-staging")
    features.capability(ExecutionCapability(
        retain_history=True, commit=CommitPolicy.terminal() if terminal else CommitPolicy.sync(),
    ), scope="runtime")
    features.tool(work, effect_policy=effect_policy)
    features.agent("default", model="default", allow_tools=("work",))
    storage = PersistenceStorage.sqlite(tmp_path / "state.db",
        coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    async with Persistence.open("pending-tool-staging", storage=storage) as persistence:
        async with Runtime.attach(persistence=persistence,
            features=model_features(_Models(_UsageFunctionModel(model)), features),
        ) as runtime:
            session = await runtime.agents.get().create_session("session")
            try:
                yield _ToolRuntime(persistence, runtime, session, gate)
            finally:
                gate.release.set()
                gate.finish.set()


@contextmanager
def _source_write_commits(state: Persistence) -> Iterator[list[None]]:
    # Count physical commits because the pending contract is one atomic source
    # publication, independent of the number of repository calls in that batch.
    engine = state.execution.executions.state_store.storage_group.context.engine.sync_engine
    writes: set[int] = set()
    commits: list[None] = []

    def statement(
        connection: Connection, cursor: object, sql: str, parameters: object,
        context: object, executemany: bool,
    ) -> None:
        if sql.lstrip().split()[0].upper() in {"INSERT", "UPDATE", "DELETE"}:
            writes.add(id(connection))

    def commit(connection: Connection) -> None:
        if id(connection) in writes:
            commits.append(None)
        writes.discard(id(connection))

    def rollback(connection: Connection) -> None:
        writes.discard(id(connection))

    event.listen(engine, "after_cursor_execute", statement)
    event.listen(engine, "commit", commit)
    event.listen(engine, "rollback", rollback)
    try:
        yield commits
    finally:
        event.remove(engine, "after_cursor_execute", statement)
        event.remove(engine, "commit", commit)
        event.remove(engine, "rollback", rollback)


@pytest.mark.asyncio
async def test_pending_replay_safe_tool_publishes_only_with_terminal_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[ToolOperationRecord, ...]] = []
    original = RuntimeToolOperationBridge.list_operations

    async def list_operations(bridge: RuntimeToolOperationBridge) -> tuple[ToolOperationRecord, ...]:
        records = await original(bridge)
        observed.append(records)
        return records

    monkeypatch.setattr(RuntimeToolOperationBridge, "list_operations", list_operations)
    async with _tool_runtime(tmp_path) as case:
        with _source_write_commits(case.persistence) as commits:
            execution = await case.start()
            assert await case.source(execution.execution_id) is None
            assert await case.operations(execution.execution_id) == ()
            case.gate.release.set()
            await asyncio.wait_for(case.gate.model_resumed.wait(), 5)
            assert await case.source(execution.execution_id) is None
            assert await case.operations(execution.execution_id) == ()
            assert commits == []
            case.gate.finish.set()
            assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
            assert len(commits) == 1
        record = await case.source(execution.execution_id)
        assert record is not None and record.status is ExecutionStatus.SUCCEEDED
        operations = await case.operations(execution.execution_id)
        assert len(operations) == 1
        operation = operations[0]
        assert operation.status is ToolOperationStatus.COMPLETED
        assert operation.execution_id == execution.execution_id and operation.tool_call_id == "call"
        assert operation.replay_safe and operation.fence > 0
        _assert_known_result(operation)
        assert observed and observed[-1] == (operation,)
        replay = await case.session.start("work", idempotency_key="request")
        assert replay.execution_id == execution.execution_id
        assert (await replay.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        assert case.gate.effects == [execution.execution_id] and case.gate.model_calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", (False, True))
async def test_pending_entered_tool_keeps_unknown_effect_after_live_failure(
    tmp_path: Path, cancel: bool,
) -> None:
    async with _tool_runtime(tmp_path, fail_after_entry=not cancel) as case:
        execution = await case.start()
        assert await case.source(execution.execution_id) is None
        if cancel:
            await asyncio.wait_for(execution.cancel(idempotency_key="cancel"), 10)
        else:
            case.gate.release.set()
        with pytest.raises(AIError) as failed:
            await execution.wait(timeout_seconds=10)
        assert failed.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
        await asyncio.wait_for(case.gate.stopped.wait(), 5)
        current = await case.source(execution.execution_id)
        assert current is not None and current.status is ExecutionStatus.RECOVERY_REQUIRED
        operations = await case.operations(execution.execution_id)
        assert len(operations) == 1
        assert operations[0].status is ToolOperationStatus.EFFECT_UNKNOWN
        assert operations[0].result_payload is None
        assert operations[0].error_code == ErrorCode.TOOL_EFFECT_UNKNOWN.value
        assert case.gate.effects == [execution.execution_id] and case.gate.model_calls == 1
        replay = await case.session.start("work", idempotency_key="request")
        assert replay.execution_id == execution.execution_id
        with pytest.raises(AIError) as repeated:
            await replay.wait(timeout_seconds=5)
        assert repeated.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
        assert case.gate.effects == [execution.execution_id]


@pytest.mark.asyncio
async def test_pending_promotion_keeps_inflight_tool_identity_and_late_result(tmp_path: Path) -> None:
    async with _tool_runtime(tmp_path) as case:
        execution = await case.start()
        assert await case.source(execution.execution_id) is None
        # Promotion is an internal boundary also used by dependency consumers;
        # invoke it directly to isolate an in-flight leaf from unrelated work.
        backend = case.runtime._execution_service.runtime_backend()
        await backend.promote_pending_execution(execution.execution_id)
        promoted = await case.source(execution.execution_id)
        assert promoted is not None and promoted.status is ExecutionStatus.STARTED
        operations = await case.operations(execution.execution_id)
        assert len(operations) == 1 and operations[0].status is ToolOperationStatus.CLAIMED
        claimed = operations[0]
        assert claimed.execution_id == execution.execution_id
        assert claimed.owner == promoted.producer_claim_id
        assert claimed.producer_generation == promoted.producer_generation
        await case.finish(execution)
        completed = await case.operations(execution.execution_id)
        assert len(completed) == 1 and completed[0].status is ToolOperationStatus.COMPLETED
        assert completed[0].tool_operation_id == claimed.tool_operation_id
        assert completed[0].fence == claimed.fence
        _assert_known_result(completed[0])
        assert case.gate.effects == [execution.execution_id]


@pytest.mark.asyncio
async def test_pending_tool_known_result_after_cancel_remains_completed(tmp_path: Path) -> None:
    async with _tool_runtime(tmp_path, return_after_cancel=True) as case:
        execution = await case.start()
        assert await case.source(execution.execution_id) is None
        cancellation = asyncio.create_task(execution.cancel(idempotency_key="cancel"))
        try:
            await asyncio.wait_for(case.gate.cancellation_seen.wait(), 5)
            claimed = await case.operations(execution.execution_id)
            assert len(claimed) == 1 and claimed[0].status is ToolOperationStatus.CLAIMED
            case.gate.release.set()
            await asyncio.wait_for(cancellation, 10)
            assert case.gate.returned.is_set()
            assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.CANCELLED
            completed = await case.operations(execution.execution_id)
            assert len(completed) == 1 and completed[0].status is ToolOperationStatus.COMPLETED
            assert completed[0].tool_operation_id == claimed[0].tool_operation_id
            assert completed[0].fence == claimed[0].fence
            _assert_known_result(completed[0])
            assert case.gate.effects == [execution.execution_id]
        finally:
            case.gate.release.set()
            await asyncio.gather(cancellation, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("effect_policy,terminal", (("non_replay_safe", True), ("replay_safe", False)))
async def test_source_tool_admission_is_retained_outside_pending_replay_safe_turns(
    tmp_path: Path, effect_policy: Literal["replay_safe", "non_replay_safe"], terminal: bool,
) -> None:
    async with _tool_runtime(tmp_path, effect_policy=effect_policy, terminal=terminal) as case:
        execution = await case.start()
        current = await case.source(execution.execution_id)
        assert current is not None and current.status is ExecutionStatus.STARTED
        assert not current.restart_checkpoints
        operations = await case.operations(execution.execution_id)
        assert len(operations) == 1 and operations[0].status is ToolOperationStatus.CLAIMED
        assert operations[0].replay_safe is (effect_policy == "replay_safe")
        await case.finish(execution)
        completed = await case.operations(execution.execution_id)
        assert len(completed) == 1 and completed[0].status is ToolOperationStatus.COMPLETED
        assert completed[0].tool_operation_id == operations[0].tool_operation_id
        assert completed[0].fence == operations[0].fence


@pytest.mark.asyncio
async def test_task_tool_without_restart_checkpoint_keeps_source_ledger(tmp_path: Path) -> None:
    async with _tool_runtime(tmp_path) as case, TaskEngine.open(persistence=case.persistence) as engine:
        task = Task.from_agent("work-task", case.runtime.agents.get())
        graph = TaskGraph("work-graph", (TaskNode("work", task=task, input=AgentTaskInput("work")),))
        run = await engine.with_definitions(task).start(
            graph, principal=case.runtime.default_principal, idempotency_key="task-request",
        )
        await asyncio.wait_for(case.gate.entered.wait(), 5)
        execution_id = case.gate.effects[0]
        current = await case.source(execution_id)
        assert current is not None and current.status is ExecutionStatus.STARTED
        assert current.requires_task_invocation_capture and not current.restart_checkpoints
        operations = await case.operations(execution_id)
        assert len(operations) == 1 and operations[0].status is ToolOperationStatus.CLAIMED
        case.gate.release.set()
        case.gate.finish.set()
        assert (await run.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
        completed = await case.operations(execution_id)
        assert len(completed) == 1 and completed[0].status is ToolOperationStatus.COMPLETED
        assert completed[0].tool_operation_id == operations[0].tool_operation_id
        assert completed[0].fence == operations[0].fence


@pytest.mark.asyncio
async def test_pending_tool_terminal_unknown_readback_keeps_one_effect_and_atomic_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _tool_runtime(tmp_path) as case:
        execution = await case.start()
        case.gate.release.set()
        await asyncio.wait_for(case.gate.model_resumed.wait(), 5)
        group = case.persistence.execution.executions.state_store.storage_group
        original = type(group).mutate
        dispatched, release_commit = asyncio.Event(), asyncio.Event()
        server_tasks: list[asyncio.Task[object]] = []

        async def delayed(owner: object, stores: object, callback: object) -> object:
            if owner is group and not dispatched.is_set():
                async def server_commit() -> object:
                    await release_commit.wait()
                    return await original(owner, stores, callback)

                server_tasks.append(asyncio.create_task(server_commit()))
                dispatched.set()
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return await original(owner, stores, callback)

        monkeypatch.setattr(type(group), "mutate", delayed)
        try:
            case.gate.finish.set()
            await asyncio.wait_for(dispatched.wait(), 5)
            with pytest.raises(AIError) as unknown:
                await execution.wait(timeout_seconds=5)
            assert unknown.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
            assert await case.source(execution.execution_id) is None
            assert await case.operations(execution.execution_id) == ()
            with pytest.raises(AIError) as repeated:
                await case.session.start("work", idempotency_key="request")
            assert repeated.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
            assert len(server_tasks) == 1
            assert case.gate.effects == [execution.execution_id] and case.gate.model_calls == 2
            release_commit.set()
            await asyncio.gather(*server_tasks)
            assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
            current = await case.source(execution.execution_id)
            assert current is not None and current.status is ExecutionStatus.SUCCEEDED
            operations = await case.operations(execution.execution_id)
            assert len(operations) == 1 and operations[0].status is ToolOperationStatus.COMPLETED
            _assert_known_result(operations[0])
            replay = await case.session.start("work", idempotency_key="request")
            assert replay.execution_id == execution.execution_id
            assert case.gate.effects == [execution.execution_id] and case.gate.model_calls == 2
        finally:
            release_commit.set()
            await asyncio.gather(*server_tasks)


@pytest.mark.asyncio
async def test_pending_tool_terminal_rollback_leaves_no_partial_effect_fact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _turn_state("sqlite", tmp_path) as state:
        turn = _pending_turn(state)
        execution = turn.commit.execution
        operation = ToolOperationRecord(
            tool_operation_id="operation", execution_id=execution.execution_id,
            agent_run_id=agent_run_id(namespace="pending-turn", tenant_id="tenant",
                execution_id=execution.execution_id, agent_run_seq=1),
            tool_call_id="call", idempotency_key_digest=canonical_sha256({"call": "call"}),
            tool_name="work", arguments_digest=canonical_sha256({}),
            binding_digest=execution.binding_digest, replay_safe=True,
            status=ToolOperationStatus.COMPLETED, owner=execution.producer_claim_id, fence=1,
            error_code=None, created_at=execution.created_at, updated_at=execution.updated_at,
            producer_generation=execution.producer_generation,
            arguments_payload=StoredPayload.inline_bytes(b"{}"),
            result_payload=StoredPayload.inline_bytes(encode_model_messages((ModelRequest(parts=[
                ToolReturnPart("runtime", "known result", tool_call_id="operation"),
            ]),))),
        )

        async def commit() -> object:
            return await state.commands.commit_pending_turn(
                turn.reservation, turn.commit, expected_session=state.session,
                next_cursor=turn.cursor, conversation_agent_run=turn.run,
                conversation_checkpoint=turn.checkpoint, audit_events=turn.audit_events,
                tool_operations=(operation,),
            )

        before = await _snapshot(state)
        group = state.storage.execution.executions.state_store.storage_group
        original = type(group).mutate
        attempted = []

        async def rollback(owner: object, stores: object, callback: object) -> object:
            if owner is not group:
                return await original(owner, stores, callback)

            async def fail(transaction: object) -> object:
                await callback(transaction)
                attempted.append(True)
                raise AIError(ErrorCode.STORAGE_CONFLICT)

            return await original(owner, stores, fail)

        with monkeypatch.context() as patch:
            patch.setattr(type(group), "mutate", rollback)
            with pytest.raises(AIError) as failed:
                await commit()
        assert attempted and failed.value.code is ErrorCode.STORAGE_CONFLICT
        assert await _snapshot(state) == before
        assert await state.storage.checkpoint.tools.get_operation("operation", tenant_id="tenant") is None
        await commit()
        assert await state.storage.checkpoint.tools.get_operation("operation", tenant_id="tenant") == operation
        confirmed = await state.commands.read_pending_turn(
            turn.reservation, expected_session=state.session, tool_operations=(operation,),
        )
        assert confirmed is not None and confirmed.execution.status is ExecutionStatus.SUCCEEDED
