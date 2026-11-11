#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool admission shares control; certain outcomes remain independently durable."""

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.core import ExecutionStatus, ToolOperationStatus, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionRecoveryDetail
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._runtime_commands import StateCommands
from linktools.ai.persistence.contracts import ToolOperationAdmission
from linktools.ai.storage import StoredPayload

from .test_history_projection_conformance import _record


def _storage(backend: str, path: Path) -> PersistenceStorage:
    return PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(path / "tool.db")


def _commands(state: Persistence) -> StateCommands:
    return StateCommands(state.execution.executions, namespace="tool-boundary", events=state.execution.events,
                         checkpoint=state.checkpoint, tools=state.checkpoint.tools, background_tasks=set())


def _request() -> ToolOperationAdmission:
    return ToolOperationAdmission("execution", "operation", "run", None, "call",
                                  "tool", canonical_sha256({}), canonical_sha256({"binding": 1}),
                                  True, "worker", producer_generation=1)


async def _admit(state: Persistence):
    repository = state.execution.executions
    initial = replace(_record(ExecutionStatus.STARTED, 1), producer_generation=1, producer_claim_id="worker",
                      recovery=ExecutionRecoveryDetail(agent_run_id="run"))
    await repository.create_execution(initial)
    observed = await repository.get_observed("execution", tenant_id="tenant")
    return await _commands(state).commit_tool_admission_boundary(_request(), execution=observed, known_new=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_tool_result_after_cancellation_never_republishes_control(tmp_path: Path, backend: str) -> None:
    async with Persistence.open("tool-boundary", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        receipt = await _admit(state)
        repository = state.execution.executions
        stopped = replace(receipt.execution.value, revision=receipt.execution.value.revision + 1, status=ExecutionStatus.CANCELLING)
        await repository.compare_and_swap("execution", tenant_id="tenant", expected_revision=receipt.execution.value.revision, next_record=stopped)
        result = await _commands(state).commit_tool_terminal_boundary(
            "operation", tenant_id="tenant", owner="worker", fence=1,
            execution=receipt.execution, observed=receipt.operation, result_payload=StoredPayload.inline_text("known"),
        )
        assert result.execution == receipt.execution
        assert result.operation.value.status is ToolOperationStatus.COMPLETED
        assert result.operation.value.result_payload.value == "known"
        assert await repository.get("execution", tenant_id="tenant") == stopped
        assert await state.checkpoint.tools.get_by_call("run", "call", tenant_id="tenant") == result.operation.value
        assert (await state.checkpoint.tools.get_tool_effect(run_id="run", tool_call_id="call")).status == "completed"
        with pytest.raises(AIError) as stale:
            await _commands(state).commit_tool_admission_boundary(
                replace(_request(), tool_operation_id="next", tool_call_id="next"),
                execution=receipt.execution, known_new=True,
            )
        assert stale.value.code is ErrorCode.STORAGE_CONFLICT
        assert await state.checkpoint.tools.get_operation("next", tenant_id="tenant") is None


@pytest.mark.asyncio
async def test_known_new_admission_and_result_use_no_select_or_alias(tmp_path: Path) -> None:
    async with Persistence.open("tool-boundary", tenant_id="tenant", storage=_storage("sqlite", tmp_path)) as state:
        repository = state.execution.executions
        initial = replace(_record(ExecutionStatus.STARTED, 1), producer_generation=1, producer_claim_id="worker",
                          recovery=ExecutionRecoveryDetail(agent_run_id="run"))
        await repository.create_execution(initial)
        observed = await repository.get_observed("execution", tenant_id="tenant")
        statements = []
        engine = repository.state_store.storage_group.context.engine.sync_engine
        def executed(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement)
        event.listen(engine, "after_cursor_execute", executed)
        try:
            receipt = await _commands(state).commit_tool_admission_boundary(_request(), execution=observed, known_new=True)
            settled = await _commands(state).commit_tool_terminal_boundary(
                "operation", tenant_id="tenant", owner="worker", fence=1, execution=receipt.execution,
                observed=receipt.operation, result_payload=StoredPayload.inline_text("known"),
            )
        finally:
            event.remove(engine, "after_cursor_execute", executed)
        assert not [sql for sql in statements if sql.lstrip().upper().startswith("SELECT")]
        assert not [sql for sql in statements if "ai_state_aliases" in sql]
        assert len([sql for sql in statements if sql.lstrip().upper().startswith("UPDATE")]) == 2
        assert settled.execution == receipt.execution
        assert receipt.execution.value == initial
        assert receipt.execution.storage_revision == observed.storage_revision + 1
        assert "payload_json" not in next(sql for sql in statements if sql.lstrip().upper().startswith("UPDATE"))


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_native_admission_failure_rolls_back_both_owners(tmp_path: Path, backend: str) -> None:
    async with Persistence.open("tool-boundary", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        repository = state.execution.executions
        initial = replace(_record(ExecutionStatus.STARTED, 1), producer_generation=1, producer_claim_id="worker",
                          recovery=ExecutionRecoveryDetail(agent_run_id="run"))
        await repository.create_execution(initial)
        observed = await repository.get_observed("execution", tenant_id="tenant")
        async def fail(_group):
            raise RuntimeError("native write failed")
        with pytest.raises((RuntimeError, AIError)):
            await _commands(state).commit_tool_admission_boundary(_request(), execution=observed, known_new=True, write=fail)
        assert await repository.get("execution", tenant_id="tenant") == initial
        assert await state.checkpoint.tools.get_operation("operation", tenant_id="tenant") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_native_result_failure_preserves_permitted_outcome(tmp_path: Path, backend: str) -> None:
    async with Persistence.open("tool-boundary", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        receipt = await _admit(state)
        async def fail(_group):
            raise RuntimeError("native write failed")
        with pytest.raises(RuntimeError, match="native write failed"):
            await _commands(state).commit_tool_terminal_boundary(
                "operation", tenant_id="tenant", owner="worker", fence=1, execution=receipt.execution,
                observed=receipt.operation, result_payload=StoredPayload.inline_text("known"), write=fail,
            )
        stored = await state.checkpoint.tools.get_operation("operation", tenant_id="tenant")
        assert stored.status is ToolOperationStatus.COMPLETED
        assert stored.result_payload.value == "known"
        assert await state.execution.executions.get("execution", tenant_id="tenant") == receipt.execution.value


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_late_result_cannot_replace_an_independent_terminal_decision(tmp_path: Path, backend: str) -> None:
    async with Persistence.open("tool-boundary", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        receipt = await _admit(state)
        first = await _commands(state).commit_tool_terminal_boundary(
            "operation", tenant_id="tenant", owner="worker", fence=1, execution=receipt.execution,
            observed=receipt.operation, result_payload=StoredPayload.inline_text("decided"),
        )
        with pytest.raises(AIError) as conflict:
            await _commands(state).commit_tool_terminal_boundary(
                "operation", tenant_id="tenant", owner="worker", fence=1, execution=receipt.execution,
                observed=receipt.operation, result_payload=StoredPayload.inline_text("late"),
            )
        assert conflict.value.code is ErrorCode.TOOL_RESULT_CONFLICT
        assert await state.checkpoint.tools.get_operation("operation", tenant_id="tenant") == first.operation.value


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("failure", (False, True))
async def test_certain_result_refreshes_stale_same_fence_unknown_receipt(tmp_path: Path, backend: str, failure: bool) -> None:
    async with Persistence.open("tool-boundary", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        receipt = await _admit(state)
        await state.checkpoint.tools.mark_effect_unknown(
            "operation", tenant_id="tenant", owner="worker", fence=1, error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value,
        )
        values = {"result_payload": StoredPayload.inline_text("known")}
        if failure:
            from linktools.ai.persistence.contracts import decode_tool_operation_failure
            payload = StoredPayload.inline_bytes(b'{"version":1,"kind":"tool_call_failed","message":"certain failure"}')
            values = {"error_code": ErrorCode.TOOL_EXECUTION_FAILED.value, "error_payload": payload}
            decode_tool_operation_failure(values["error_code"], payload)
        settled = await _commands(state).commit_tool_terminal_boundary(
            "operation", tenant_id="tenant", owner="worker", fence=1, execution=receipt.execution,
            observed=receipt.operation, **values,
        )
        assert settled.operation.value.status is (ToolOperationStatus.FAILED if failure else ToolOperationStatus.COMPLETED)
        assert await state.checkpoint.tools.get_operation("operation", tenant_id="tenant") == settled.operation.value


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_late_outcome_cannot_overwrite_manual_resolution_and_new_fence(tmp_path: Path, backend: str) -> None:
    from datetime import datetime, timezone
    from .test_execution_recovery_commands import _commands as recovery_commands, _resolution_operation
    async with Persistence.open("tool-boundary", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        receipt = await _admit(state)
        tools = state.checkpoint.tools
        await tools.mark_effect_unknown("operation", tenant_id="tenant", owner="worker", fence=1,
                                        error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value)
        operation = replace(_resolution_operation(datetime.now(timezone.utc)), resource_id="operation", result_ref="operation")
        await recovery_commands(state).resolve_tool_effect(operation, expected_fence=1, target_status=ToolOperationStatus.PENDING)
        successor = await tools.admit(replace(_request(), owner="successor", producer_generation=2))
        assert successor.fence == 2
        with pytest.raises(AIError) as conflict:
            await _commands(state).commit_tool_terminal_boundary(
                "operation", tenant_id="tenant", owner="worker", fence=1, execution=receipt.execution,
                observed=receipt.operation, result_payload=StoredPayload.inline_text("late"),
            )
        assert conflict.value.code is ErrorCode.TOOL_OPERATION_CONFLICT
        assert await tools.get_operation("operation", tenant_id="tenant") == successor


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("change_business", (False, True))
async def test_admission_refreshes_only_unchanged_control_after_trace_guard(
    tmp_path: Path, backend: str, change_business: bool,
) -> None:
    async with Persistence.open("tool-boundary", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        repository = state.execution.executions
        initial = replace(_record(ExecutionStatus.STARTED, 1), producer_generation=1, producer_claim_id="worker",
                          recovery=ExecutionRecoveryDetail(agent_run_id="run"))
        await repository.create_execution(initial)
        observed = await repository.get_observed("execution", tenant_id="tenant")
        if change_business:
            await repository.compare_and_swap("execution", tenant_id="tenant", expected_revision=initial.revision,
                                              next_record=replace(initial, revision=initial.revision + 1,
                                                                  safe_error_details={"changed": True}))
        else:
            await repository.state_store.mutate(lambda tx: tx.guard_record_revision(
                repository.execution_record_key("execution"), expected_storage_revision=observed.storage_revision,
            ))
        writes = []
        async def publish(_group):
            writes.append("native")
        if change_business:
            with pytest.raises(AIError) as failure:
                await _commands(state).commit_tool_admission_boundary(
                    _request(), execution=observed, known_new=True, write=publish,
                )
            assert failure.value.code is ErrorCode.STORAGE_CONFLICT
            assert writes == []
            assert await state.checkpoint.tools.get_operation("operation", tenant_id="tenant") is None
        else:
            confirmed = await _commands(state).commit_tool_admission_boundary(
                _request(), execution=observed, known_new=True, write=publish,
            )
            assert confirmed.execution.value == initial
            assert confirmed.execution.storage_revision == observed.storage_revision + 2
            assert writes == ["native"]
            assert confirmed.operation.value.status is ToolOperationStatus.CLAIMED


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_unknown_native_write_is_atomic_and_repeatable_without_lifecycle_change(tmp_path: Path, backend: str) -> None:
    async with Persistence.open("tool-boundary", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        receipt = await _admit(state)
        tools = state.checkpoint.tools
        seen = []
        async def reject(transaction):
            stored = await transaction.get_record(tools._tool_key("operation"))
            assert stored.state == ToolOperationStatus.EFFECT_UNKNOWN.value
            raise RuntimeError("native event unavailable")
        with pytest.raises((RuntimeError, AIError)):
            await tools.mark_effect_unknown("operation", tenant_id="tenant", owner="worker", fence=1,
                                            error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value, write=reject)
        assert await tools.get_operation("operation", tenant_id="tenant") == receipt.operation.value
        async def publish(transaction):
            stored = await transaction.get_record(tools._tool_key("operation"))
            seen.append(stored.state)
        first = await tools.mark_effect_unknown("operation", tenant_id="tenant", owner="worker", fence=1,
                                               error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value, write=publish)
        repeated = await tools.mark_effect_unknown("operation", tenant_id="tenant", owner="worker", fence=1,
                                                  error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value, write=publish)
        assert first == repeated and first.status is ToolOperationStatus.EFFECT_UNKNOWN
        assert seen == [ToolOperationStatus.EFFECT_UNKNOWN.value] * 2
        assert await state.execution.executions.get("execution", tenant_id="tenant") == receipt.execution.value
