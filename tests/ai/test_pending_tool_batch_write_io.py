#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Terminal pending tool receipts share bounded inserts without replaying effects."""

from pathlib import Path

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine

from linktools.ai.core import ToolOperationStatus, agent_run_id, canonical_sha256
from linktools.ai.persistence.contracts import ToolOperationRecord
from linktools.ai.storage import StoredPayload

from .test_pending_turn_commit import _pending_turn, _snapshot, _turn_state


@pytest.mark.asyncio
@pytest.mark.parametrize("count", (3, 65))
@pytest.mark.parametrize("custom", (False, True))
@pytest.mark.parametrize("rollback", (False, True))
async def test_pending_terminal_batches_tool_receipt_inserts(
    tmp_path: Path, count: int, custom: bool, rollback: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _turn_state("sqlite", tmp_path) as state:
        turn = _pending_turn(state)
        execution = turn.commit.execution
        operations = tuple(ToolOperationRecord(
            tool_operation_id=f"operation-{index}", execution_id=execution.execution_id,
            agent_run_id=agent_run_id(namespace="pending-turn", tenant_id="tenant",
                execution_id=execution.execution_id, agent_run_seq=1),
            tool_call_id=f"call-{index}", tool_name="work", arguments_digest=canonical_sha256({}),
            binding_digest=execution.binding_digest, replay_safe=True,
            status=ToolOperationStatus.COMPLETED, owner=execution.producer_claim_id, fence=1,
            error_code=None, created_at=execution.created_at, updated_at=execution.updated_at,
            producer_generation=execution.producer_generation,
            arguments_payload=StoredPayload.inline_bytes(b"{}"),
            result_payload=StoredPayload.inline_bytes(b"result"),
        ) for index in range(count))
        reserves = []
        if custom:
            from linktools.ai.persistence._checkpoint_repositories import ToolRepositoryImpl

            class CustomTools(ToolRepositoryImpl):
                async def reserve(self, record):
                    reserves.append(record.tool_operation_id)
                    return await super().reserve(record)

                async def reserve_many_in_transaction(self, transaction, records):
                    raise AssertionError("native batching must not bypass custom reserve")

            state.commands._tools = CustomTools(state.storage.checkpoint.tools.state_store,
                namespace="pending-turn", tenant_id="tenant")
        inserts = []

        def sql(connection, cursor, statement, parameters, context, many):
            if statement.startswith("INSERT INTO ai_step_tool_operations"):
                inserts.append(len(parameters))

        async def commit():
            return await state.commands.commit_pending_turn(
                turn.reservation, turn.commit, expected_session=state.session,
                next_cursor=turn.cursor, conversation_agent_run=turn.run,
                conversation_checkpoint=turn.checkpoint, audit_events=turn.audit_events,
                tool_operations=operations,
            )

        event.listen(Engine, "before_cursor_execute", sql)
        try:
            if rollback:
                from linktools.ai.errors import AIError, ErrorCode

                before = await _snapshot(state)
                group = state.storage.execution.executions.state_store.storage_group
                original = type(group).mutate

                async def fail_after_write(owner, stores, callback):
                    if owner is not group:
                        return await original(owner, stores, callback)

                    async def fail(transaction):
                        await callback(transaction)
                        raise AIError(ErrorCode.STORAGE_CONFLICT)

                    return await original(owner, stores, fail)

                with monkeypatch.context() as patch:
                    patch.setattr(type(group), "mutate", fail_after_write)
                    with pytest.raises(AIError) as rejected:
                        await commit()
                    assert rejected.value.code is ErrorCode.STORAGE_CONFLICT
                assert await _snapshot(state) == before
                for operation in operations:
                    assert await state.storage.checkpoint.tools.get_operation(
                        operation.tool_operation_id, tenant_id="tenant") is None
                inserts.clear()
                reserves.clear()
            first = await commit()
            assert len(inserts) == (count if custom else (count + 31) // 32)
            assert reserves == ([value.tool_operation_id for value in operations] if custom else [])
            assert max(inserts) <= 999
            inserts.clear()
            assert await commit() == first
            assert inserts == []
            repository = state.storage.checkpoint.tools
            for expected in operations:
                assert await repository.get_operation(expected.tool_operation_id, tenant_id="tenant") == expected
        finally:
            event.remove(Engine, "before_cursor_execute", sql)
