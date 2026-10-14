#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Producer revocation is established from one authoritative metadata view."""

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeDomain, RuntimeStorage
from linktools.ai.runtime.state._contracts import (
    ExecutionHistoryHeadRecord, ExecutionHistorySealRecord, ExecutionHistoryState,
)
from linktools.ai.runtime.state._store import active_state_scope

from .test_step_projection import _execution


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome,expected", (
    ("sealed", None),
    ("new_generation", None),
    ("same_generation", ErrorCode.STORAGE_CONFLICT),
    ("active", ErrorCode.STORAGE_CONFLICT),
    ("missing_seal", ErrorCode.STORAGE_INTEGRITY_ERROR),
    ("wrong_seal", ErrorCode.STORAGE_INTEGRITY_ERROR),
))
async def test_revoked_producer_check_uses_one_sql_metadata_snapshot(
    tmp_path: Path, outcome: str, expected: ErrorCode | None,
) -> None:
    state = RuntimeStorage.sqlite(tmp_path / "producer.db")
    await state.initialize(namespace="producer-read", tenant_id="tenant")
    repository = state.execution.executions
    archive = state.run_store.read_store(RuntimeDomain.EXECUTION)
    seal = ExecutionHistorySealRecord("execution", (), 0)
    open_head = outcome in {"new_generation", "same_generation"}
    head = ExecutionHistoryHeadRecord(
        "execution", ExecutionHistoryState.OPEN if open_head else ExecutionHistoryState.SEALED,
        0, None if open_head else "incorrect" if outcome == "wrong_seal" else seal.seal_digest,
        2 if outcome == "new_generation" else 1,
    )
    await repository.create(replace(
        _execution(), status=ExecutionStatus.STARTED if outcome == "active" else ExecutionStatus.SUCCEEDED,
    ))

    async def seed(transaction):
        await repository.insert_history_head_in_transaction(transaction, head)
        if not open_head and outcome != "missing_seal":
            await repository.put_history_seal_in_transaction(transaction, seal)

    await repository.state_store.mutate(seed)
    engine = repository.state_store.storage_group.context.engine.sync_engine
    begins = []

    def begin(connection):
        begins.append(connection)

    event.listen(engine, "begin", begin)
    try:
        if expected is None:
            await archive.require_revoked_execution_producer(execution_id="execution", producer_generation=1)
        else:
            with pytest.raises(AIError) as raised:
                await archive.require_revoked_execution_producer(execution_id="execution", producer_generation=1)
            assert raised.value.code is expected
        assert len(begins) <= 1
        assert active_state_scope() is None
    finally:
        event.remove(engine, "begin", begin)
        await state.close()
