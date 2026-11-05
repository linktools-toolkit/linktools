#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Terminal history is immutable detail referenced by execution control."""

from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionEventType, ExecutionStatus, StopReason, UsageMetrics
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence
from linktools.ai.persistence._execution_commands import ExecutionStateCommands
from linktools.ai.persistence.contracts import ExecutionTerminalCommit, ResultRecord

from .test_history_projection_conformance import _record
from .test_incremental_projection_boundaries import _storage


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("trace", (False, True))
async def test_terminal_seal_is_referenced_by_execution(tmp_path: Path, backend: str, trace: bool) -> None:
    persistence = Persistence(_storage(backend, tmp_path))
    await persistence.initialize(namespace="seal-detail", tenant_id="tenant")
    repository = persistence.execution.executions
    current = replace(_record(ExecutionStatus.STARTED, 0), durable_trace=trace)
    try:
        await repository.create_execution(current)
        commit = ExecutionTerminalCommit(
            current.revision, current.event_seq,
            replace(current, status=ExecutionStatus.FAILED, error_code=ErrorCode.EXECUTION_FAILED.value),
            ResultRecord(None, StopReason.ERROR, UsageMetrics(), current.updated_at),
            ExecutionEventType.EXECUTION_FAILED, {"error_code": ErrorCode.EXECUTION_FAILED.value, "safe_error_details": {}},
        )
        commands = ExecutionStateCommands(repository.state_store, repository, None, background_tasks=set())
        result = await commands.commit_terminal_checkpoint(commit, agent_run=None)
        cold = await repository.get(current.execution_id, tenant_id="tenant")
        seal = await repository.get_history_seal(current.execution_id, tenant_id="tenant")
        assert cold == result.execution
        assert cold.history_seal_digest == (None if seal is None else seal.seal_digest)
        assert (seal is not None) is trace
        projected = await repository.get(current.execution_id, tenant_id="tenant")
        assert (projected.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED} and (projected.history_seal_digest is not None) is trace)
        records = await repository.state_store.read(lambda tx: tx.scan_records())
        assert not any(record.kind == "execution_history_head" for record in records)
        with pytest.raises(AIError) as rejected:
            await repository.state_store.mutate(lambda tx: repository.require_execution_history_write_in_transaction(
                tx, current.execution_id,
            ))
        assert rejected.value.code is ErrorCode.STORAGE_CONFLICT
        if seal is not None:
            with pytest.raises(AIError) as immutable:
                await repository.state_store.mutate(lambda tx: repository.put_history_seal_in_transaction(
                    tx, replace(seal, execution_event_high_water=seal.execution_event_high_water + 1),
                ))
            assert immutable.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await persistence.close()
