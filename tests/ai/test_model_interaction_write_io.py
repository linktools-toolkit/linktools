#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution projections batch metadata while retaining producer and owner fences."""

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._contracts import RuntimePayloadRef
from linktools.ai.storage import StoredPayload

from .test_model_interaction_records import _prepared, _running, _terminal
from .test_step_archive_read_boundaries import _archive


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_execution_interaction_lifecycle_bounds_record_reads(
    tmp_path: Path, backend: str,
) -> None:
    async with _archive(tmp_path, backend) as (_state, archive, run):
        admitted = _running()
        prepared = replace(
            _prepared(admitted),
            request_envelope=RuntimePayloadRef(
                StoredPayload.inline_bytes(b"{}"), RuntimeDomain.EXECUTION,
            ),
        )
        engine = (
            archive.state_store.storage_group.context.engine.sync_engine
            if backend == "sqlite" else None
        )
        for value in (admitted, prepared, _terminal(prepared)):
            statements: list[str] = []

            def capture(_conn, _cursor, statement, _parameters, _context, _many):
                statements.append(statement.lstrip().upper())

            if engine is not None:
                event.listen(engine, "before_cursor_execute", capture)
            try:
                await archive.sync_prepared_projection(
                    run, events=(), checkpoints=(), interactions=(value,),
                    execution_id="execution", producer_generation=1,
                )
            finally:
                if engine is not None:
                    event.remove(engine, "before_cursor_execute", capture)
            if engine is not None:
                record_reads = [
                    sql for sql in statements
                    if sql.startswith("SELECT") and "FROM AI_STATE_RECORDS" in sql
                ]
                assert len(record_reads) <= 2, (
                    "projection needs the execution fence and one combined owner/interaction read"
                )
            assert await archive.list_model_interactions(agent_run_id="run") == [value]
            assert await archive.model_interaction_count(agent_run_id="run") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("fence", ("producer", "owner"))
async def test_execution_interaction_conflict_rolls_back_and_can_retry(
    tmp_path: Path, backend: str, fence: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _archive(tmp_path, backend) as (state, archive, run):
        current = _running()
        update = replace(current, model={"route_id": "updated"})
        await archive.sync_prepared_projection(
            run, events=(), checkpoints=(), interactions=(current,),
            execution_id="execution", producer_generation=1,
        )
        repository = state.execution.executions
        retry_generation = 1
        if fence == "producer":
            execution = await repository.get("execution", tenant_id="tenant")
            assert execution is not None
            retry_generation = await repository.state_store.mutate(
                lambda tx: repository.admit_history_producer_in_transaction(
                    tx, replace(execution, revision=execution.revision + 1),
                )
            )
        before = await repository.get_history_head("execution", tenant_id="tenant")

        async def project(transaction):
            original_guard = transaction.guard_record
            owner_key = archive._agent_run_key(run.agent_run_id)

            async def guard(key, *, expected_storage_revision):
                if fence == "owner" and key == owner_key:
                    return None
                return await original_guard(
                    key, expected_storage_revision=expected_storage_revision,
                )

            with monkeypatch.context() as patch:
                patch.setattr(transaction, "guard_record", guard)
                await archive.sync_projection_in_transaction(
                    transaction, run, events=(), checkpoints=(), interactions=(update,),
                    execution_id="execution", producer_generation=1,
                )

        with pytest.raises(AIError) as raised:
            await archive.state_store.mutate(project)
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        assert await archive.list_model_interactions(agent_run_id="run") == [current]
        assert await repository.get_history_head("execution", tenant_id="tenant") == before
        await archive.sync_prepared_projection(
            run, events=(), checkpoints=(), interactions=(update,),
            execution_id="execution", producer_generation=retry_generation,
        )
        assert await archive.list_model_interactions(agent_run_id="run") == [update]
        assert await archive.model_interaction_count(agent_run_id="run") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("invalid_kind", ("type", "foreign_owner", "unencodable_owner"))
async def test_invalid_execution_interaction_preserves_errors_and_rolls_back(
    tmp_path: Path, backend: str, invalid_kind: str,
) -> None:
    async with _archive(tmp_path, backend) as (state, archive, run):
        invalid = (
            object() if invalid_kind == "type"
            else replace(_running(), agent_run_id="other" if invalid_kind == "foreign_owner" else object())
        )
        repository = state.execution.executions
        before = await repository.get_history_head("execution", tenant_id="tenant")
        error_type = TypeError if invalid_kind == "type" and backend == "memory" else AIError
        with pytest.raises(error_type) as raised:
            await archive.sync_prepared_projection(
                run, events=(), checkpoints=(), interactions=(invalid,),
                execution_id="execution", producer_generation=1,
            )
        if error_type is AIError:
            assert raised.value.code is (
                ErrorCode.INTERNAL_ERROR if invalid_kind == "type" else ErrorCode.STORAGE_INTEGRITY_ERROR
            )
        assert await archive.get_agent_run(agent_run_id="run") is None
        assert await archive.model_interaction_count(agent_run_id="run") == 0
        assert await repository.get_history_head("execution", tenant_id="tenant") == before
