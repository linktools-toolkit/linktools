#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Imported context roots publish atomically with their admission owner."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart

from linktools.ai.core import ExecutionStatus, IdempotencyStatus, ResourceKind
from linktools.ai.execution import ExecutionStartReservation, IdempotencyRecord
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionInputContext
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.inputs import ExecutionContextArchive
from linktools.ai.persistence import PersistenceDomain, Persistence

from .test_history_projection_conformance import _record


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ['memory', 'filesystem', 'sqlite'])
@pytest.mark.parametrize('abort', [False, True])
async def test_imported_context_admission_commits_or_rolls_back_one_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, abort: bool,
) -> None:
    storage = (PersistenceStorage.in_memory() if backend == 'memory' else
               PersistenceStorage.filesystem(tmp_path / 'state') if backend == 'filesystem' else
               PersistenceStorage.sqlite(tmp_path / 'state.db'))
    state = Persistence(storage)
    await state.initialize(namespace='imported-context', tenant_id='tenant')
    try:
        repository = state.execution.executions
        archive = ExecutionContextArchive(
            repository.state_store, state.archive_object_store(PersistenceDomain.EXECUTION),
            namespace='imported-context', tenant_id='tenant',
        )
        context = ExecutionInputContext.from_messages(
            (ModelRequest(parts=[UserPromptPart('accepted history')]),),
            session_metadata={'note': 'accepted metadata'},
        )
        write = await archive.prepare(context)
        execution = replace(_record(ExecutionStatus.PENDING_START, 0),
                            input_context=write.reference, context_imported=True)
        identity = IdempotencyRecord(
            scope='test', idempotency_key_digest='a' * 64, request_digest='b' * 64,
            resource_kind=ResourceKind.EXECUTION, resource_id=execution.execution_id,
            status=IdempotencyStatus.RESERVED, result_digest=None, error_code=None,
            created_at=execution.created_at, updated_at=execution.updated_at,
        )
        reservation = ExecutionStartReservation(execution, identity)
        original_publish = write.archive.publish

        async def fail_after_content(transaction, prepared):
            await original_publish(transaction, prepared)
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        if abort:
            monkeypatch.setattr(write.archive, 'publish', fail_after_content)
            with pytest.raises(AIError) as raised:
                await repository.reserve_start(reservation, input_context_write=(write.archive, write.records))
            assert raised.value.code is ErrorCode.STORAGE_CONFLICT
            assert await repository.get(execution.execution_id, tenant_id='tenant') is None
            records = await write.archive.store.read(lambda tx: tx.get_records(tuple(write.records.records)))
            assert records == {}
        else:
            result = await repository.reserve_start(reservation, input_context_write=(write.archive, write.records))
            assert result.created
            assert (await archive.load(result.execution.input_context)).digest == context.digest
            monkeypatch.setattr(write.archive, 'publish', fail_after_content)
            replay = await repository.reserve_start(reservation, input_context_write=(write.archive, write.records))
            assert not replay.created and replay.execution == result.execution
    finally:
        await state.close()
