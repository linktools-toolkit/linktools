#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Successful CAS receipts retain the actual physical owner revision."""

from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage

from .test_history_projection_conformance import _record


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ('memory', 'sqlite'))
async def test_cas_receipt_matches_committed_owner_and_outer_rollback(tmp_path: Path, backend: str) -> None:
    storage = (PersistenceStorage.in_memory() if backend == 'memory'
               else PersistenceStorage.sqlite(tmp_path / 'receipt.db'))
    async with Persistence.open('receipt', tenant_id='tenant', storage=storage) as state:
        repository = state.execution.executions
        initial = _record(ExecutionStatus.STARTED, 0)
        await repository.create_execution(initial)
        records = await repository.state_store.read(lambda tx: tx.scan_records())
        owner, = [value for value in records if value.kind == 'execution']
        # A physical guard advances storage revision without a logical state update.
        await repository.state_store.mutate(lambda tx: tx.guard_record_revision(
            owner.key, expected_storage_revision=owner.storage_revision))
        candidate = replace(initial, revision=1)
        receipt = await repository.compare_and_swap_observed(
            initial.execution_id, tenant_id='tenant', expected_revision=0, next_record=candidate,
        )
        assert receipt == await repository.get_observed(initial.execution_id, tenant_id='tenant')
        assert receipt.storage_revision == owner.storage_revision + 2
        assert receipt.storage_revision != candidate.revision
        with pytest.raises(AIError) as caught:
            await repository.compare_and_swap_observed(
                initial.execution_id, tenant_id='tenant', expected_revision=0, next_record=candidate,
            )
        assert caught.value.code is ErrorCode.STORAGE_CONFLICT

        async def abort(transaction):
            await repository.compare_and_swap_observed(
                initial.execution_id, tenant_id='tenant', expected_revision=1,
                next_record=replace(candidate, revision=2),
            )
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

        with pytest.raises(AIError) as aborted:
            await repository.state_store.mutate(abort)
        assert aborted.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
        assert await repository.get_observed(initial.execution_id, tenant_id='tenant') == receipt


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ('unknown', 'replaced-owner'))
async def test_input_context_does_not_publish_unconfirmed_or_retired_owner(outcome: str, monkeypatch) -> None:
    from types import SimpleNamespace
    from linktools.ai.execution import ExecutionInputContext
    from linktools.ai.persistence import PersistenceDomain
    from linktools.ai.persistence.contracts import RuntimePayloadRef
    from linktools.ai.runtime._local import LocalExecutionBackend
    from linktools.ai.storage import StoredPayload

    async with Persistence.open('receipt', tenant_id='tenant', storage=PersistenceStorage.in_memory()) as state:
        repository = state.execution.executions
        initial = _record(ExecutionStatus.STARTED, 0)
        await repository.create_execution(initial)
        observed = await repository.get_observed(initial.execution_id, tenant_id='tenant')
        owner = SimpleNamespace(observed=observed)
        backend = LocalExecutionBackend.__new__(LocalExecutionBackend)
        backend._execution = SimpleNamespace(executions=repository)
        backend._tenant_id = 'tenant'
        backend._turn_owners = {initial.execution_id: owner}
        context = ExecutionInputContext.from_messages(())
        reference = RuntimePayloadRef(StoredPayload.inline_json(context.to_payload()), PersistenceDomain.EXECUTION)

        async def publish():
            return None

        async def prepare(*args, **kwargs):
            return SimpleNamespace(reference=reference, publish=publish)

        backend._context_archive = SimpleNamespace(prepare=prepare)
        original = repository.state_store.mutate
        depth = 0

        async def mutate(callback):
            nonlocal depth
            depth += 1
            try:
                result = await original(callback)
                if depth == 1:
                    if outcome == 'unknown':
                        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
                    backend._turn_owners[initial.execution_id] = SimpleNamespace(observed=None)
                return result
            finally:
                depth -= 1

        monkeypatch.setattr(repository.state_store, 'mutate', mutate)
        if outcome == 'unknown':
            with pytest.raises(AIError) as caught:
                await backend._persist_input_context(initial, context, owner=owner)
            assert caught.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        else:
            await backend._persist_input_context(initial, context, owner=owner)
        assert owner.observed is observed
        assert (await repository.get_observed(initial.execution_id, tenant_id='tenant')).value.input_context == reference
