#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One execution row owns dispatch, settlement and producer takeover."""
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.codec import decode_domain, encode_domain
from linktools.ai.execution import ExecutionRecord, ExecutionRecoveryDetail, PendingToolContinuation, PendingDeferredCall
from linktools.ai.storage import StoredPayload
from .test_step_projection import _execution


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ('memory', 'sqlite'))
async def test_admission_and_takeover_atomically_replace_exact_producer(tmp_path: Path, backend: str) -> None:
    storage = PersistenceStorage.in_memory() if backend == 'memory' else PersistenceStorage.sqlite(tmp_path / 'control.db')
    async with Persistence.open('producer-control', tenant_id='tenant', storage=storage) as state:
        repo = state.execution.executions
        initial = replace(_execution(), revision=1, agent_run_seq=0)
        await repo.create_execution(initial)
        assert decode_domain(encode_domain(initial), ExecutionRecord) == initial
        head = await repo.get('execution', tenant_id='tenant')
        first = await repo.claim_next_agent_run('execution', tenant_id='tenant', expected_revision=1,
                                               expected_agent_run_seq=0, producer_claim_id='owner-1')
        assert (first.producer_generation, first.producer_claim_id) == (2, 'owner-1')
        assert head == initial
        assert first.revision == initial.revision + 1 and first.agent_run_seq == 1
        assert await repo.get('execution', tenant_id='tenant') == first

        async def aborted(tx):
            await repo.claim_next_agent_run_in_transaction(tx, 'execution', tenant_id='tenant',
                expected_revision=2, expected_agent_run_seq=1, producer_claim_id='aborted')
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        with pytest.raises(AIError):
            await repo.state_store.mutate(aborted)
        assert await repo.get('execution', tenant_id='tenant') == first
        second = await repo.claim_next_agent_run('execution', tenant_id='tenant', expected_revision=2,
                                                expected_agent_run_seq=1, producer_claim_id='owner-2')
        assert (second.producer_generation, second.producer_claim_id) == (3, 'owner-2')
        with pytest.raises(AIError) as stale:
            await repo.state_store.mutate(lambda tx: repo.require_execution_producer_in_transaction(
                tx, 'execution', expected_producer_generation=2))
        assert stale.value.code is ErrorCode.STORAGE_CONFLICT
        await repo.state_store.mutate(lambda tx: repo.require_execution_producer_in_transaction(
            tx, 'execution', expected_producer_generation=3))
        assert (await repo.get('execution', tenant_id='tenant')).producer_claim_id == 'owner-2'


@pytest.mark.asyncio
@pytest.mark.parametrize('status', (ExecutionStatus.STARTED, ExecutionStatus.CANCELLING,
    ExecutionStatus.FINALIZING, ExecutionStatus.WAITING_DEFERRED, ExecutionStatus.CANCELLED,
    ExecutionStatus.RECOVERY_REQUIRED))
@pytest.mark.parametrize('settlement', (False, True))
async def test_source_control_status_distinguishes_dispatch_and_settlement(tmp_path: Path, status, settlement: bool) -> None:
    async with Persistence.open('producer-control', tenant_id='tenant', storage=PersistenceStorage.sqlite(tmp_path / 'control.db')) as state:
        repo = state.execution.executions
        await repo.create_execution(replace(_execution(), revision=4, producer_generation=2,
                                                    producer_claim_id='owner', status=status, recovery=(
            ExecutionRecoveryDetail(agent_run_id='run', pending_tools=PendingToolContinuation('run', calls=(
                PendingDeferredCall('call', 'tool', StoredPayload.inline_json({})),
            ))) if status is ExecutionStatus.WAITING_DEFERRED else None)))
        statements = []
        engine = repo.state_store.storage_group.context.engine.sync_engine
        def executed(_connection, _cursor, statement, _parameters, _context, _many):
            if statement.lstrip().upper().startswith('SELECT'):
                statements.append(statement)
        event.listen(engine, 'after_cursor_execute', executed)
        allowed = status is ExecutionStatus.STARTED or settlement and status in {ExecutionStatus.CANCELLING, ExecutionStatus.FINALIZING}
        try:
            operation = repo.state_store.mutate(lambda tx: repo.require_execution_producer_in_transaction(
                tx, 'execution', expected_producer_generation=2, allow_cancelling=settlement))
            if allowed:
                value, _receipt = await operation
                assert value.producer_claim_id == 'owner'
            else:
                with pytest.raises(AIError):
                    await operation
            assert len(statements) == 1
        finally:
            event.remove(engine, 'after_cursor_execute', executed)


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ('memory', 'sqlite'))
async def test_recovery_required_revokes_capture_without_closing_execution(tmp_path: Path, backend: str) -> None:
    from linktools.ai.persistence import PersistenceDomain

    storage = PersistenceStorage.in_memory() if backend == 'memory' else PersistenceStorage.sqlite(tmp_path / 'control.db')
    async with Persistence.open('producer-control', tenant_id='tenant', storage=storage) as state:
        repo = state.execution.executions
        current = replace(_execution(), revision=4, producer_generation=2, producer_claim_id='owner',
                          status=ExecutionStatus.RECOVERY_REQUIRED)
        await repo.create_execution(current)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        await archive.require_revoked_execution_producer(execution_id='execution', producer_generation=2)
        with pytest.raises(AIError) as invalid:
            await archive.require_revoked_execution_producer(execution_id='execution', producer_generation=3)
        assert invalid.value.code is ErrorCode.STORAGE_CONFLICT
        assert await repo.get('execution', tenant_id='tenant') == current


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ('memory', 'sqlite'))
@pytest.mark.parametrize('change', ('takeover', 'cancel'))
async def test_old_trace_writer_racing_source_control_cannot_dispatch(tmp_path: Path, backend: str, change: str) -> None:
    import asyncio
    from linktools.ai.persistence import PersistenceDomain
    from linktools.ai.persistence._step_contracts import AgentRunRecord

    storage = PersistenceStorage.in_memory() if backend == 'memory' else PersistenceStorage.sqlite(tmp_path / 'race.db')
    async with Persistence.open('producer-race', tenant_id='tenant', storage=storage) as state:
        repo = state.execution.executions
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        initial = replace(_execution(), revision=2, producer_generation=2, producer_claim_id='old')
        await repo.create_execution(initial)
        captured, changed = asyncio.Event(), asyncio.Event()

        async def old_writer():
            observed = await repo.get('execution', tenant_id='tenant')
            captured.set()
            await changed.wait()
            async def publish(group):
                transaction = group.transaction(archive.state_store)
                await archive.guard_execution_producer(transaction, execution_id='execution',
                    producer_generation=observed.producer_generation, allow_cancelling=False)
                await archive.sync_projection_in_transaction(transaction, AgentRunRecord('old-run'),
                    events=(), checkpoints=(), execution_id='execution',
                    producer_generation=observed.producer_generation)
            with pytest.raises(AIError) as rejected:
                await repo.state_store.storage_group.mutate((repo.state_store, archive.state_store), publish)
            assert rejected.value.code is (ErrorCode.STORAGE_CONFLICT if change == 'takeover' else ErrorCode.EXECUTION_CANCELLED)

        async def control():
            await captured.wait()
            if change == 'takeover':
                successor = await repo.claim_next_agent_run('execution', tenant_id='tenant', expected_revision=2,
                    expected_agent_run_seq=1, producer_claim_id='new')
                assert successor.producer_claim_id == 'new'
            else:
                await repo.compare_and_swap('execution', tenant_id='tenant', expected_revision=2,
                    next_record=replace(initial, revision=3, status=ExecutionStatus.CANCELLING))
            changed.set()

        await asyncio.wait_for(asyncio.gather(old_writer(), control()), 10)
        assert await archive.get_agent_run(agent_run_id='old-run') is None
        current = await repo.get('execution', tenant_id='tenant')
        assert current.producer_claim_id == ('new' if change == 'takeover' else 'old')
