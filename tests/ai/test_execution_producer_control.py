#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One execution row owns dispatch, settlement and producer takeover."""
import asyncio
from collections.abc import Callable, Sequence
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
                PendingDeferredCall('call', 'tool', StoredPayload.inline_json({}), operation_id="test-operation"),
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


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_producer_authority_never_hydrates_large_execution_bodies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    from linktools.ai.core import StopReason, UsageMetrics
    from linktools.ai.execution import ResultRecord
    from linktools.ai.persistence import PersistenceDomain
    from linktools.ai.persistence._archive import ContentArchive
    from linktools.ai.persistence._archive_closure import iter_archive_content_refs
    from linktools.ai.persistence.contracts import ExecutionHistorySealRecord
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "bodies.db")
    async with Persistence.open("producer-control", tenant_id="tenant", storage=storage) as state:
        repository = state.execution.executions
        active = replace(_execution(), producer_generation=1, producer_claim_id="producer",
            binding=replace(_execution().binding, output_schema={"type": "string", "description": "schema " * 20000}),
            stored_user_input=replace(_execution().stored_user_input, payload=StoredPayload.inline_text("input " * 20000)))
        seal = ExecutionHistorySealRecord("terminal", (), 0)
        result = ResultRecord(StoredPayload.inline_json({"text": "output " * 20000}), StopReason.END_TURN, UsageMetrics(), active.created_at)
        terminal = replace(active, execution_id="terminal", root_execution_id="terminal", status=ExecutionStatus.SUCCEEDED,
            history_seal_digest=seal.seal_digest, result=result).finalize_terminal(result)
        await repository.create_execution(active)
        await repository.create_execution(terminal)
        await repository.state_store.mutate(lambda tx: repository.put_history_seal_in_transaction(tx, seal))
        row = await repository.state_store.read_record(repository.execution_record_key("terminal"))
        assert len(tuple(iter_archive_content_refs(row.data))) == 5
        read_contents = ContentArchive.read_contents
        async def reject_body(archive: ContentArchive, identities: Sequence[str], *, codec: str,
                              reserve: Callable[[str, int], None] | None = None) -> tuple[bytes, ...]:
            assert codec == "runtime-policy", "producer authority must not hydrate execution bodies"
            return await read_contents(archive, identities, codec=codec, reserve=reserve)
        monkeypatch.setattr(ContentArchive, "read_contents", reject_body)
        state.run_store.bind_execution_producer("active-run", execution_id="execution", producer_generation=1)
        await state.run_store.require_active_producer("active-run")
        state.run_store.bind_execution_producer("terminal-run", execution_id="terminal", producer_generation=1)
        with pytest.raises(AIError) as inactive:
            await state.run_store.require_active_producer("terminal-run")
        assert inactive.value.code is ErrorCode.STORAGE_CONFLICT
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        await archive.require_revoked_execution_producer(execution_id="terminal", producer_generation=1, terminal_execution=terminal)


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", (
    {"producer_generation": True}, {"producer_generation": -1},
    {"producer_claim_id": ""}, {"producer_generation": 0, "producer_claim_id": "claim"},
    {"durable_trace": 1}, {"history_seal_digest": "bad"},
    {"durable_trace": False, "history_seal_digest": "a" * 64},
    {"execution_id": "foreign"}, {"_kind": "unexpected"},
))
async def test_producer_control_reads_reject_malformed_source_metadata(tmp_path: Path, changes: dict) -> None:
    from copy import deepcopy
    async with Persistence.open("producer-control", tenant_id="tenant", storage=PersistenceStorage.in_memory()) as state:
        repository = state.execution.executions
        await repository.create_execution(replace(_execution(), producer_generation=1, producer_claim_id="claim"))
        async def corrupt(transaction):
            key = repository.execution_record_key("execution")
            record = await transaction.get_record(key)
            data = deepcopy(dict(record.data))
            data["value"]["payload"]["fields"].update({key: item for key, item in changes.items() if key != "_kind"})
            await transaction.delete_record(key)
            await transaction.insert_record(replace(record, kind=changes.get("_kind", record.kind), data=data))
        await repository.state_store.mutate(corrupt)
        for operation in (
            lambda: repository.require_active_producer("execution", tenant_id="tenant", expected_producer_generation=1, fresh=True),
            lambda: repository.require_revoked_producer("execution", tenant_id="tenant", producer_generation=0),
        ):
            with pytest.raises(AIError) as invalid:
                await operation()
            assert invalid.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_missing_custom_producer_control_implementation_cannot_grant_authority() -> None:
    from linktools.ai.persistence.contracts import ExecutionRepository
    class MissingControl(ExecutionRepository):
        pass
    repository = MissingControl()
    with pytest.raises(NotImplementedError):
        await repository.require_active_producer("execution", tenant_id="tenant", expected_producer_generation=1)
    with pytest.raises(NotImplementedError):
        await repository.require_revoked_producer("execution", tenant_id="tenant", producer_generation=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", (
    ("execution_id", " "), ("execution_id", 42),
    ("producer_generation", True), ("producer_claim_id", ""),
    ("producer_claim_id", 42), ("budget_scope_id", " "),
    ("budget_scope_id", 42), ("retention_closed", 1),
))
async def test_publication_control_projection_rejects_corrupt_metadata(field, value) -> None:
    from copy import deepcopy
    async with Persistence.open("producer-control", tenant_id="tenant", storage=PersistenceStorage.in_memory()) as state:
        repository = state.execution.executions
        await repository.create_execution(replace(_execution(), producer_generation=1, producer_claim_id="claim"))
        async def corrupt(transaction):
            key = repository.execution_record_key("execution")
            record = await transaction.get_record(key)
            data = deepcopy(dict(record.data))
            data["value"]["payload"]["fields"][field] = value
            await transaction.delete_record(key)
            await transaction.insert_record(replace(record, data=data))
        await repository.state_store.mutate(corrupt)
        with pytest.raises(AIError) as invalid:
            await repository.state_store.mutate(lambda tx: repository.require_execution_producer_in_transaction(
                tx, "execution", expected_producer_generation=1))
        assert invalid.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_reopened_mutation_enlists_existing_binding_before_callback(tmp_path: Path) -> None:
    storage = PersistenceStorage.sqlite(tmp_path / "reopened-control.db")
    initial = replace(_execution(), producer_generation=1, producer_claim_id="claim")
    async with Persistence.open("producer-control", tenant_id="tenant", storage=storage) as state:
        await state.execution.executions.create_execution(initial)
    async with Persistence.open("producer-control", tenant_id="tenant", storage=storage) as state:
        repository = state.execution.executions
        assert repository.state_store.storage_group.opened_archive is None
        calls = 0
        async def read_full(transaction):
            nonlocal calls
            calls += 1
            return await repository.get_in_transaction(transaction, "execution", tenant_id="tenant")
        assert await repository.state_store.mutate(read_full) == initial
        assert calls == 1


@pytest.mark.asyncio
async def test_first_mutation_enlists_a_competing_writers_archive(tmp_path: Path, monkeypatch) -> None:
    from linktools.ai.persistence.state import STATE_GROUP_ARCHIVE_KEY
    storage = PersistenceStorage.sqlite(tmp_path / "first-writer.db")
    async with Persistence.open("producer-control", tenant_id="tenant", storage=storage) as reader:
        repository = reader.execution.executions
        group = repository.state_store.storage_group
        open_partition = group._open_partition
        published = False
        async def publish_before_snapshot(key):
            nonlocal published
            handle = await open_partition(key)
            if key == STATE_GROUP_ARCHIVE_KEY and not published:
                published = True
                async with Persistence.open("producer-control", tenant_id="tenant", storage=storage) as writer:
                    await writer.execution.executions.create_execution(_execution())
            return handle
        monkeypatch.setattr(group, "_open_partition", publish_before_snapshot)
        calls = 0
        async def read_full(transaction):
            nonlocal calls
            calls += 1
            return await repository.get_in_transaction(transaction, "execution", tenant_id="tenant")
        result = await repository.state_store.mutate(read_full)
        assert published and calls == 1
        assert result.execution_id == "execution"


@pytest.mark.asyncio
async def test_first_scope_identity_is_published_only_after_confirmed_commit(tmp_path: Path) -> None:
    storage = PersistenceStorage.sqlite(tmp_path / "scope-commit.db")
    async with Persistence.open("producer-control", tenant_id="tenant", storage=storage) as state:
        store = state.execution.executions.state_store
        group = store.storage_group
        async def abort(_transaction):
            assert group.opened_archive is None
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        with pytest.raises(AIError):
            await store.mutate(abort)
        assert group.opened_archive is None
        await store.mutate(lambda _transaction: asyncio.sleep(0))
        assert group.opened_archive is not None
