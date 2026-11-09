#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired ownership and cost contracts at real durable writer/reader boundaries.

One persistence lifecycle per size exercises the same archive operations used by
Runtime. Provider execution and repeated Runtime startup are deliberately absent:
this measures message preparation, SQL publication and cold materialization, not
model latency. CPU and allocation observations are diagnostics, not timing gates.
"""

import json
import time
import tracemalloc
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord
from sqlalchemy import event, text

from linktools.ai.execution import ExecutionInputContext
from linktools.ai.inputs import ExecutionContextArchive
from linktools.ai.inputs import _context_archive
from linktools.ai.persistence import NativeMessageRoot, Persistence, PersistenceDomain, PersistenceStorage, SqlStepStore
from linktools.ai.persistence import _transcript_staging
from linktools.ai.persistence._archive import ContentRecord, PreparedArchive
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint, AgentRunRecord
from linktools.ai.persistence._step_message_content import NativeMessageContent
from linktools.ai.persistence.state import RecordQuery

from .test_step_projection import _execution


class _Traffic:
    def __init__(self) -> None:
        self.statements: Counter[str] = Counter()
        self.rows = 0
        self.returned_body_bytes = 0
        self.encodings = 0
        self.preparations = 0
        self.encoded_bytes = 0
        self.hash_input_bytes = 0
        self.context_hash_input_bytes = 0
        self.native_bind_bytes = 0

    def sql(self, connection, cursor, statement, parameters, context, many) -> None:
        operation = statement.lstrip().split(None, 1)[0].upper()
        # Recursive sequence joins start WITH, not SELECT.
        self.statements['SELECT' if operation == 'WITH' else operation] += 1
        if operation in ('INSERT', 'UPDATE', 'DELETE') and any(
            table in statement.lower() for table in ('ai_step_messages', 'ai_step_sequences')
        ):
            # JSON size is a reproducible bind-payload accounting unit, not a
            # claim about driver framing or database protocol wire bytes.
            self.native_bind_bytes += len(json.dumps(parameters, default=str).encode('utf-8'))

    def observe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        encode = _transcript_staging.encode_model_messages
        prepare = SqlStepStore._prepare_message_payload
        hydrate = NativeMessageContent.hydrate_ranges
        digest = _transcript_staging.MessageSnapshot.encoded_digest
        context_sha256 = _context_archive.hashlib.sha256

        def encoded(messages):
            self.encodings += len(messages)
            result = encode(messages)
            self.encoded_bytes += len(result)
            return result

        def hashed(snapshot):
            result = digest(snapshot)
            self.hash_input_bytes += result[1]
            return result

        def context_hashed(data=b'', **kwargs):
            self.context_hash_input_bytes += len(data)
            return context_sha256(data, **kwargs)

        async def prepared(store, raw):
            self.preparations += 1
            return await prepare(store, raw)

        async def hydrated(owner, roots, rows, **kwargs):
            # Observe the rows actually materialized by the scoped join, rather
            # than estimating returned traffic from a standalone table scan.
            self.rows += len(rows)
            self.returned_body_bytes += sum(
                len(row['message'].encode('utf-8'))
                for row in rows if isinstance(row['message'], str)
            )
            return await hydrate(owner, roots, rows, **kwargs)

        monkeypatch.setattr(_transcript_staging, 'encode_model_messages', encoded)
        monkeypatch.setattr(_transcript_staging.MessageSnapshot, 'encoded_digest', hashed)
        monkeypatch.setattr(_context_archive, 'hashlib', SimpleNamespace(sha256=context_hashed))
        monkeypatch.setattr(SqlStepStore, '_prepare_message_payload', prepared)
        monkeypatch.setattr(NativeMessageContent, 'hydrate_ranges', hydrated)

    def report(self, record_property, phase: str, cpu: int, peak: int) -> None:
        record_property(phase, json.dumps({
            'sql': dict(self.statements), 'materialized_rows': self.rows,
            'returned_body_utf8_bytes': self.returned_body_bytes,
            'encoded_messages': self.encodings, 'prepared_payloads': self.preparations,
            'encoded_bytes': self.encoded_bytes, 'snapshot_digest_input_bytes': self.hash_input_bytes,
            'context_history_digest_input_bytes': self.context_hash_input_bytes,
            'native_bind_json_bytes': self.native_bind_bytes,
            'process_cpu_ns': cpu, 'traced_peak_bytes': peak,
        }, sort_keys=True))


async def _native_counts(engine) -> tuple[int, int, int]:
    async with engine.connect() as connection:
        bodies = (await connection.execute(text('SELECT message FROM ai_step_messages'))).scalars().all()
        nodes = await connection.scalar(text('SELECT count(*) FROM ai_step_sequences'))
    return len(bodies), nodes, sum(len(body.encode('utf-8')) for body in bodies)


@pytest.mark.asyncio
@pytest.mark.parametrize('size', (10, 100, 1000))
async def test_shared_owner_writes_and_cold_reads_have_linear_message_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record_property, size: int,
) -> None:
    path = tmp_path / 'owners.db'
    namespace, tenant = 'ownership-cost', 'tenant'
    messages = tuple(ModelRequest(parts=[UserPromptPart(
        f'body-{index:04d}', timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )], timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc)) for index in range(size))
    context = ExecutionInputContext.from_messages(messages)
    snapshot = ContinuableSnapshot(run_id='run', step_index=size, idempotency_key='complete', messages=list(messages))
    state = Persistence(PersistenceStorage.sqlite(path))
    await state.initialize(namespace=namespace, tenant_id=tenant)
    traces = [state.run_store.read_store(domain) for domain in (
        PersistenceDomain.TRACE, PersistenceDomain.CONVERSATION,
    )]
    store = state.execution.executions.state_store
    engine = store.storage_group.context.engine
    native = SqlStepStore.for_state_store(store, namespace=namespace, tenant_id=tenant)
    assert native is not None
    state.run_store.bind_step_store(native)
    service = ExecutionContextArchive(store, state.archive_object_store(PersistenceDomain.EXECUTION),
                                      namespace=namespace, tenant_id=tenant)
    # Provision/open metadata is outside the steady-state message budget.
    for archive in traces:
        await archive.transcript_repository.content_archive(create=True)
    await service._open(create=True)
    registered = RunRecord(run_id='run', registration_id='registered')
    await native.register_run(registered)
    for archive in traces:
        archive.confirm_native_run(registered)
    traffic = _Traffic()
    traffic.observe(monkeypatch)
    event.listen(engine.sync_engine, 'before_cursor_execute', traffic.sql)
    tracing_started = not tracemalloc.is_tracing()
    if tracing_started:
        tracemalloc.start()
    started = time.process_time_ns()
    try:
        await native.save_snapshot(snapshot)
        assert traffic.statements['SELECT'] == 0
        roots = []
        for archive in traces:
            repository = archive.transcript_repository
            bundle = PreparedArchive()
            before = traffic.statements['SELECT']
            projection = await repository.prepare_model_projection(messages, bundle)
            assert traffic.statements['SELECT'] == before
            root = projection.items[0].root
            assert isinstance(root, NativeMessageRoot)
            roots.append(root)

            async def publish(transaction):
                await repository.store_projection(transaction, 'projection', projection)
                await repository.publish_native_archive(transaction, bundle, owner_written=True)

            await archive.state_store.mutate(publish)
            # Existing metadata read for projection insertion; native publication
            # must not add a message-existence SELECT on this hot path.
            assert traffic.statements['SELECT'] == before + 1

        before = traffic.statements['SELECT']
        accepted = await service.prepare(context)
        assert traffic.statements['SELECT'] == before
        roots.append(NativeMessageRoot.from_json(accepted.reference.payload.decode()['history']))

        async def admit(transaction):
            await accepted.publish()
            await state.execution.executions.create_execution_in_transaction(
                transaction, replace(_execution(), input_context=accepted.reference),
            )

        await store.mutate(admit)
        assert traffic.statements['SELECT'] == before + 1
        assert roots[0] == roots[1] == roots[2]
        assert 0 < traffic.encodings <= 4 * size
        assert 0 < traffic.preparations <= 2 * size
        assert 0 < sum(traffic.statements[op] for op in ('INSERT', 'UPDATE', 'DELETE')) <= 4 * size + 4
        traffic.report(record_property, 'shared_write', time.process_time_ns() - started,
                       tracemalloc.get_traced_memory()[1])
        event.remove(engine.sync_engine, 'before_cursor_execute', traffic.sql)
        body_count, node_count, stored_bytes = await _native_counts(engine)
        assert body_count == node_count == size
        record_property('stored_message_utf8_bytes', stored_bytes)
        assert 0 < stored_bytes <= size * 2048

        # Exercise complete current archive writers as well as their isolated
        # projection budget. Chunked transcript roots may add sequence nodes,
        # but cannot acquire a second durable message-body owner.
        traffic.statements.clear()
        traffic.encodings = traffic.preparations = 0
        traffic.encoded_bytes = traffic.hash_input_bytes = traffic.context_hash_input_bytes = traffic.native_bind_bytes = 0
        event.listen(engine.sync_engine, 'before_cursor_execute', traffic.sql)
        started = time.process_time_ns()
        checkpoint = AgentRunCheckpoint('run', size, list(messages), transcript_message_count_before=0)
        for archive in traces:
            await archive.materialize_checkpoint(
                AgentRunRecord('run'), checkpoint,
                execution_id='execution' if archive.runtime_domain is PersistenceDomain.TRACE else None,
            )
        traffic.report(record_property, 'checkpoint_publication', time.process_time_ns() - started,
                       tracemalloc.get_traced_memory()[1])
        event.remove(engine.sync_engine, 'before_cursor_execute', traffic.sql)
        assert (await _native_counts(engine))[0] == size
        content_archive = await traces[0].transcript_repository.content_archive()
        records = await content_archive.store.read(lambda tx: tx.list_records(RecordQuery(kind='archive_content')))
        assert not any(ContentRecord.from_json(record.data).codec == 'model-message' for record in records)
    finally:
        if event.contains(engine.sync_engine, 'before_cursor_execute', traffic.sql):
            event.remove(engine.sync_engine, 'before_cursor_execute', traffic.sql)
        if tracing_started:
            tracemalloc.stop()
        await state.close()

    # New persistence/context/engine objects, with no writer cache surviving.
    cold = Persistence(PersistenceStorage.sqlite(path))
    await cold.initialize(namespace=namespace, tenant_id=tenant)
    cold_store = cold.execution.executions.state_store
    cold_engine = cold_store.storage_group.context.engine
    reader = SqlStepStore.for_state_store(cold_store, namespace=namespace, tenant_id=tenant)
    assert reader is not None
    cold.run_store.bind_step_store(reader)
    cold_service = ExecutionContextArchive(cold_store, cold.archive_object_store(PersistenceDomain.EXECUTION),
                                           namespace=namespace, tenant_id=tenant)
    cold_archives = [cold.run_store.read_store(domain) for domain in (
        PersistenceDomain.TRACE, PersistenceDomain.CONVERSATION,
    )]
    for archive in cold_archives:
        await archive.transcript_repository.content_archive()
    await cold_service._open()
    traffic.statements.clear()
    traffic.rows = traffic.returned_body_bytes = 0
    traffic.encodings = traffic.preparations = 0
    traffic.encoded_bytes = traffic.hash_input_bytes = traffic.context_hash_input_bytes = traffic.native_bind_bytes = 0
    tracing_started = not tracemalloc.is_tracing()
    if tracing_started:
        tracemalloc.start()
    event.listen(cold_engine.sync_engine, 'before_cursor_execute', traffic.sql)
    started = time.process_time_ns()
    try:
        assert await reader.latest_snapshot(run_id='run') == snapshot
        assert traffic.statements['SELECT'] == 2
        for archive in cold_archives:
            before = traffic.statements['SELECT']
            repository = archive.transcript_repository
            projection = await repository.load_projection('projection')
            assert projection is not None
            loaded = await repository.load_projected_context('projection', projection)
            assert loaded.model_messages() == messages
            assert traffic.statements['SELECT'] == before + 2
        before = traffic.statements['SELECT']
        execution = await cold.execution.executions.get('execution', tenant_id=tenant)
        assert execution is not None and execution.input_context is not None
        assert await cold_service.load(execution.input_context) == context
        assert traffic.statements['SELECT'] == before + 2
        assert traffic.rows == 4 * size
        assert traffic.returned_body_bytes == 4 * stored_bytes
        assert not any(traffic.statements[op] for op in ('INSERT', 'UPDATE', 'DELETE'))
        traffic.report(record_property, 'cold_read', time.process_time_ns() - started,
                       tracemalloc.get_traced_memory()[1])
        traffic.statements.clear()
        traffic.rows = traffic.returned_body_bytes = 0
        started = time.process_time_ns()
        for archive in cold_archives:
            recovered = await archive.latest_checkpoint(agent_run_id='run')
            assert recovered is not None and recovered.messages == list(messages)
            assert tuple([message async for message in archive.iter_messages(agent_run_id='run')]) == messages
        traffic.report(record_property, 'checkpoint_recovery', time.process_time_ns() - started,
                       tracemalloc.get_traced_memory()[1])
        empty = PreparedArchive()
        empty_root = await reader.prepare_archived_messages((), empty)
        before = traffic.statements['SELECT']
        assert await reader.read_message_roots((empty_root,)) == ((),)
        assert traffic.statements['SELECT'] == before
    finally:
        event.remove(cold_engine.sync_engine, 'before_cursor_execute', traffic.sql)
        if tracing_started:
            tracemalloc.stop()
        await cold.close()


@pytest.mark.asyncio
async def test_cold_native_media_object_is_read_by_snapshot_and_execution_context(
    tmp_path: Path,
) -> None:
    from pydantic_ai.messages import TextContent

    path = tmp_path / 'media.db'
    namespace, tenant = 'media-ownership', 'tenant'
    binary = bytes(range(256)) * 128
    message = ModelRequest(parts=[UserPromptPart([
        TextContent('evidence', metadata={'source': {'labels': ['retained']}}),
        BinaryContent(binary, media_type='application/octet-stream', identifier='evidence'),
    ])])
    snapshot = ContinuableSnapshot(run_id='media', step_index=1, idempotency_key='media', messages=[message])
    context = ExecutionInputContext.from_messages((message,))
    async with Persistence.open(namespace, storage=PersistenceStorage.sqlite(path), tenant_id=tenant) as state:
        store = state.execution.executions.state_store
        native = SqlStepStore.for_state_store(store, namespace=namespace, tenant_id=tenant)
        assert native is not None
        state.run_store.bind_step_store(native)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        await archive.transcript_repository.content_archive(create=True)
        await native.save_snapshot(snapshot)
        service = ExecutionContextArchive(store, state.archive_object_store(PersistenceDomain.EXECUTION),
                                          namespace=namespace, tenant_id=tenant)
        prepared = await service.prepare(context)

        async def publish(transaction):
            await prepared.publish()
            await state.execution.executions.create_execution_in_transaction(
                transaction, replace(_execution(), input_context=prepared.reference),
            )

        await store.mutate(publish)
        async with store.storage_group.context.engine.connect() as connection:
            bodies = (await connection.execute(text('SELECT message FROM ai_step_messages'))).scalars().all()
        assert len(bodies) == 1
        content = ContentRecord.from_json(json.loads(bodies[0]))
        assert content.payload.kind == 'object' and content.payload.ref is not None
        reference = content.payload.ref

    async with Persistence.open(namespace, storage=PersistenceStorage.sqlite(path), tenant_id=tenant) as cold:
        store = cold.execution.executions.state_store
        native = SqlStepStore.for_state_store(store, namespace=namespace, tenant_id=tenant)
        assert native is not None
        cold.run_store.bind_step_store(native)
        archive = cold.run_store.read_store(PersistenceDomain.TRACE)
        await archive.transcript_repository.content_archive()
        assert await native.latest_snapshot(run_id='media') == snapshot
        service = ExecutionContextArchive(store, cold.archive_object_store(PersistenceDomain.EXECUTION),
                                          namespace=namespace, tenant_id=tenant)
        execution = await cold.execution.executions.get('execution', tenant_id=tenant)
        assert execution is not None and execution.input_context is not None
        restored = await service.load(execution.input_context)
        assert restored == context
        assert restored.model_messages() == (message,)
        assert reference.size > 0


@pytest.mark.asyncio
@pytest.mark.parametrize('turns', (10, 100, 1000))
async def test_incremental_owner_chain_reuses_confirmed_prefixes_each_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record_property, turns: int,
) -> None:
    """Count cumulative work over N commits, not one batch containing N bodies."""
    from linktools.ai.model.messages import combine_model_message_payloads, decode_model_messages

    path = tmp_path / 'incremental-owners.db'
    namespace, tenant = 'incremental-ownership', 'tenant'
    state = Persistence(PersistenceStorage.sqlite(path))
    await state.initialize(namespace=namespace, tenant_id=tenant)
    store = state.execution.executions.state_store
    engine = store.storage_group.context.engine
    native = SqlStepStore.for_state_store(store, namespace=namespace, tenant_id=tenant)
    assert native is not None
    state.run_store.bind_step_store(native)
    archives = [state.run_store.read_store(domain) for domain in (
        PersistenceDomain.TRACE, PersistenceDomain.CONVERSATION,
    )]
    for archive in archives:
        await archive.transcript_repository.content_archive(create=True)
    service = ExecutionContextArchive(store, state.archive_object_store(PersistenceDomain.EXECUTION),
                                      namespace=namespace, tenant_id=tenant)
    await service._open(create=True)
    registered = RunRecord(run_id='growing', registration_id='growing-registration')
    await native.register_run(registered)
    for archive in archives:
        archive.confirm_native_run(registered)

    cold = Persistence(PersistenceStorage.sqlite(path))
    await cold.initialize(namespace=namespace, tenant_id=tenant)
    cold_store = cold.execution.executions.state_store
    cold_engine = cold_store.storage_group.context.engine
    reader = SqlStepStore(cold_store.context, namespace=namespace, tenant_id=tenant, max_cached_messages=0)
    cold.run_store.bind_step_store(reader)
    await cold.run_store.read_store(PersistenceDomain.TRACE).transcript_repository.content_archive()
    writer_traffic, reader_traffic = _Traffic(), _Traffic()
    writer_traffic.observe(monkeypatch)
    event.listen(engine.sync_engine, 'before_cursor_execute', writer_traffic.sql)
    event.listen(cold_engine.sync_engine, 'before_cursor_execute', reader_traffic.sql)
    tracing_started = not tracemalloc.is_tracing()
    if tracing_started:
        tracemalloc.start()
    started = time.process_time_ns()
    messages = []
    frozen_owner = _transcript_staging.MessageArchive()
    frozen = frozen_owner.snapshot(())
    baselines = [None, None]
    phase_encodings: Counter[str] = Counter()
    maximum_tail_rows = 0
    try:
        for turn in range(1, turns + 1):
            message = ModelRequest(parts=[UserPromptPart(
                f'turn-{turn:04d}', timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )], timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))
            messages.append(message)
            before = writer_traffic.encodings
            frozen = frozen_owner.concatenate(frozen, (message,))
            phase_encodings['delta_freeze'] += writer_traffic.encodings - before
            assert writer_traffic.encodings - before == 1
            snapshot = ContinuableSnapshot(run_id='growing', step_index=turn,
                                            idempotency_key=str(turn), messages=list(messages))
            before, reads = writer_traffic.encodings, writer_traffic.statements['SELECT']
            await native.save_snapshot(snapshot)
            phase_encodings['mutable_snapshot_freeze'] += writer_traffic.encodings - before
            assert writer_traffic.statements['SELECT'] == reads
            roots = []
            for index, archive in enumerate(archives):
                repository = archive.transcript_repository
                prepared = PreparedArchive()
                before, reads = writer_traffic.encodings, writer_traffic.statements['SELECT']
                projection = await repository.prepare_model_projection(
                    frozen, prepared, baseline=baselines[index],
                )
                phase_encodings['frozen_projection'] += writer_traffic.encodings - before
                assert writer_traffic.encodings == before
                assert writer_traffic.statements['SELECT'] == reads

                async def publish(transaction):
                    await repository.store_projection(transaction, 'growing', projection)
                    await repository.publish_native_archive(transaction, prepared, owner_written=True)

                await archive.state_store.mutate(publish)
                assert writer_traffic.statements['SELECT'] == reads + 1
                baselines[index] = projection
                roots.append(projection.items[0].root)
            assert roots[0] == roots[1]
            context = ExecutionInputContext(combine_model_message_payloads(frozen.encoded_payloads()))
            before, reads = writer_traffic.encodings, writer_traffic.statements['SELECT']
            accepted = await service.prepare(
                context, history_root=roots[0], history_digest=baselines[0].items[0].raw_digest,
            )
            phase_encodings['confirmed_context'] += writer_traffic.encodings - before
            assert writer_traffic.encodings == before
            assert writer_traffic.statements['SELECT'] == reads
            assert NativeMessageRoot.from_json(accepted.reference.payload.decode()['history']) == roots[0]
            execution_id = f'execution-{turn}'

            async def admit(transaction):
                await accepted.publish()
                await state.execution.executions.create_execution_in_transaction(
                    transaction, replace(_execution(), execution_id=execution_id,
                                         root_execution_id=execution_id, input_context=accepted.reference),
                )

            await store.mutate(admit)
            assert writer_traffic.statements['SELECT'] == reads + 1

            # Each committed tail is read through a separate physical connection
            # with the native proof cache disabled. Earlier bodies must not be
            # returned merely to decode the one new occurrence.
            read_count = reader_traffic.statements['SELECT']
            rows, body_bytes = writer_traffic.rows, writer_traffic.returned_body_bytes
            tail, = await reader.read_message_roots((roots[0].slice(turn - 1, turn),))
            assert tuple(decode_model_messages(tail[0])) == (message,)
            assert reader_traffic.statements['SELECT'] == read_count + 1
            assert writer_traffic.rows - rows <= 8 * turn.bit_length()
            assert 0 < writer_traffic.returned_body_bytes - body_bytes <= 2048
            maximum_tail_rows = max(maximum_tail_rows, writer_traffic.rows - rows)

        assert phase_encodings['delta_freeze'] == turns
        # Mutable SDK history must still be checked for mutations. Report this
        # separately from the zero-encode frozen projection/context reuse path.
        assert phase_encodings['mutable_snapshot_freeze'] <= turns * (turns + 1) // 2
        assert phase_encodings['frozen_projection'] == phase_encodings['confirmed_context'] == 0
        assert writer_traffic.statements['SELECT'] == 3 * turns
        assert reader_traffic.statements['SELECT'] == turns
        assert 0 < writer_traffic.preparations <= 2 * turns
        assert sum(writer_traffic.statements[op] for op in ('INSERT', 'UPDATE', 'DELETE')) <= 12 * turns
        writer_traffic.report(record_property, 'incremental_lifecycle', time.process_time_ns() - started,
                              tracemalloc.get_traced_memory()[1])
        record_property('incremental_encoding_phases', json.dumps(dict(phase_encodings), sort_keys=True))
        record_property('incremental_turns_completed', turns)
        record_property('incremental_max_cold_tail_rows', maximum_tail_rows)
        # A descriptor supplied after proof invalidation is not authority to
        # skip verification. The current writer must fall back to full freezing
        # without probing SQL or trusting a naked root from the caller.
        SqlStepStore.invalidate_context(store.context)
        before, reads = writer_traffic.encodings, writer_traffic.statements['SELECT']
        fallback = await service.prepare(
            context, history_root=roots[0], history_digest=baselines[0].items[0].raw_digest,
        )
        assert writer_traffic.encodings - before == turns
        assert writer_traffic.statements['SELECT'] == reads
        assert NativeMessageRoot.from_json(fallback.reference.payload.decode()['history']) == roots[0]
        record_property('unconfirmed_context_encoded_messages', writer_traffic.encodings - before)
        event.remove(engine.sync_engine, 'before_cursor_execute', writer_traffic.sql)
        bodies, nodes, stored_bytes = await _native_counts(engine)
        assert bodies == nodes == turns
        assert writer_traffic.native_bind_bytes <= 3 * stored_bytes + 2048 * turns
        record_property('incremental_stored_message_bytes', stored_bytes)
        content_archive = await archives[0].transcript_repository.content_archive()
        records = await content_archive.store.read(lambda tx: tx.list_records(RecordQuery(kind='archive_content')))
        assert not any(ContentRecord.from_json(record.data).codec == 'model-message' for record in records)
        # Recover the complete final value only once: N growing full reads would
        # themselves create a triangular cost unrelated to the write regression.
        assert await reader.latest_snapshot(run_id='growing') == snapshot
        restored_service = ExecutionContextArchive(
            cold_store, cold.archive_object_store(PersistenceDomain.EXECUTION),
            namespace=namespace, tenant_id=tenant,
        )
        execution = await cold.execution.executions.get(execution_id, tenant_id=tenant)
        assert execution is not None and execution.input_context is not None
        assert await restored_service.load(execution.input_context) == context
    finally:
        if event.contains(engine.sync_engine, 'before_cursor_execute', writer_traffic.sql):
            event.remove(engine.sync_engine, 'before_cursor_execute', writer_traffic.sql)
        event.remove(cold_engine.sync_engine, 'before_cursor_execute', reader_traffic.sql)
        if tracing_started:
            tracemalloc.stop()
        await cold.close()
        await state.close()
