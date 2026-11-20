#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Owner/content captures retain native admission counters in the same snapshot."""

import pytest
from sqlalchemy import event, update

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence._archive import archive_scope_key
from linktools.ai.persistence._run_counters import run_counter_streams
from linktools.ai.persistence.state import active_state_scope, record_key

from .test_native_run_references import _source, _raw_run, _flush


@pytest.mark.asyncio
@pytest.mark.parametrize('missing', (False, True))
@pytest.mark.parametrize('warm_owner', (False, True))
@pytest.mark.parametrize('requested', (False, True))
async def test_owner_capture_includes_exact_admission_counters(tmp_path, missing, warm_owner, requested):
    async with _source(tmp_path) as (state, _, _, _run):
        await _flush(state)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        source = await _raw_run(archive)
        store = archive.state_store
        key = record_key('native-run-facts', 'tenant', 'trace', 'agent_run', ('missing',)) if missing else source.key
        streams = run_counter_streams(key, store_key=store.store_key)
        expected = dict(zip(streams, (0, 0, 0) if missing else (3, 5, 7)))
        if not missing:
            await store.mutate(lambda tx: tx.reserve_sequences(expected))
            expected = await store.read(lambda tx: tx.get_sequences(streams))
        owner = state.execution.executions.state_store
        _, content = await owner.storage_group.open_archive(create=True)
        statements = []

        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            if statement.lstrip().upper().startswith(('SELECT', 'WITH')):
                statements.append(statement)

        async def read(group):
            if warm_owner:
                await group.transaction(owner).get_record(state.execution.executions.execution_record_key('execution'))
                statements.clear()
            await active_state_scope().record_reader.read_content_references(
                owner_store=owner, owner_keys=(state.execution.executions.execution_record_key('execution'),),
                content_store=content, content_scope_key=archive_scope_key('native-run-facts', 'tenant'),
                content_paths=(('unused',),), additional_records={store: (key,)},
                terminal_write_sources=requested,
            )
            assert len(statements) == 1
            assert await group.transaction(store).get_sequences(streams) == expected
            assert len(statements) == 1 + int(warm_owner or not requested)
            before_clock = len(statements)
            await group.transaction(store).now()
            assert len(statements) == before_clock + int(warm_owner or not requested)

        event.listen(owner.context.engine.sync_engine, 'before_cursor_execute', capture)
        try:
            await owner.storage_group.mutate((owner, store, content), read)
        finally:
            event.remove(owner.context.engine.sync_engine, 'before_cursor_execute', capture)


@pytest.mark.asyncio
@pytest.mark.parametrize('requested', (False, True))
async def test_owner_capture_rejects_corrupt_native_counter(tmp_path, requested):
    async with _source(tmp_path) as (state, _, _, _run):
        await _flush(state)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        source = await _raw_run(archive)
        store = archive.state_store
        table = store.storage_group.metadata.tables['ai_step_run_admissions']
        async with store.context.sessions() as connection, connection.begin():
            await connection.execute(update(table).where(table.c.store_key == store.store_key,
                table.c.key == source.key).values(checkpoint_high_water=-1))
        owner = state.execution.executions.state_store
        _, content = await owner.storage_group.open_archive(create=True)

        async def read(group):
            await active_state_scope().record_reader.read_content_references(
                owner_store=owner, owner_keys=(state.execution.executions.execution_record_key('execution'),),
                content_store=content, content_scope_key=archive_scope_key('native-run-facts', 'tenant'),
                content_paths=(('unused',),), additional_records={store: (source.key,)},
                terminal_write_sources=requested,
            )

            if not requested:
                # Generic extra-owner consumers do not validate unused counters.
                await group.transaction(store).get_sequences(run_counter_streams(source.key, store_key=store.store_key))

        with pytest.raises(AIError) as caught:
            await owner.storage_group.mutate((owner, store, content), read)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ('sqlite', 'mysql', 'postgresql'))
async def test_terminal_source_time_keeps_dialect_audit_semantics(tmp_path, monkeypatch, backend):
    from sqlalchemy.dialects import mysql, postgresql, sqlite
    from linktools.ai.persistence import _sql
    from linktools.ai.storage import SQLiteDialect, MySQLDialect, PostgreSQLDialect
    from .test_state_group_record_batch import _database, _record

    class Captured(Exception):
        pass

    async with _database(tmp_path / 'clock.db') as (_, owner, _):
        record = _record('owner')
        await owner.mutate(lambda tx: tx.insert_record(record))
        _, content = await owner.storage_group.open_archive(create=True)
        sql = []

        async def inspect_statement(self, statement):
            dialect = {'sqlite': sqlite, 'mysql': mysql, 'postgresql': postgresql}[backend].dialect()
            sql.append(str(statement.compile(dialect=dialect, compile_kwargs={'render_postcompile': True})))
            raise Captured()

        async def capture(group):
            with monkeypatch.context() as patch:
                patch.setattr(owner.context, 'dialect', {'sqlite': SQLiteDialect, 'mysql': MySQLDialect,
                    'postgresql': PostgreSQLDialect}[backend]())
                patch.setattr(_sql._SqlTransaction, '_execute', inspect_statement)
                with pytest.raises(Captured):
                    await active_state_scope().record_reader.read_content_references(
                        owner_store=owner, owner_keys=(record.key,), content_store=content,
                        content_scope_key=archive_scope_key('overlap', 'tenant'),
                        content_paths=(('unused',),), terminal_write_sources=True,
                        additional_records={owner: (record_key('overlap', 'tenant', 'trace', 'agent_run', ('run',)),)},
                    )

        await owner.storage_group.mutate((owner, content), capture)
        assert len(sql) == 1 and 'terminal_source_time' in sql[0]
        assert 'event_high_water' in sql[0] and 'checkpoint_high_water' in sql[0]
        if backend == 'postgresql':
            assert 'now() AS terminal_source_time' in sql[0]
            assert 'statement_timestamp' not in sql[0] and 'clock_timestamp' not in sql[0]
        elif backend == 'mysql':
            assert 'UTC_TIMESTAMP' in sql[0].upper()
        else:
            assert 'strftime' in sql[0]
