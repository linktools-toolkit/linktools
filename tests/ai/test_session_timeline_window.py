#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL timeline windows preserve the counter and bounded page contract."""

import pytest
from sqlalchemy import event
from sqlalchemy.dialects import mysql, postgresql, sqlite

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage

from .test_session_timeline import _session


@pytest.mark.asyncio
@pytest.mark.parametrize('count', (0, 1, 4))
@pytest.mark.parametrize('pending', (False, True))
async def test_window_matches_counter_and_page(tmp_path, count, pending):
    async with Persistence.open('window', tenant_id='tenant', storage=PersistenceStorage.sqlite(tmp_path / 'state.db')) as state:
        sessions = state.conversation.sessions
        await sessions.create(_session())
        for index in range(count):
            await sessions.admit_execution('session', tenant_id='tenant', execution_id=f'e{index}', expected=None)
            await sessions.release_execution('session', tenant_id='tenant', execution_id=f'e{index}')
        statements = []
        engine = sessions.state_store.context.engine.sync_engine

        def observe(conn, clause, multiparams, params, execution_options):
            if clause.is_select:
                statements.append(clause)

        for before in (None, 1, count + int(pending) + 1, count + int(pending) + 2, 0):
            for limit in (1, 2, 5):
                event.listen(engine, 'before_execute', observe)
                try:
                    if before is not None and not 1 <= before <= count + int(pending) + 1:
                        with pytest.raises(AIError) as caught:
                            await sessions.capture_timeline_window('session', tenant_id='tenant', before=before, limit=limit, pending=pending)
                        assert caught.value.code is ErrorCode.CURSOR_INVALID
                    else:
                        head, values = await sessions.capture_timeline_window('session', tenant_id='tenant', before=before, limit=limit, pending=pending)
                        assert head == count
                        end = count + int(pending) + 1 if before is None else before
                        start = max(1, end - limit)
                        start = min(start, count) if pending and count else start
                        assert tuple(value.turn_seq for value in values) == tuple(range(start, min(end, count + 1)))
                finally:
                    event.remove(engine, 'before_execute', observe)
                assert len(statements) == 1
                for dialect in (sqlite.dialect(), mysql.dialect(), postgresql.dialect()):
                    statements[0].compile(dialect=dialect)
                statements.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize('damage', ('missing-fact', 'wrong-kind', 'counter-key', 'counter-value'))
async def test_window_keeps_corruption_checks(tmp_path, damage):
    from sqlalchemy import delete, update

    async with Persistence.open('window', tenant_id='tenant', storage=PersistenceStorage.sqlite(tmp_path / 'state.db')) as state:
        sessions = state.conversation.sessions
        await sessions.create(_session())
        await sessions.admit_execution('session', tenant_id='tenant', execution_id='e', expected=None)
        store = sessions.state_store
        facts = store.storage_group.metadata.tables['ai_state_facts']
        counters = store.storage_group.metadata.tables['ai_state_counters']
        stream = sessions._timeline_stream('session')
        if damage == 'missing-fact':
            statement = delete(facts).where(facts.c.store_key == store.store_key, facts.c.stream_key == stream)
        elif damage == 'wrong-kind':
            statement = update(facts).where(facts.c.store_key == store.store_key, facts.c.stream_key == stream).values(kind='wrong')
        else:
            values = {'stream_key': b'wrong'} if damage == 'counter-key' else {'value': -1}
            statement = update(counters).where(counters.c.store_key == store.store_key, counters.c.stream_key == stream).values(**values)
        async with store.context.sessions() as connection, connection.begin():
            await connection.execute(statement)
        if damage == 'wrong-kind':
            with pytest.raises(AIError) as cursor_error:
                await sessions.capture_timeline_window('session', tenant_id='tenant', before=0, limit=1, pending=False)
            assert cursor_error.value.code is ErrorCode.CURSOR_INVALID
        with pytest.raises(AIError) as caught:
            await sessions.capture_timeline_window('session', tenant_id='tenant', before=None, limit=1, pending=False)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_memory_window_uses_existing_fallback():
    async with Persistence.open('window', tenant_id='tenant', storage=PersistenceStorage.in_memory()) as state:
        assert await state.conversation.sessions.capture_timeline_window(
            'session', tenant_id='tenant', before=None, limit=1, pending=False,
        ) is None


def test_joined_fact_adapter_does_not_require_sequence_windows():
    from linktools.ai.persistence.state import SqlStateFactReader, SqlStateSequenceReader

    class JoinedOnly:
        async def read_joined_facts(self, context, statement):
            return ()

    adapter = JoinedOnly()
    assert isinstance(adapter, SqlStateFactReader)
    assert not isinstance(adapter, SqlStateSequenceReader)
