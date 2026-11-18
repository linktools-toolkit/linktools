#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native import reuses conflict rows without weakening owner or replay checks."""

import pytest
from pydantic_ai_harness.step_persistence import RunRecord
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.sql import Select

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence._step_store import SqlStepStore
from linktools.ai.storage import MySQLDialect


class _Rows:
    def __init__(self, row):
        self.row = row

    def mappings(self):
        return self

    def first(self):
        return self.row

    def one(self):
        if self.row is None:
            raise NoResultFound()
        return self.row


class _ConflictSession:
    def __init__(self, row, error):
        self.row, self.error = row, error
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        if not isinstance(statement, Select):
            raise self.error
        return _Rows(self.row)

    async def scalar(self, statement):
        self.statements.append(statement)
        return None if self.row is None else 1


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ('same', 'metadata', 'owner', 'run', 'invisible', 'other-error', 'unknown'))
async def test_import_duplicate_preserves_identity_and_failure(case, monkeypatch):
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    store = SqlStepStore(engine, namespace='test', tenant_id='tenant')
    monkeypatch.setattr(store._context, 'dialect', MySQLDialect())
    record = RunRecord(run_id='run', metadata={'value': 'original'})
    row = store._run_values(record.run_id) | store._record_values(record)
    if case == 'metadata':
        row = store._run_values(record.run_id) | store._record_values(
            RunRecord(run_id='run', metadata={'value': 'changed'}))
    elif case == 'owner':
        row['store_key'] = b'other owner'
    elif case == 'run':
        row['run_id'] = 'run '
    elif case == 'invisible':
        row = None
    error = IntegrityError('insert', {}, Exception(9999 if case == 'unknown' else 1452 if case == 'other-error' else 1062, 'conflict'))
    session = _ConflictSession(row, error)
    try:
        if case == 'same':
            await store._import_run(session, record)
        elif case in ('metadata', 'run'):
            with pytest.raises(AIError) as caught:
                await store._import_run(session, record)
            assert caught.value.code is (ErrorCode.STORAGE_INTEGRITY_ERROR if case == 'run' else ErrorCode.STORAGE_CONFLICT)
        elif case == 'owner':
            with pytest.raises(NoResultFound):
                await store._import_run(session, record)
        else:
            with pytest.raises(IntegrityError) as caught:
                await store._import_run(session, record)
            assert caught.value is error
        selects = [statement for statement in session.statements if isinstance(statement, Select)]
        assert len(selects) == (0 if case in ('other-error', 'unknown') else 1)
        if selects:
            assert set(selects[0].selected_columns.keys()) == set(store._table('runs').c.keys())
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_default_duplicate_probe_stays_id_only():
    from sqlalchemy import Column, Integer, MetaData, Table
    table = Table('example', MetaData(), Column('id', Integer), Column('key', Integer))
    error = IntegrityError('insert', {}, Exception(1062, 'conflict'))
    session = _ConflictSession({'id': 1}, error)
    result = await MySQLDialect().insert_ignore_conflict(
        session, table=table, values={'key': 2}, index_elements=('key',),
    )
    assert not result.inserted and result.conflict_row is None
    assert tuple(session.statements[-1].selected_columns.keys()) == ('id',)


@pytest.mark.asyncio
async def test_duplicate_import_reuses_existing_conflict_observation(monkeypatch):
    class IdOnlyConflictDialect(MySQLDialect):
        async def insert_ignore_conflict(self, session, *, conflict_columns=(), **kwargs):
            return await super().insert_ignore_conflict(session, **kwargs)

    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    store = SqlStepStore(engine, namespace='test', tenant_id='tenant')
    record = RunRecord(run_id='run', metadata={'value': 'original'})
    row = store._run_values(record.run_id) | store._record_values(record)
    error = IntegrityError('insert', {}, Exception(1062, 'conflict'))
    counts = []
    try:
        for dialect in (IdOnlyConflictDialect(), MySQLDialect()):
            monkeypatch.setattr(store._context, 'dialect', dialect)
            session = _ConflictSession(row, error)
            await store._import_run(session, record)
            counts.append(sum(isinstance(statement, Select) for statement in session.statements))
        assert counts == [2, 1]
    finally:
        await engine.dispose()
