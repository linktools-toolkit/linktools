#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recovery operation receipts retain integrity under bounded batch reads."""

from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.errors import AIError
from linktools.ai.persistence import Persistence, PersistenceStorage, _sql
from linktools.ai.persistence.state import OperationQuery, OperationReader, StateTransaction, StoredOperation
from linktools.ai.task import TaskGraphAdmission

from .test_state_batch_delete_backends import _open_store
from .test_task_durable_recovery import _request

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("backend", ("memory", "sql"))
async def test_operation_batch_preserves_missing_duplicate_and_staged_reads(
    tmp_path: Path, backend: str,
) -> None:
    values = tuple(StoredOperation(f"operation-{index}".encode(), b"stream", index + 1,
                                  "pending", False, {}) for index in range(129))
    async with _open_store(backend, tmp_path) as store:
        async def seed(transaction: StateTransaction) -> None:
            for value in values:
                await transaction.insert_operation(value)
            assert isinstance(transaction, OperationReader)
            assert await transaction.get_operations((values[0].key, b"missing", values[0].key)) == {
                values[0].key: values[0],
            }
        await store.mutate(seed)
        assert await store.read(lambda tx: tx.get_operations(())) == {}
        assert await store.read(lambda tx: tx.get_operations(tuple(value.key for value in values))) == {
            value.key: value for value in values
        }


async def test_operation_batch_rejects_locator_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_sql, "_key_locator", lambda value: "a" * 64)
    async with _open_store("sql", tmp_path) as store:
        await store.mutate(lambda tx: tx.insert_operation(
            StoredOperation(b"original", b"stream", 1, "pending", False, {}),
        ))
        with pytest.raises(AIError, match="STORAGE_INTEGRITY_ERROR"):
            await store.read(lambda tx: tx.get_operations((b"different",)))


async def test_recovery_page_batches_receipts_without_reading_lookahead(tmp_path: Path) -> None:
    async with Persistence.open("task-test", tenant_id="tenant", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as state:
        expected = []
        for index in range(21):
            request = _request(f"page-{index:03d}")
            admission = TaskGraphAdmission.from_request(request)
            await state.task.admissions.admit(admission, request.graph)
            expected.append(admission.launch())
        queries = []
        engine = state.task.admissions.state_store.storage_group.context.engine.sync_engine
        def capture(_connection, _cursor, statement, parameters, _context, _many):
            if statement.lstrip().upper().startswith("SELECT") and "ai_state_operations" in statement:
                queries.append((statement, parameters))
        event.listen(engine, "before_cursor_execute", capture)
        try:
            first = await state.task.admissions.list_recoverable_page(cursor=None, limit=20)
            assert first.items == tuple(expected[:20])
            assert first.next_cursor is not None
            # One keyed receipt batch, rather than one read per graph.
            assert len(queries) == 1
            assert len(queries[0][1]) == 21  # store scope plus twenty receipt locators
            second = await state.task.admissions.list_recoverable_page(cursor=first.next_cursor, limit=20)
            assert second.items == tuple(expected[20:])
            assert second.next_cursor is None
        finally:
            event.remove(engine, "before_cursor_execute", capture)


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_recovery_page_requires_every_admission_receipt(tmp_path: Path, backend: str) -> None:
    storage = (PersistenceStorage.in_memory() if backend == "memory"
               else PersistenceStorage.sqlite(tmp_path / "state.db"))
    async with Persistence.open("task-test", tenant_id="tenant", storage=storage) as state:
        request = _request("missing-receipt")
        await state.task.admissions.admit(TaskGraphAdmission.from_request(request), request.graph)
        await state.task.operations.state_store.mutate(lambda tx: tx.delete_operations(OperationQuery()))
        with pytest.raises(AIError, match="STORAGE_INTEGRITY_ERROR"):
            await state.task.admissions.list_recoverable_page(cursor=None, limit=20)


class _LegacyOperationTransaction:
    def __init__(self, transaction: StateTransaction) -> None:
        self._transaction = transaction

    def __getattr__(self, name: str) -> object:
        if name == "get_operations":
            raise AttributeError(name)
        return getattr(self._transaction, name)


@pytest.mark.parametrize("backend", ("memory", "sql"))
async def test_custom_transaction_without_batch_capability_keeps_recovery_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    from .test_state_fact_reader import _custom_transactions
    from linktools.ai.persistence.state import _ReadOnlyStateTransaction

    storage = (PersistenceStorage.in_memory() if backend == "memory"
               else PersistenceStorage.sqlite(tmp_path / "state.db"))
    async with Persistence.open("task-test", tenant_id="tenant", storage=storage) as state:
        request = _request("legacy")
        admission = TaskGraphAdmission.from_request(request)
        await state.task.admissions.admit(admission, request.graph)
        _custom_transactions(monkeypatch, backend, _LegacyOperationTransaction)
        assert (await state.task.admissions.list_recoverable_page(cursor=None, limit=20)).items == (admission.launch(),)
        async def read(transaction: StateTransaction) -> None:
            assert not isinstance(transaction, OperationReader)
            view = _ReadOnlyStateTransaction(transaction)
            assert await view.get_operations((b"missing",)) == {}
        await state.task.operations.state_store.read(read)
