#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""First claim may stage incomplete graph, but must only publish complete truth."""
import asyncio
from dataclasses import replace
import pytest
from .test_state_turn_authority import _backend, _claim
from .test_state_batch_delete_backends import _open_store
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence.turns import validate_turn_records
from linktools.ai.persistence._sql import SqlStateStore, _SqlTransaction

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sql"))
@pytest.mark.parametrize("failure", ("acquire_rejected", "cancel_after_acquire", "final_validation"))
async def test_initial_claim_never_commits_incomplete_session(tmp_path, monkeypatch, backend, failure):
    async with _open_store(backend, tmp_path / backend) as store:
        authority = _backend(store)
        async def transaction_type(tx):
            return type(tx)
        cls = await store.mutate(transaction_type)
        original = cls.transition_record_lease
        async def fault(tx, key, *, expected_storage_revision, transition):
            if transition.operation == "acquire":
                staged = await tx.get_record(key)
                assert staged.state == "ACTIVE" and staged.data["execution_id"] == "e"
                assert staged.lease_owner is None
                if failure == "acquire_rejected":
                    return None
                value = await original(tx, key, expected_storage_revision=expected_storage_revision, transition=transition)
                if failure == "cancel_after_acquire":
                    raise asyncio.CancelledError
                return value
            return await original(tx, key, expected_storage_revision=expected_storage_revision, transition=transition)
        def invalid(*args):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        with monkeypatch.context() as patch:
            patch.setattr(cls, "transition_record_lease", fault)
            if failure == "final_validation":
                patch.setattr(authority._records, "graph", invalid)
            with pytest.raises(asyncio.CancelledError if failure == "cancel_after_acquire" else AIError) as caught:
                await _claim(authority)
        if failure == "acquire_rejected":
            assert caught.value.code is ErrorCode.STORAGE_CONFLICT
        if failure == "final_validation":
            assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await store.read(lambda tx: tx.scan_records()) == ()
        claimed = await _claim(authority)
        assert claimed is not None and claimed.lease is not None
        records = await store.read(lambda tx: tx.scan_records())
        validate_turn_records(records, namespace="turn-authority", tenant_id="tenant")
        assert await authority.read_turn("scope", "e") == replace(claimed, lease=None)


async def test_initial_claim_two_sql_connections_admit_exactly_one(tmp_path, monkeypatch):
    async with _open_store("sql", tmp_path / "sql") as store:
        other = SqlStateStore(store.context.engine)
        await other.initialize()
        original = _SqlTransaction.insert_records
        arrived = set()
        ready = asyncio.Event()
        async def synchronized(tx, records):
            values = tuple(records)
            if any(record.kind == "turn_session" for record in values):
                arrived.add(asyncio.current_task())
                if len(arrived) == 2:
                    ready.set()
                await asyncio.wait_for(ready.wait(), 5)
            return await original(tx, values)
        try:
            with monkeypatch.context() as patch:
                patch.setattr(_SqlTransaction, "insert_records", synchronized)
                results = await asyncio.gather(*(
                    _claim(_backend(target), request=f"r{i}", execution=f"e{i}", owner=f"o{i}")
                    for i, target in enumerate((store, other))
                ))
            assert len([result for result in results if result is not None]) == 1
            rows = await store.read(lambda tx: tx.scan_records())
            assert len(rows) == 3
            validate_turn_records(rows, namespace="turn-authority", tenant_id="tenant")
            winner = next(result for result in results if result is not None)
            assert await _backend(other).read_turn_by_session("scope", "s") == replace(winner, lease=None)
        finally:
            await other.close()


async def test_initial_claim_is_not_visible_before_acquire_or_after_cancel(tmp_path, monkeypatch):
    async with _open_store("sql", tmp_path / "sql") as store:
        other = SqlStateStore(store.context.engine)
        await other.initialize()
        original = _SqlTransaction.transition_record_lease
        staged, release = asyncio.Event(), asyncio.Event()
        async def pause(tx, key, *, expected_storage_revision, transition):
            if transition.operation == "acquire":
                staged.set()
                await release.wait()
            return await original(tx, key, expected_storage_revision=expected_storage_revision, transition=transition)
        worker = None
        try:
            with monkeypatch.context() as patch:
                patch.setattr(_SqlTransaction, "transition_record_lease", pause)
                worker = asyncio.create_task(_claim(_backend(store)))
                await asyncio.wait_for(staged.wait(), 5)
                assert await other.read(lambda tx: tx.scan_records()) == ()
                worker.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await worker
            assert await other.read(lambda tx: tx.scan_records()) == ()
        finally:
            release.set()
            if worker is not None and not worker.done():
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            await other.close()


async def test_initial_claim_sql_statement_budget(tmp_path):
    from collections import Counter
    from sqlalchemy import event

    async with _open_store("sql", tmp_path / "sql") as store:
        authority = _backend(store)
        counts = Counter()
        def count(connection, cursor, statement, parameters, context, executemany):
            counts[statement.lstrip().split(None, 1)[0].upper()] += 1
        engine = store.context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", count)
        try:
            claimed = await _claim(authority)
            assert claimed is not None and claimed.lease is not None
            assert counts["SELECT"] <= 3
            assert counts["INSERT"] == 1 and counts["UPDATE"] == 1
            assert counts["DELETE"] == 0
        finally:
            event.remove(engine, "before_cursor_execute", count)
        records = await store.read(lambda tx: tx.scan_records())
        assert len(records) == 3
        validate_turn_records(records, namespace="turn-authority", tenant_id="tenant")
