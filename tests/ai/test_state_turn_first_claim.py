#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""First claim may stage incomplete graph, but must only publish complete truth."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from sqlalchemy import Insert, Update, select
from .test_state_turn_authority import _backend, _claim, _open_store
from linktools.ai.storage import TurnScopeIdentity
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence.turns import validate_turn_records
from linktools.ai.persistence._sql import SqlStateStore, _SqlTransaction
from linktools.ai.persistence._turn_sql import SqlTurnProof
from linktools.ai.persistence._record_owner import EXECUTION_SHELL_KIND

pytestmark = pytest.mark.asyncio

_SCOPE = TurnScopeIdentity("turn-authority", "tenant", "runtime-pending-turn")


async def _receipt_rows(store):
    table = store.storage_group.metadata.tables["ai_execution_records"]
    async with store.context.engine.connect() as conn:
        return tuple((await conn.execute(select(table).where(
            table.c.store_key == store.store_key, table.c.turn_receipt.is_not(None),
        ))).mappings())


@pytest.mark.parametrize("failure", ("acquire_rejected", "cancel_after_acquire", "final_validation"))
async def test_initial_claim_never_commits_incomplete_session(tmp_path, monkeypatch, failure):
    async with _open_store("memory", tmp_path / "memory") as store:
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
        assert await authority.read_turn(_SCOPE, "e") == replace(claimed, lease=None)


@pytest.mark.parametrize("failure", ("acquire_rejected", "cancel_after_acquire", "final_validation"))
async def test_initial_sql_claim_rolls_back_native_execution_receipt(tmp_path, monkeypatch, failure):
    async with _open_store("sql", tmp_path / "sql") as store:
        authority = _backend(store)
        original = _SqlTransaction._execute
        original_coordination = SqlTurnProof.coordination
        bound = False
        faults = 0

        async def fault(tx, statement):
            nonlocal bound, faults
            if isinstance(statement, Insert) and statement.table.name == "ai_execution_records":
                if failure == "acquire_rejected":
                    faults += 1
                    return SimpleNamespace(rowcount=0)
            result = await original(tx, statement)
            if isinstance(statement, Update) and statement.table.name == "ai_execution_records":
                assert result.rowcount == 1
                bound = True
                if failure == "cancel_after_acquire":
                    faults += 1
                    raise asyncio.CancelledError
            return result

        def fail_result(proof, scope, *, token=None):
            nonlocal faults
            if token is not None:
                assert bound and proof.active and proof.execution_id == "e"
                faults += 1
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return original_coordination(proof, scope, token=token)

        with monkeypatch.context() as patch:
            patch.setattr(_SqlTransaction, "_execute", fault)
            if failure == "final_validation":
                patch.setattr(SqlTurnProof, "coordination", fail_result)
            with pytest.raises(asyncio.CancelledError if failure == "cancel_after_acquire" else AIError) as caught:
                await _claim(authority)
        assert faults > 0 and bound == (failure != "acquire_rejected")
        if failure != "cancel_after_acquire":
            expected = ErrorCode.STORAGE_CONFLICT if failure == "acquire_rejected" else ErrorCode.STORAGE_INTEGRITY_ERROR
            assert caught.value.code is expected
        assert await _receipt_rows(store) == ()
        assert await authority.read_turn(_SCOPE, "e") is None
        claimed = await _claim(authority)
        assert claimed is not None and claimed.lease is not None
        rows = await _receipt_rows(store)
        assert len(rows) == 1 and rows[0]["kind"] == EXECUTION_SHELL_KIND
        assert rows[0]["turn_token"] == claimed.lease.token.encode("ascii")
        assert await authority.read_turn(_SCOPE, "e") == replace(claimed, lease=None)


async def test_initial_claim_two_sql_connections_admit_exactly_one(tmp_path, monkeypatch):
    async with _open_store("sql", tmp_path / "sql") as store:
        other = SqlStateStore(store.context.engine, store_key=store.store_key)
        await other.initialize()
        original = _SqlTransaction._execute
        arrived = set()
        ready = asyncio.Event()

        async def synchronized(tx, statement):
            if isinstance(statement, Insert) and statement.table.name == "ai_execution_records":
                arrived.add(tx)
                if len(arrived) == 2:
                    ready.set()
                await asyncio.wait_for(ready.wait(), 5)
            return await original(tx, statement)
        try:
            with monkeypatch.context() as patch:
                patch.setattr(_SqlTransaction, "_execute", synchronized)
                results = await asyncio.gather(*(
                    _claim(_backend(target), request=f"r{i}", execution=f"e{i}", owner=f"o{i}")
                    for i, target in enumerate((store, other))
                ))
            assert len([result for result in results if result is not None]) == 1
            assert len(arrived) == 2
            rows = await _receipt_rows(store)
            assert len(rows) == 1 and rows[0]["kind"] == EXECUTION_SHELL_KIND
            winner = next(result for result in results if result is not None)
            assert rows[0]["turn_token"] == winner.lease.token.encode("ascii")
            assert await _backend(other).read_turn_by_session(_SCOPE, "s") == replace(winner, lease=None)
        finally:
            await other.close()


async def test_initial_claim_is_not_visible_before_acquire_or_after_cancel(tmp_path, monkeypatch):
    async with _open_store("sql", tmp_path / "sql") as store:
        other = SqlStateStore(store.context.engine, store_key=store.store_key)
        await other.initialize()
        original = _SqlTransaction._execute
        staged, release = asyncio.Event(), asyncio.Event()

        async def pause(tx, statement):
            if isinstance(statement, Update) and statement.table.name == "ai_execution_records":
                staged.set()
                await release.wait()
            return await original(tx, statement)
        worker = None
        try:
            with monkeypatch.context() as patch:
                patch.setattr(_SqlTransaction, "_execute", pause)
                worker = asyncio.create_task(_claim(_backend(store)))
                await asyncio.wait_for(staged.wait(), 5)
                assert await _receipt_rows(other) == ()
                worker.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await worker
            assert await _receipt_rows(other) == ()
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
            # One outer-joined ownership snapshot, then fresh source time after the
            # unique native INSERT; neither SELECT is a readback of our own write.
            assert counts["SELECT"] == 2
            assert counts["INSERT"] == 1 and counts["UPDATE"] == 1
            assert counts["DELETE"] == 0
        finally:
            event.remove(engine, "before_cursor_execute", count)
        records = await _receipt_rows(store)
        assert len(records) == 1 and records[0]["kind"] == EXECUTION_SHELL_KIND
        assert records[0]["turn_token"] == claimed.lease.token.encode("ascii")
