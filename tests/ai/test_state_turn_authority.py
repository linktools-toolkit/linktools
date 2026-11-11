#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Source-owned turn authority preserves admission, expiry and unknown outcomes."""

import asyncio
from pathlib import Path

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from linktools.ai.storage import TurnScopeIdentity
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence.turns import StateTurnCoordination, validate_turn_records
from linktools.ai.storage import TurnAdmissionReader, TurnCoordinationBackend, TurnSessionReader
from .test_state_batch_delete_backends import _open_store
from .test_state_lease_availability import _clock

pytestmark = pytest.mark.asyncio
_BACKENDS = ("memory", "sql")

_SCOPE = TurnScopeIdentity("turn-authority", "tenant", "scope")


def _backend(store):
    return StateTurnCoordination(store, namespace="turn-authority", tenant_id="tenant")


async def _claim(backend, *, session="s", request="r", execution="e", owner="o", digest="d", seconds=60):
    return await backend.claim_turn(_SCOPE, session_key=session, request_key=request, request_digest=digest,
                                    execution_id=execution, owner_id=owner, lease_seconds=seconds)


@pytest.mark.parametrize("backend", _BACKENDS)
async def test_state_turn_claim_replay_renew_cancel_release(tmp_path, backend):
    async with _open_store(backend, tmp_path / backend) as store:
        authority = _backend(store)
        assert all(isinstance(authority, protocol) for protocol in
                   (TurnCoordinationBackend, TurnAdmissionReader, TurnSessionReader))
        assert await authority.read_turn_by_session(_SCOPE, "s") is None
        first = await _claim(authority)
        assert first is not None and first.lease is not None
        replay = await _claim(authority, execution="ignored", owner="ignored")
        assert replay == replace(first, lease=None)
        assert await _claim(authority, request="other", execution="other", owner="other") is None
        with pytest.raises(AIError) as caught:
            await _claim(authority, session="other", execution="other", owner="other")
        assert caught.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        for read, identity in ((authority.read_turn, "e"), (authority.read_turn_by_request, "r"),
                               (authority.read_turn_by_session, "s")):
            assert await read(_SCOPE, identity) == replay
        assert await authority.cancel_turn(_SCOPE, "e")
        renewed = await authority.renew_turn(first.lease, lease_seconds=60.001)
        assert renewed is not None and renewed.cancel_requested
        assert renewed.lease == first.lease
        assert await authority.release_turn(first.lease)
        assert not await authority.release_turn(first.lease)
        assert await authority.renew_turn(first.lease, lease_seconds=60) is None
        assert not await authority.cancel_turn(_SCOPE, "e")
        assert await authority.read_turn_by_session(_SCOPE, "s") is None
        assert await authority.read_turn_by_request(_SCOPE, "r") is None
        assert await authority.read_turn(_SCOPE, "e") is None
        successor = await _claim(authority, execution="e2", owner="o2")
        assert successor is not None and successor.lease.fence == first.lease.fence + 1
        assert not await authority.release_turn(first.lease)
        assert await authority.cancel_turn(_SCOPE, "e2")
        records = await store.read(lambda tx: tx.scan_records())
        validate_turn_records(records, namespace="turn-authority", tenant_id="tenant")


async def _expire(store, *, session):
    async def expire(tx):
        for record in await tx.scan_records():
            if record.kind == "turn_session" and record.data["session_key"] == session:
                assert await tx.replace_record(replace(record, storage_revision=record.storage_revision + 1,
                                                       lease_expires_at=datetime(2000, 1, 1, tzinfo=timezone.utc)),
                                               expected_storage_revision=record.storage_revision)
    await store.mutate(expire)


@pytest.mark.parametrize("backend", _BACKENDS)
async def test_state_turn_cross_session_request_reclamation_retires_both_old_generations(tmp_path, backend):
    async with _open_store(backend, tmp_path / backend) as store:
        authority = _backend(store)
        old1 = await _claim(authority, session="s1", request="r1", execution="e1", owner="o1")
        old2 = await _claim(authority, session="s2", request="r2", execution="e2", owner="o2")
        await _expire(store, session="s1")
        await _expire(store, session="s2")
        assert await authority.read_turn_by_session(_SCOPE, "s1") is None
        claimed = await _claim(authority, session="s1", request="r2", execution="e3", owner="o3", digest="new")
        assert claimed is not None and claimed.lease is not None
        assert await authority.read_turn_by_session(_SCOPE, "s2") is None
        assert await authority.read_turn_by_request(_SCOPE, "r1") is None
        assert await authority.read_turn_by_request(_SCOPE, "r2") == replace(claimed, lease=None)
        assert await authority.read_turn(_SCOPE, "e1") is None
        assert await authority.read_turn(_SCOPE, "e2") is None
        assert not await authority.release_turn(old1.lease)
        assert not await authority.release_turn(old2.lease)
        records = await store.read(lambda tx: tx.scan_records())
        assert len(records) == 7
        validate_turn_records(records, namespace="turn-authority", tenant_id="tenant")


@pytest.mark.parametrize("backend", _BACKENDS)
async def test_state_turn_milliseconds_are_rounded_up_without_renewing_reads(tmp_path, monkeypatch, backend):
    async with _open_store(backend, tmp_path / backend) as store:
        authority = _backend(store)
        stamp = datetime(2030, 1, 1, tzinfo=timezone.utc)
        _clock(monkeypatch, store, backend, lambda: stamp)
        claim = await _claim(authority, seconds=60.0001)
        before = await store.read(lambda tx: tx.scan_records())
        await authority.read_turn_by_session(_SCOPE, "s")
        assert await store.read(lambda tx: tx.scan_records()) == before
        session = next(record for record in before if record.kind == "turn_session")
        assert session.lease_expires_at == stamp + timedelta(seconds=60, milliseconds=1)
        assert claim.lease is not None


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("operation", ("claim", "renew", "cancel", "release"))
async def test_state_turn_commit_unknown_is_confirmed_by_original_receipt(tmp_path, monkeypatch, backend, operation):
    async with _open_store(backend, tmp_path / backend) as store:
        authority = _backend(store)
        first = None if operation == "claim" else await _claim(authority)
        mutation = store.mutate
        attempted = 0
        unknown = AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        async def lose_response(callback):
            nonlocal attempted
            attempted += 1
            await mutation(callback)
            raise unknown
        with monkeypatch.context() as patch:
            patch.setattr(store, "mutate", lose_response)
            if operation == "claim":
                result = await _claim(authority)
                assert result.lease is not None and result.execution_id == "e"
            elif operation == "renew":
                result = await authority.renew_turn(first.lease, lease_seconds=90)
                assert result.lease == first.lease
            elif operation == "cancel":
                assert await authority.cancel_turn(_SCOPE, "e") is True
            else:
                assert await authority.release_turn(first.lease) is True
        assert attempted == 1
        records = await store.read(lambda tx: tx.scan_records())
        assert len(records) == 3
        validate_turn_records(records, namespace="turn-authority", tenant_id="tenant")


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("committed", (False, True))
async def test_state_turn_unknown_without_matching_receipt_stays_unknown(tmp_path, monkeypatch, backend, committed):
    async with _open_store(backend, tmp_path / backend) as store:
        authority = _backend(store)
        first = await _claim(authority)
        before = await store.read(lambda tx: tx.scan_records())
        mutation = store.mutate
        unknown = AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        attempted = 0
        async def uncertain(callback):
            nonlocal attempted
            attempted += 1
            if committed:
                assert await mutation(callback) is None
            raise unknown
        with monkeypatch.context() as patch:
            patch.setattr(store, "mutate", uncertain)
            with pytest.raises(AIError) as caught:
                await _claim(authority, request="other", execution="other", owner="other")
        assert caught.value is unknown and attempted == 1
        assert await authority.read_turn_by_session(_SCOPE, "s") == replace(first, lease=None)
        after = await store.read(lambda tx: tx.scan_records())
        assert tuple(replace(record, storage_revision=0) for record in after) == tuple(
            replace(record, storage_revision=0) for record in before
        )


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("missing", ("turn_execution", "turn_request", "turn_session"))
async def test_state_turn_reachable_broken_edges_are_never_idle(tmp_path, backend, missing):
    async with _open_store(backend, tmp_path / backend) as store:
        authority = _backend(store)
        await _claim(authority)
        async def damage(tx):
            record = next(record for record in await tx.scan_records() if record.kind == missing)
            await tx.delete_record(record.key_digest)
        await store.mutate(damage)
        reader, identity = ((authority.read_turn_by_request, "r") if missing == "turn_session"
                            else (authority.read_turn_by_session, "s"))
        with pytest.raises(AIError) as caught:
            await reader(_SCOPE, identity)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        with pytest.raises(AIError):
            validate_turn_records(await store.read(lambda tx: tx.scan_records()),
                                  namespace="turn-authority", tenant_id="tenant")


@pytest.mark.parametrize("backend", _BACKENDS)
async def test_state_turn_renews_during_observation_without_combining_generations(tmp_path, monkeypatch, backend):
    async with _open_store(backend, tmp_path / backend) as store:
        authority = _backend(store)
        first = await _claim(authority)
        read = store.read
        calls = 0
        async def racing(callback):
            nonlocal calls
            calls += 1
            result = await read(callback)
            if calls == 2:
                assert await authority.renew_turn(first.lease, lease_seconds=90) is not None
            return result
        with monkeypatch.context() as patch:
            patch.setattr(store, "read", racing)
            assert await authority.read_turn_by_session(_SCOPE, "s") == replace(first, lease=None)
        assert calls == 6


async def test_state_turn_independent_sql_connections_choose_one_session_winner(tmp_path):
    from linktools.ai.persistence._sql import SqlStateStore
    async with _open_store("sql", tmp_path / "sql") as store:
        other = SqlStateStore(store.context.engine)
        await other.initialize()
        try:
            left, right = _backend(store), _backend(other)
            results = await asyncio.gather(
                _claim(left, request="one", execution="one", owner="one"),
                _claim(right, request="two", execution="two", owner="two"),
            )
            assert sum(result is not None for result in results) == 1
            winner = next(result for result in results if result is not None)
            assert await right.read_turn_by_session(_SCOPE, "s") == replace(winner, lease=None)
            records = await store.read(lambda tx: tx.scan_records())
            assert len(records) == 3
        finally:
            await other.close()


@pytest.mark.parametrize("operation", ("claim", "reclaim", "renew", "cancel", "release"))
async def test_state_turn_sql_commit_then_lost_transport_confirms_exact_receipt(tmp_path, monkeypatch, operation):
    from sqlalchemy.ext.asyncio import AsyncSessionTransaction
    from linktools.ai.storage import SqlTransactionDisposition, SqlTransactionPhase
    async with _open_store("sql", tmp_path / "sql") as store:
        authority = _backend(store)
        first = None if operation == "claim" else await _claim(authority)
        if operation == "reclaim":
            assert await authority.release_turn(first.lease)
        exit_transaction = AsyncSessionTransaction.__aexit__
        classify = store.context.dialect.classify_transaction_error
        lost = OSError("commit reply lost")
        commits = 0
        async def committed_without_reply(transaction, kind, error, traceback):
            nonlocal commits
            result = await exit_transaction(transaction, kind, error, traceback)
            if kind is None and commits == 0:
                commits += 1
                raise lost
            return result
        def disposition(error, *, phase, connection_invalidated):
            if error is lost and phase is SqlTransactionPhase.COMMIT:
                return SqlTransactionDisposition.COMMIT_UNKNOWN
            return classify(error, phase=phase, connection_invalidated=connection_invalidated)
        with monkeypatch.context() as patch:
            patch.setattr(AsyncSessionTransaction, "__aexit__", committed_without_reply)
            patch.setattr(store.context.dialect, "classify_transaction_error", disposition)
            if operation == "claim":
                result = await _claim(authority)
                assert result.lease is not None
            elif operation == "reclaim":
                result = await _claim(authority, owner="new")
                assert result.lease is not None and result.owner_id == "new"
                assert result.lease != first.lease
            elif operation == "renew":
                result = await authority.renew_turn(first.lease, lease_seconds=90)
                assert result.lease == first.lease
            elif operation == "cancel":
                assert await authority.cancel_turn(_SCOPE, "e")
            else:
                assert await authority.release_turn(first.lease)
        assert commits == 1
        records = await store.read(lambda tx: tx.scan_records())
        assert len(records) == 3
        execution = next(record for record in records if record.kind == "turn_execution")
        assert execution.data["receipt_kind"] == ("claim" if operation == "reclaim" else operation)
        validate_turn_records(records, namespace="turn-authority", tenant_id="tenant")


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("damage", ("state", "receipt_kind", "claim_receipt", "cancel_without_flag", "format"))
async def test_state_turn_impossible_receipts_are_never_absence(tmp_path, backend, damage):
    async with _open_store(backend, tmp_path / backend) as store:
        authority = _backend(store)
        await _claim(authority)
        async def corrupt(tx):
            record = next(record for record in await tx.scan_records() if record.kind == "turn_execution")
            data = dict(record.data)
            state = record.state
            if damage == "state":
                state = "RETIRED"
            elif damage == "receipt_kind":
                data["receipt_kind"] = "release"
            elif damage == "claim_receipt":
                data["receipt_id"] = "unrelated"
            elif damage == "cancel_without_flag":
                data["receipt_kind"] = "cancel"
            else:
                data["format"] = 1.0
            assert await tx.replace_record(replace(record, storage_revision=record.storage_revision + 1,
                                                   data=data, state=state),
                                           expected_storage_revision=record.storage_revision)
        await store.mutate(corrupt)
        with pytest.raises(AIError) as caught:
            await authority.read_turn(_SCOPE, "e")
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        with pytest.raises(AIError):
            validate_turn_records(await store.read(lambda tx: tx.scan_records()),
                                  namespace="turn-authority", tenant_id="tenant")


@pytest.mark.parametrize("backend", _BACKENDS)
async def test_state_turn_failed_dual_retirement_rolls_back_every_old_edge(tmp_path, monkeypatch, backend):
    async with _open_store(backend, tmp_path / backend) as store:
        authority = _backend(store)
        await _claim(authority, session="s1", request="r1", execution="e1", owner="o1")
        await _claim(authority, session="s2", request="r2", execution="e2", owner="o2")
        await _expire(store, session="s1")
        await _expire(store, session="s2")
        before = await store.read(lambda tx: tx.scan_records())
        retire = authority._retire
        attempted = 0
        unavailable = AIError(ErrorCode.STORAGE_UNAVAILABLE)
        async def failed(tx, graph, **changes):
            nonlocal attempted
            attempted += 1
            if attempted == 2:
                raise unavailable
            return await retire(tx, graph, **changes)
        with monkeypatch.context() as patch:
            patch.setattr(authority, "_retire", failed)
            with pytest.raises(AIError) as caught:
                await _claim(authority, session="s1", request="r2", execution="e3", owner="o3")
        assert caught.value is unavailable
        assert await store.read(lambda tx: tx.scan_records()) == before


async def test_state_turn_sql_observation_clock_expiry_and_failure_preserve_source(tmp_path, monkeypatch):
    async with _open_store("sql", tmp_path / "sql") as store:
        authority = _backend(store)
        await _claim(authority)
        before = await store.read(lambda tx: tx.scan_records())
        session = next(record for record in before if record.kind == "turn_session")
        async def expired(connection):
            return session.lease_expires_at
        with monkeypatch.context() as patch:
            patch.setattr(store.context.dialect, "database_now", expired)
            assert await authority.read_turn_by_session(_SCOPE, "s") is None
        unavailable = AIError(ErrorCode.STORAGE_UNAVAILABLE)
        async def failed(connection):
            raise unavailable
        with monkeypatch.context() as patch:
            patch.setattr(store.context.dialect, "database_now", failed)
            with pytest.raises(AIError) as caught:
                await authority.read_turn_by_request(_SCOPE, "r")
        assert caught.value is unavailable
        assert await store.read(lambda tx: tx.scan_records()) == before


async def test_state_turn_sql_observer_cancellation_waits_for_clock_and_releases_connection(tmp_path, monkeypatch):
    async with _open_store("sql", tmp_path / "sql") as store:
        authority = _backend(store)
        await _claim(authority)
        before = await store.read(lambda tx: tx.scan_records())
        clock = store.context.dialect.database_now
        entered, release = asyncio.Event(), asyncio.Event()
        async def delayed(connection):
            entered.set()
            await release.wait()
            return await clock(connection)
        observer = None
        with monkeypatch.context() as patch:
            patch.setattr(store.context.dialect, "database_now", delayed)
            try:
                observer = asyncio.create_task(authority.read_turn_by_session(_SCOPE, "s"))
                await asyncio.wait_for(entered.wait(), 5)
                observer.cancel()
                await asyncio.sleep(0)
                assert not observer.done()
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(observer, 5)
                assert store.context.engine.pool.checkedout() == 0
            finally:
                release.set()
                if observer is not None:
                    if not observer.done():
                        observer.cancel()
                    await asyncio.gather(observer, return_exceptions=True)
        assert await store.read(lambda tx: tx.scan_records()) == before


async def test_state_turn_sql_cross_session_reclamation_with_two_initial_snapshots(tmp_path, monkeypatch):
    from linktools.ai.persistence._sql import SqlStateStore, _SqlTransaction
    async with _open_store("sql", tmp_path / "sql") as store:
        authority = _backend(store)
        await _claim(authority, session="s1", request="r1", execution="e1", owner="o1")
        await _claim(authority, session="s2", request="r2", execution="e2", owner="o2")
        await _expire(store, session="s1")
        await _expire(store, session="s2")
        other = SqlStateStore(store.context.engine)
        await other.initialize()
        ready = asyncio.Event()
        arrived = set()
        keys = {record.key_digest for record in await store.read(lambda tx: tx.scan_records())
                if record.kind == "turn_session"}
        read = _SqlTransaction.get_record
        async def initial_snapshot(tx, key):
            value = await read(tx, key)
            task = asyncio.current_task()
            if key in keys and task not in arrived:
                arrived.add(task)
                if len(arrived) == 2:
                    ready.set()
                await asyncio.wait_for(ready.wait(), 5)
            return value
        tasks = []
        try:
            with monkeypatch.context() as patch:
                patch.setattr(_SqlTransaction, "get_record", initial_snapshot)
                tasks = [asyncio.create_task(_claim(authority, session="s1", request="r2", execution="e3", owner="o3")),
                         asyncio.create_task(_claim(_backend(other), session="s2", request="r1", execution="e4", owner="o4"))]
                results = await asyncio.wait_for(asyncio.gather(*tasks), 10)
            assert all(result is not None and result.lease is not None for result in results)
            assert await authority.read_turn_by_session(_SCOPE, "s1") == replace(results[0], lease=None)
            assert await authority.read_turn_by_session(_SCOPE, "s2") == replace(results[1], lease=None)
            records = await store.read(lambda tx: tx.scan_records())
            assert len(records) == 8
            validate_turn_records(records, namespace="turn-authority", tenant_id="tenant")
        finally:
            ready.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await other.close()


@pytest.mark.parametrize("backend", _BACKENDS)
async def test_state_turn_expiry_never_renews_cancels_or_releases_old_owner(tmp_path, backend):
    async with _open_store(backend, tmp_path / backend) as store:
        authority = _backend(store)
        first = await _claim(authority)
        await _expire(store, session="s")
        before = await store.read(lambda tx: tx.scan_records())
        assert await authority.renew_turn(first.lease, lease_seconds=60) is None
        assert await authority.cancel_turn(_SCOPE, "e") is False
        assert await authority.release_turn(first.lease) is False
        assert await store.read(lambda tx: tx.scan_records()) == before


@pytest.mark.parametrize("backend", _BACKENDS)
async def test_state_turn_fence_is_scoped_and_full_lease_identity_is_required(tmp_path, backend):
    import json
    async with _open_store(backend, tmp_path / backend) as store:
        authority = _backend(store)
        left = await _claim(authority)
        right = await authority.claim_turn(TurnScopeIdentity('turn-authority', "tenant", "route-owner"), session_key="s", request_key="r", request_digest="d",
                                           execution_id="e", owner_id="route-origin-and-generation", lease_seconds=60)
        assert left.lease.fence == right.lease.fence == 1
        wrong_key = replace(left.lease, key=right.lease.key)
        wrong_token = replace(left.lease, token=right.lease.token)
        wrong_fence = replace(left.lease, fence=left.lease.fence + 1)
        bool_fence = replace(left.lease, fence=True)
        foreign = json.loads(left.lease.key)
        foreign[3] = "other-tenant"
        foreign_lease = replace(left.lease, key=json.dumps(foreign))
        for lease in (wrong_key, wrong_token, wrong_fence, bool_fence, foreign_lease):
            assert not await authority.release_turn(lease)
            assert await authority.renew_turn(lease, lease_seconds=60) is None
        assert await authority.read_turn_by_session(_SCOPE, "s") == replace(left, lease=None)
        assert await authority.read_turn_by_session(TurnScopeIdentity('turn-authority', "tenant", "route-owner"), "s") == replace(right, lease=None)
        assert await authority.read_turn_by_session(TurnScopeIdentity('turn-authority', "tenant", "unrelated"), "s") is None


@pytest.mark.parametrize("backend", _BACKENDS)
async def test_state_turn_never_borrows_an_existing_read_or_mutation_scope(tmp_path, backend):
    from linktools.ai.persistence.state import StateTransactionNestingError
    async with _open_store(backend, tmp_path / backend) as store:
        authority = _backend(store)
        await _claim(authority)
        before = await store.read(lambda tx: tx.scan_records())
        async def rejected(tx):
            with pytest.raises(StateTransactionNestingError):
                await authority.read_turn_by_session(_SCOPE, "s")
            with pytest.raises(StateTransactionNestingError):
                await authority.cancel_turn(_SCOPE, "e")
        for invoke in (store.read, store.mutate):
            await invoke(rejected)
            assert await store.read(lambda tx: tx.scan_records()) == before


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("failure", ("cas", "cancel_after_batch"))
async def test_state_turn_retirement_batch_failure_preserves_every_edge(tmp_path, monkeypatch, backend, failure):
    async with _open_store(backend, tmp_path / backend) as store:
        authority = _backend(store)
        claimed = await _claim(authority)
        before = await store.read(lambda tx: tx.scan_records())
        async def transaction_type(tx):
            return type(tx)
        cls = await store.mutate(transaction_type)
        original = cls.replace_records
        batches = []
        async def fail_retirement(tx, replacements):
            values = tuple(replacements)
            if {value.record.kind for value in values} == {"turn_session", "turn_request", "turn_execution"}:
                batches.append(values)
                if failure == "cas":
                    values = tuple(replace(value, expected_storage_revision=value.expected_storage_revision + 1,
                                           record=replace(value.record, storage_revision=value.record.storage_revision + 1))
                                   if value.record.kind == "turn_request" else value for value in values)
                    return await original(tx, values)
                await original(tx, values)
                raise asyncio.CancelledError
            return await original(tx, values)
        with monkeypatch.context() as patch:
            patch.setattr(cls, "replace_records", fail_retirement)
            with pytest.raises(AIError if failure == "cas" else asyncio.CancelledError) as caught:
                await authority.release_turn(claimed.lease)
        if failure == "cas":
            assert caught.value.code is ErrorCode.STORAGE_CONFLICT
        assert batches
        assert await store.read(lambda tx: tx.scan_records()) == before
        assert await authority.read_turn(_SCOPE, "e") == replace(claimed, lease=None)
        assert await authority.release_turn(claimed.lease)


async def test_state_turn_sqlite_known_graph_statement_budgets(tmp_path):
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
            first = await _claim(authority)
            assert first is not None and first.lease is not None
            assert counts["SELECT"] <= 3 and counts["INSERT"] <= 2 and counts["UPDATE"] <= 2
            counts.clear()
            assert await authority.read_turn_by_session(_SCOPE, "s") == replace(first, lease=None)
            assert counts["SELECT"] <= 5 and counts["UPDATE"] == 0
            counts.clear()
            assert await authority.release_turn(first.lease)
            assert counts["SELECT"] <= 4 and counts["UPDATE"] <= 2
        finally:
            event.remove(engine, "before_cursor_execute", count)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("expired", (False, True))
async def test_recover_logical_turn_with_fresh_owner_after_release_or_expiry(
    tmp_path: Path, backend: str, expired: bool,
) -> None:
    async with _open_store(backend, tmp_path / backend) as store:
        authority = _backend(store)
        first = await _claim(authority)
        assert first is not None and first.lease is not None
        assert await _claim(authority, owner="new") == replace(first, lease=None)
        if expired:
            await _expire(store, session="s")
        else:
            assert await authority.release_turn(first.lease)
        recovered = await _claim(authority, owner="new")
        assert recovered is not None and recovered.lease is not None
        assert recovered.execution_id == first.execution_id
        assert recovered.owner_id == "new" and recovered.lease != first.lease
        assert recovered.lease.fence > first.lease.fence
        assert await authority.renew_turn(first.lease, lease_seconds=60) is None
        assert not await authority.release_turn(first.lease)
        assert await authority.read_turn(_SCOPE, "e") == replace(recovered, lease=None)
        assert await authority.cancel_turn(_SCOPE, "e")
        cancelled = await authority.renew_turn(recovered.lease, lease_seconds=60)
        assert cancelled is not None and cancelled.cancel_requested
        assert cancelled.owner_id == recovered.owner_id
        assert not await authority.release_turn(first.lease)
        records = await store.read(lambda tx: tx.scan_records())
        validate_turn_records(records, namespace="turn-authority", tenant_id="tenant")


@pytest.mark.parametrize("backend", _BACKENDS)
async def test_state_turn_sort_reuses_identity_across_pages_and_snapshot(tmp_path, backend):
    from linktools.ai.persistence import PersistenceDomain
    from linktools.ai.persistence._snapshot_validation import validate_snapshot_domain
    from linktools.ai.persistence.codec import decode_record, encode_record
    from linktools.ai.persistence.state import RecordQuery

    async with _open_store(backend, tmp_path / backend) as store:
        authority = _backend(store)
        for index in range(3):
            claim = await _claim(authority, session=f"s{index}", request=f"r{index}",
                                 execution=f"e{index}", owner=f"o{index}")
            assert claim is not None and claim.lease is not None
            assert await authority.read_turn_by_request(_SCOPE, f"r{index}") == replace(claim, lease=None)
        records = await store.read(lambda tx: tx.scan_records())
        assert all(record.sort_key == record.key_digest for record in records)
        for kind in ("turn_session", "turn_request", "turn_execution"):
            expected = tuple(sorted((record for record in records if record.kind == kind),
                                    key=lambda record: record.key_digest))
            observed = []
            after = None
            while True:
                page = await store.read(lambda tx: tx.list_records(RecordQuery(
                    kind=kind, limit=1,
                    after_sort_key=None if after is None else after.sort_key,
                    after_key_digest=None if after is None else after.key_digest,
                )))
                if not page:
                    break
                observed.extend(page)
                after = page[-1]
            assert tuple(observed) == expected
        restored = tuple(decode_record(encode_record(record)) for record in records)
        assert restored == records
        validate_snapshot_domain(
            namespace="turn-authority", tenant_id="tenant", domain=PersistenceDomain.EXECUTION,
            records=restored, aliases=(), facts=(), operations=(), sequences={},
        )
        invalid = (replace(restored[0], sort_key=b"not-its-identity"), *restored[1:])
        with pytest.raises(AIError) as caught:
            validate_snapshot_domain(
                namespace="turn-authority", tenant_id="tenant", domain=PersistenceDomain.EXECUTION,
                records=invalid, aliases=(), facts=(), operations=(), sequences={},
            )
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("field", ("namespace", "tenant_id"))
async def test_state_turn_rejects_foreign_scope_without_mutation(tmp_path, backend, field):
    async with _open_store(backend, tmp_path / backend) as store:
        authority = _backend(store)
        foreign = replace(_SCOPE, **{field: "other"})
        identity = dict(session_key="s", request_key="r", request_digest="d",
                        execution_id="e", owner_id="o")
        with pytest.raises(AIError) as caught:
            authority.enter(foreign, **identity)
        assert caught.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
        for action in (
            lambda: authority.claim_turn(foreign, **identity, lease_seconds=60),
            lambda: authority.read_turn(foreign, "e"),
            lambda: authority.read_turn_by_request(foreign, "r"),
            lambda: authority.read_turn_by_session(foreign, "s"),
            lambda: authority.cancel_turn(foreign, "e"),
        ):
            with pytest.raises(AIError) as caught:
                await action()
            assert caught.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
        assert await store.read(lambda tx: tx.scan_records()) == ()


@pytest.mark.parametrize("backend", _BACKENDS)
async def test_pending_turn_uses_local_scope_with_maximum_namespace_and_compound_keys(tmp_path, backend):
    from linktools.ai.persistence.coordination import PendingTurnCoordination

    namespace, tenant = '"' * 256, '"' * 128
    request_key = "s" * 128 + ":" + "k" * 256
    session_key = "request:" + request_key
    async with _open_store(backend, tmp_path / backend) as store:
        authority = StateTurnCoordination(store, namespace=namespace, tenant_id=tenant)
        coordinator = PendingTurnCoordination(None, authority, namespace=namespace, tenant_id=tenant)
        lifetime = coordinator.enter(session_key=session_key, request_key=request_key, request_digest="digest",
                                     execution_id="e" * 128, owner_id="owner")
        async with lifetime as owner:
            records = await store.read(lambda tx: tx.scan_records())
            assert len(records) == 3
            assert {record.data["scope"] for record in records} == {"runtime-pending-turn"}
            validate_turn_records(records, namespace=namespace, tenant_id=tenant)
            assert await coordinator.read_owner("e" * 128, tenant_id=tenant) == replace(owner, lease=None)
            assert await authority.renew_turn(owner.lease, lease_seconds=60) == owner
        assert await coordinator.read_owner("e" * 128, tenant_id=tenant) is None


@pytest.mark.parametrize("backend", _BACKENDS)
async def test_turn_reader_rejects_oversize_local_scope_with_matching_addresses(tmp_path, backend):
    from linktools.ai.persistence.state import record_key_digest, scope_digest

    async with _open_store(backend, tmp_path / backend) as store:
        authority = _backend(store)
        scope = replace(_SCOPE, scope="é" * 128)
        owner = await authority.claim_turn(scope, session_key="session:" + "s" * 128,
                                          request_key="r" * 385, request_digest="digest",
                                          execution_id="e", owner_id="o", lease_seconds=60)
        assert await authority.read_turn(scope, "e") == replace(owner, lease=None)
        records = await store.read(lambda tx: tx.scan_records())
        validate_turn_records(records, namespace=scope.namespace, tenant_id=scope.tenant_id)
        invalid_scope = scope.scope + "x"
        invalid_records = []
        for record in records:
            identity = record.data[{"turn_session": "session_key", "turn_request": "request_key",
                                    "turn_execution": "execution_id"}[record.kind]]
            key = record_key_digest(scope.namespace, scope.tenant_id, "execution", record.kind,
                                    [invalid_scope, identity])
            invalid_records.append(replace(
                record, key_digest=key, sort_key=key,
                scope_digest=scope_digest(scope.namespace, scope.tenant_id, "execution", "turn", "scope", invalid_scope),
                data={**record.data, "scope": invalid_scope},
            ))
        with pytest.raises(AIError) as caught:
            validate_turn_records(invalid_records, namespace=scope.namespace, tenant_id=scope.tenant_id)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
