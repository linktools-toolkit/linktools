#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Available lease guards preserve expired ownership while authorizing retirement."""

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSessionTransaction

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import _filesystem_transaction, _memory_transaction
from linktools.ai.persistence._sql import _SqlTransaction
from linktools.ai.persistence.state import RecordLeaseTransition, StateTransactionNestingError
from linktools.ai.storage import SqlTransactionDisposition, SqlTransactionPhase

from .test_state_batch_delete_backends import _open_store, _record

pytestmark = pytest.mark.asyncio
_BACKENDS = ("memory", "filesystem", "sql")
_STAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _clock(monkeypatch, store, backend, read):
    if backend == "sql":
        async def current(session):
            return read()
        monkeypatch.setattr(store.context.dialect, "database_current_time", current)
    else:
        class Clock:
            @staticmethod
            def now(tz):
                return read()
        monkeypatch.setattr(_memory_transaction if backend == "memory" else _filesystem_transaction,
                            "datetime", Clock)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("owner", (None, "expired"))
async def test_available_guard_preserves_exact_owner_and_all_lease_fields(tmp_path, monkeypatch, backend, owner):
    record = replace(_record(1), lease_owner=owner, lease_fence=9,
                     lease_expires_at=_STAMP if owner is not None else None)
    async with _open_store(backend, tmp_path / backend) as store:
        await store.mutate(lambda tx: tx.insert_record(record))
        _clock(monkeypatch, store, backend, lambda: _STAMP)

        async def retire(tx):
            guarded = await tx.transition_record_lease(
                record.key_digest, expected_storage_revision=record.storage_revision,
                transition=RecordLeaseTransition("guard_available", owner, 9),
            )
            assert guarded is not None
            assert replace(guarded, storage_revision=record.storage_revision) == record
            retired = replace(guarded, storage_revision=guarded.storage_revision + 1,
                              data={"retired": True})
            assert await tx.replace_record(retired, expected_storage_revision=guarded.storage_revision)
            return retired

        retired = await store.mutate(retire)
        assert await store.read(lambda tx: tx.get_record(record.key_digest)) == retired
        with pytest.raises(StateTransactionNestingError):
            await store.read(lambda tx: tx.transition_record_lease(
                record.key_digest, expected_storage_revision=retired.storage_revision,
                transition=RecordLeaseTransition("guard_available", owner, 9),
            ))


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("condition", ("live", "infinite", "owner", "none_owner", "fence", "revision", "missing"))
async def test_available_guard_rejects_wrong_or_live_generation(tmp_path, monkeypatch, backend, condition):
    record = replace(_record(1), lease_owner="old", lease_fence=9, lease_expires_at=_STAMP)
    if condition == "live":
        record = replace(record, lease_expires_at=_STAMP + timedelta(microseconds=1))
    elif condition == "infinite":
        record = replace(record, lease_expires_at=None)
    expected_owner = {"owner": "other", "none_owner": None}.get(condition, "old")
    async with _open_store(backend, tmp_path / backend) as store:
        if condition != "missing":
            await store.mutate(lambda tx: tx.insert_record(record))
        _clock(monkeypatch, store, backend, lambda: _STAMP)
        assert await store.mutate(lambda tx: tx.transition_record_lease(
            record.key_digest, expected_storage_revision=record.storage_revision + (condition == "revision"),
            transition=RecordLeaseTransition("guard_available", expected_owner, 9 + (condition == "fence")),
        )) is None
        assert await store.read(lambda tx: tx.get_record(record.key_digest)) == (
            None if condition == "missing" else record
        )


@pytest.mark.parametrize("protected", ("none", "guard", "insert"))
async def test_sql_available_guard_checks_fresh_clock_after_cas_and_aborts_caught_conflict(
    tmp_path, monkeypatch, protected,
):
    record = replace(_record(1), lease_owner="old", lease_fence=9, lease_expires_at=_STAMP)
    sampled = _STAMP
    guard = _SqlTransaction.guard_record
    async with _open_store("sql", tmp_path / "sql") as store:
        if protected != "insert":
            await store.mutate(lambda tx: tx.insert_record(record))
        _clock(monkeypatch, store, "sql", lambda: sampled)
        async def delayed_guard(self, key, *, expected_storage_revision):
            nonlocal sampled
            guarded = await guard(self, key, expected_storage_revision=expected_storage_revision)
            # Database wall-clock changes are not made monotonic by a transaction snapshot.
            sampled = _STAMP - timedelta(microseconds=1)
            return guarded

        async def callback(tx):
            await tx.insert_record(_record(2))
            current = record
            if protected == "insert":
                await tx.insert_record(record)
            elif protected == "guard":
                current = await guard(tx, record.key_digest, expected_storage_revision=record.storage_revision)
            with monkeypatch.context() as patch:
                patch.setattr(_SqlTransaction, "guard_record", delayed_guard)
                with pytest.raises(AIError) as caught:
                    await tx.transition_record_lease(
                        record.key_digest, expected_storage_revision=current.storage_revision,
                        transition=RecordLeaseTransition("guard_available", "old", 9),
                    )
            assert caught.value.code is ErrorCode.STORAGE_CONFLICT
            await tx.insert_record(_record(3))

        with pytest.raises(AIError) as caught:
            await store.mutate(callback)
        assert caught.value.code is ErrorCode.STORAGE_CONFLICT
        assert await store.read(lambda tx: tx.scan_records()) == (() if protected == "insert" else (record,))


async def test_available_guard_cancel_after_cas_aborts_even_when_callback_catches(tmp_path, monkeypatch):
    record = replace(_record(1), lease_owner="old", lease_fence=9, lease_expires_at=_STAMP)
    entered, release = asyncio.Event(), asyncio.Event()
    async with _open_store("sql", tmp_path / "sql") as store:
        await store.mutate(lambda tx: tx.insert_record(record))
        calls = 0
        async def clock(session):
            nonlocal calls
            calls += 1
            if calls == 2:
                entered.set()
                await release.wait()
            return _STAMP
        monkeypatch.setattr(store.context.dialect, "database_current_time", clock)

        async def callback(tx):
            try:
                await tx.transition_record_lease(
                    record.key_digest, expected_storage_revision=record.storage_revision,
                    transition=RecordLeaseTransition("guard_available", "old", 9),
                )
            except asyncio.CancelledError:
                await tx.insert_record(_record(2))

        worker = asyncio.create_task(store.mutate(callback))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            worker.cancel()
            await asyncio.sleep(0)
            assert not worker.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(worker, 5)
            assert await store.read(lambda tx: tx.scan_records()) == (record,)
        finally:
            release.set()
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)


async def test_available_guard_unknown_commit_never_replays_or_reassigns_lease(tmp_path, monkeypatch):
    record = replace(_record(1), lease_owner="old", lease_fence=9, lease_expires_at=_STAMP)
    failure = OSError("lost commit response")
    async with _open_store("sql", tmp_path / "sql") as store:
        await store.mutate(lambda tx: tx.insert_record(record))
        _clock(monkeypatch, store, "sql", lambda: _STAMP)
        original_exit = AsyncSessionTransaction.__aexit__
        classify = store.context.dialect.classify_transaction_error
        attempted = 0
        injected = False
        async def commit_then_lose_reply(transaction, kind, error, traceback):
            nonlocal injected
            result = await original_exit(transaction, kind, error, traceback)
            if kind is None and not injected:
                injected = True
                raise failure
            return result
        def disposition(error, *, phase, connection_invalidated):
            if error is failure and phase is SqlTransactionPhase.COMMIT:
                return SqlTransactionDisposition.COMMIT_UNKNOWN
            return classify(error, phase=phase, connection_invalidated=connection_invalidated)
        async def callback(tx):
            nonlocal attempted
            attempted += 1
            return await tx.transition_record_lease(
                record.key_digest, expected_storage_revision=record.storage_revision,
                transition=RecordLeaseTransition("guard_available", "old", 9),
            )
        with monkeypatch.context() as patch:
            patch.setattr(AsyncSessionTransaction, "__aexit__", commit_then_lose_reply)
            patch.setattr(store.context.dialect, "classify_transaction_error", disposition)
            with pytest.raises(AIError) as caught:
                await store.mutate(callback)
        assert caught.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        assert attempted == 1
        assert await store.read(lambda tx: tx.get_record(record.key_digest)) == replace(
            record, storage_revision=record.storage_revision + 1,
        )


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("seconds", (0.000001, 0.001, 1.234))
async def test_fractional_lease_duration_roundtrips_without_whole_second_inflation(
    tmp_path, monkeypatch, backend, seconds,
):
    record = _record(1)
    async with _open_store(backend, tmp_path / backend) as store:
        await store.mutate(lambda tx: tx.insert_record(record))
        _clock(monkeypatch, store, backend, lambda: _STAMP)
        acquired = await store.mutate(lambda tx: tx.transition_record_lease(
            record.key_digest, expected_storage_revision=record.storage_revision,
            transition=RecordLeaseTransition("acquire", "owner", 0, seconds),
        ))
        assert acquired.lease_expires_at == _STAMP + timedelta(seconds=seconds)
        assert await store.read(lambda tx: tx.get_record(record.key_digest)) == acquired
        renewed = await store.mutate(lambda tx: tx.transition_record_lease(
            record.key_digest, expected_storage_revision=acquired.storage_revision,
            transition=RecordLeaseTransition("renew", "owner", 1, seconds * 2),
        ))
        assert renewed.lease_expires_at == _STAMP + timedelta(seconds=seconds * 2)
        assert await store.read(lambda tx: tx.get_record(record.key_digest)) == renewed


@pytest.mark.parametrize("seconds", (None, False, 0, -1, float("nan"), float("inf"), 1e-10, 1e100, "1"))
async def test_fractional_lease_duration_validation(seconds):
    with pytest.raises(ValueError, match="lease_seconds"):
        RecordLeaseTransition("acquire", "owner", 0, seconds)


@pytest.mark.parametrize("operation", ("acquire", "renew", "release", "guard"))
async def test_only_available_guard_allows_none_as_expected_owner(operation):
    with pytest.raises(ValueError, match="lease owner"):
        RecordLeaseTransition(operation, None, 0, 1 if operation in ("acquire", "renew") else None)


@pytest.mark.parametrize("duration", (False, 0, 0.001, 1))
async def test_available_guard_rejects_duration(duration):
    with pytest.raises(ValueError, match="guard_available does not accept"):
        RecordLeaseTransition("guard_available", None, 0, duration)
