#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live lease guards authorize a mutation without renewing or transferring ownership."""

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


def _leased():
    return replace(_record(1), lease_owner="owner", lease_fence=7,
                   lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=10))


@pytest.mark.parametrize("backend", _BACKENDS)
async def test_live_lease_guard_preserves_lease_and_authorizes_following_cas(tmp_path, backend):
    record = _leased()
    async with _open_store(backend, tmp_path / backend) as store:
        await store.mutate(lambda tx: tx.insert_record(record))

        async def cancel(tx):
            guarded = await tx.transition_record_lease(
                record.key_digest, expected_storage_revision=record.storage_revision,
                transition=RecordLeaseTransition("guard", "owner", 7),
            )
            assert guarded == replace(record, storage_revision=record.storage_revision + 1)
            cancelled = replace(guarded, data={"cancelled": True},
                                storage_revision=guarded.storage_revision + 1)
            assert await tx.replace_record(cancelled, expected_storage_revision=guarded.storage_revision)
            return cancelled

        cancelled = await store.mutate(cancel)
        assert await store.read(lambda tx: tx.get_record(record.key_digest)) == cancelled
        with pytest.raises(StateTransactionNestingError):
            await store.read(lambda tx: tx.transition_record_lease(
                record.key_digest, expected_storage_revision=cancelled.storage_revision,
                transition=RecordLeaseTransition("guard", "owner", 7),
            ))
        assert await store.read(lambda tx: tx.get_record(record.key_digest)) == cancelled


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("condition", ("owner", "fence", "revision", "expired", "no_expiry", "missing"))
async def test_rejected_lease_guard_does_not_change_record_or_lease(tmp_path, backend, condition):
    record = _leased()
    if condition == "expired":
        record = replace(record, lease_expires_at=datetime(2000, 1, 1, tzinfo=timezone.utc))
    elif condition == "no_expiry":
        record = replace(record, lease_expires_at=None)
    async with _open_store(backend, tmp_path / backend) as store:
        if condition != "missing":
            await store.mutate(lambda tx: tx.insert_record(record))
        assert await store.mutate(lambda tx: tx.transition_record_lease(
            record.key_digest,
            expected_storage_revision=record.storage_revision + (condition == "revision"),
            transition=RecordLeaseTransition("guard", "other" if condition == "owner" else "owner",
                                             8 if condition == "fence" else 7),
        )) is None
        assert await store.read(lambda tx: tx.get_record(record.key_digest)) == (
            None if condition == "missing" else record
        )


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("at_expiry", (False, True))
async def test_lease_guard_uses_fresh_time_without_replacing_transaction_snapshot(
    tmp_path, monkeypatch, backend, at_expiry,
):
    record = _leased()
    sampled = record.lease_expires_at - timedelta(seconds=1)
    async with _open_store(backend, tmp_path / backend) as store:
        await store.mutate(lambda tx: tx.insert_record(record))
        if backend == "sql":
            async def clock(session):
                return sampled
            monkeypatch.setattr(store.context.dialect, "database_current_time", clock)
        else:
            class Clock:
                @staticmethod
                def now(tz):
                    return sampled
            monkeypatch.setattr(_memory_transaction if backend == "memory" else _filesystem_transaction,
                                "datetime", Clock)

        async def callback(tx):
            nonlocal sampled
            original_time = await tx.now()
            if at_expiry:
                sampled = record.lease_expires_at
            guarded = await tx.transition_record_lease(
                record.key_digest, expected_storage_revision=record.storage_revision,
                transition=RecordLeaseTransition("guard", "owner", 7),
            )
            assert await tx.now() == original_time
            return guarded

        result = await store.mutate(callback)
        assert (result is None) is at_expiry
        if result is not None:
            assert result.lease_expires_at == record.lease_expires_at
        assert await store.read(lambda tx: tx.get_record(record.key_digest)) == (record if at_expiry else result)


@pytest.mark.parametrize("protected", ("none", "guard", "insert"))
async def test_sql_guard_expiry_after_cas_aborts_even_when_callback_catches(tmp_path, monkeypatch, protected):
    record = _leased()
    prior, following = _record(2), _record(3)
    sampled = record.lease_expires_at - timedelta(seconds=1)
    async with _open_store("sql", tmp_path / "sql") as store:
        if protected != "insert":
            await store.mutate(lambda tx: tx.insert_record(record))
        async def clock(session):
            return sampled
        monkeypatch.setattr(store.context.dialect, "database_current_time", clock)
        guard = _SqlTransaction.guard_record
        async def delayed_guard(self, key, *, expected_storage_revision):
            nonlocal sampled
            result = await guard(self, key, expected_storage_revision=expected_storage_revision)
            sampled = record.lease_expires_at
            return result

        async def callback(tx):
            await tx.insert_record(prior)
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
                        transition=RecordLeaseTransition("guard", "owner", 7),
                    )
            assert caught.value.code is ErrorCode.STORAGE_CONFLICT
            await tx.insert_record(following)

        with pytest.raises(AIError) as caught:
            await store.mutate(callback)
        assert caught.value.code is ErrorCode.STORAGE_CONFLICT
        assert await store.read(lambda tx: tx.scan_records()) == (() if protected == "insert" else (record,))


@pytest.mark.parametrize("backend", _BACKENDS)
async def test_lease_guard_rechecks_conditions_for_an_already_guarded_record(tmp_path, backend):
    record = _leased()
    async with _open_store(backend, tmp_path / backend) as store:
        await store.mutate(lambda tx: tx.insert_record(record))

        async def callback(tx):
            guarded = await tx.guard_record(record.key_digest, expected_storage_revision=record.storage_revision)
            assert guarded is not None
            for revision, owner, fence in (
                (record.storage_revision, "owner", 7),
                (guarded.storage_revision, "other", 7),
                (guarded.storage_revision, "owner", 8),
            ):
                assert await tx.transition_record_lease(
                    record.key_digest, expected_storage_revision=revision,
                    transition=RecordLeaseTransition("guard", owner, fence),
                ) is None
                assert await tx.get_record(record.key_digest) == guarded
            result = await tx.transition_record_lease(
                record.key_digest, expected_storage_revision=guarded.storage_revision,
                transition=RecordLeaseTransition("guard", "owner", 7),
            )
            assert result is not None
            assert replace(result, storage_revision=guarded.storage_revision) == guarded
            return result

        result = await store.mutate(callback)
        assert await store.read(lambda tx: tx.get_record(record.key_digest)) == result


async def test_sql_lease_guard_cancellation_after_cas_cannot_be_caught_to_commit(tmp_path, monkeypatch):
    record = _leased()
    following = _record(2)
    entered = asyncio.Event()
    release = asyncio.Event()
    sampled = record.lease_expires_at - timedelta(seconds=1)
    worker = None
    async with _open_store("sql", tmp_path / "sql") as store:
        await store.mutate(lambda tx: tx.insert_record(record))
        reads = 0
        async def clock(session):
            nonlocal reads
            reads += 1
            if reads == 2:
                entered.set()
                await release.wait()
            return sampled
        monkeypatch.setattr(store.context.dialect, "database_current_time", clock)

        async def callback(tx):
            try:
                await tx.transition_record_lease(
                    record.key_digest, expected_storage_revision=record.storage_revision,
                    transition=RecordLeaseTransition("guard", "owner", 7),
                )
            except asyncio.CancelledError:
                await tx.insert_record(following)

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


async def test_unknown_guard_commit_keeps_canonical_revision_without_replaying(tmp_path, monkeypatch):
    record = _leased()
    async with _open_store("sql", tmp_path / "sql") as store:
        await store.mutate(lambda tx: tx.insert_record(record))
        failure = OSError("lost commit response")
        exit_transaction = AsyncSessionTransaction.__aexit__
        classify = store.context.dialect.classify_transaction_error
        attempted = 0
        injected = False
        async def commit_then_lose_response(transaction, kind, error, traceback):
            nonlocal injected
            result = await exit_transaction(transaction, kind, error, traceback)
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
                transition=RecordLeaseTransition("guard", "owner", 7),
            )
        with monkeypatch.context() as patch:
            patch.setattr(AsyncSessionTransaction, "__aexit__", commit_then_lose_response)
            patch.setattr(store.context.dialect, "classify_transaction_error", disposition)
            with pytest.raises(AIError) as caught:
                await store.mutate(callback)
        assert caught.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        assert attempted == 1
        assert await store.read(lambda tx: tx.get_record(record.key_digest)) == replace(
            record, storage_revision=record.storage_revision + 1,
        )


@pytest.mark.parametrize("duration", (0, 1, True, 0.5, "1"))
async def test_lease_guard_rejects_duration(duration):
    with pytest.raises(ValueError, match="guard does not accept a lease duration"):
        RecordLeaseTransition("guard", "owner", 7, duration)
