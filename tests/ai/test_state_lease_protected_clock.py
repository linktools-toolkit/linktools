#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Existing write protection still requires a fresh lease decision, not another CAS."""

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence._sql import _SqlTransaction
from linktools.ai.persistence.state import RecordLeaseTransition, RecordReplacement

from .test_state_batch_delete_backends import _open_store, _record

pytestmark = pytest.mark.asyncio
_STAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)
_PROTECTED = ("insert", "guard", "replace")


async def _protect(tx, record, source):
    if source == "insert":
        await tx.insert_record(record)
    elif source == "guard":
        record = await tx.guard_record(record.key_digest, expected_storage_revision=record.storage_revision)
    elif source == "replace":
        candidate = replace(record, storage_revision=record.storage_revision + 1, data={"protected": True})
        await tx.replace_records((RecordReplacement(candidate, record.storage_revision),))
        record = candidate
    else:
        assert await tx.get_record(record.key_digest) == record
    return record


@pytest.mark.parametrize("source", (*_PROTECTED, "cached"))
@pytest.mark.parametrize("operation", ("acquire", "renew", "release", "guard", "guard_available"))
async def test_sql_lease_decision_samples_once_after_existing_write_protection(
    tmp_path, monkeypatch, source, operation,
):
    record = replace(_record(1), lease_owner="owner", lease_fence=7,
                     lease_expires_at=_STAMP + timedelta(seconds=30))
    if operation == "acquire":
        record = replace(record, lease_owner=None, lease_expires_at=None)
    elif operation == "guard_available":
        record = replace(record, lease_expires_at=_STAMP)
    sampled = []
    async with _open_store("sql", tmp_path / "sql") as store:
        if source != "insert":
            await store.mutate(lambda tx: tx.insert_record(record))
        async def clock(session):
            sampled.append(_STAMP)
            return _STAMP
        monkeypatch.setattr(store.context.dialect, "database_current_time", clock)

        async def callback(tx):
            current = await _protect(tx, record, source)
            result = await tx.transition_record_lease(
                current.key_digest, expected_storage_revision=current.storage_revision,
                transition=RecordLeaseTransition(operation, "owner", 7,
                                                 5 if operation in {"acquire", "renew"} else None),
            )
            assert result is not None
            if operation in {"guard", "guard_available"}:
                assert replace(result, storage_revision=current.storage_revision) == current
            elif operation in {"acquire", "renew"}:
                assert result.lease_expires_at == _STAMP + timedelta(seconds=5)
            else:
                assert result.lease_owner is None and result.lease_expires_at is None
            return result

        result = await store.mutate(callback)
        assert len(sampled) == (2 if source == "cached" else 1)
        assert await store.read(lambda tx: tx.get_record(record.key_digest)) == result


@pytest.mark.parametrize("source", _PROTECTED)
@pytest.mark.parametrize("condition", ("expiry", "clock_backwards", "owner", "fence", "revision", "infinite"))
async def test_sql_preprotected_lease_rejects_fresh_invalid_conditions_without_its_own_write(
    tmp_path, monkeypatch, source, condition,
):
    record = replace(_record(1), lease_owner="owner", lease_fence=7,
                     lease_expires_at=_STAMP if condition in {"expiry", "clock_backwards"} else
                     _STAMP + timedelta(seconds=30))
    if condition == "infinite":
        record = replace(record, lease_expires_at=None)
    async with _open_store("sql", tmp_path / "sql") as store:
        if source != "insert":
            await store.mutate(lambda tx: tx.insert_record(record))
        sampled = _STAMP + timedelta(seconds=1) if condition == "clock_backwards" else _STAMP - timedelta(seconds=1)
        async def clock(session):
            return sampled
        monkeypatch.setattr(store.context.dialect, "database_current_time", clock)

        async def callback(tx):
            nonlocal sampled
            current = await _protect(tx, record, source)
            snapshot = await tx.now()
            sampled = _STAMP if condition != "clock_backwards" else _STAMP - timedelta(microseconds=1)
            operation = "guard_available" if condition == "clock_backwards" else "guard"
            assert await tx.transition_record_lease(
                current.key_digest,
                expected_storage_revision=current.storage_revision + (condition == "revision"),
                transition=RecordLeaseTransition(operation, "other" if condition == "owner" else "owner",
                                                 8 if condition == "fence" else 7),
            ) is None
            assert await tx.now() == snapshot
            assert await tx.get_record(current.key_digest) == current
            return current

        result = await store.mutate(callback)
        assert await store.read(lambda tx: tx.get_record(record.key_digest)) == result


@pytest.mark.parametrize("source", _PROTECTED)
@pytest.mark.parametrize("error_kind", ("unavailable", "cancel"))
async def test_sql_preprotected_first_clock_error_keeps_existing_caught_error_contract(
    tmp_path, monkeypatch, source, error_kind,
):
    record = replace(_record(1), lease_owner="owner", lease_fence=7,
                     lease_expires_at=_STAMP + timedelta(seconds=30))
    failure = AIError(ErrorCode.STORAGE_UNAVAILABLE) if error_kind == "unavailable" else asyncio.CancelledError()
    async with _open_store("sql", tmp_path / "sql") as store:
        if source != "insert":
            await store.mutate(lambda tx: tx.insert_record(record))
        async def clock(session):
            raise failure
        monkeypatch.setattr(store.context.dialect, "database_current_time", clock)

        async def callback(tx):
            current = await _protect(tx, record, source)
            with pytest.raises(type(failure)) as caught:
                await tx.transition_record_lease(
                    current.key_digest, expected_storage_revision=current.storage_revision,
                    transition=RecordLeaseTransition("renew", "owner", 7, 5),
                )
            assert caught.value is failure
            await tx.insert_record(_record(2))
            return current

        result = await store.mutate(callback)
        records = await store.read(lambda tx: tx.scan_records())
        assert {item.key_digest: item for item in records} == {item.key_digest: item for item in (result, _record(2))}


@pytest.mark.parametrize("source", _PROTECTED)
@pytest.mark.parametrize("error_kind", ("unavailable", "cancel"))
async def test_sql_preprotected_lease_write_failure_aborts_even_when_caught(
    tmp_path, monkeypatch, source, error_kind,
):
    record = replace(_record(1), lease_owner="owner", lease_fence=7,
                     lease_expires_at=_STAMP + timedelta(seconds=30))
    failure = AIError(ErrorCode.STORAGE_UNAVAILABLE) if error_kind == "unavailable" else asyncio.CancelledError()
    async with _open_store("sql", tmp_path / "sql") as store:
        if source != "insert":
            await store.mutate(lambda tx: tx.insert_record(record))
        async def clock(session):
            return _STAMP
        monkeypatch.setattr(store.context.dialect, "database_current_time", clock)
        update = _SqlTransaction.update_record_lease
        async def failed(tx, *args, **kwargs):
            assert await update(tx, *args, **kwargs)
            raise failure
        monkeypatch.setattr(_SqlTransaction, "update_record_lease", failed)

        async def callback(tx):
            current = await _protect(tx, record, source)
            with pytest.raises(type(failure)):
                await tx.transition_record_lease(
                    current.key_digest, expected_storage_revision=current.storage_revision,
                    transition=RecordLeaseTransition("renew", "owner", 7, 5),
                )
            await tx.insert_record(_record(2))

        with pytest.raises(type(failure)) as caught:
            await store.mutate(callback)
        assert caught.value is failure
        assert await store.read(lambda tx: tx.scan_records()) == (() if source == "insert" else (record,))


@pytest.mark.parametrize("reinsert", (False, True))
async def test_sql_deleted_record_does_not_reuse_previous_lease_protection(tmp_path, monkeypatch, reinsert):
    record = replace(_record(1), lease_owner="old", lease_fence=7,
                     lease_expires_at=_STAMP + timedelta(seconds=30))
    async with _open_store("sql", tmp_path / "sql") as store:
        await store.mutate(lambda tx: tx.insert_record(record))
        async def clock(session):
            return _STAMP
        monkeypatch.setattr(store.context.dialect, "database_current_time", clock)

        async def callback(tx):
            guarded = await tx.guard_record(record.key_digest, expected_storage_revision=record.storage_revision)
            await tx.delete_records((record.key_digest,))
            assert await tx.transition_record_lease(
                record.key_digest, expected_storage_revision=guarded.storage_revision,
                transition=RecordLeaseTransition("guard", "old", 7),
            ) is None
            if not reinsert:
                return None
            successor = replace(record, lease_owner="new", lease_fence=8)
            await tx.insert_record(successor)
            assert await tx.transition_record_lease(
                record.key_digest, expected_storage_revision=successor.storage_revision,
                transition=RecordLeaseTransition("guard", "old", 7),
            ) is None
            return await tx.transition_record_lease(
                record.key_digest, expected_storage_revision=successor.storage_revision,
                transition=RecordLeaseTransition("guard", "new", 8),
            )

        result = await store.mutate(callback)
        assert await store.read(lambda tx: tx.get_record(record.key_digest)) == result
