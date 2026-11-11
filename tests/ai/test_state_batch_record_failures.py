#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch preconditions fail without staging a partial record mutation."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence.state import RecordReplacement, StateTransaction
from tests.ai.test_state_batch_delete_backends import _open_store, _record

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("backend", ("memory", "sql"))
async def test_record_storage_revision_is_independent_of_domain_revision(
    backend: str, tmp_path: Path,
) -> None:
    record = replace(_record(1), data={"revision": 7})
    expires_at = datetime(2030, 1, 1, tzinfo=timezone.utc)
    async with _open_store(backend, tmp_path / backend) as store:
        await store.mutate(lambda tx: tx.insert_record(record))

        guarded = await store.mutate(lambda tx: tx.guard_record(
            record.key, expected_storage_revision=0,
        ))
        assert guarded == replace(record, storage_revision=1)
        assert await store.mutate(lambda tx: tx.update_record_lease(
            record.key, expected_storage_revision=1,
            lease_owner="worker", lease_fence=1, lease_expires_at=expires_at,
        ))
        leased = replace(
            record, storage_revision=2, lease_owner="worker", lease_fence=1,
            lease_expires_at=expires_at,
        )
        assert await store.read(lambda tx: tx.get_record(record.key)) == leased

        async def stale_mutations(transaction: StateTransaction) -> None:
            assert await transaction.guard_record(
                record.key, expected_storage_revision=1,
            ) is None
            assert not await transaction.update_record_lease(
                record.key, expected_storage_revision=1,
                lease_owner="stale", lease_fence=2, lease_expires_at=expires_at,
            )
            assert not await transaction.replace_record(
                replace(leased, data={"revision": 8}), expected_storage_revision=1,
            )
            assert not await transaction.delete_record(
                record.key, expected_storage_revision=1,
            )

        await store.mutate(stale_mutations)
        assert await store.read(lambda tx: tx.get_record(record.key)) == leased
        updated = replace(leased, storage_revision=3, data={"revision": 8})
        assert await store.mutate(lambda tx: tx.replace_record(
            updated, expected_storage_revision=2,
        ))
        assert await store.read(lambda tx: tx.get_record(record.key)) == updated
        assert not await store.mutate(lambda tx: tx.delete_record(
            record.key, expected_storage_revision=2,
        ))
        assert await store.mutate(lambda tx: tx.delete_record(
            record.key, expected_storage_revision=3,
        ))
        assert await store.read(lambda tx: tx.get_record(record.key)) is None


@pytest.mark.parametrize("backend", ("memory", "sql"))
async def test_caught_insert_conflict_preserves_batch_atomicity(
    backend: str, tmp_path: Path,
) -> None:
    first, existing = sorted((_record(1), _record(2)), key=lambda r: r.key)
    other = _record(3)
    async with _open_store(backend, tmp_path / backend) as store:
        await store.mutate(lambda tx: tx.insert_record(existing))

        async def callback(transaction):
            with pytest.raises((AIError, IntegrityError)):
                await transaction.insert_records((first, existing))
            await transaction.insert_record(other)

        await store.mutate(callback)
        assert await store.read(lambda tx: tx.get_record(first.key)) is None
        assert await store.read(lambda tx: tx.get_record(existing.key)) == existing
        assert await store.read(lambda tx: tx.get_record(other.key)) == other


@pytest.mark.parametrize("backend", ("memory", "sql"))
async def test_caught_replace_conflict_aborts_unrelated_write(
    backend: str, tmp_path: Path,
) -> None:
    records = tuple(sorted((_record(1), _record(2)), key=lambda r: r.key))
    other = _record(3)
    async with _open_store(backend, tmp_path / backend) as store:
        await store.mutate(lambda tx: tx.insert_records(records))

        async def callback(transaction):
            with pytest.raises(AIError) as raised:
                await transaction.replace_records(tuple(
                    RecordReplacement(replace(record, storage_revision=index + 1), index)
                    for index, record in enumerate(records)
                ))
            assert raised.value.code is ErrorCode.STORAGE_CONFLICT
            assert not await transaction.replace_record(
                replace(records[-1], storage_revision=2), expected_storage_revision=1,
            )
            await transaction.insert_record(other)

        with pytest.raises(AIError):
            await store.mutate(callback)
        assert await store.read(lambda tx: tx.get_records(tuple(r.key for r in records))) == {
            record.key: record for record in records
        }
        assert await store.read(lambda tx: tx.get_record(other.key)) is None
