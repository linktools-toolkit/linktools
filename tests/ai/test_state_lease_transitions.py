#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lease transitions preserve fencing and reject expiry during SQL CAS waits."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence._sql import _SqlTransaction
from linktools.ai.persistence.state import (
    RecordLeaseTransition, StateTransaction, StateTransactionNestingError, StoredRecord,
)
from tests.ai.test_state_batch_delete_backends import _open_store, _record

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sql"))
async def test_lease_transitions_preserve_fencing_and_snapshot_data(
    backend: str, tmp_path: Path,
) -> None:
    record = _record(1)
    async with _open_store(backend, tmp_path / backend) as store:
        await store.mutate(lambda tx: tx.insert_record(record))

        async def transition(
            current: StoredRecord, operation: Literal["acquire", "renew", "release"],
            owner: str, fence: int, seconds: int | None = None,
        ) -> StoredRecord | None:
            return await store.mutate(lambda tx: tx.transition_record_lease(
                current.key_digest, expected_storage_revision=current.storage_revision,
                transition=RecordLeaseTransition(operation, owner, fence, seconds),
            ))

        acquired = await transition(record, "acquire", "first", 0, 60)
        assert acquired is not None and acquired.lease_fence == 1
        assert acquired.lease_owner == "first" and acquired.data == record.data
        for operation, owner, fence in (("renew", "other", 1), ("release", "first", 0),
                                        ("acquire", "other", 1)):
            assert await transition(acquired, operation, owner, fence,
                                    None if operation == "release" else 60) is None
        assert await store.read(lambda tx: tx.get_record(record.key_digest)) == acquired
        renewed = await transition(acquired, "renew", "first", 1, 120)
        assert renewed is not None and renewed.lease_fence == 1
        assert renewed.lease_expires_at is not None and acquired.lease_expires_at is not None
        assert renewed.lease_expires_at > acquired.lease_expires_at
        released = await transition(renewed, "release", "first", 1)
        assert released is not None and released.lease_owner is None
        assert released.lease_expires_at is None and released.lease_fence == 1
        successor = await transition(released, "acquire", "second", 1, 60)
        assert successor is not None and successor.lease_fence == 2
        assert await transition(successor, "release", "first", 1) is None
        assert await store.read(lambda tx: tx.get_record(record.key_digest)) == successor
        with pytest.raises(StateTransactionNestingError):
            await store.read(lambda tx: tx.transition_record_lease(
                record.key_digest, expected_storage_revision=successor.storage_revision,
                transition=RecordLeaseTransition("release", "second", 2),
            ))


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sql"))
async def test_expired_lease_cannot_renew_or_release_but_can_be_taken_over(
    backend: str, tmp_path: Path,
) -> None:
    record = replace(_record(1), lease_owner="old", lease_fence=7,
                     lease_expires_at=datetime(2000, 1, 1, tzinfo=timezone.utc))
    async with _open_store(backend, tmp_path / backend) as store:
        await store.mutate(lambda tx: tx.insert_record(record))
        for operation, seconds in (("renew", 60), ("release", None)):
            assert await store.mutate(lambda tx: tx.transition_record_lease(
                record.key_digest, expected_storage_revision=record.storage_revision,
                transition=RecordLeaseTransition(operation, "old", 7, seconds),
            )) is None
        acquired = await store.mutate(lambda tx: tx.transition_record_lease(
            record.key_digest, expected_storage_revision=record.storage_revision,
            transition=RecordLeaseTransition("acquire", "new", 7, 60),
        ))
        assert acquired is not None and acquired.lease_fence == 8
        assert acquired.lease_owner == "new"


@pytest.mark.parametrize("after_wait", (37, 50))
async def test_sql_lease_resamples_time_after_cas_and_rolls_back_caught_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, after_wait: int,
) -> None:
    epoch = datetime(2001, 1, 1, tzinfo=timezone.utc)
    sampled = epoch + timedelta(seconds=35)
    record = replace(_record(1), lease_owner="owner", lease_fence=1,
                     lease_expires_at=epoch + timedelta(seconds=40))
    unrelated = _record(2)
    async with _open_store("sql", tmp_path / "sql") as store:
        await store.mutate(lambda tx: tx.insert_record(record))

        async def clock(session: "AsyncSession") -> datetime:
            return sampled

        monkeypatch.setattr(store.context.dialect, "database_now", clock)
        monkeypatch.setattr(store.context.dialect, "database_current_time", clock)
        guard = _SqlTransaction.guard_record

        async def delayed_guard(
            self: _SqlTransaction, key: bytes, *, expected_storage_revision: int,
        ) -> StoredRecord | None:
            nonlocal sampled
            result = await guard(self, key, expected_storage_revision=expected_storage_revision)
            sampled = epoch + timedelta(seconds=after_wait)
            return result

        monkeypatch.setattr(_SqlTransaction, "guard_record", delayed_guard)

        async def callback(tx: StateTransaction) -> StoredRecord | None:
            assert await tx.now() == epoch + timedelta(seconds=35)
            if after_wait == 50:
                with pytest.raises(AIError) as caught:
                    await tx.transition_record_lease(
                        record.key_digest, expected_storage_revision=record.storage_revision,
                        transition=RecordLeaseTransition("renew", "owner", 1, 30),
                    )
                assert caught.value.code is ErrorCode.STORAGE_CONFLICT
                await tx.insert_record(unrelated)
                return None
            result = await tx.transition_record_lease(
                record.key_digest, expected_storage_revision=record.storage_revision,
                transition=RecordLeaseTransition("renew", "owner", 1, 30),
            )
            assert await tx.now() == epoch + timedelta(seconds=35)
            return result

        if after_wait == 50:
            with pytest.raises(AIError) as caught:
                await store.mutate(callback)
            assert caught.value.code is ErrorCode.STORAGE_CONFLICT
            assert await store.read(lambda tx: tx.get_record(record.key_digest)) == record
            assert await store.read(lambda tx: tx.get_record(unrelated.key_digest)) is None
        else:
            result = await store.mutate(callback)
            assert result is not None
            assert result.lease_expires_at == epoch + timedelta(seconds=67)
            assert result.lease_fence == 1
            assert await store.read(lambda tx: tx.get_record(record.key_digest)) == result


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sql"))
async def test_guarded_record_still_requires_current_revision_for_lease_transition(
    backend: str, tmp_path: Path,
) -> None:
    record = replace(_record(1), lease_owner="owner", lease_fence=1,
                     lease_expires_at=datetime(2999, 1, 1, tzinfo=timezone.utc))
    async with _open_store(backend, tmp_path / backend) as store:
        await store.mutate(lambda tx: tx.insert_record(record))

        async def callback(tx: StateTransaction) -> StoredRecord:
            guarded = await tx.guard_record(
                record.key_digest, expected_storage_revision=record.storage_revision,
            )
            assert guarded is not None
            assert await tx.transition_record_lease(
                record.key_digest, expected_storage_revision=record.storage_revision,
                transition=RecordLeaseTransition("release", "owner", 1),
            ) is None
            assert await tx.get_record(record.key_digest) == guarded
            return guarded

        guarded = await store.mutate(callback)
        assert await store.read(lambda tx: tx.get_record(record.key_digest)) == guarded
