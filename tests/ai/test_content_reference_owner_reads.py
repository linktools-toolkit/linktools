#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overlapping content-read roles keep one physical owner and exact partitions."""

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from linktools.ai.persistence import _sql
from linktools.ai.persistence._archive import ContentRecord, archive_scope_key
from linktools.ai.persistence.state import StoredRecord, record_key, active_state_scope

from .test_state_group_record_batch import _database, _key, _record


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_count", (0, 1, 40))
async def test_content_read_overlap_preserves_current_owner_and_missing_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owner_count: int,
) -> None:
    async with _database(tmp_path / "overlap.db") as (_, first, second):
        owners = tuple(replace(_record(f"owner-{index}"), data={"body": "x" * 65536, "index": index})
                       for index in range(owner_count))
        common = owners[-1].key if owners else _key("extra-only")
        extra = _record("extra", key=common)
        foreign = replace(extra, data={"partition": "second"})
        await first.mutate(lambda tx: tx.insert_records(owners or (extra,)))
        await second.mutate(lambda tx: tx.insert_record(foreign))
        _, content = await first.storage_group.open_archive(create=True)
        missing = _key("missing")
        decoded = []
        original = _sql._record_from_row

        def capture(row):
            value = original(row)
            decoded.append((row["store_key"], value.key))
            return value

        monkeypatch.setattr(_sql, "_record_from_row", capture)

        async def read(group):
            values = await active_state_scope().record_reader.read_content_references(
                owner_store=first, owner_keys=(*(record.key for record in owners), missing) if owners else (),
                content_store=content, content_scope_key=archive_scope_key("overlap", "tenant"),
                content_paths=tuple(("unused", str(index)) for index in range(16)),
                additional_records={first: (common, missing), second: (common, missing)},
            )
            first_tx, second_tx = group.transaction(first), group.transaction(second)
            assert await first_tx.get_record(common) == (owners[-1] if owners else extra)
            assert await second_tx.get_record(common) == foreign
            assert first_tx.cached_records((missing,)) == {missing: None}
            assert second_tx.cached_records((missing,)) == {missing: None}
            if owners:
                assert values[first] == {record.key: record for record in owners}
            before = len(decoded)
            await group.get_records({first: (common, missing), second: (common, missing)})
            assert len(decoded) == before

        await first.storage_group.read_many((first, second, content), read)
        assert decoded.count((first.store_key, common)) == 1
        assert decoded.count((second.store_key, common)) == 1


@pytest.mark.asyncio
async def test_large_additional_batch_keeps_typed_payload_route(tmp_path: Path) -> None:
    async with _database(tmp_path / "payload.db") as (_, first, _second):
        _, archive = await first.storage_group.open_archive(create=True)
        body = b"canonical payload"
        value = ContentRecord("execution-payload", hashlib.sha256(body).hexdigest(), len(body), "raw", body=body)
        key = record_key("overlap", "tenant", "archive", "archive_content", (value.identity,))
        payload = StoredRecord(key, archive_scope_key("overlap", "tenant"), None, "archive_content",
                               value.identity.encode(), None, 0, None, 0, None, value.to_json(), payload_bytes=body)
        owners = tuple(_record(f"owner-{index}") for index in range(32))
        await archive.mutate(lambda tx: tx.insert_records((*owners, payload)))

        async def read(group):
            await active_state_scope().record_reader.read_content_references(
                owner_store=archive, owner_keys=tuple(record.key for record in owners), content_store=archive,
                content_scope_key=archive_scope_key("overlap", "tenant"), content_paths=(("unused",),),
                additional_records={archive: (*(record.key for record in owners), key)},
            )
            tx = group.transaction(archive)
            assert await tx.get_record(key) == payload
            assert await tx.get_records(tuple(record.key for record in owners)) == {
                record.key: record for record in owners}

        await first.storage_group.read_many((archive,), read)
