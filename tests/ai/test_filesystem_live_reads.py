#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live filesystem readers observe only complete committed metadata."""

import asyncio
import hashlib
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state._filesystem import (
    FilesystemStateStorageGroup,
    FilesystemStateStore,
)
from linktools.ai.runtime.state._store import (
    FactQuery, OperationQuery, RecordQuery, RecordReplacement, StateTransaction,
    StoredAlias, StoredFact, StoredOperation, StoredRecord,
)

pytestmark = pytest.mark.asyncio


def _key(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def _record(value: str) -> StoredRecord:
    return StoredRecord(
        _key(value), None, None, "probe", value, "pending", 0,
        None, 0, None, {"value": 0},
    )


def _store(root: Path, *, grouped: bool, read_only: bool = False) -> FilesystemStateStore:
    group = FilesystemStateStorageGroup(
        root, namespace="live", tenant_id="tenant", scope_digest="scope",
        standalone=not grouped, read_only=read_only,
    )
    return FilesystemStateStore(
        root / "domain" if grouped else root,
        namespace="live", tenant_id="tenant", runtime_domain="execution",
        _range_index=True, group=group,
    )


@pytest.mark.parametrize("grouped", (False, True))
async def test_live_reader_invalidates_mutable_and_negative_caches(
    tmp_path: Path, grouped: bool,
) -> None:
    writer = _store(tmp_path / "state", grouped=grouped)
    reader = _store(tmp_path / "state", grouped=grouped, read_only=True)
    record = _record("record")
    alias, stream, sequence, operation_key = map(_key, ("alias", "stream", "seq", "op"))
    operation = StoredOperation(operation_key, stream, 1, "pending", True, {})
    fact = StoredFact(stream, 1, record.key_digest, "event", None, None, {})

    async def observe(transaction: StateTransaction) -> tuple[object, ...]:
        return (
            await transaction.get_record(record.key_digest),
            await transaction.resolve_alias(alias),
            await transaction.get_sequence(sequence),
            await transaction.list_facts(FactQuery(stream)),
            await transaction.get_operation(operation_key),
            await transaction.list_records(RecordQuery(kind="probe", limit=10)),
        )

    await writer.initialize()
    await reader.initialize()
    try:
        assert await reader.read(observe) == (None, None, 0, (), None, ())

        async def create(transaction: StateTransaction) -> None:
            await transaction.insert_record(record)
            await transaction.insert_alias(StoredAlias(alias, record.key_digest))
            await transaction.next_sequence(sequence)
            await transaction.insert_fact(fact)
            await transaction.insert_operation(operation)

        await writer.mutate(create)
        assert await reader.read(observe) == (
            record, record.key_digest, 1, (fact,), operation, (record,),
        )
        updated = replace(record, storage_version=1, data={"value": 1})
        completed = replace(operation, state="done")
        next_fact = replace(fact, sequence=2)

        async def update(transaction: StateTransaction) -> None:
            assert await transaction.replace_record(updated, expected_storage_version=0)
            await transaction.next_sequence(sequence)
            await transaction.insert_fact(next_fact)
            assert await transaction.replace_operation(completed, expected_state="pending")

        await writer.mutate(update)
        assert await reader.read(observe) == (
            updated, record.key_digest, 2, (fact, next_fact), completed, (updated,),
        )

        async def delete(transaction: StateTransaction) -> None:
            assert await transaction.delete_record(record.key_digest)
            await transaction.delete_sequence(sequence)
            await transaction.delete_operations(OperationQuery(stream_digest=stream))

        await writer.mutate(delete)
        assert await reader.read(observe) == (None, None, 0, (), None, ())
        await writer.mutate(create)
        assert await reader.read(observe) == (
            record, record.key_digest, 1, (fact,), operation, (record,),
        )
    finally:
        await reader.close()
        await writer.close()


@pytest.mark.parametrize("grouped", (False, True))
async def test_reader_waits_for_entire_multirecord_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, grouped: bool,
) -> None:
    writer = _store(tmp_path / "state", grouped=grouped)
    reader = _store(tmp_path / "state", grouped=grouped, read_only=True)
    records = (_record("first"), _record("second"))
    await writer.initialize()
    await writer.mutate(lambda transaction: transaction.insert_records(records))
    await reader.initialize()
    published = threading.Event()
    release = threading.Event()
    original_replace = os.replace

    def paused_replace(source: str | Path, destination: str | Path) -> None:
        original_replace(source, destination)
        if "stage" in Path(source).parts and "records" in Path(destination).parts:
            if not published.is_set():
                published.set()
                assert release.wait(5), "publication barrier timed out"

    monkeypatch.setattr(os, "replace", paused_replace)
    changed = tuple(replace(record, storage_version=1, data={"value": 1}) for record in records)
    commit = asyncio.create_task(writer.mutate(
        lambda transaction: transaction.replace_records(
            tuple(RecordReplacement(record, 0) for record in changed)
        )
    ))
    read = None
    try:
        assert await asyncio.to_thread(published.wait, 5)
        read = asyncio.create_task(reader.read(
            lambda transaction: transaction.get_records(tuple(record.key_digest for record in records))
        ))
        done, _ = await asyncio.wait((read,), timeout=0.05)
        assert not done, "a reader escaped while only one record was published"
        release.set()
        await asyncio.wait_for(commit, 5)
        result = await asyncio.wait_for(read, 5)
        assert result == {record.key_digest: record for record in changed}
    finally:
        release.set()
        await asyncio.gather(commit, *(() if read is None else (read,)), return_exceptions=True)
        await reader.close()
        await writer.close()


def _crash_during_publish(root: str, grouped: bool) -> None:
    async def run() -> None:
        writer = _store(Path(root), grouped=grouped)
        await writer.initialize()
        original_replace = os.replace

        def crash_replace(source: str | Path, destination: str | Path) -> None:
            original_replace(source, destination)
            if "stage" in Path(source).parts and "records" in Path(destination).parts:
                os._exit(91)

        os.replace = crash_replace
        await writer.mutate(lambda transaction: transaction.replace_records(tuple(
            RecordReplacement(replace(_record(value), storage_version=1, data={"value": 1}), 0)
            for value in ("first", "second")
        )))

    asyncio.run(run())


@pytest.mark.parametrize("grouped", (False, True))
async def test_reader_rejects_writer_death_until_writer_recovers(
    tmp_path: Path, grouped: bool,
) -> None:
    root = tmp_path / "state"
    writer = _store(root, grouped=grouped)
    records = (_record("first"), _record("second"))
    await writer.initialize()
    await writer.mutate(lambda transaction: transaction.insert_records(records))
    await writer.close()
    reader = _store(root, grouped=grouped, read_only=True)
    await reader.initialize()
    await reader.read(lambda transaction: transaction.get_record(records[0].key_digest))
    command = (
        "from tests.ai.test_filesystem_live_reads import _crash_during_publish; "
        f"_crash_during_publish({str(root)!r}, {grouped!r})"
    )
    process = await asyncio.to_thread(
        subprocess.run, (sys.executable, "-c", command),
        capture_output=True, text=True, timeout=15, check=False,
    )
    assert process.returncode == 91, process.stderr
    journal = root / (".txn-scope" if grouped else ".txn")
    before = {path.relative_to(journal): path.read_bytes() for path in journal.rglob("*") if path.is_file()}
    fresh = _store(root, grouped=grouped, read_only=True)
    recovered = _store(root, grouped=grouped)
    try:
        with pytest.raises(AIError) as error:
            await reader.read(lambda transaction: transaction.get_records(tuple(record.key_digest for record in records)))
        assert error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        with pytest.raises(AIError) as error:
            await fresh.initialize()
        assert error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        assert {path.relative_to(journal): path.read_bytes() for path in journal.rglob("*") if path.is_file()} == before
        await recovered.initialize()
        result = await reader.read(lambda transaction: transaction.get_records(tuple(record.key_digest for record in records)))
        assert len(result) == 2
        assert all(record.data["value"] == 1 for record in result.values())
        assert not journal.exists()
    finally:
        await fresh.close()
        await reader.close()
        await recovered.close()


async def test_live_read_open_and_refresh_never_scan_all_business_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "state"
    writer = _store(root, grouped=True)
    await writer.initialize()
    records = tuple(_record(f"item-{index:04}") for index in range(128))
    await writer.mutate(lambda transaction: transaction.insert_records(records))
    reader = _store(root, grouped=True, read_only=True)
    original_glob = Path.glob

    def bounded_glob(path: Path, pattern: str, **kwargs: object):
        if path == writer.root / "records":
            raise AssertionError("a live bounded read enumerated every stored record")
        return original_glob(path, pattern, **kwargs)

    monkeypatch.setattr(Path, "glob", bounded_glob)
    try:
        await reader.initialize()
        for index in range(8):
            value = replace(records[index], storage_version=1, data={"value": 1})
            await writer.mutate(lambda transaction: transaction.replace_record(value, expected_storage_version=0))
            assert await reader.read(lambda transaction: transaction.get_record(value.key_digest)) == value
    finally:
        await reader.close()
        await writer.close()


async def test_live_reader_preserves_offline_writer_exclusivity(tmp_path: Path) -> None:
    root = tmp_path / "state"
    writer = _store(root, grouped=True)
    reader = _store(root, grouped=True, read_only=True)
    offline = _store(root, grouped=True)
    await writer.initialize()
    await reader.initialize()
    try:
        with pytest.raises(AIError) as error:
            async with offline.storage_group.offline_exclusivity():
                pytest.fail("offline access admitted an active writer")
        assert error.value.code is ErrorCode.STORAGE_CONFLICT
        await writer.close()
        async with offline.storage_group.offline_exclusivity():
            assert await reader.read(lambda transaction: transaction.get_record(_key("missing"))) is None
    finally:
        await reader.close()
        await writer.close()
        await offline.close()


async def test_cancelled_live_read_releases_publication_fence(tmp_path: Path) -> None:
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
    root = tmp_path / "state"
    writer = _store(root, grouped=True)
    reader = _store(root, grouped=True, read_only=True)
    await writer.initialize()
    await reader.initialize()
    record = _record("record")
    entered = asyncio.Event()

    async def blocked_read(transaction: StateTransaction) -> None:
        assert await transaction.get_record(record.key_digest) is None
        entered.set()
        await asyncio.Event().wait()

    read = asyncio.create_task(reader.read(blocked_read))
    commit = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        commit = asyncio.create_task(writer.mutate(lambda transaction: transaction.insert_record(record)))
        done, _ = await asyncio.wait((commit,), timeout=0.05)
        assert not done
        read.cancel()
        with pytest.raises(asyncio.CancelledError):
            await read
        await asyncio.wait_for(commit, 5)
        assert await reader.read(lambda transaction: transaction.get_record(record.key_digest)) == record
    finally:
        read.cancel()
        await asyncio.gather(read, *(() if commit is None else (commit,)), return_exceptions=True)
        await reader.close()
        await writer.close()


@pytest.mark.parametrize("grouped", (False, True))
async def test_deleting_absent_and_unpublished_sequences_is_idempotent(
    tmp_path: Path, grouped: bool,
) -> None:
    writer = _store(tmp_path / "state", grouped=grouped)
    await writer.initialize()
    missing, transient, retained = map(_key, ("missing", "transient", "retained"))

    async def mutate(transaction: StateTransaction) -> None:
        await transaction.delete_sequences((missing, missing))
        await transaction.next_sequence(transient)
        await transaction.delete_sequences((transient, missing))
        assert await transaction.get_sequence(transient) == 0
        await transaction.next_sequence(retained)

    try:
        await writer.mutate(mutate)
        assert await writer.read(lambda transaction: transaction.scan_sequences()) == {retained: 1}
        await writer.mutate(lambda transaction: transaction.delete_sequences((missing, retained)))
        assert await writer.read(lambda transaction: transaction.scan_sequences()) == {}
        await writer.validate_integrity()
    finally:
        await writer.close()


async def test_live_reader_tracks_optional_index_changes_across_writer_restart(
    tmp_path: Path,
) -> None:
    root = tmp_path / "state"
    writer = _store(root, grouped=False)
    reader = _store(root, grouped=False, read_only=True)
    record = replace(_record("record"), scope_digest=_key("scope"))
    await writer.initialize()
    await writer.mutate(lambda transaction: transaction.insert_record(record))
    await reader.initialize()
    query = RecordQuery(kind="probe", scope_digest=record.scope_digest, limit=1)
    try:
        assert await reader.read(lambda transaction: transaction.list_records(query)) == (record,)
        await writer.close()
        writer = FilesystemStateStore(
            root, namespace="live", tenant_id="tenant", runtime_domain="execution",
        )
        await writer.initialize()
        assert await reader.read(lambda transaction: transaction.list_records(query)) == (record,)
    finally:
        await writer.close()
        await reader.close()
