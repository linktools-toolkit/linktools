#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capture bindings own their archive independently of execution retention."""

import sqlite3

from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine

from linktools.ai.agent import AgentBindingContract
from linktools.ai.core import ExecutionStatus, Principal, TenantAuthorizationPolicy, canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.inputs import CaptureInputRequest, InputCaptures
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._archive import ContentArchive, PreparedArchive, archive_content_scope
from linktools.ai.persistence.captures import (
    capture_record, capture_record_digest, capture_record_key, capture_record_value, input_capture_state_dependency,
)
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.persistence.state import StateGroupTransaction, StateTransaction, StoredRecord, active_state_scope
from linktools.ai.storage import InMemoryObjectStore, StorageRoute

from .test_archive_closure import Offline
from .test_history_projection_conformance import _record

pytestmark = pytest.mark.asyncio
_PRINCIPAL = Principal("principal", "tenant", "service")
_BODY = "CAPTURE_BINDING_ONLY_OWNER_" * 2048


def _storage(path: Path, *, split: bool = False) -> PersistenceStorage:
    storage = PersistenceStorage.sqlite(path / "execution.db")
    return PersistenceStorage(replace(storage.plan, task=StorageRoute.sqlite(path / "capture.db"))) if split else storage


def _captures(state: Persistence) -> InputCaptures:
    return InputCaptures.from_persistence(state, authorization=TenantAuthorizationPolicy("tenant"),
                                         execution=SimpleNamespace())


async def _seed(state: Persistence) -> AgentBindingContract:
    record = _record(ExecutionStatus.STARTED, 1)
    binding = replace(record.binding, agent_spec=replace(record.binding.agent_spec, system_prompt=_BODY))
    await state.execution.executions.create_execution(replace(record, binding=binding))
    return binding


async def _capture_rows(state: Persistence) -> tuple[StoredRecord, ...]:
    rows = await state.task.tasks.state_store.read(lambda tx: tx.scan_records())
    return tuple(row for row in rows if row.kind == "input_capture"
                 and capture_record_value(row).get("kind") == "agent")


@pytest.mark.parametrize("split", (False, True))
async def test_capture_binding_survives_execution_removal_gc_and_snapshot(tmp_path: Path, split: bool) -> None:
    transport = InMemoryObjectStore("capture-binding-snapshot")
    limits = SnapshotLimits(max_entries=4096, max_bytes=16 * 1024 * 1024)
    async with Persistence.open("binding-owner", storage=_storage(tmp_path, split=split), tenant_id="tenant") as state:
        binding = await _seed(state)
        service = _captures(state)
        references = tuple([await service.capture_input("execution", CaptureInputRequest(_PRINCIPAL, key))
                            for key in ("first", "second")])
        rows = await _capture_rows(state)
        assert len(rows) == 2
        pointers = [capture_record_value(row)["payload"]["binding"] for row in rows]
        assert pointers[0] == pointers[1] and "$execution_binding" in pointers[0]
        assert _BODY.encode() not in canonical_json_bytes([capture_record_value(row) for row in rows])
        assert all(len(canonical_json_bytes(capture_record_value(row))) < 2048 for row in rows)
        group = state.task.tasks.state_store.storage_group
        assert pointers[0]["archive_scope"] == group.opened_archive[0]
        if split:
            assert group is not state.execution.executions.state_store.storage_group
        archive_rows = await group.opened_archive[1].read(lambda tx: tx.scan_records())
        binding_rows = [row for row in archive_rows if row.kind == "archive_content" and row.data.get("codec") == "execution-binding"]
        assert len(binding_rows) == 1 and binding_rows[0].payload_bytes is not None
        source_rows = await state.execution.executions.state_store.read(lambda tx: tx.scan_records())
        await state.execution.executions.state_store.mutate(lambda tx: tx.delete_records(tuple(
            row.key for row in source_rows if row.kind == "execution")))
        await state.purge_unreferenced_objects((), exclusive=Offline())
        for reference in references:
            assert (await service.read_agent(reference, principal=_PRINCIPAL)).binding == binding
    async with Persistence.open("binding-owner", storage=_storage(tmp_path, split=split), tenant_id="tenant", read_only=True) as state:
        snapshot = await state.export_snapshot(object_store=transport, limits=limits)
    await Persistence.restore_snapshot(snapshot, object_store=transport, root=tmp_path / "restored", limits=limits)
    async with Persistence.open("binding-owner", storage=PersistenceStorage.from_root(tmp_path / "restored"), tenant_id="tenant") as state:
        for reference in references:
            assert (await _captures(state).read_agent(reference, principal=_PRINCIPAL)).binding == binding


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_capture_binding_and_owner_rollback_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    storage = PersistenceStorage.in_memory() if backend == "memory" else _storage(tmp_path, split=True)
    async with Persistence.open("binding-rollback", storage=storage, tenant_id="tenant") as state:
        binding = await _seed(state)
        service = _captures(state)
        original = ContentArchive.publish
        archive_store = (await state.task.tasks.state_store.storage_group.open_archive(create=True))[1]
        original_rows = await archive_store.read(lambda tx: tx.scan_records())

        async def fail_after_content(self: ContentArchive, transaction: StateTransaction, prepared: PreparedArchive) -> None:
            await original(self, transaction, prepared)
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

        with monkeypatch.context() as patch:
            patch.setattr(ContentArchive, "publish", fail_after_content)
            with pytest.raises(AIError) as failure:
                await service.capture_input("execution", CaptureInputRequest(_PRINCIPAL, "retry"))
        assert failure.value.code is ErrorCode.STORAGE_UNAVAILABLE
        assert await _capture_rows(state) == ()
        archive_rows = await archive_store.read(lambda tx: tx.scan_records())
        assert archive_rows == original_rows
        if backend == "sqlite":
            assert not any(row.kind == "archive_content" and row.data.get("codec") == "execution-binding" for row in archive_rows)
        reference = await service.capture_input("execution", CaptureInputRequest(_PRINCIPAL, "retry"))
        assert (await service.read_agent(reference, principal=_PRINCIPAL)).binding == binding


@pytest.mark.parametrize("corruption", ("missing", "body", "scope", "summary"))
async def test_capture_binding_reference_never_falls_back_to_execution_or_latest(
    tmp_path: Path, corruption: str,
) -> None:
    async with Persistence.open("binding-integrity", storage=_storage(tmp_path), tenant_id="tenant") as state:
        await _seed(state)
        service = _captures(state)
        reference = await service.capture_input("execution", CaptureInputRequest(_PRINCIPAL, "capture"))
        row, = await _capture_rows(state)
        value = capture_record_value(row)
        pointer = value["payload"]["binding"]
        assert set(pointer) == {"$execution_binding", "archive_scope", "content_scope"}
        if corruption in {"missing", "body"}:
            archive_store = state.task.tasks.state_store.storage_group.opened_archive[1]
            rows = await archive_store.read(lambda tx: tx.scan_records())
            owner = next(item for item in rows if item.kind == "archive_content"
                         and item.data.get("codec") == "execution-binding")
            if corruption == "missing":
                await archive_store.mutate(lambda tx: tx.delete_records((owner.key,)))
            else:
                assert await archive_store.mutate(lambda tx: tx.replace_record(
                    replace(owner, payload_bytes=b"corrupt", storage_revision=owner.storage_revision + 1),
                    expected_storage_revision=owner.storage_revision))
        else:
            if corruption == "scope":
                pointer["content_scope"] = archive_content_scope("binding-integrity", "another-tenant")
            else:
                pointer["summary"] = {"id": "another-agent"}
            changed = capture_record("binding-integrity", "tenant", capture_record_key(row), value,
                storage_revision=row.storage_revision + 1, source_digest=capture_record_digest(row))
            if corruption == "summary":
                with pytest.raises(AIError) as rejected_write:
                    await state.task.tasks.state_store.mutate(lambda tx: tx.replace_record(
                        changed, expected_storage_revision=row.storage_revision))
                assert rejected_write.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
                # Bypass the validated writer to exercise an actually corrupt source.
                with sqlite3.connect(tmp_path / "execution.db") as connection:
                    updated = connection.execute(
                        "UPDATE ai_state_records SET payload_json = ? WHERE kind = ? AND storage_revision = ?",
                        (canonical_json_bytes(changed.data).decode("utf-8"), "input_capture", row.storage_revision),
                    )
                    assert updated.rowcount == 1
            else:
                assert await state.task.tasks.state_store.mutate(lambda tx: tx.replace_record(
                    changed, expected_storage_revision=row.storage_revision))
        with pytest.raises(AIError) as failure:
            await service.read_agent(reference, principal=_PRINCIPAL)
        assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("remove_owner", (False, True))
async def test_cold_capture_read_rebuilds_owner_and_binding_in_one_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remove_owner: bool,
) -> None:
    async with Persistence.open("binding-cold", storage=_storage(tmp_path), tenant_id="tenant") as state:
        binding = await _seed(state)
        reference = await _captures(state).capture_input("execution", CaptureInputRequest(_PRINCIPAL, "capture"))
        row, = await _capture_rows(state)
    async with Persistence.open("binding-cold", storage=_storage(tmp_path), tenant_id="tenant") as state:
        store = state.task.tasks.state_store
        group = store.storage_group
        assert group.opened_archive is None
        original_open = group.open_archive
        original_read = ContentArchive.read_contents
        body_reads = 0
        opened = 0

        async def open_archive(*, create: bool = False) -> object:
            nonlocal opened
            assert active_state_scope() is None and create is False
            opened += 1
            result = await original_open(create=create)
            if remove_owner:
                archive_store = result[1]
                records = await archive_store.read(lambda tx: tx.scan_records())

                async def remove(transactions: StateGroupTransaction) -> None:
                    await transactions.transaction(store).delete_record(row.key)
                    await transactions.transaction(archive_store).delete_records(tuple(item.key for item in records
                        if item.kind == "archive_content" and item.data.get("codec") == "execution-binding"))

                await group.mutate((store, archive_store), remove)
            return result

        async def read_contents(self: ContentArchive, *args: object, **kwargs: object) -> object:
            nonlocal body_reads
            scope = active_state_scope()
            assert scope is not None and scope.group is group and not scope.writable
            assert store in scope.transactions and self.store in scope.transactions
            body_reads += 1
            return await original_read(self, *args, **kwargs)

        monkeypatch.setattr(group, "open_archive", open_archive)
        monkeypatch.setattr(ContentArchive, "read_contents", read_contents)
        if remove_owner:
            with pytest.raises(AIError) as absent:
                await _captures(state).read_agent(reference, principal=_PRINCIPAL)
            assert absent.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
            assert body_reads == 0
        else:
            assert (await _captures(state).read_agent(reference, principal=_PRINCIPAL)).binding == binding
            assert body_reads == 1
        assert opened == 1


@pytest.mark.parametrize("split", (False, True))
async def test_capture_binding_warm_and_cold_readonly_sql_budget(tmp_path: Path, split: bool) -> None:
    storage = _storage(tmp_path, split=split)
    async with Persistence.open("binding-reads", storage=storage, tenant_id="tenant") as state:
        binding = await _seed(state)
        reference = await _captures(state).capture_input("execution", CaptureInputRequest(_PRINCIPAL, "capture"))
    async with Persistence.open("binding-reads", storage=storage, tenant_id="tenant", read_only=True) as state:
        service = _captures(state)
        statements: list[str] = []

        def collect(_connection: object, _cursor: object, statement: str,
                    _parameters: object, _context: object, _many: bool) -> None:
            statements.append(statement)

        event.listen(Engine, "before_cursor_execute", collect)
        try:
            for expected in ({"BEGIN": 4, "SELECT": 5}, {"BEGIN": 2, "SELECT": 3}):
                statements.clear()
                assert (await service.read_agent(reference, principal=_PRINCIPAL)).binding == binding
                assert dict(Counter(sql.split()[0].upper() for sql in statements)) == expected
                assert not any(name in sql for sql in statements for name in ("ai_objects", "ai_step_messages"))
        finally:
            event.remove(Engine, "before_cursor_execute", collect)


@pytest.mark.parametrize("split", (False, True))
async def test_capture_binding_is_collected_after_its_final_owner_is_removed(tmp_path: Path, split: bool) -> None:
    async with Persistence.open("binding-release", storage=_storage(tmp_path, split=split), tenant_id="tenant") as state:
        binding = await _seed(state)
        service = _captures(state)
        references = [await service.capture_input("execution", CaptureInputRequest(_PRINCIPAL, key))
                      for key in ("first", "second")]
        source = state.execution.executions.state_store
        records = await source.read(lambda tx: tx.scan_records())
        await source.mutate(lambda tx: tx.delete_records(tuple(row.key for row in records if row.kind == "execution")))
        archive_store = state.task.tasks.state_store.storage_group.opened_archive[1]
        for remaining, reference in zip((1, 0), references, strict=True):
            key, _ = input_capture_state_dependency(reference)
            deleted = await state.snapshot_operations().purge_unreferenced_captures((key,), exclusive=Offline())
            assert deleted.deleted == (key,)
            await state.purge_unreferenced_objects((), exclusive=Offline())
            archived = await archive_store.read(lambda tx: tx.scan_records())
            bindings = [row for row in archived if row.kind == "archive_content" and row.data.get("codec") == "execution-binding"]
            assert len(bindings) == remaining
            if remaining:
                assert (await service.read_agent(references[1], principal=_PRINCIPAL)).binding == binding
            with pytest.raises(AIError) as unavailable:
                await service.read_agent(reference, principal=_PRINCIPAL)
            assert unavailable.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
        assert await _capture_rows(state) == ()
