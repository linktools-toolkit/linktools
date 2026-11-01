#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Memory receipts retain durable replay and integrity semantics."""

import asyncio
import json
from dataclasses import fields
from pathlib import Path

import pytest
from pydantic_ai_harness.memory import MemoryOperation, MemoryOperationConflictError

from linktools.ai.core import OperationKind, OperationLedgerInput, OperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.runtime._memory import RuntimeMemoryStore, _operation_id

pytestmark = pytest.mark.asyncio


def _store(state: Persistence) -> RuntimeMemoryStore:
    return RuntimeMemoryStore(
        state.memory, object_store=state.object_store(PersistenceDomain.MEMORY),
        namespace="receipts", tenant_id="tenant", execution_id="execution",
        memory_scope="workspace",
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_memory_receipts_roundtrip_all_mutation_statuses(
    backend: str, tmp_path: Path,
) -> None:
    storage = (PersistenceStorage.in_memory() if backend == "memory"
               else PersistenceStorage.sqlite(tmp_path / "receipts.db"))
    state = Persistence(storage)
    await state.initialize(namespace="receipts", tenant_id="tenant")
    try:
        store = _store(state)
        prior = None
        for index, kind in enumerate(("write", "write", "delete", "delete")):
            operation = MemoryOperation(id=f"operation-{index}", fingerprint=str(index) * 64)
            if kind == "write":
                mutation = await store.write("memory/a.md", str(index), expected_version=prior, operation=operation)
            else:
                mutation = await store.delete("memory/a.md", expected_version=prior, operation=operation)
            assert mutation.existed is (index in (1, 2))
            assert mutation.replayed is False
            if backend == "sqlite":
                await state.close()
                state = Persistence(storage)
                await state.initialize(namespace="receipts", tenant_id="tenant")
                store = _store(state)
            replay = await store.get_operation(operation)
            assert replay is not None
            assert (replay.version, replay.existed, replay.replayed) == (mutation.version, mutation.existed, True)
            if kind == "write":
                assert await store.write("memory/a.md", str(index), expected_version=prior, operation=operation) == replay
            else:
                assert await store.delete("memory/a.md", expected_version=prior, operation=operation) == replay
            current = await store.read("memory/a.md", max_chars=100)
            assert (current.content if current else None) == (str(index) if kind == "write" else None)
            prior = mutation.version
    finally:
        await state.close()


@pytest.mark.parametrize("damage", [
    "json", "version", "malformed_version", "path", "appended", "shape", "token", "resource", "kind", "envelope", "extra", "conflict",
])
async def test_memory_receipt_validates_durable_ledger_readback(damage: str) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="receipts", tenant_id="tenant")
    try:
        store = _store(state)
        original = MemoryOperation(id="original", fingerprint="a" * 64)
        mutation = await store.write("memory/a.md", "content", expected_version=None, operation=original)
        record = await state.memory.operations.get(
            _operation_id(store._memory_scope_digest, original.id), tenant_id="tenant",
        )
        assert record is not None
        payload = json.loads(record.result_ref)
        # Rebind a real writer's receipt to a new ledger entry. Deletes have no
        # version token, so malformed envelopes reach their intended validation.
        payload["result"].update(version=None, status="deleted")
        values = {field.name: getattr(record, field.name) for field in fields(OperationLedgerInput)}
        operation = MemoryOperation(id="damaged", fingerprint="a" * 64)
        values.update(operation_id=_operation_id(store._memory_scope_digest, operation.id), operation_kind=OperationKind.MEMORY_DELETE)
        if damage == "version":
            payload["version"] = 2
        elif damage == "malformed_version":
            payload["version"] = True
        elif damage == "path":
            payload["result"]["file"] = "../invalid"
        elif damage == "appended":
            payload["result"]["status"] = "appended"
        elif damage == "shape":
            payload["result"] = []
        elif damage == "token":
            payload["result"].update(status="created", version=mutation.version)
            values["operation_kind"] = OperationKind.MEMORY_WRITE
        elif damage == "resource":
            values["resource_id"] = "wrong-resource"
        elif damage == "kind":
            values["operation_kind"] = OperationKind.MEMORY_WRITE
        elif damage == "envelope":
            values["status"] = OperationStatus.FAILED
        elif damage == "extra":
            payload["additional"] = True
            payload["result"]["additional"] = "allowed"
        values["result_ref"] = "{" if damage in ("json", "conflict") else json.dumps(payload)
        await state.memory.operations.append(OperationLedgerInput(**values))
        if damage == "conflict":
            operation = MemoryOperation(id=operation.id, fingerprint="b" * 64)
            with pytest.raises(MemoryOperationConflictError):
                await store.get_operation(operation)
        elif damage == "extra":
            replay = await store.get_operation(operation)
            assert replay is not None and replay.existed and replay.replayed
        else:
            with pytest.raises(AIError) as raised:
                await store.get_operation(operation)
            expected = (ErrorCode.STORAGE_VERSION_UNSUPPORTED if damage == "version"
                        else ErrorCode.STORAGE_INTEGRITY_ERROR)
            assert raised.value.code is expected
            if damage in {"json", "version", "malformed_version", "path", "appended", "shape"}:
                with pytest.raises(AIError) as repository_error:
                    await state.memory.records.apply_delete(
                        values["resource_id"], tenant_id="tenant", expected_revision=0,
                        operation=OperationLedgerInput(**values),
                    )
                assert repository_error.value.code is expected
        assert (await store.read("memory/a.md", max_chars=100)).content == "content"
    finally:
        await state.close()


@pytest.mark.parametrize("target", ["path", "kind"])
async def test_memory_replay_rejects_different_target(target: str) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="receipts", tenant_id="tenant")
    try:
        store = _store(state)
        operation = MemoryOperation(id="original", fingerprint="a" * 64)
        created = await store.write("memory/a.md", "content", expected_version=None, operation=operation)
        with pytest.raises(MemoryOperationConflictError):
            if target == "path":
                await store.write("memory/b.md", "content", expected_version=None, operation=operation)
            else:
                await store.delete("memory/a.md", expected_version=created.version, operation=operation)
        assert (await store.read("memory/a.md", max_chars=100)).content == "content"
        assert await store.read("memory/b.md", max_chars=100) is None
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("kind", ["write", "delete", "delete_existing"])
async def test_memory_concurrent_mutation_replays_committed_receipt(
    kind: str, backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = (PersistenceStorage.in_memory() if backend == "memory"
               else PersistenceStorage.sqlite(tmp_path / "concurrent.db"))
    state = Persistence(storage)
    other = Persistence(storage) if backend == "sqlite" else state
    await state.initialize(namespace="receipts", tenant_id="tenant")
    try:
        if other is not state:
            await other.initialize(namespace="receipts", tenant_id="tenant")
        loser, winner = _store(state), _store(other)
        operation = MemoryOperation(id="concurrent", fingerprint="a" * 64)
        version = None
        if kind == "delete_existing":
            version = (await winner.write("memory/a.md", "initial", expected_version=None)).version
        observed = asyncio.Event()
        committed = asyncio.Event()
        read_record = loser._record

        async def pause_after_read(path: str):
            record = await read_record(path)
            observed.set()
            await committed.wait()
            return record

        monkeypatch.setattr(loser, "_record", pause_after_read)

        async def mutate(store: RuntimeMemoryStore):
            if kind == "write":
                return await store.write("memory/a.md", "content", expected_version=None, operation=operation)
            return await store.delete("memory/a.md", expected_version=version, operation=operation)

        pending = asyncio.create_task(mutate(loser))
        try:
            await asyncio.wait_for(observed.wait(), timeout=10)
            mutation = await mutate(winner)
            committed.set()
            replay = await asyncio.wait_for(pending, timeout=10)
            assert mutation.replayed is False
            assert (replay.version, replay.existed, replay.replayed) == (
                mutation.version, mutation.existed, True,
            )
            assert await winner.get_operation(operation) == replay
        finally:
            committed.set()
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
    finally:
        if other is not state:
            await other.close()
        await state.close()


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("failure", ["unknown", "cancel"])
async def test_memory_receipt_readback_resolves_storage_boundary_failure(
    committed: bool, failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = Persistence(PersistenceStorage.sqlite(tmp_path / "outcome.db"))
    await state.initialize(namespace="receipts", tenant_id="tenant")
    try:
        store = _store(state)
        operation = MemoryOperation(id="interrupted", fingerprint="a" * 64)
        version = (await store.write("memory/a.md", "initial", expected_version=None)).version
        boundary = state.memory.records.state_store
        mutate = boundary.mutate
        reached = asyncio.Event()
        release = asyncio.Event()

        async def interrupt(callback):
            if committed:
                await mutate(callback)
            reached.set()
            if failure == "unknown":
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            await release.wait()

        with monkeypatch.context() as patch:
            patch.setattr(boundary, "mutate", interrupt)
            pending = asyncio.create_task(store.delete(
                "memory/a.md", expected_version=version, operation=operation,
            ))
            try:
                await asyncio.wait_for(reached.wait(), timeout=10)
                if failure == "cancel":
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                else:
                    with pytest.raises(AIError) as raised:
                        await pending
                    assert raised.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
            finally:
                release.set()
                if not pending.done():
                    pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
        receipt = await store.get_operation(operation)
        assert (receipt is not None) is committed
        current = await store.read("memory/a.md", max_chars=100)
        assert (current is None) is committed
        if current is not None:
            assert current.content == "initial" and current.version == version
        result = await store.delete("memory/a.md", expected_version=version, operation=operation)
        assert result.replayed is committed
        assert result.version is None and result.existed is True
        assert await store.read("memory/a.md", max_chars=100) is None
        assert await store.get_operation(operation) is not None
    finally:
        await state.close()


async def test_memory_receipt_writer_preserves_wire_bytes() -> None:
    from linktools.ai.persistence.contracts import MemoryMutationReceipt

    for kind, existed, status in (
        ("write", False, "created"), ("write", True, "updated"),
        ("delete", True, "deleted"), ("delete", False, "not_found"),
    ):
        version = "m1:" + "a" * 64 if kind == "write" else None
        receipt = MemoryMutationReceipt("memory/a.md", version, existed, kind)
        expected = json.dumps(
            {"version": 1, "result": {"file": "memory/a.md", "version": version, "status": status}},
            ensure_ascii=False, separators=(",", ":"), sort_keys=True,
        )
        assert receipt.encode() == expected
        assert MemoryMutationReceipt.decode(expected) == receipt


@pytest.mark.parametrize("change", [
    {"kind": "append"}, {"existed": 1}, {"path": "../invalid"}, {"version": "invalid"},
])
async def test_memory_receipt_writer_rejects_invalid_fields(change: dict) -> None:
    from linktools.ai.persistence.contracts import MemoryMutationReceipt

    values = dict(path="memory/a.md", version="m1:" + "a" * 64, existed=False, kind="write")
    values.update(change)
    with pytest.raises(AIError) as raised:
        MemoryMutationReceipt(**values).encode()
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("version", [0, True, "1"])
async def test_memory_receipt_rejects_malformed_wire_version(version: object) -> None:
    from linktools.ai.persistence.contracts import MemoryMutationReceipt

    value = json.dumps({"version": version, "result": {
        "file": "memory/a.md", "version": None, "status": "deleted",
    }})
    with pytest.raises(AIError) as raised:
        MemoryMutationReceipt.decode(value)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


async def test_memory_replay_preserves_input_validation_precedence() -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="receipts", tenant_id="tenant")
    try:
        store = _store(state)
        operation = MemoryOperation(id="original", fingerprint="a" * 64)
        await store.write("memory/a.md", "content", expected_version=None, operation=operation)
        for path, content, version in (
            ("../invalid", "content", None),
            ("memory/a.md", "\x00", None),
            ("memory/a.md", "content", "invalid"),
        ):
            with pytest.raises(ValueError):
                await store.write(path, content, expected_version=version, operation=operation)
        with pytest.raises(ValueError):
            await store.delete("memory/a.md", expected_version="invalid", operation=operation)
        assert (await store.read("memory/a.md", max_chars=100)).content == "content"
    finally:
        await state.close()
