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
    "json", "version", "appended", "shape", "token", "resource", "kind", "envelope", "extra", "conflict",
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
            assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
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


@pytest.mark.parametrize("kind", ["write", "delete"])
async def test_memory_concurrent_mutation_replays_committed_receipt(
    kind: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="receipts", tenant_id="tenant")
    try:
        loser, winner = _store(state), _store(state)
        operation = MemoryOperation(id="concurrent", fingerprint="a" * 64)
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
            return await store.delete("memory/a.md", expected_version=None, operation=operation)

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
        await state.close()
