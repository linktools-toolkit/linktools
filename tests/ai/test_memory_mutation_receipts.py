#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Memory receipt bytes and repository replay keep one mutation result."""

import json
from pathlib import Path

import pytest
from pydantic_ai_harness.memory import MemoryConflictError, MemoryOperation, MemoryOperationConflictError

from linktools.ai.core import canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.runtime._memory import RuntimeMemoryStore, _MutationReceipt


@pytest.mark.parametrize("kind,existed,status", [
    ("write", False, "created"), ("write", True, "updated"),
    ("delete", True, "deleted"), ("delete", False, "not_found"),
])
def test_memory_receipt_preserves_exact_bytes_and_mutation(kind: str, existed: bool, status: str) -> None:
    version = "m1:" + "a" * 64 if kind == "write" else None
    receipt = _MutationReceipt("memory/a.md", version, existed, kind)
    token = json.dumps(version)
    expected = '{"result":{"file":"memory/a.md","status":"' + status + '","version":' + token + '},"version":1}'
    assert receipt.encode() == expected
    assert _MutationReceipt.decode(expected) == receipt
    for replayed in (False, True):
        mutation = receipt.mutation(replayed=replayed)
        assert (mutation.version, mutation.existed, mutation.replayed) == (version, existed, replayed)
    receipt.require_target("memory/a.md", kind)
    for path, target_kind in (("memory/b.md", kind), (receipt.path, "delete" if kind == "write" else "write")):
        with pytest.raises(MemoryOperationConflictError):
            receipt.require_target(path, target_kind)
    extended = json.loads(expected)
    extended["unused"] = "accepted extension"
    extended["result"]["unused"] = True
    assert _MutationReceipt.decode(json.dumps(extended)) == receipt


@pytest.mark.parametrize("payload", [
    "not-json", "[]", '{"version":true,"result":{}}', '{"version":2,"result":{}}',
    '{"version":1}', '{"version":1,"result":[]}',
    *[json.dumps({"version": 1, "result": result}) for result in (
        {"file": "../a", "version": "m1:" + "a" * 64, "status": "created"},
        {"file": "memory/a", "version": None, "status": "created"},
        {"file": "memory/a", "version": "bad", "status": "updated"},
        {"file": "memory/a", "version": "m1:" + "a" * 64, "status": "deleted"},
        {"file": "memory/a", "version": None, "status": "appended"},
        {"file": "memory/a", "status": "not_found"},
    )],
])
def test_memory_receipt_preserves_integrity_errors(payload: str) -> None:
    with pytest.raises(AIError) as raised:
        _MutationReceipt.decode(payload)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_memory_receipt_real_repository_replay_and_conflicts(tmp_path: Path, backend: str) -> None:
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.from_root(tmp_path)
    state = Persistence(storage)
    await state.initialize(namespace="receipt-test", tenant_id="tenant")
    try:
        store = RuntimeMemoryStore(state.memory, object_store=state.object_store(PersistenceDomain.MEMORY),
                                   namespace="receipt-test", tenant_id="tenant", execution_id="execution", memory_scope="scope")
        operation = MemoryOperation("create", canonical_sha256({"content": "first"}))
        created = await store.write("memory/a.md", "first", expected_version=None, operation=operation)
        assert not created.replayed and not created.existed
        original = await store.read("memory/a.md", max_chars=100)
        assert original is not None and original.content == "first"
        replay = await store.write("memory/a.md", "first", expected_version=None, operation=operation)
        assert (replay.version, replay.replayed, replay.existed) == (created.version, True, False)
        assert await store.get_operation(operation) == replay
        assert await store.read("memory/a.md", max_chars=100) == original
        for path, op in (("memory/b.md", operation), ("memory/a.md", MemoryOperation("create", "b" * 64))):
            with pytest.raises(MemoryOperationConflictError):
                await store.write(path, "different", expected_version=None, operation=op)
        with pytest.raises(MemoryOperationConflictError):
            await store.delete("memory/a.md", expected_version=created.version, operation=operation)
        assert await store.read("memory/a.md", max_chars=100) == original
        assert await store.read("memory/b.md", max_chars=100) is None
        updated = await store.write("memory/a.md", "second", expected_version=created.version)
        assert updated.existed and not updated.replayed and updated.version != created.version
        current = await store.read("memory/a.md", max_chars=100)
        assert current is not None and current.content == "second"
        with pytest.raises(MemoryConflictError):
            await store.write("memory/a.md", "stale", expected_version=created.version)
        assert await store.write("memory/a.md", "first", expected_version=None, operation=operation) == replay
        assert await store.read("memory/a.md", max_chars=100) == current
        delete_op = MemoryOperation("delete", "c" * 64)
        deleted = await store.delete("memory/a.md", expected_version=updated.version, operation=delete_op)
        assert (deleted.version, deleted.replayed, deleted.existed) == (None, False, True)
        assert await store.read("memory/a.md", max_chars=100) is None
        replay_delete = await store.delete("memory/a.md", expected_version=updated.version, operation=delete_op)
        assert (replay_delete.version, replay_delete.replayed, replay_delete.existed) == (None, True, True)
        assert await store.get_operation(delete_op) == replay_delete
        missing_op = MemoryOperation("missing", "d" * 64)
        missing = await store.delete("memory/a.md", expected_version=None, operation=missing_op)
        assert (missing.version, missing.replayed, missing.existed) == (None, False, False)
        assert (await store.get_operation(missing_op)).existed is False
        recreated = await store.write("memory/a.md", "new incarnation", expected_version=None)
        assert not recreated.replayed and not recreated.existed
        after_recreate = await store.read("memory/a.md", max_chars=100)
        assert after_recreate is not None and after_recreate.content == "new incarnation"
        assert await store.delete("memory/a.md", expected_version=updated.version, operation=delete_op) == replay_delete
        assert await store.read("memory/a.md", max_chars=100) == after_recreate
    finally:
        await state.close()
