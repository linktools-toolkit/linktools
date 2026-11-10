#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Current Memory snapshots preserve content and operation receipts."""

import hashlib
from pathlib import Path

import pytest
from pydantic_ai_harness.memory import MemoryOperation

from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.memory import PersistentMemoryStore
from linktools.ai.snapshot import SnapshotLimits
from linktools.ai.storage import InMemoryObjectStore


@pytest.mark.asyncio
async def test_current_memory_snapshot_preserves_content_and_operation_receipt(tmp_path: Path) -> None:
    namespace, tenant = "memory-snapshot", "tenant"
    content = "快照内存\n" * 5000
    operation = MemoryOperation(id="write-once", fingerprint=hashlib.sha256(content.encode()).hexdigest())
    objects = InMemoryObjectStore("snapshots")
    limits = SnapshotLimits(max_entries=1000, max_bytes=2_000_000)

    def memory(state: Persistence) -> PersistentMemoryStore:
        return PersistentMemoryStore(
            state.memory, object_store=state.object_store(PersistenceDomain.MEMORY), namespace=namespace,
            tenant_id=tenant, execution_id="execution", memory_scope="retained",

        )

    async with Persistence.open(namespace, tenant_id=tenant,
                                storage=PersistenceStorage.from_root(tmp_path / "current")) as state:
        store = memory(state)
        first = await store.write("memory/note.md", content, expected_version=None, operation=operation)
        captured = await store.capture()
    async with Persistence.open(namespace, tenant_id=tenant, read_only=True,
                                storage=PersistenceStorage.from_root(tmp_path / "current")) as state:
        snapshot = await state.export_snapshot(object_store=objects, limits=limits)
    target = tmp_path / "restored"
    await Persistence.restore_snapshot(snapshot, object_store=objects, root=target, limits=limits)
    async with Persistence.open(namespace, tenant_id=tenant,
                                storage=PersistenceStorage.from_root(target)) as state:
        store = memory(state)
        assert await store.capture() == captured
        receipt = await store.get_operation(operation)
        assert receipt is not None and receipt.replayed and receipt.version == first.version
        replay = await store.write("memory/note.md", content, expected_version=None, operation=operation)
        assert replay.replayed and replay.version == first.version
        assert await store.capture() == captured
