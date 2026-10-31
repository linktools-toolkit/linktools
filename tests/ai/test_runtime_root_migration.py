#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A local SQLite root never silently replaces recognizable filesystem state."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from pydantic_ai_harness.memory import MemoryOperation

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence.runs import AgentRunRecord
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.runtime._memory import RuntimeMemoryStore
from linktools.ai.storage import InMemoryObjectStore, PayloadPolicy, read_object


def _contents(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


async def _write_retained_run(storage: PersistenceStorage, run_id: str) -> AgentRunRecord:
    record = AgentRunRecord(run_id)
    async with Persistence.open("root-migration", storage=storage, tenant_id="tenant") as persistence:
        await persistence.run_store.read_store(PersistenceDomain.CHECKPOINT).register_agent_run(record)
    return record


async def _retained_run(persistence: Persistence, run_id: str) -> AgentRunRecord | None:
    return await persistence.run_store.read_store(PersistenceDomain.CHECKPOINT).get_agent_run(agent_run_id=run_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("manifest", ("group", "members", "direct_member"))
async def test_from_root_rejects_recognizable_filesystem_state_without_touching_it(tmp_path: Path, manifest: str) -> None:
    root = tmp_path / "retained"
    retained = await _write_retained_run(PersistenceStorage.filesystem(root), "retained-run")
    if manifest == "members":
        shutil.rmtree(root / ".state-groups")
    elif manifest == "direct_member":
        root = next((root / "checkpoint").glob("*/*/manifest.json")).parent
    before = _contents(root)
    with pytest.raises(AIError) as rejected:
        PersistenceStorage.from_root(root)
    assert rejected.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    assert "filesystem" in str(rejected.value).lower()
    assert _contents(root) == before
    assert not (root / "runtime.db").exists()
    if manifest == "group":
        async with Persistence.open("root-migration", storage=PersistenceStorage.filesystem(root),
                                    tenant_id="tenant", read_only=True) as persistence:
            assert await _retained_run(persistence, "retained-run") == retained


@pytest.mark.asyncio
async def test_from_root_does_not_treat_objects_cache_or_unrelated_manifests_as_state(tmp_path: Path) -> None:
    examples = {
        "objects/execution/n/t/manifest.json": {"format": "linktools-ai-state"},
        "cache/.state-groups/scope/manifest.json": {"format": "linktools-ai-state-group"},
        ".state-groups/unrelated/manifest.json": {"format": "application-settings"},
        "execution/n/t/manifest.json": {"format": "application-data"},
        "manifest.json": {"format": ["linktools-ai-state"]},
        ".state-groups/non-object/manifest.json": ["not a state manifest"],
    }
    for name, value in examples.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    malformed = tmp_path / "task/n/t/manifest.json"
    malformed.parent.mkdir(parents=True)
    malformed.write_text("ordinary non-JSON content", encoding="utf-8")
    before = _contents(tmp_path)
    await _write_retained_run(PersistenceStorage.from_root(tmp_path), "sqlite-run")
    assert (tmp_path / "runtime.db").is_file()
    assert all((tmp_path / name).read_bytes() == value for name, value in before.items())


@pytest.mark.asyncio
async def test_existing_sqlite_root_keeps_its_explicit_source_with_filesystem_state_present(tmp_path: Path) -> None:
    sqlite_run = await _write_retained_run(PersistenceStorage.from_root(tmp_path), "sqlite-run")
    filesystem_run = await _write_retained_run(PersistenceStorage.filesystem(tmp_path), "filesystem-run")
    async with Persistence.open("root-migration", storage=PersistenceStorage.from_root(tmp_path),
                                tenant_id="tenant", read_only=True) as persistence:
        assert await _retained_run(persistence, "sqlite-run") == sqlite_run
        assert await _retained_run(persistence, "filesystem-run") is None
    async with Persistence.open("root-migration", storage=PersistenceStorage.filesystem(tmp_path),
                                tenant_id="tenant", read_only=True) as persistence:
        assert await _retained_run(persistence, "filesystem-run") == filesystem_run


@pytest.mark.asyncio
@pytest.mark.parametrize("exclusive_writer", (False, True))
async def test_filesystem_snapshot_restores_retained_run_into_a_new_sqlite_root(tmp_path: Path, exclusive_writer: bool) -> None:
    root, target = tmp_path / "source", tmp_path / "restored"
    retained = await _write_retained_run(PersistenceStorage.filesystem(root), "retained-run")
    objects = InMemoryObjectStore()
    limits = SnapshotLimits(max_entries=1000, max_bytes=1_000_000)
    async with Persistence.open("root-migration", storage=PersistenceStorage.filesystem(root),
                                tenant_id="tenant", read_only=True) as persistence:
        reference = await persistence.export_snapshot(object_store=objects, limits=limits)
    before = _contents(root)
    await Persistence.restore_snapshot(reference, object_store=objects, root=target, limits=limits,
                                       exclusive_writer=exclusive_writer)
    async with Persistence.open("root-migration", storage=PersistenceStorage.from_root(target, exclusive_writer=exclusive_writer),
                                tenant_id="tenant", read_only=True) as persistence:
        assert await _retained_run(persistence, "retained-run") == retained
        assert await persistence.export_snapshot(object_store=objects, limits=limits) == reference
    assert _contents(root) == before
    assert not (root / "runtime.db").exists() and (target / "runtime.db").is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize("exclusive_writer", (False, True))
async def test_filesystem_snapshot_preserves_core_memory_objects_and_replay_receipts(tmp_path: Path, exclusive_writer: bool) -> None:
    source, target = tmp_path / "source", tmp_path / "target"
    content = "大内容迁移\n" * 5000
    operation = MemoryOperation(id="write-once", fingerprint=hashlib.sha256(content.encode()).hexdigest())
    objects = InMemoryObjectStore()
    limits = SnapshotLimits(max_entries=1000, max_bytes=2_000_000)

    def memory(persistence: Persistence) -> RuntimeMemoryStore:
        return RuntimeMemoryStore(
            persistence.memory, object_store=persistence.object_store(PersistenceDomain.MEMORY),
            namespace="memory-migration", tenant_id="tenant", execution_id="execution",
            memory_scope="retained", payload_policy=PayloadPolicy(inline_limit_bytes=64),
        )

    async with Persistence.open("memory-migration", storage=PersistenceStorage.filesystem(source),
                                tenant_id="tenant") as persistence:
        original = await memory(persistence).write("memory/note.md", content, expected_version=None,
                                                   operation=operation)
        captured = await memory(persistence).capture()
        assert not persistence.durable_capability_ids
    async with Persistence.open("memory-migration", storage=PersistenceStorage.filesystem(source),
                                tenant_id="tenant", read_only=True) as persistence:
        reference = await persistence.export_snapshot(object_store=objects, limits=limits)
    before = _contents(source)
    manifest = json.loads(await read_object(objects, reference.key, expected_digest=reference.digest,
                                            expected_size=reference.size))
    assert len(manifest["domains"]["memory"]["records"]) == 1
    assert len(manifest["domains"]["memory"]["operations"]) == 1
    assert sum(item["domain"] == "memory" for item in manifest["objects"]) == 1
    await Persistence.restore_snapshot(reference, object_store=objects, root=target, limits=limits,
                                       exclusive_writer=exclusive_writer)
    async with Persistence.open("memory-migration", storage=PersistenceStorage.from_root(target, exclusive_writer=exclusive_writer),
                                tenant_id="tenant", read_only=True) as persistence:
        assert await memory(persistence).capture() == captured
        assert await persistence.export_snapshot(object_store=objects, limits=limits) == reference
        receipt = await memory(persistence).get_operation(operation)
        assert receipt is not None and receipt.replayed and receipt.version == original.version
    async with Persistence.open("memory-migration", storage=PersistenceStorage.from_root(target, exclusive_writer=exclusive_writer),
                                tenant_id="tenant") as persistence:
        replay = await memory(persistence).write("memory/note.md", content, expected_version=None,
                                                operation=operation)
        assert replay.replayed and replay.version == original.version
        assert await memory(persistence).capture() == captured
    assert _contents(source) == before
