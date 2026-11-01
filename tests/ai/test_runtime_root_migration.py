#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A local SQLite root never silently replaces recognizable filesystem state."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest
from pydantic_ai_harness.memory import MemoryOperation

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import FilesystemSnapshotReader, Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence.codec import _decode_enveloped_domain, decode_record
from linktools.ai.persistence.contracts import MemoryRecord
from linktools.ai.persistence.state import StoredRecord
from linktools.ai.persistence.runs import AgentRunRecord
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.runtime._memory import RuntimeMemoryStore
from linktools.ai.storage import FilesystemObjectStore, InMemoryObjectStore, ObjectRef, PayloadPolicy, read_object

from ._filesystem_legacy_fixture_helpers import _extract, legacy_provenance, load_legacy_source


NAMESPACE, TENANT = "legacy-reader", "tenant"


def _contents(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file() and path.name not in {"state.lock", "publication.lock"}}


async def _write_retained_run(storage: PersistenceStorage, run_id: str) -> AgentRunRecord:
    record = AgentRunRecord(run_id)
    async with Persistence.open(NAMESPACE, storage=storage, tenant_id=TENANT) as persistence:
        await persistence.run_store.read_store(PersistenceDomain.CHECKPOINT).register_agent_run(record)
    return record


async def _retained_run(persistence: Persistence, run_id: str) -> AgentRunRecord | None:
    return await persistence.run_store.read_store(PersistenceDomain.CHECKPOINT).get_agent_run(agent_run_id=run_id)


def _legacy_record(root: Path, domain: str, kind: str, provenance: dict | None = None) -> StoredRecord:
    provenance = legacy_provenance() if provenance is None else provenance
    names = [name for name in provenance["business_file_sha256"]
             if name.startswith(domain + "/") and "/records/" + kind + "/" in name]
    assert len(names) == 1
    return decode_record(json.loads((root / names[0]).read_bytes()))


def _legacy_run(root: Path, provenance: dict | None = None) -> AgentRunRecord:
    return _decode_enveloped_domain(_legacy_record(root, "checkpoint", "agent_run", provenance).data, AgentRunRecord)


async def _export_legacy(
    root: Path, objects: InMemoryObjectStore, limits: SnapshotLimits, *, provenance: dict | None = None,
) -> ObjectRef:
    provenance = legacy_provenance() if provenance is None else provenance
    reader = FilesystemSnapshotReader(root, namespace=provenance["namespace"], tenant_id=provenance["tenant_id"])
    try:
        return await reader.export_snapshot(object_store=objects, limits=limits)
    finally:
        await reader.close()


def _minimal_source(root: Path, name: str) -> dict:
    fixtures = Path(__file__).parent / "fixtures" / "persistence"
    provenance = json.loads((fixtures / "filesystem_minimal_roots_v1.provenance.json").read_text())
    bundle = fixtures / "filesystem_minimal_roots_v1.zip"
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == provenance["zip_sha256"]
    variant = provenance["variants"][name]
    root.mkdir()
    with zipfile.ZipFile(bundle) as archive:
        names = [value for value in archive.namelist() if value.startswith(name + "/")]
        assert sorted(names) == sorted(name + "/" + value for value in variant["business_file_sha256"])
        for path, digest in variant["business_file_sha256"].items():
            _extract(root, archive, name + "/" + path, path, digest)
    return variant


@pytest.mark.asyncio
@pytest.mark.parametrize("manifest", ("group", "members", "direct_member"))
async def test_from_root_rejects_recognizable_filesystem_state_without_touching_it(tmp_path: Path, manifest: str) -> None:
    root = tmp_path / "retained"
    load_legacy_source(root)
    retained = _legacy_run(root)
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
        objects = InMemoryObjectStore(legacy_provenance()["snapshot_object_store_id"])
        limits = SnapshotLimits(max_entries=1000, max_bytes=2_000_000)
        reference = await _export_legacy(root, objects, limits)
        assert reference.digest == legacy_provenance()["snapshot_digest"]
        target = tmp_path / "readback"
        await Persistence.restore_snapshot(reference, object_store=objects, root=target, limits=limits)
        async with Persistence.open(NAMESPACE, storage=PersistenceStorage.from_root(target),
                                    tenant_id=TENANT, read_only=True) as persistence:
            assert await _retained_run(persistence, "run") == retained
        assert _contents(root) == before


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
    root = tmp_path / "mixed"
    load_legacy_source(root)
    before = _contents(root)
    filesystem_run = _legacy_run(root)
    sqlite_run = await _write_retained_run(PersistenceStorage.sqlite(root / "runtime.db"), "sqlite-run")
    async with Persistence.open(NAMESPACE, storage=PersistenceStorage.from_root(root),
                                tenant_id=TENANT, read_only=True) as persistence:
        assert await _retained_run(persistence, "sqlite-run") == sqlite_run
        assert await _retained_run(persistence, "run") is None
    objects = InMemoryObjectStore(legacy_provenance()["snapshot_object_store_id"])
    limits = SnapshotLimits(max_entries=1000, max_bytes=2_000_000)
    reference = await _export_legacy(root, objects, limits)
    assert reference.digest == legacy_provenance()["snapshot_digest"]
    target = tmp_path / "legacy-only"
    await Persistence.restore_snapshot(reference, object_store=objects, root=target, limits=limits)
    async with Persistence.open(NAMESPACE, storage=PersistenceStorage.from_root(target),
                                tenant_id=TENANT, read_only=True) as persistence:
        assert await _retained_run(persistence, "run") == filesystem_run
        assert await _retained_run(persistence, "sqlite-run") is None
    assert all((root / name).read_bytes() == value for name, value in before.items())


@pytest.mark.asyncio
@pytest.mark.parametrize("exclusive_writer", (False, True))
async def test_filesystem_snapshot_restores_retained_run_into_a_new_sqlite_root(tmp_path: Path, exclusive_writer: bool) -> None:
    root, target = tmp_path / "source", tmp_path / "restored"
    provenance = _minimal_source(root, "run")
    before = _contents(root)
    retained = _legacy_run(root, provenance)
    objects = InMemoryObjectStore(provenance["snapshot_object_store_id"])
    limits = SnapshotLimits(max_entries=1000, max_bytes=1_000_000)
    reference = await _export_legacy(root, objects, limits, provenance=provenance)
    assert reference.digest == provenance["snapshot_digest"]
    await Persistence.restore_snapshot(reference, object_store=objects, root=target, limits=limits,
                                       exclusive_writer=exclusive_writer)
    async with Persistence.open(provenance["namespace"], storage=PersistenceStorage.from_root(target, exclusive_writer=exclusive_writer),
                                tenant_id=provenance["tenant_id"], read_only=True) as persistence:
        assert await _retained_run(persistence, "retained-run") == retained
        assert await persistence.export_snapshot(object_store=objects, limits=limits) == reference
    assert _contents(root) == before
    assert not (root / "runtime.db").exists() and (target / "runtime.db").is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize("exclusive_writer", (False, True))
async def test_filesystem_snapshot_preserves_core_memory_objects_and_replay_receipts(tmp_path: Path, exclusive_writer: bool) -> None:
    source, target = tmp_path / "source", tmp_path / "target"
    provenance = _minimal_source(source, "memory")
    before = _contents(source)
    record = _decode_enveloped_domain(_legacy_record(source, "memory", "memory", provenance).data, MemoryRecord)
    assert record.content.kind == "object" and record.content.ref is not None
    reference = record.content.ref
    content = (await read_object(FilesystemObjectStore(source / "memory" / "objects"), reference.key,
                                expected_digest=reference.digest, expected_size=reference.size)).decode("utf-8")
    assert content == "大内容迁移\n" * 5000
    assert len(content.encode()) == provenance["memory_bytes"]
    assert hashlib.sha256(content.encode()).hexdigest() == provenance["memory_digest"]
    assert record.metadata["version"] == provenance["memory_version"]
    captured = {record.metadata["path"]: {"content": content, "version": provenance["memory_version"]}}
    operation = MemoryOperation(id="write-once", fingerprint=hashlib.sha256(content.encode()).hexdigest())
    objects = InMemoryObjectStore(provenance["snapshot_object_store_id"])
    limits = SnapshotLimits(max_entries=1000, max_bytes=2_000_000)

    def memory(persistence: Persistence) -> RuntimeMemoryStore:
        return RuntimeMemoryStore(
            persistence.memory, object_store=persistence.object_store(PersistenceDomain.MEMORY),
            namespace=provenance["namespace"], tenant_id=provenance["tenant_id"], execution_id="execution",
            memory_scope="retained", payload_policy=PayloadPolicy(inline_limit_bytes=64),
        )

    reference = await _export_legacy(source, objects, limits, provenance=provenance)
    assert reference.digest == provenance["snapshot_digest"]
    manifest = json.loads(await read_object(objects, reference.key, expected_digest=reference.digest,
                                            expected_size=reference.size))
    assert len(manifest["domains"]["memory"]["records"]) == 1
    assert len(manifest["domains"]["memory"]["operations"]) == 1
    assert sum(item["domain"] == "memory" for item in manifest["objects"]) == 1
    await Persistence.restore_snapshot(reference, object_store=objects, root=target, limits=limits,
                                       exclusive_writer=exclusive_writer)
    async with Persistence.open(provenance["namespace"], storage=PersistenceStorage.from_root(target, exclusive_writer=exclusive_writer),
                                tenant_id=provenance["tenant_id"], read_only=True) as persistence:
        assert not persistence.durable_capability_ids
        assert await memory(persistence).capture() == captured
        assert await persistence.export_snapshot(object_store=objects, limits=limits) == reference
        receipt = await memory(persistence).get_operation(operation)
        assert receipt is not None and receipt.replayed and receipt.version == provenance["memory_version"]
    async with Persistence.open(provenance["namespace"], storage=PersistenceStorage.from_root(target, exclusive_writer=exclusive_writer),
                                tenant_id=provenance["tenant_id"]) as persistence:
        replay = await memory(persistence).write("memory/note.md", content, expected_version=None,
                                                operation=operation)
        assert replay.replayed and replay.version == provenance["memory_version"]
        assert await memory(persistence).capture() == captured
    assert _contents(source) == before
    assert not (source / "runtime.db").exists() and (target / "runtime.db").is_file()
