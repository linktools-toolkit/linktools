#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact archive reachability across shared prefixes and namespace partitions."""

import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from linktools.ai.core import canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state._archive import ContentArchive, ContentRecord, PreparedArchive
from linktools.ai.runtime.state._archive_closure import ArchiveClosure, iter_archive_content_refs, iter_archive_roots, rekey_archive_roots
from linktools.ai.runtime.state._codec import _encode_persisted_domain, decode_domain, encode_domain
from linktools.ai.runtime.state._contracts import ArchivedContextBlock, RuntimePayloadRef
from linktools.ai.runtime.state._memory import InMemoryStateStore
from linktools.ai.runtime.state._object_cleanup import purge_unreferenced_objects
from linktools.ai.runtime.state._plan import RuntimeDomain
from linktools.ai.runtime.state._store import StoredRecord
from linktools.ai.storage import InMemoryObjectStore, ObjectRef, StoredPayload

pytestmark = pytest.mark.asyncio


class Offline:
    @asynccontextmanager
    async def offline_exclusivity(self) -> AsyncIterator[None]:
        yield


async def _prepare(archive: ContentArchive, values: tuple[bytes, ...]):
    prepared = PreparedArchive()
    identities = await archive.prepare_content(tuple(("bytes", value) for value in values), prepared)
    root = await archive.append(archive.empty_root(), identities, prepared)
    await archive.store.mutate(lambda tx: archive.publish(tx, prepared))
    return root, identities


async def test_archive_closure_preserves_partial_visible_content_and_shared_dag() -> None:
    owner = InMemoryStateStore()
    await owner.initialize()
    scope, store = await owner.storage_group.open_archive(create=True)
    archive = ContentArchive(store, scope, namespace="archive", tenant_id="tenant", object_store=None)
    root, identities = await _prepare(archive, (b"A", b"B", b"C", b"D", b"E"))
    partial = root.slice(1, 2)
    records = await store.read(lambda tx: tx.scan_records())
    closure = ArchiveClosure(scope, records)
    closure.mark(partial)
    closure.mark(partial)
    retained = closure.retained_records
    assert {record.sort_key for record in retained if record.kind == "archive_content"} == {identities[1]}
    for record in closure.unreachable_records:
        await store.mutate(lambda tx, record=record: tx.delete_record(record.key_digest))
    assert await archive.read_contents(await archive.content_ids(partial), codec="bytes") == (b"B",)
    restored = ArchiveClosure(scope, retained)
    restored.mark(partial)
    with pytest.raises(AIError) as error:
        restored.mark(root)
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


async def test_archive_closure_keys_content_roots_by_logical_tenant_scope() -> None:
    owner = InMemoryStateStore()
    await owner.initialize()
    scope, store = await owner.storage_group.open_archive(create=True)
    first = ContentArchive(store, scope, namespace="archive", tenant_id="first", object_store=None)
    second = ContentArchive(store, scope, namespace="archive", tenant_id="second", object_store=None)
    root, identities = await _prepare(first, (b"same",))
    await _prepare(second, (b"same",))
    records = await store.read(lambda tx: tx.scan_records())
    closure = ArchiveClosure(scope, records)
    direct = {"archive_scope": scope, "content_scope": canonical_sha256(["archive", "first"]), "content_id": identities[0]}
    manifest = {"kind": "execution-context", "version": 1, "history": root.to_json(),
                "session_metadata": {"value": direct}, "memory": None, "repository_instructions": None,
                "semantic_digest": "digest", "replace_history_system_prompt": False, "unavailable_reason": None}
    wire = {"$dataclass": "execution_record", "fields": {"input_context": encode_domain(
        RuntimePayloadRef(StoredPayload.inline_json(manifest), RuntimeDomain.EXECUTION))}}
    assert tuple(iter_archive_roots(wire)) == (root,)
    assert tuple(iter_archive_content_refs(wire)) == ((scope, direct["content_scope"], identities[0]),)
    closure.mark_content(*tuple(iter_archive_content_refs(wire))[0])
    assert len(closure.retained_records) == 1
    assert closure.retained_records[0].scope_digest.hex() == direct["content_scope"]
    rewritten = rekey_archive_roots(wire, {scope: "restored"})
    assert tuple(iter_archive_content_refs(rewritten))[0][0] == "restored"
    repaired = decode_domain(rewritten["fields"]["input_context"], RuntimePayloadRef)
    assert repaired.payload.decode()["session_metadata"]["value"]["archive_scope"] == "restored"
    assert tuple(iter_archive_roots(rewritten))[0].scope_id == "restored"
    block = ArchivedContextBlock(root, hashlib.sha256(b"semantic").hexdigest(), 8)
    block_wire = encode_domain(block)
    assert tuple(iter_archive_roots(block_wire)) == (root,)
    restored_block = decode_domain(rekey_archive_roots(block_wire, {scope: "restored"}), ArchivedContextBlock)
    assert restored_block.root.scope_id == "restored"
    assert (restored_block.raw_digest, restored_block.raw_size) == (block.raw_digest, block.raw_size)


async def test_archive_scanner_preserves_business_keys_and_codec_lookalikes() -> None:
    business = {"metadata": {"archive_scope": "business-source", "content_scope": "customer-category",
                             "slices": [], "content_id": "business-id", "other": "unchanged"},
                "output": {"$dataclass": "archive_root", "fields": {"scope_id": "business-source"}}}
    wire = encode_domain(business)
    assert not tuple(iter_archive_roots(wire))
    assert not tuple(iter_archive_content_refs(wire))
    assert rekey_archive_roots(wire, {"business-source": "physical-target"}) == wire
    payload = encode_domain(StoredPayload.inline_json(business))
    assert rekey_archive_roots(payload, {"business-source": "physical-target"}) == payload


async def test_archive_gc_scans_unrouted_owner_and_collects_only_excluded_content() -> None:
    owner = InMemoryStateStore()
    hidden = InMemoryStateStore(owner._storage_group)
    await owner.initialize()
    await hidden.initialize()
    scope, store = await owner.storage_group.open_archive(create=True)
    objects = InMemoryObjectStore()
    archive = ContentArchive(store, scope, namespace="archive", tenant_id="tenant", object_store=objects)
    values = tuple(b"".join(hashlib.sha256(f"{item}:{index}".encode()).digest() for index in range(1024)) for item in range(3))
    root, identities = await _prepare(archive, values)
    record = StoredRecord(hashlib.sha256(b"hidden-owner").digest(), None, None, "probe", "hidden", None,
                          0, None, 0, None, {"root": encode_domain(root.slice(1, 2))})
    await hidden.mutate(lambda tx: tx.insert_record(record))
    contents = {item.sort_key: ContentRecord.from_json(item.data) for item in await store.read(lambda tx: tx.scan_records()) if item.kind == "archive_content"}
    candidates = tuple((RuntimeDomain.EXECUTION, contents[identity].payload.ref) for identity in identities)
    result = await purge_unreferenced_objects(candidates, namespace="archive", tenant_id="tenant",
        stores={RuntimeDomain.EXECUTION: owner}, object_stores={domain: objects for domain in RuntimeDomain}, exclusive=Offline())
    assert result.retained == (candidates[1],)
    assert set(result.deleted) == {candidates[0], candidates[2]}
    assert await archive.read_contents(await archive.content_ids(root.slice(1, 2)), codec="bytes") == (values[1],)
    remaining = await store.read(lambda tx: tx.scan_records())
    assert {item.sort_key for item in remaining if item.kind == "archive_content"} == {identities[1]}


async def test_archive_closure_full_prefix_work_scales_with_unique_nodes_and_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    async def measure(size: int) -> int:
        owner = InMemoryStateStore()
        await owner.initialize()
        scope, store = await owner.storage_group.open_archive(create=True)
        archive = ContentArchive(store, scope, namespace="archive", tenant_id="tenant", object_store=None)
        prepared = PreparedArchive()
        root = archive.empty_root()
        roots = []
        for index in range(size):
            identities = await archive.prepare_content((("bytes", str(index).encode()),), prepared)
            root = await archive.append(root, identities, prepared)
            roots.append(root)
        await store.mutate(lambda tx: archive.publish(tx, prepared))
        closure = ArchiveClosure(scope, await store.read(lambda tx: tx.scan_records()))
        calls = 0

        def counted(method):
            def invoke(*args, **kwargs):
                nonlocal calls
                calls += 1
                return method(*args, **kwargs)
            return invoke

        for name in ("_record", "_node", "_mark_structure", "_forest", "_mark_full_forest", "_mark_tree"):
            monkeypatch.setattr(closure, name, counted(getattr(closure, name)))
        for value in (*roots, *roots):
            closure.mark(value)
        return calls

    small = await measure(16)
    large = await measure(256)
    assert large < small * 18


async def test_archive_scanner_accepts_persisted_portable_object_context() -> None:
    payload = StoredPayload.object(ObjectRef("runtime", "v1/runtime/portable-context", "0" * 64, 80000))
    wire = {"$dataclass": "execution_record", "schema": 1, "fields": {"input_context":
        _encode_persisted_domain(RuntimePayloadRef(payload, RuntimeDomain.EXECUTION))}}
    assert not tuple(iter_archive_roots(wire))
    assert not tuple(iter_archive_content_refs(wire))
    assert rekey_archive_roots(wire, {}) == wire
