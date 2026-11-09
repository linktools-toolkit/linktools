#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detached snapshot inventory and portable closure accounting stay bounded."""

import hashlib
import json
from copy import deepcopy

import pytest

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import NativeMessageRoot, NativeMessageSlice
from linktools.ai.persistence._archive import ContentRecord, ImmutableSequenceNode
from linktools.ai.persistence.codec import encode_domain, encode_record
from linktools.ai.persistence.contracts import ArchivedContextBlock
from linktools.ai.persistence.snapshots import (
    SnapshotLimits, _SnapshotCaptureBudget, _capture_snapshot_reader, snapshot_manifest_identity,
)
from linktools.ai.persistence.state import StoredRecord
from linktools.ai.snapshot._snapshot import _storage_snapshot_usage
from linktools.ai.storage import ObjectRef, StoredPayload


def _manifest(group: str) -> dict:
    raw = b"[]"
    digest = hashlib.sha256(raw).hexdigest()
    content = ContentRecord("model-message", digest, len(raw), "raw", StoredPayload.object(
        ObjectRef("runtime", f"v1/archive/{group}/scope/{digest}", digest, len(raw))))
    node = ImmutableSequenceNode(digest, 1, None, None, None)
    root = NativeMessageRoot(group, "a" * 64, (NativeMessageSlice(node.identity, 1, 0, 1),))
    reference = content.payload.ref
    return {
        "kind": "runtime-storage-snapshot", "format_version": 1, "namespace": "test", "tenant_id": "default",
        "domains": {"trace": {"records": [encode_domain(ArchivedContextBlock(root, digest, len(raw)))],
                               "aliases": [], "facts": [], "operations": [], "sequences": []}},
        "archives": [],
        "native_messages": [{"group_id": group, "domain": "trace", "messages": [{
            "store_digest": "a" * 64, "sha256": digest, "message": canonical_json_bytes(content.to_json()).decode(),
        }], "sequences": [{"store_digest": "a" * 64, "sha256": node.identity, "message_sha256": digest,
                           "tree_count": 1, "left_sha256": None, "right_sha256": None, "next_sha256": None}]}],
        "objects": [{"domain": "trace", "source": {"store_id": "runtime", "key": reference.key,
                     "digest": digest, "size": len(raw)}, "content": {
                     "store_id": "runtime", "key": "transport/" + group, "digest": digest, "size": len(raw)}}],
    }


def test_closure_usage_counts_each_transport_row_once() -> None:
    manifest = _manifest("1" * 32)
    manifest["archives"] = [{"scope_id": "1" * 32, "domain": "trace", "records": [{}, {}]}]
    assert _storage_snapshot_usage(manifest) == (6, 2)


def test_manifest_identity_preserves_relocated_native_content() -> None:
    assert snapshot_manifest_identity(_manifest("1" * 32)) == snapshot_manifest_identity(_manifest("2" * 32))


@pytest.mark.parametrize("component", ["messages", "sequences"])
def test_manifest_identity_includes_native_closure(component: str) -> None:
    original = _manifest("1" * 32)
    changed = deepcopy(original)
    if component == "messages":
        row = changed["native_messages"][0][component][0]
        body = json.loads(row["message"])
        body["raw_size"] += 1
        row["message"] = json.dumps(body)
    else:
        changed["native_messages"][0][component][0]["tree_count"] = 3
    assert snapshot_manifest_identity(changed) != snapshot_manifest_identity(original)


def test_manifest_identity_includes_archive_records() -> None:
    original = _manifest("1" * 32)
    content = ContentRecord("bytes", hashlib.sha256(b"x").hexdigest(), 1, "raw", StoredPayload.inline_bytes(b"x"))
    record = StoredRecord(b"a" * 32, b"b" * 32, None, "archive_content", content.identity,
                          None, 0, None, 0, None, content.to_json())
    original["archives"] = [{"scope_id": "1" * 32, "domain": "trace", "records": [encode_record(record)]}]
    changed = deepcopy(original)
    changed["archives"][0]["records"] = []
    assert snapshot_manifest_identity(changed) != snapshot_manifest_identity(original)


@pytest.mark.asyncio
async def test_capture_stops_on_shared_entry_budget_before_caching_more_sources() -> None:
    class Reader:
        consumed = 0

        async def scan_records_page(self, *, after, limit):
            start = 0 if after is None else int.from_bytes(after.key_digest, "big") + 1
            result = tuple(StoredRecord(index.to_bytes(32, "big"), None, None, "fixture", str(index),
                                        None, 0, None, 0, None, {}) for index in range(start, min(start + limit, 1000)))
            self.consumed += len(result)
            return result

        async def scan_facts_page(self, *, after, limit):
            raise AssertionError("overflow must stop before another source scan")

    reader = Reader()
    budget = _SnapshotCaptureBudget(SnapshotLimits(2, 100000))
    with pytest.raises(AIError) as rejected:
        await _capture_snapshot_reader(reader, budget=budget)
    assert rejected.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED
    assert budget.entries == 2
    assert reader.consumed == 3


def test_capture_byte_budget_rejects_before_retaining_a_large_body() -> None:
    budget = _SnapshotCaptureBudget(SnapshotLimits(10, 10))
    with pytest.raises(AIError) as rejected:
        budget.accept({"body": "x" * 100})
    assert rejected.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED
    assert (budget.entries, budget.byte_size) == (0, 0)


@pytest.mark.asyncio
async def test_capture_reads_only_bodies_selected_by_the_shared_native_closure() -> None:
    from linktools.ai.persistence.snapshots import _read_native_messages

    partition = "a" * 64
    wanted = hashlib.sha256(b"wanted").hexdigest()
    unrelated = hashlib.sha256(b"unrelated").hexdigest()
    nodes = [ImmutableSequenceNode(identity, 1, None, None, None) for identity in (wanted, unrelated)]
    rows = [{"store_digest": partition, "sha256": node.identity, "message_sha256": node.content_id,
             "tree_count": 1, "left_sha256": None, "right_sha256": None, "next_sha256": None} for node in nodes]
    content = ContentRecord("model-message", wanted, 6, "raw", StoredPayload.inline_bytes(b"wanted"))
    body = {"store_digest": partition, "sha256": wanted, "message": json.dumps(content.to_json())}
    body_requests = []

    class Transaction:
        async def read_sql_rows(self, context, statement):
            if "message" not in statement.selected_columns.keys():
                return tuple(sorted(rows, key=lambda row: row["sha256"]))
            requested = [value for value in statement.compile().params.values() if isinstance(value, list)]
            assert requested == [[wanted]], "unreachable message bodies must not enter the detached inventory"
            body_requests.extend(requested[0])
            return (body,)

    class Store:
        context = object()

        async def read(self, callback):
            return await callback(Transaction())

    root = NativeMessageRoot("group", partition, (NativeMessageSlice(nodes[0].identity, 1, 0, 1),))
    retained_nodes, retained_bodies = await _read_native_messages(
        Store(), [root, root], budget=_SnapshotCaptureBudget(SnapshotLimits(10, 10000)))
    assert [row["sha256"] for row in retained_nodes] == [nodes[0].identity]
    assert retained_bodies == [body]
    assert body_requests == [wanted]


def _inline_native(manifest: dict, *, compressed: bool = False) -> None:
    import zlib

    raw = b"[]"
    encoded = zlib.compress(raw) if compressed else raw
    row = manifest["native_messages"][0]["messages"][0]
    content = ContentRecord("model-message", row["sha256"], len(raw), "zlib" if compressed else "raw",
                            StoredPayload.inline_bytes(encoded))
    row["message"] = json.dumps(content.to_json())
    manifest["objects"] = []


@pytest.mark.parametrize("compressed", (False, True))
@pytest.mark.parametrize("retained_inline", (False, True))
def test_manifest_identity_merges_split_native_payload_representations(
    compressed: bool, retained_inline: bool,
) -> None:
    source = _manifest("1" * 32)
    other = _manifest("2" * 32)
    _inline_native(other, compressed=compressed)
    source["domains"]["conversation"] = other["domains"]["trace"]
    other["native_messages"][0]["domain"] = "conversation"
    source["native_messages"].extend(other["native_messages"])

    merged = _manifest("3" * 32)
    merged["domains"]["conversation"] = deepcopy(merged["domains"]["trace"])
    if retained_inline:
        _inline_native(merged, compressed=compressed)
    assert snapshot_manifest_identity(source) == snapshot_manifest_identity(merged)


def test_native_representation_normalization_keeps_shared_archive_object_dependency() -> None:
    source = _manifest("1" * 32)
    body = json.loads(source["native_messages"][0]["messages"][0]["message"])
    content = ContentRecord.from_json(body)
    record = StoredRecord(b"a" * 32, b"b" * 32, None, "archive_content", content.identity,
                          None, 0, None, 0, None, content.to_json())
    source["archives"] = [{"scope_id": "1" * 32, "domain": "trace", "records": [encode_record(record)]}]
    changed = deepcopy(source)
    _inline_native(changed)
    changed["objects"] = deepcopy(source["objects"])
    assert snapshot_manifest_identity(source) == snapshot_manifest_identity(changed)
    changed["objects"] = []
    assert snapshot_manifest_identity(source) != snapshot_manifest_identity(changed)


def test_native_semantic_identity_still_rejects_corrupt_inline_bytes() -> None:
    manifest = _manifest("1" * 32)
    _inline_native(manifest)
    row = manifest["native_messages"][0]["messages"][0]
    content = ContentRecord("model-message", row["sha256"], 2, "raw", StoredPayload.inline_bytes(b"{}"))
    row["message"] = json.dumps(content.to_json())
    with pytest.raises(AIError) as rejected:
        snapshot_manifest_identity(manifest)
    assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
