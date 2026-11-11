#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detached snapshot inventory and portable closure accounting stay bounded."""

import hashlib
import json
import zlib
from copy import deepcopy
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, UserPromptPart

from linktools.ai.steps import native_store_key, encode_native_store_key
from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import NativeMessageRoot, NativeMessageSlice
from linktools.ai.model.messages import encode_model_messages
from linktools.ai.persistence._snapshot import encode_native_message_row, decode_native_message_row
from linktools.ai.persistence._archive import archive_content_scope, archive_scope_key, ContentRecord, ImmutableSequenceNode
from linktools.ai.persistence.codec import encode_domain, encode_record
from linktools.ai.persistence.contracts import ArchivedContextBlock
from linktools.ai.persistence.snapshots import (
    SnapshotLimits, _SnapshotCaptureBudget, _capture_snapshot_reader, snapshot_manifest_identity,
)
from linktools.ai.persistence.state import record_key, StoredRecord
from linktools.ai.snapshot._snapshot import _storage_snapshot_usage
from linktools.ai.storage import ObjectRef, StoredPayload


def _raw() -> bytes:
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return encode_model_messages((ModelRequest(parts=[UserPromptPart([
        BinaryContent(b"[]", media_type="application/octet-stream"),
    ], timestamp=stamp)], timestamp=stamp),))


def _content(group: str, *, external: bool, compressed: bool = False) -> ContentRecord:
    raw = _raw()
    digest = hashlib.sha256(raw).hexdigest()
    if not external:
        return ContentRecord("model-message", digest, len(raw), "zlib" if compressed else "raw",
                             body=zlib.compress(raw) if compressed else raw)
    file_digest = hashlib.sha256(b"[]").hexdigest()
    ref = ObjectRef("runtime", f"v1/archive/{group}/{archive_content_scope('archive', 'tenant')}/{file_digest}", file_digest, 2)
    value = json.loads(raw)
    value[0]["parts"][0]["content"][0]["data"] = None
    payload = StoredPayload("inline", "json-attachments", digest, len(raw), {
        "format": "model-messages", "content": value, "attachments": [{
            "path": [0, "parts", 0, "content", 0, "data"],
            "ref": {"store_id": ref.store_id, "key": ref.key, "digest": ref.digest, "size": ref.size},
        }],
    })
    return ContentRecord("model-message", digest, len(raw), "raw", payload)


def _row(content: ContentRecord) -> dict:
    return {"store_key": native_store_key("a", "tenant"), "sha256": content.raw_digest, **content.to_native_values()}


def _manifest(group: str) -> dict:
    content = _content(group, external=True)
    node = ImmutableSequenceNode(content.raw_digest, 1, None, None, None)
    root = NativeMessageRoot(group, native_store_key("a", "tenant"), (NativeMessageSlice(node.identity, 1, 0, 1),))
    reference, = content.object_refs()
    return {
        "kind": "runtime-storage-snapshot", "format_version": 1, "namespace": "test", "tenant_id": "default",
        "domains": {"trace": {"records": [encode_domain(ArchivedContextBlock(root, content.raw_digest, content.raw_size))],
                               "aliases": [], "facts": [], "operations": [], "sequences": []}},
        "archives": [],
        "native_messages": [{"group_id": group, "domain": "trace", "messages": [encode_native_message_row(_row(content))],
            "sequences": [{"store_key": encode_native_store_key(native_store_key("a", "tenant")), "sha256": node.identity, "message_sha256": content.raw_digest,
                           "tree_count": 1, "left_sha256": None, "right_sha256": None, "next_sha256": None}]}],
        "objects": [{"domain": "trace", "source": {"store_id": "runtime", "key": reference.key,
                     "digest": reference.digest, "size": reference.size}, "content": {
                     "store_id": "runtime", "key": "transport/" + group, "digest": reference.digest, "size": reference.size}}],
    }


@pytest.mark.parametrize("external", (False, True))
@pytest.mark.parametrize("compressed", (False, True))
def test_closure_usage_counts_each_transport_row_once(external: bool, compressed: bool) -> None:
    manifest = _manifest("1" * 32)
    manifest["archives"] = [{"scope_id": "1" * 32, "domain": "trace", "records": [{}, {}]}]
    content = _content("1" * 32, external=external, compressed=compressed)
    manifest["native_messages"][0]["messages"][0] = encode_native_message_row(_row(content))
    if not external:
        manifest["objects"] = []

    # Inline body bytes are already transported; hydration reserves decoded bytes.
    entries = 5 + int(external)
    outside_bytes = len(_raw()) + (2 if external else 0)
    assert _storage_snapshot_usage(manifest) == (entries, outside_bytes)
    manifest["domains"]["conversation"] = deepcopy(manifest["domains"]["trace"])
    # Another root reference adds a domain row, not another native row or object body.
    assert _storage_snapshot_usage(manifest) == (entries + 1, outside_bytes)


def test_manifest_identity_preserves_relocated_native_content() -> None:
    assert snapshot_manifest_identity(_manifest("1" * 32)) == snapshot_manifest_identity(_manifest("2" * 32))


@pytest.mark.parametrize("component", ["messages", "sequences"])
def test_manifest_identity_includes_native_closure(component: str) -> None:
    original = _manifest("1" * 32)
    changed = deepcopy(original)
    if component == "messages":
        row = changed["native_messages"][0][component][0]
        replacement = ContentRecord("model-message", hashlib.sha256(b"[]").hexdigest(), 2, "raw", body=b"[]")
        row.update(encode_native_message_row(_row(replacement)))
    else:
        changed["native_messages"][0][component][0]["tree_count"] = 3
    assert snapshot_manifest_identity(changed) != snapshot_manifest_identity(original)


def test_manifest_identity_includes_archive_records() -> None:
    original = _manifest("1" * 32)
    content = ContentRecord("bytes", hashlib.sha256(b"x").hexdigest(), 1, "raw", body=b"x")
    record = StoredRecord(record_key("archive", "tenant", "archive", "archive_content", (content.identity,)),
                          archive_scope_key("archive", "tenant"), None, "archive_content", content.identity.encode("utf-8"),
                          None, 0, None, 0, None, content.to_json(), payload_bytes=content.body)
    original["archives"] = [{"scope_id": "1" * 32, "domain": "trace", "records": [encode_record(record)]}]
    changed = deepcopy(original)
    changed["archives"][0]["records"] = []
    assert snapshot_manifest_identity(changed) != snapshot_manifest_identity(original)


@pytest.mark.asyncio
async def test_capture_stops_on_shared_entry_budget_before_caching_more_sources() -> None:
    class Reader:
        consumed = 0

        async def scan_records_page(self, *, after, limit):
            start = 0 if after is None else int.from_bytes(after.key, "big") + 1
            result = tuple(StoredRecord(index.to_bytes(32, "big"), None, None, "fixture", (str(index)).encode("ascii"),
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

    partition = native_store_key("a", "tenant")
    wanted = hashlib.sha256(b"wanted").hexdigest()
    unrelated = hashlib.sha256(b"unrelated").hexdigest()
    nodes = [ImmutableSequenceNode(identity, 1, None, None, None) for identity in (wanted, unrelated)]
    rows = [{"store_key": partition, "sha256": node.identity, "message_sha256": node.content_id,
             "tree_count": 1, "left_sha256": None, "right_sha256": None, "next_sha256": None} for node in nodes]
    content = ContentRecord("model-message", wanted, 6, "raw", body=b"wanted")
    body = {"store_key": partition, "sha256": wanted, **content.to_native_values()}
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
    entry = manifest["native_messages"][0]
    content = _content(entry["group_id"], external=False, compressed=compressed)
    entry["messages"][0] = encode_native_message_row(_row(content))
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
    content = ContentRecord.from_native_row(decode_native_message_row(
        source["native_messages"][0]["messages"][0]))
    record = StoredRecord(record_key("archive", "tenant", "archive", "archive_content", (content.identity,)),
                          archive_scope_key("archive", "tenant"), None, "archive_content", content.identity.encode("utf-8"),
                          None, 0, None, 0, None, content.to_json(), payload_bytes=content.body)
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
    content = ContentRecord("model-message", row["sha256"], len(_raw()), "raw", body=b"{}")
    row.update(encode_native_message_row(_row(content)))
    with pytest.raises(AIError) as rejected:
        snapshot_manifest_identity(manifest)
    assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_captured_record_pages_use_kind_order_independently_of_source_key_order() -> None:
    from linktools.ai.persistence.snapshots import _CapturedSnapshotReader
    from linktools.ai.persistence.state import RecordScanCursor

    def record(kind: str) -> StoredRecord:
        key = record_key("namespace", "tenant", "execution", kind, ("owner",))
        return StoredRecord(key, None, None, kind, key, None, 0, None, 0, None, {})

    source_order = tuple(sorted((record("aa"), record("z")), key=lambda value: value.key))
    assert tuple(value.kind for value in source_order) == ("z", "aa")
    captured = _CapturedSnapshotReader(source_order, (), ())
    first = await captured.scan_records_page(after=None, limit=1)
    assert tuple(value.kind for value in first) == ("aa",)
    second = await captured.scan_records_page(after=RecordScanCursor(first[0].kind, first[0].key), limit=1)
    assert tuple(value.kind for value in second) == ("z",)
    assert await captured.scan_records_page(after=RecordScanCursor(second[0].kind, second[0].key), limit=1) == ()
