#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Archive owner tuples and content IDs stay exact across durable boundaries."""

import base64
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, UserPromptPart

from linktools.ai.steps import native_store_key
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model.messages import encode_model_messages
from linktools.ai.persistence import Persistence
from linktools.ai.persistence._archive import (
    ArchiveRoot, ContentArchive, ContentRecord, ImmutableSequenceNode, PreparedArchive,
    archive_content_scope, archive_scope_key, archive_scope_parts, archive_scope_token,
    decode_archive_scope, validate_archive_content_identity,
)
from linktools.ai.persistence._archive_closure import (
    ArchiveClosure, NativeMessageClosure, rekey_archive_roots, validate_native_message_content,
)
from linktools.ai.persistence._memory import InMemoryStateStore
from linktools.ai.persistence.codec import binding_reference, decode_domain, encode_domain
from linktools.ai.persistence.state import record_key
from linktools.ai.storage import InMemoryObjectStore, ObjectStat, StoredPayload

from .test_incremental_projection_boundaries import _storage


def test_archive_scope_uses_exact_bounded_utf8_tuple_and_one_transport() -> None:
    namespace, tenant = "界" * 42 + "xx", "😀" * 32
    raw = archive_scope_key(namespace, tenant)
    expected = b"".join(len(value.encode()).to_bytes(2, "big") + value.encode()
                        for value in ("archive_scope", namespace, tenant))
    assert raw == expected and len(raw) == 275
    token = archive_scope_token(raw)
    assert len(token) == 367 and "=" not in token
    assert decode_archive_scope(token) == raw
    assert archive_scope_parts(raw) == (namespace, tenant)
    assert archive_content_scope("é", "tenant") != archive_content_scope("e\u0301", "tenant")
    root = ArchiveRoot("group", token)
    assert ArchiveRoot.from_json(root.to_json()) == decode_domain(encode_domain(root), ArchiveRoot) == root
    assert token not in repr(root)
    for malformed in (token + "=", token + "\n", "0" * 64,
                      base64.urlsafe_b64encode(raw + b"\0").decode().rstrip("="),
                      base64.urlsafe_b64encode(raw.replace(b"archive_scope", b"wrong_scope__", 1)).decode().rstrip("=")):
        with pytest.raises(AIError) as error:
            ArchiveRoot.from_json({**root.to_json(), "content_scope": malformed})
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    for malformed in (raw[:-1], raw + b"\0", b"\0\1\xff", b"x" * 404):
        with pytest.raises(AIError) as error:
            archive_scope_parts(malformed)
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("codec", ("model-message", "scheme:a/b:界", "😀" * 16))
def test_archive_content_identity_retains_codec_and_exact_checksum(codec: str) -> None:
    raw = b"content"
    digest = hashlib.sha256(raw).hexdigest()
    content = ContentRecord(codec, digest, len(raw), "raw", body=raw)
    assert content.identity == codec + ":" + digest
    assert validate_archive_content_identity(content.identity) == content.identity
    assert ContentRecord.from_json(content.to_json(), body=raw) == content
    node = ImmutableSequenceNode(content.identity)
    assert ImmutableSequenceNode.from_json(node.to_json()) == node
    token = archive_content_scope("namespace", "tenant")
    reference = {"$execution_binding": content.identity, "archive_scope": "group", "content_scope": token,
                 "summary": {"kind": "agent", "id": "agent", "revision": 1, "max_attempts": None,
                             "capability_refs": {"$tuple": []}, "binding_digest": digest}}
    assert binding_reference(reference) == ("group", token, content.identity)
    for invalid in (digest, codec + ":" + "A" * 64, codec + ":" + "g" * 64):
        with pytest.raises(AIError) as error:
            binding_reference({**reference, "$execution_binding": invalid})
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("codec", ("", "x" * 65, "😀" * 17, "\ud800", "codec\0name"))
def test_archive_codec_limit_applies_to_current_record_reader(codec: str) -> None:
    with pytest.raises(AIError) as error:
        ContentRecord.from_json({"codec": codec, "raw_digest": "0" * 64,
            "raw_size": 0, "compression": "raw", "payload": None}, body=b"")
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_archive_and_native_sequence_readers_keep_distinct_content_owners() -> None:
    native = ImmutableSequenceNode("a" * 64)
    row = {"store_key": native_store_key("b", "tenant"), "sha256": native.identity, "message_sha256": "a" * 64,
           "tree_count": 1, "left_sha256": None, "right_sha256": None, "next_sha256": None}
    NativeMessageClosure((row,))
    with pytest.raises(AIError) as error:
        ImmutableSequenceNode.from_json(native.to_json())
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    with pytest.raises(AIError) as error:
        NativeMessageClosure(({**row, "message_sha256": "bytes:" + "a" * 64},))
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "local_sqlite", "sqlite"))
async def test_archive_shared_store_isolates_exact_owners_and_round_trips_unicode_codec(
    tmp_path: Path, backend: str,
) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="archive", tenant_id="tenant")
    try:
        scope, store = await state.execution.executions.state_store.storage_group.open_archive(create=True)
        codec = "scheme:a/b:界"
        roots = []
        for namespace, tenant in (("é", "tenant"), ("e\u0301", "tenant"), ("é", "other")):
            archive = ContentArchive(store, scope, namespace=namespace, tenant_id=tenant, object_store=None)
            prepared = PreparedArchive()
            root = await archive.prepare_append(archive.empty_root(), ((codec, b"same"), (codec, b"same")), prepared)
            await store.mutate(lambda tx: archive.publish(tx, prepared))
            assert await archive.read_ranges((root,), codec=codec) == ((b"same", b"same"),)
            content, = [row for row in prepared.records.values() if row.kind == "archive_content"]
            identity = codec + ":" + hashlib.sha256(b"same").hexdigest()
            assert content.scope_key == archive_scope_key(namespace, tenant)
            assert content.sort_key == identity.encode("utf-8")
            assert content.key == record_key(namespace, tenant, "archive", "archive_content", (identity,))
            assert archive.content_scope.encode() not in content.key
            roots.append((archive, root, content))
        records = await store.read(lambda tx: tx.scan_records())
        assert len({item[2].key for item in roots}) == 3
        closure = ArchiveClosure(scope, records)
        closure.mark(roots[0][1])
        assert {row.scope_key for row in closure.retained_records} == {roots[0][2].scope_key}
        assert roots[1][2] in closure.unreachable_records
        with pytest.raises(AIError) as error:
            await roots[1][0].read_ranges((roots[0][1],), codec=codec)
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        damaged = replace(roots[0][2], scope_key=roots[1][2].scope_key)
        for read in (lambda: ContentRecord.from_record(damaged), lambda: ArchiveClosure(scope, (damaged,))):
            with pytest.raises(AIError) as error:
                read()
            assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        wire = encode_domain(roots[0][1])
        restored = decode_domain(rekey_archive_roots(wire, {scope: "custom/group"}), ArchiveRoot)
        assert restored.scope_id == "custom/group" and restored.content_scope == roots[0][1].content_scope
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_archive_object_paths_keep_maximum_scope_and_hide_reversible_values() -> None:
    store, objects = InMemoryStateStore(), InMemoryObjectStore()
    await store.initialize()
    namespace, tenant, group = "界" * 42 + "xx", "😀" * 32, "g/" + "x" * 126
    archive = ContentArchive(store, group, namespace=namespace, tenant_id=tenant, object_store=objects)
    raw = encode_model_messages((ModelRequest(parts=[UserPromptPart(
        [BinaryContent(b"attachment", media_type="application/octet-stream")],
    )]),))
    try:
        prepared = PreparedArchive()
        identity, = await archive.prepare_content((("model-message", raw),), prepared)
        record, = prepared.records.values()
        content = ContentRecord.from_record(record)
        reference, = content.object_refs()
        assert identity.startswith("model-message:") and len(identity) == 78
        assert len(reference.key.encode("utf-8")) == 572
        assert reference.key == f"v1/archive/{group}/{archive.content_scope}/{reference.digest}"
        assert await archive.read_payload_object(reference) == b"attachment"
        stat = await objects.stat(reference.key)
        assert stat == ObjectStat(reference.key, reference.digest, reference.size)
        assert replace(reference) == reference and replace(stat) == stat
        for value in (reference, stat, StoredPayload.object(reference)):
            assert reference.key not in repr(value) and archive.content_scope not in repr(value)
        row = {"store_key": native_store_key(namespace, tenant),
               "sha256": content.raw_digest, **content.to_native_values()}
        assert validate_native_message_content(row, group_id=group, content_scope=archive.content_scope) == content
        with pytest.raises(AIError) as error:
            validate_native_message_content(row, group_id=group, content_scope=archive_content_scope("other", tenant))
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        for invalid in ("", "😀" * 33, "\ud800", "group\0id"):
            with pytest.raises(AIError) as error:
                ContentArchive(store, invalid, namespace=namespace, tenant_id=tenant, object_store=objects)
            assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            with pytest.raises(AIError):
                ArchiveRoot(invalid, archive.content_scope)
        for codec in ("x" * 65, "😀" * 17, "codec\0name"):
            with pytest.raises(AIError) as error:
                await archive.prepare_content(((codec, b"raw"),), PreparedArchive())
            assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await store.close()
