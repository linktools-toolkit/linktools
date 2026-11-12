#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native result projections retain one authenticated, relocatable body owner."""

import hashlib
from copy import deepcopy
from dataclasses import replace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.errors import AIError
from linktools.ai.model.messages import encode_model_messages
from linktools.ai.persistence._archive import ImmutableSequenceNode, archive_content_scope
from linktools.ai.persistence._archive_closure import (
    iter_native_message_roots, rekey_native_message_roots,
)
from linktools.ai.persistence._message_root import NativeMessageRoot, NativeMessageSlice, NativeTextReference
from linktools.ai.persistence.codec import (
    _encode_persisted_domain, decode_envelope, encode_envelope, native_text_reference,
)
from linktools.ai.persistence.snapshots import _expand_snapshot_native_content
from linktools.ai.persistence.state import StoredRecord
from linktools.ai.steps import native_store_key
from linktools.ai.storage import StoredPayload


def _projection() -> tuple[StoredRecord, dict, NativeTextReference, StoredPayload, dict, dict]:
    text = "native result " * 6000
    raw = encode_model_messages([ModelResponse(parts=[TextPart(text)])])
    identity = hashlib.sha256(raw).hexdigest()
    node = ImmutableSequenceNode(identity)
    store = native_store_key("test", "tenant")
    group = "a" * 32
    root = NativeMessageRoot(group, store, (NativeMessageSlice(node.identity, 1, 0, 1),))
    expected = StoredPayload.inline_json(text)
    reference = NativeTextReference(root)
    binding = {"$execution_binding": "execution-binding:" + "b" * 64,
        "archive_scope": group, "content_scope": archive_content_scope("test", "tenant")}
    value = {"$dataclass": "execution_record", "schema": 1, "fields": {
        "binding": binding, "result": {"$dataclass": "result_record", "schema": 1, "fields": {"output": {
            "$dataclass": "stored_payload", "schema": 1, "fields": {"content": reference.to_json()},
        }}}}}
    record = StoredRecord(b"execution", None, None, "execution", b"execution", None, 1, None, 0, None,
        encode_envelope({"type": "execution_record", "payload": value}))
    sequence = {"store_key": store, "sha256": node.identity, "message_sha256": identity,
        "tree_count": 1, "left_sha256": None, "right_sha256": None, "next_sha256": None}
    body = {"store_key": store, "sha256": identity, "message": raw, "compression": "raw",
            "raw_size": len(raw), "payload_json": None}
    return record, value, reference, expected, {group: ([sequence], [body])}, {group: {(store, identity): raw}}


def test_native_text_snapshot_expands_before_payload_domain_decode() -> None:
    record, value, reference, expected, rows, payloads = _projection()
    restored = _expand_snapshot_native_content(record, rows, payloads, namespace="test", tenant_id="tenant")
    output = decode_envelope(restored.data).value["payload"]["fields"]["result"]["fields"]["output"]
    assert output == _encode_persisted_domain(expected)
    assert tuple(iter_native_message_roots(record.data)) == (reference.root,)
    assert native_text_reference(value["fields"]["result"]["fields"]["output"]) == reference


@pytest.mark.parametrize("damage", ["bytes", "digest", "missing", "binding_group", "partition"])
def test_native_text_snapshot_rejects_corrupt_or_foreign_sources(damage: str) -> None:
    record, value, reference, expected, rows, payloads = _projection()
    if damage == "bytes":
        payloads[reference.root.group_id][next(iter(payloads[reference.root.group_id]))] = b"[]"
    elif damage == "missing":
        rows = {}
    elif damage == "binding_group":
        value["fields"]["binding"]["archive_scope"] = "d" * 32
    elif damage == "partition":
        content = value["fields"]["result"]["fields"]["output"]["fields"]["content"]
        content["$native_text"] = replace(reference.root, store_key=native_store_key("test", "other")).to_json()
    else:
        rows[reference.root.group_id][0][0]["message_sha256"] = "e" * 64
    record = replace(record, data=encode_envelope({"type": "execution_record", "payload": value}))
    with pytest.raises(AIError):
        _expand_snapshot_native_content(record, rows, payloads, namespace="test", tenant_id="tenant")


def test_native_text_rekey_changes_only_declared_owner_slot() -> None:
    record, value, reference, expected, rows, payloads = _projection()
    destination = "d" * 32
    moved = rekey_native_message_roots(value, {reference.root.group_id: destination})
    output = moved["fields"]["result"]["fields"]["output"]
    assert native_text_reference(output) == replace(reference, root=replace(reference.root, group_id=destination))
    opaque = _encode_persisted_domain(StoredPayload.inline_json(deepcopy(value)))
    assert tuple(iter_native_message_roots(opaque)) == ()
    assert rekey_native_message_roots(opaque, {}) == opaque
    with pytest.raises(AIError):
        rekey_native_message_roots(value, {})
