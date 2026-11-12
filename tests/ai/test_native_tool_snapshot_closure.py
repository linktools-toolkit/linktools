#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native tool receipt snapshots preserve exact original wrappers and roots."""

import hashlib
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import ModelRequest, ToolReturnPart, UserPromptPart

from linktools.ai.errors import AIError
from linktools.ai.model.messages import encode_model_messages
from linktools.ai.persistence._archive import ImmutableSequenceNode
from linktools.ai.persistence._archive_closure import iter_native_message_roots, rekey_native_message_roots
from linktools.ai.persistence._message_root import NativeMessageRoot, NativeMessageSlice, NativeToolResultReference
from linktools.ai.persistence._native_tool_result import prepare_native_tool_result
from linktools.ai.persistence.codec import (
    _encode_persisted_domain, decode_envelope, encode_envelope, native_tool_result_reference,
)
from linktools.ai.persistence.snapshots import _expand_snapshot_native_content
from linktools.ai.persistence.state import StoredRecord
from linktools.ai.steps import native_store_key
from linktools.ai.storage import StoredPayload


def _projection() -> tuple[StoredRecord, dict, NativeToolResultReference, StoredPayload, dict, dict]:
    text = "native tool receipt " * 6000
    now = datetime(2026, 11, 12, tzinfo=timezone.utc)
    raw = encode_model_messages((ModelRequest(parts=[UserPromptPart("other part", timestamp=now),
        ToolReturnPart("actual-tool", text, tool_call_id="model-call", timestamp=now)]),))
    expected = StoredPayload.inline_bytes(encode_model_messages((ModelRequest(parts=[
        ToolReturnPart("runtime", text, tool_call_id="operation", timestamp=now)]),)))
    identity = hashlib.sha256(raw).hexdigest()
    node = ImmutableSequenceNode(identity)
    store = native_store_key("test", "tenant")
    group = "a" * 32
    root = NativeMessageRoot(group, store, (NativeMessageSlice(node.identity, 1, 0, 1),))
    reference = prepare_native_tool_result(root, raw, 1, expected, operation_id="operation")
    assert reference is not None
    value = {"$dataclass": "tool_operation", "schema": 1, "fields": {
        "result_payload": {"$dataclass": "stored_payload", "schema": 1,
                           "fields": {"content": reference.to_json()}},
    }}
    record = StoredRecord(b"tool-operation", None, None, "tool_operation", b"operation", None,
        1, None, 0, None, encode_envelope({"type": "tool_operation", "payload": value}))
    sequence = {"store_key": store, "sha256": node.identity, "message_sha256": identity,
        "tree_count": 1, "left_sha256": None, "right_sha256": None, "next_sha256": None}
    body = {"store_key": store, "sha256": identity, "message": raw, "compression": "raw",
            "raw_size": len(raw), "payload_json": None}
    return record, value, reference, expected, {group: ([sequence], [body])}, {group: {(store, identity): raw}}


def test_native_tool_snapshot_recovers_original_receipt_not_source_wrapper() -> None:
    record, value, reference, expected, rows, payloads = _projection()
    restored = _expand_snapshot_native_content(record, rows, payloads, namespace="test", tenant_id="tenant")
    output = decode_envelope(restored.data).value["payload"]["fields"]["result_payload"]
    assert output == _encode_persisted_domain(expected)
    assert tuple(iter_native_message_roots(record.data)) == (reference.root,)
    assert "native tool receipt" not in reference.template


@pytest.mark.parametrize("damage", ["bytes", "digest", "missing", "part", "call", "template", "partition"])
def test_native_tool_snapshot_rejects_invalid_projection(damage: str) -> None:
    record, value, reference, expected, rows, payloads = _projection()
    content = value["fields"]["result_payload"]["fields"]["content"]
    if damage == "bytes":
        payloads[reference.root.group_id][next(iter(payloads[reference.root.group_id]))] = b"[]"
    elif damage == "missing":
        rows = {}
    elif damage == "part":
        content["part_index"] = 0
    elif damage == "call":
        content["tool_call_id"] = "another-model-call"
    elif damage == "template":
        content["template"] = content["template"].replace("operation", "other-operation")
    elif damage == "partition":
        content["$native_tool_result"] = replace(reference.root, store_key=native_store_key("test", "other")).to_json()
    else:
        content["digest"] = "e" * 64
    record = replace(record, data=encode_envelope({"type": "tool_operation", "payload": value}))
    with pytest.raises(AIError):
        _expand_snapshot_native_content(record, rows, payloads, namespace="test", tenant_id="tenant")


def test_native_tool_rekey_keeps_template_and_business_json_opaque() -> None:
    record, value, reference, expected, rows, payloads = _projection()
    moved = rekey_native_message_roots(value, {reference.root.group_id: "d" * 32})
    output = moved["fields"]["result_payload"]
    assert native_tool_result_reference(output) == replace(reference, root=replace(reference.root, group_id="d" * 32))
    opaque = _encode_persisted_domain(StoredPayload.inline_json(value))
    assert tuple(iter_native_message_roots(opaque)) == ()
    assert rekey_native_message_roots(opaque, {}) == opaque
    with pytest.raises(AIError):
        rekey_native_message_roots(value, {})
