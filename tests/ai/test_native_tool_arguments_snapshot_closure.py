#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed Native call arguments survive portable closure without interpretation."""

import hashlib
from dataclasses import replace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model.messages import encode_model_messages
from linktools.ai.persistence._archive import ImmutableSequenceNode
from linktools.ai.persistence._archive_closure import iter_native_message_roots, rekey_native_message_roots
from linktools.ai.persistence._message_root import NativeMessageRoot, NativeMessageSlice, NativeToolArgumentsReference
from linktools.ai.persistence._native_tool_arguments import prepare_native_tool_arguments
from linktools.ai.persistence.codec import (
    _encode_persisted_domain, decode_envelope, encode_envelope, native_tool_arguments_reference,
)
from linktools.ai.persistence.snapshots import _expand_snapshot_native_content
from linktools.ai.persistence.state import StoredRecord
from linktools.ai.steps import native_store_key
from linktools.ai.storage import StoredPayload

from .test_tool_command_convergence import _record


def _projection() -> tuple[StoredRecord, dict, NativeToolArgumentsReference, StoredPayload, dict, dict]:
    arguments = {"query": "small", "nested": [None, True, {"unicode": "界"}]}
    raw = encode_model_messages((ModelResponse(parts=[TextPart("unrelated"),
        ToolCallPart("actual-tool", arguments, tool_call_id="model-call")]),))
    expected = StoredPayload.inline_bytes(canonical_json_bytes(arguments))
    identity = hashlib.sha256(raw).hexdigest()
    node = ImmutableSequenceNode(identity)
    store, group = native_store_key("test", "tenant"), "a" * 32
    root = NativeMessageRoot(group, store, (NativeMessageSlice(node.identity, 1, 0, 1),))
    reference = prepare_native_tool_arguments(root, raw, 1, expected)
    assert reference is not None
    operation = replace(_record(), tool_name="actual-tool", tool_call_id="model-call",
                        arguments_payload=expected, arguments_digest=expected.digest)
    value = _encode_persisted_domain(operation)
    value["fields"]["arguments_payload"] = {"$dataclass": "stored_payload", "schema": 1,
                                             "fields": {"content": reference.to_json()}}
    record = StoredRecord(b"tool-operation", None, None, "tool_operation", b"operation", operation.status.value,
        1, operation.owner, operation.fence, None, encode_envelope({"type": "tool_operation", "payload": value}))
    sequence = {"store_key": store, "sha256": node.identity, "message_sha256": identity,
        "tree_count": 1, "left_sha256": None, "right_sha256": None, "next_sha256": None}
    body = {"store_key": store, "sha256": identity, "message": raw, "compression": "raw",
            "raw_size": len(raw), "payload_json": None}
    return record, value, reference, expected, {group: ([sequence], [body])}, {group: {(store, identity): raw}}


def test_native_arguments_snapshot_recovers_exact_accepted_bytes() -> None:
    record, value, reference, expected, rows, payloads = _projection()
    restored = _expand_snapshot_native_content(record, rows, payloads, namespace="test", tenant_id="tenant")
    arguments = decode_envelope(restored.data).value["payload"]["fields"]["arguments_payload"]
    assert arguments == _encode_persisted_domain(expected)
    assert tuple(iter_native_message_roots(record.data)) == (reference.root,)


@pytest.mark.parametrize("damage", ("bytes", "digest", "size", "missing_digest", "missing", "part", "call", "tool", "partition", "slot"))
def test_native_arguments_snapshot_rejects_invalid_projection(damage: str) -> None:
    record, value, reference, expected, rows, payloads = _projection()
    content = value["fields"]["arguments_payload"]["fields"]["content"]
    if damage == "bytes":
        payloads[reference.root.group_id][next(iter(payloads[reference.root.group_id]))] = b"[]"
    elif damage == "missing":
        rows = {}
    elif damage == "part":
        content["part_index"] = 0
    elif damage == "call":
        content["tool_call_id"] = "other-call"
    elif damage == "tool":
        content["tool_name"] = "other-tool"
    elif damage == "size":
        rows[reference.root.group_id][1][0]["raw_size"] += 1
    elif damage == "missing_digest":
        value["fields"].pop("arguments_digest")
    elif damage == "partition":
        content["$native_tool_arguments"] = replace(reference.root, store_key=native_store_key("test", "other")).to_json()
    elif damage == "slot":
        value["fields"]["result_payload"] = value["fields"].pop("arguments_payload")
    else:
        value["fields"]["arguments_digest"] = "e" * 64
    record = replace(record, data=encode_envelope({"type": "tool_operation", "payload": value}))
    with pytest.raises(AIError) as invalid:
        _expand_snapshot_native_content(record, rows, payloads, namespace="test", tenant_id="tenant")
    assert invalid.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_native_arguments_rekey_keeps_payload_semantics_and_opaque_json() -> None:
    record, value, reference, expected, rows, payloads = _projection()
    moved = rekey_native_message_roots(value, {reference.root.group_id: "d" * 32})
    assert native_tool_arguments_reference(moved["fields"]["arguments_payload"]) == replace(
        reference, root=replace(reference.root, group_id="d" * 32))
    opaque = _encode_persisted_domain(StoredPayload.inline_json(value))
    assert tuple(iter_native_message_roots(opaque)) == ()
    assert rekey_native_message_roots(opaque, {}) == opaque
    with pytest.raises(AIError):
        rekey_native_message_roots(value, {})
