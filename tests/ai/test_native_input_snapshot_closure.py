#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plain native input projections preserve UTF-8 bytes and typed owner slots."""

import hashlib
from dataclasses import replace

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart

from linktools.ai.errors import AIError
from linktools.ai.model import StoredUserInput
from linktools.ai.model.messages import encode_model_messages
from linktools.ai.persistence._archive import ImmutableSequenceNode
from linktools.ai.persistence._archive_closure import iter_native_message_roots, rekey_native_message_roots
from linktools.ai.persistence._message_root import NativeMessageRoot, NativeMessageSlice, NativeUserInputReference
from linktools.ai.persistence.codec import (
    _encode_persisted_domain, decode_envelope, encode_envelope, native_user_input_reference,
)
from linktools.ai.persistence.snapshots import _expand_snapshot_native_content
from linktools.ai.persistence.state import StoredRecord
from linktools.ai.steps import native_store_key
from linktools.ai.storage import StoredPayload

from .test_native_text_snapshot_closure import _projection as _terminal_projection


def _projection(*, structured: bool = False) -> tuple[StoredRecord, dict, NativeUserInputReference, StoredUserInput, dict, dict]:
    text = "用户 input\n" * 6000
    raw = encode_model_messages((ModelRequest(parts=[UserPromptPart("unrelated"),
        UserPromptPart([text] if structured else text)]),))
    expected = StoredUserInput("text", StoredPayload.inline_text(text))
    digest = hashlib.sha256(raw).hexdigest()
    node = ImmutableSequenceNode(digest)
    store = native_store_key("test", "tenant")
    group = "a" * 32
    root = NativeMessageRoot(group, store, (NativeMessageSlice(node.identity, 1, 0, 1),))
    reference = NativeUserInputReference(root, 1, expected.payload.digest, expected.payload.size)
    record, value, *_ = _terminal_projection()
    del value["fields"]["result"]
    value["fields"]["stored_user_input"] = {"$dataclass": "stored_user_input", "schema": 1,
        "fields": {"content": reference.to_json()}}
    record = replace(record, data=encode_envelope({"type": "execution_record", "payload": value}))
    sequence = {"store_key": store, "sha256": node.identity, "message_sha256": digest,
        "tree_count": 1, "left_sha256": None, "right_sha256": None, "next_sha256": None}
    body = {"store_key": store, "sha256": digest, "message": raw, "compression": "raw",
            "raw_size": len(raw), "payload_json": None}
    return record, value, reference, expected, {group: ([sequence], [body])}, {group: {(store, digest): raw}}


def test_native_input_snapshot_restores_exact_utf8_semantics() -> None:
    record, value, reference, expected, rows, payloads = _projection()
    restored = _expand_snapshot_native_content(record, rows, payloads, namespace="test", tenant_id="tenant")
    actual = decode_envelope(restored.data).value["payload"]["fields"]["stored_user_input"]
    assert actual == _encode_persisted_domain(expected)
    assert tuple(iter_native_message_roots(record.data)) == (reference.root,)
    assert native_user_input_reference(value["fields"]["stored_user_input"]) == reference


@pytest.mark.parametrize("damage", ["bytes", "digest", "part", "missing", "group", "partition", "schema", "structured"])
def test_native_input_snapshot_rejects_invalid_projection(damage: str) -> None:
    record, value, reference, expected, rows, payloads = _projection(structured=damage == "structured")
    wrapper = value["fields"]["stored_user_input"]
    content = wrapper["fields"]["content"]
    if damage == "bytes":
        payloads[reference.root.group_id][next(iter(payloads[reference.root.group_id]))] = b"[]"
    elif damage == "digest":
        content["digest"] = "e" * 64
    elif damage == "part":
        content["part_index"] = 0
    elif damage == "missing":
        rows = {}
    elif damage == "group":
        value["fields"]["binding"]["archive_scope"] = "d" * 32
    elif damage == "partition":
        content["$native_user_input"] = replace(reference.root, store_key=native_store_key("test", "other")).to_json()
    elif damage == "schema":
        wrapper["schema"] = 2
    record = replace(record, data=encode_envelope({"type": "execution_record", "payload": value}))
    with pytest.raises(AIError):
        _expand_snapshot_native_content(record, rows, payloads, namespace="test", tenant_id="tenant")


def test_native_input_rekey_ignores_business_payloads() -> None:
    record, value, reference, expected, rows, payloads = _projection()
    moved = rekey_native_message_roots(value, {reference.root.group_id: "d" * 32})
    assert native_user_input_reference(moved["fields"]["stored_user_input"]) == replace(
        reference, root=replace(reference.root, group_id="d" * 32))
    opaque = _encode_persisted_domain(StoredPayload.inline_json(value))
    assert tuple(iter_native_message_roots(opaque)) == ()
    assert rekey_native_message_roots(opaque, {}) == opaque
    with pytest.raises(AIError):
        rekey_native_message_roots(value, {})
