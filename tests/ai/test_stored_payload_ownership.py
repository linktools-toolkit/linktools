#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retained payloads and input provenance own their nested JSON values."""

import hashlib
import json

import pytest
from pydantic_ai.messages import BinaryContent, TextContent

from linktools.ai.core import JsonValue, PromptLimits
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import MaterializedUserContent, StoredUserInput
from linktools.ai.inputs import ExecutionInputMaterializer
from linktools.ai.persistence.codec import decode_domain, encode_domain
from linktools.ai.storage import ObjectRef, StoredPayload


@pytest.mark.parametrize("surface", ("value", "decode", "wire"))
@pytest.mark.parametrize("root", ("object", "array"))
def test_payload_views_cannot_mutate_retained_json(surface: str, root: str) -> None:
    source: dict[str, JsonValue] = {"items": [{"label": "original"}]}
    value: JsonValue = source if root == "object" else [source]
    stored = StoredPayload.inline_json(value)
    original_wire = stored.to_json()
    original_digest = stored.digest
    source["items"][0]["label"] = "caller annotation"

    view = stored.to_json()["value"] if surface == "wire" else (
        stored.value if surface == "value" else stored.decode()
    )
    nested = view if root == "object" else view[0]
    nested["items"][0]["label"] = "consumer annotation"

    expected = {"items": [{"label": "original"}]}
    assert stored.value == (expected if root == "object" else [expected])
    assert stored.digest == original_digest
    assert stored.to_json() == original_wire
    assert StoredPayload.from_json(original_wire) == stored


@pytest.mark.parametrize("value", ({"empty": []}, [None, True, 2, 3.5], "你好", 0, False, None))
def test_inline_json_preserves_root_wire_values(value: JsonValue) -> None:
    payload = StoredPayload.inline_json(value)
    wire = payload.to_json()
    reconstructed = StoredPayload(
        kind=wire["kind"], encoding=wire["encoding"], digest=wire["digest"],
        size=wire["size"], value=wire["value"],
    )

    assert reconstructed.decode() == value
    assert reconstructed.to_json() == wire
    assert StoredPayload.from_json(wire) == payload
    encoded = encode_domain(payload)
    assert encoded == {
        "$dataclass": "stored_payload",
        "fields": {
            "kind": "inline", "encoding": "json", "digest": payload.digest,
            "size": payload.size, "value": encode_domain(value), "ref": None,
        },
    }
    assert decode_domain(encoded, StoredPayload) == payload


@pytest.mark.parametrize("encoding", ("utf-8", "base64", "object"))
def test_payload_ownership_preserves_non_json_wire_values(encoding: str) -> None:
    if encoding == "object":
        payload = StoredPayload.object(ObjectRef("objects", "body", "a" * 64, 3))
    elif encoding == "utf-8":
        payload = StoredPayload.inline_text("你好")
    else:
        # Accepted base64 spelling is wire data, not a value to normalize away.
        payload = StoredPayload(
            "inline", "base64", hashlib.sha256(b"a").hexdigest(), 1, "YR==",
        )
    wire = payload.to_json()
    restored = StoredPayload.from_json(wire)

    assert restored == payload
    assert restored.to_json() == wire
    assert decode_domain(encode_domain(payload), StoredPayload) == payload
    if encoding == "object":
        assert restored.value is None
        with pytest.raises(ValueError, match="ObjectStore"):
            restored.decode()
    else:
        assert restored.decode() == (b"a" if encoding == "base64" else "你好")
        assert restored.value == ("YR==" if encoding == "base64" else "你好")


@pytest.mark.parametrize("field", ("digest", "size", "value"))
def test_payload_reader_rejects_changed_inline_integrity(field: str) -> None:
    wire = StoredPayload.inline_json({"nested": ["original"]}).to_json()
    wire[field] = {"digest": "a" * 64, "size": 0, "value": {"nested": ["changed"]}}[field]

    with pytest.raises(ValueError, match="digest or size"):
        StoredPayload.from_json(wire)


@pytest.mark.parametrize("field", ("digest", "size", "value"))
def test_domain_payload_reader_rejects_changed_inline_integrity(field: str) -> None:
    wire = encode_domain(StoredPayload.inline_json({"nested": ["original"]}))
    wire["fields"][field] = {
        "digest": "a" * 64, "size": 0, "value": encode_domain({"nested": ["changed"]}),
    }[field]

    with pytest.raises(AIError) as caught:
        decode_domain(wire, StoredPayload)

    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_input_provenance_detaches_caller_and_consumer_views() -> None:
    view = {
        "version": 1,
        "prompt": {"kind": "items", "items": [{"kind": "text", "text": "prompt"}]},
        "files": [{"path": "evidence.txt"}],
        "attachments": [{"attachment_id": "accepted"}],
        "": {"extension": []},
    }
    expected = json.loads(json.dumps(view))
    materialized = MaterializedUserContent(("prompt",), view)
    stored = StoredUserInput("text", StoredPayload.inline_text("prompt"), view)
    digest = stored.digest
    for candidate in (view, materialized.view, stored.view):
        candidate["prompt"]["items"][0]["text"] = "changed"
        candidate["files"][0]["path"] = "changed.txt"
        candidate["attachments"].append({"attachment_id": "annotation"})
        candidate[""]["extension"].append("annotation")

    assert materialized.view == expected
    assert stored.view == expected
    assert stored.digest == digest == StoredUserInput("text", stored.payload).digest
    appended = materialized.append_text("suffix")
    assert appended == ("prompt", "suffix")
    assert appended.view["prompt"]["items"][-1] == {"kind": "text", "text": "suffix"}
    assert materialized.view == expected
    assert decode_domain(encode_domain(stored), StoredUserInput).view == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("text", "binary"))
async def test_restored_sdk_metadata_cannot_mutate_retained_input(kind: str) -> None:
    metadata = {"nested": {"labels": ["original"]}}
    content = TextContent("hello", metadata=metadata) if kind == "text" else BinaryContent(
        b"hello", media_type="text/plain", vendor_metadata=metadata,
    )
    materializer = ExecutionInputMaterializer(None, PromptLimits())
    stored = await materializer.store((content,), tenant_id="tenant")
    original_wire = encode_domain(stored)
    original_view = dict(stored.view)
    metadata["nested"]["labels"].append("author annotation")
    first = await materializer.restore(stored)
    restored_metadata = first[0].metadata if kind == "text" else first[0].vendor_metadata
    restored_metadata["nested"]["labels"].append("consumer annotation")
    first.view["attachments"].append({"display_annotation": "local"})
    second = await materializer.restore(stored)
    second_metadata = second[0].metadata if kind == "text" else second[0].vendor_metadata

    assert second_metadata == {"nested": {"labels": ["original"]}}
    assert stored.view == second.view == original_view
    assert encode_domain(stored) == original_wire


@pytest.mark.parametrize("kind", ("json", "text", "bytes"))
def test_inline_factories_hash_once_and_external_descriptors_remain_checked(
    monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    from linktools.ai.storage import _payload

    original = hashlib.sha256
    hashes = []

    def counted(value: bytes):
        hashes.append(value)
        return original(value)

    monkeypatch.setattr(_payload.hashlib, "sha256", counted)
    payload = (StoredPayload.inline_json({"values": [1, 1.0, True, -0.0]}) if kind == "json"
               else StoredPayload.inline_text("你好") if kind == "text"
               else StoredPayload.inline_bytes(b"\x00\xffpayload"))
    assert len(hashes) == 1
    assert payload.digest == original(hashes[0]).hexdigest()
    assert payload.size == len(hashes[0])
    wire = payload.to_json()
    assert StoredPayload.from_json(wire) == payload
    assert len(hashes) == 2
    with pytest.raises(ValueError, match="does not match"):
        StoredPayload("inline", payload.encoding, "0" * 64, payload.size, payload.value)
    with pytest.raises(ValueError, match="does not match"):
        StoredPayload.from_json({**wire, "size": payload.size + 1})
    with pytest.raises(ValueError):
        StoredPayload.from_json({**wire, "value": "changed"})


@pytest.mark.parametrize("kind", ("json", "text", "bytes"))
def test_inline_factories_preserve_subclass_initialization(kind: str) -> None:
    calls = []

    class CustomPayload(StoredPayload):
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))
            super().__init__(*args, **kwargs)

    payload = (CustomPayload.inline_json({"value": 1.0}) if kind == "json"
               else CustomPayload.inline_text("text") if kind == "text"
               else CustomPayload.inline_bytes(b"bytes"))
    assert isinstance(payload, CustomPayload)
    assert len(calls) == 1
    assert StoredPayload.from_json(payload.to_json()).to_json() == payload.to_json()
