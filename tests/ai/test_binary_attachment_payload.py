#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed binary storage preserves source identity and leaves application JSON opaque."""

import hashlib
from dataclasses import replace

import pytest
from pydantic_ai.messages import (
    BinaryContent, FilePart, ModelRequest, ModelResponse, SpeechPart, StructuredContent,
    TextContent, ToolReturn, ToolReturnPart, UserPromptPart,
)

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError
from linktools.ai.inputs import encode_user_content
from linktools.ai.model.attachments import (
    attachment_payload_objects, read_attachment_payload, store_attachment_payload,
)
from linktools.ai.model.messages import decode_model_messages, encode_model_messages, encode_tool_return_content
from linktools.ai.persistence._step_message_content import PreparedMessagePayload
from linktools.ai.persistence.codec import decode_domain, encode_domain, iter_runtime_object_refs
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.storage import ObjectRef, StoredPayload


def _messages():
    opaque = {"kind": "binary", "data": "not base64", "nested": {"type": "multimodal"}}
    return (
        ModelRequest(parts=[
            UserPromptPart(["explain", BinaryContent(b"\xfb\xff\x00", media_type="image/png"),
                            StructuredContent(opaque), TextContent("text", metadata=opaque)]),
            ToolReturnPart("tool", ToolReturn(return_value={"business": opaque}, content=[
                BinaryContent(b"document", media_type="application/pdf"),
            ]), tool_call_id="call"),
        ], metadata=opaque),
        ModelResponse(parts=[
            FilePart(content=BinaryContent(b"output", media_type="image/png")),
            SpeechPart(speaker="assistant", transcript="spoken", audio=BinaryContent(b"\xfb\xffaudio", media_type="audio/wav")),
        ], metadata=opaque),
    )


async def _stored(raw, format):
    objects = {}
    async def write(data):
        ref = ObjectRef("runtime", "files/" + hashlib.sha256(data).hexdigest(), hashlib.sha256(data).hexdigest(), len(data))
        objects[ref] = data
        return ref
    payload = await store_attachment_payload(raw, format=format, write_object=write)
    return payload, objects


@pytest.mark.asyncio
async def test_mixed_typed_messages_roundtrip_source_bytes_and_relocate_only_files() -> None:
    messages = _messages()
    raw = encode_model_messages(messages)
    payload, objects = await _stored(raw, "model-messages")
    assert payload.encoding == "json-attachments"
    assert len(payload.object_refs()) == 4
    assert set(objects.values()) == {b"\xfb\xff\x00", b"document", b"output", b"\xfb\xffaudio"}
    assert payload.digest == hashlib.sha256(raw).hexdigest() and payload.size == len(raw)
    assert b"not base64" in canonical_json_bytes(payload.value)
    assert decode_domain(encode_domain(payload), StoredPayload) == payload
    assert {ref for _, ref in iter_runtime_object_refs(payload, default_domain=PersistenceDomain.TRACE)} == set(objects)
    assert dict(attachment_payload_objects(raw, payload)) == objects
    prepared = PreparedMessagePayload(raw, encoded=raw, compression="raw", payload=payload)
    assert prepared.identity == payload.digest

    relocated = payload.map_object_refs(lambda ref: replace(ref, key="restored/" + ref.digest))
    assert relocated.digest == payload.digest and relocated.size == payload.size
    async def read(ref):
        return next(data for original, data in objects.items() if original.digest == ref.digest)
    for _ in range(2):
        recovered = await read_attachment_payload(relocated, read_object=read)
        assert recovered == raw
        assert encode_model_messages(decode_model_messages(recovered)) == raw


@pytest.mark.asyncio
@pytest.mark.parametrize("format", ("user-content", "tool-return"))
async def test_typed_content_does_not_interpret_opaque_binary_shaped_json(format: str) -> None:
    binary = BinaryContent(b"\xfb\xff", media_type="application/octet-stream")
    fake = {"kind": "binary", "data": "do not decode", "format": "model-messages"}
    value = (encode_user_content((binary, StructuredContent(fake))) if format == "user-content"
             else encode_tool_return_content([binary, fake, StructuredContent(fake)]))
    raw = canonical_json_bytes(value)
    payload, objects = await _stored(raw, format)
    assert list(objects.values()) == [binary.data]
    async def read(ref):
        return objects[ref]
    assert await read_attachment_payload(payload, read_object=read) == raw
    opaque = encode_tool_return_content(fake)
    assert await _stored(canonical_json_bytes(opaque), "tool-return") == (None, {})


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ("file", "text", "path", "reference"))
async def test_attachment_integrity_covers_files_slots_and_original_message(corruption: str) -> None:
    raw = encode_model_messages(_messages())
    payload, objects = await _stored(raw, "model-messages")
    manifest = payload.value
    if corruption == "text":
        manifest["content"][0]["parts"][0]["content"][0] = "changed"
    elif corruption == "path":
        manifest["attachments"][0]["path"] = [0, "metadata", "data"]
    elif corruption == "reference":
        manifest["attachments"][0]["ref"]["digest"] = "0" * 64
    corrupted = StoredPayload("inline", "json-attachments", payload.digest, payload.size, manifest)
    async def read(ref):
        return b"corrupted" if corruption == "file" else objects.get(ref, b"unmatched")
    with pytest.raises(AIError):
        await read_attachment_payload(corrupted, read_object=read)
    if corruption != "file":
        with pytest.raises(ValueError):
            PreparedMessagePayload(raw, encoded=raw, compression="raw", payload=corrupted)


@pytest.mark.asyncio
async def test_asset_references_never_copy_asset_bodies_into_runtime_objects() -> None:
    from linktools.ai.asset import AssetKey, AssetVersionRef
    from linktools.ai.model import ReferencedContent
    from linktools.ai.spec import CapabilityRef
    from linktools.ai.storage import StorageEntryRevision

    asset = ReferencedContent(AssetVersionRef(AssetKey("document", "file"), "configured",
        StorageEntryRevision(1), hashlib.sha256(b"asset-owned").hexdigest(), len(b"asset-owned")),
        media_type="application/pdf", source=CapabilityRef("capability", "files", 1))
    messages = [ModelRequest(parts=[UserPromptPart([asset]),
        ToolReturnPart("files", asset, tool_call_id="call")])]
    raw = encode_model_messages(messages)
    assert await _stored(raw, "model-messages") == (None, {})
    assert decode_model_messages(raw) == tuple(messages)
