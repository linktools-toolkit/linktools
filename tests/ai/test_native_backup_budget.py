#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable capture charges decoded native and attachment bytes before hydration."""

import hashlib

import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, UserPromptPart

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model.attachments import store_attachment_payload
from linktools.ai.model.messages import encode_model_messages
from linktools.ai.persistence._archive import ContentRecord, archive_content_scope
from linktools.ai.persistence._payload import encode_payload_bytes
from linktools.ai.persistence._snapshot import encode_native_message_row
from linktools.ai.persistence.snapshots import _SnapshotCaptureBudget, _capture_native_payloads, SnapshotLimits
from linktools.ai.storage import InMemoryObjectStore, ObjectRef


async def _content(*, external: bool, text: str = ""):
    bodies = {}

    async def write(data):
        digest = hashlib.sha256(data).hexdigest()
        ref = ObjectRef("runtime", "v1/archive/group/" + archive_content_scope("budget", "tenant") + "/" + digest, digest, len(data))
        bodies[ref.key] = data
        return ref

    raw = encode_model_messages((ModelRequest(parts=[UserPromptPart(
        [text, BinaryContent(b"file bytes", media_type="application/octet-stream")] if external else text,
    )]),))
    payload = await store_attachment_payload(raw, format="model-messages", write_object=write)
    compression, encoded = encode_payload_bytes(raw)
    digest, size = hashlib.sha256(raw).hexdigest(), len(raw)
    content = (ContentRecord("model-message", digest, size, "raw", payload)
               if payload is not None else ContentRecord("model-message", digest, size, compression, body=encoded))
    row = {"store_digest": "c" * 64, "sha256": content.raw_digest, **content.to_native_values()}
    return content, raw, row, bodies


@pytest.mark.asyncio
@pytest.mark.parametrize("external", (False, True))
async def test_native_raw_budget_rejects_before_object_open_or_decompression(monkeypatch, external: bool) -> None:
    from linktools.ai.persistence import snapshots

    async def forbidden_read(*args, **kwargs):
        raise AssertionError("object read preceded the decoded byte budget")

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("decompression preceded the decoded byte budget")

    _, _, row, _ = await _content(external=external, text="x" * 10001)
    monkeypatch.setattr(snapshots, "read_object", forbidden_read)
    monkeypatch.setattr(snapshots, "decode_payload_bytes", forbidden_decode)
    with pytest.raises(AIError) as rejected:
        await _capture_native_payloads("group", [], [row],
            budget=_SnapshotCaptureBudget(SnapshotLimits(100, 10000)), objects=InMemoryObjectStore(), backup=None)
    assert rejected.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED


@pytest.mark.asyncio
async def test_native_capture_retains_authenticated_payload_and_reads_only_file_bytes(monkeypatch) -> None:
    from linktools.ai.persistence import snapshots

    _, raw, row, bodies = await _content(external=True)
    reads = []

    async def read(*args, **kwargs):
        reads.append(args[1])
        return bodies[args[1]]

    monkeypatch.setattr(snapshots, "read_object", read)
    result = await _capture_native_payloads("group", [], [row],
        budget=_SnapshotCaptureBudget(SnapshotLimits(100, 10000)), objects=InMemoryObjectStore(), backup=None)
    assert result == {("c" * 64, row["sha256"]): raw}
    assert reads == list(bodies)


@pytest.mark.asyncio
@pytest.mark.parametrize("external", (False, True))
async def test_storage_and_outer_snapshot_share_exact_decoded_byte_boundary(external: bool) -> None:
    from linktools.ai.core import canonical_json_bytes
    from linktools.ai.persistence.snapshots import snapshot_manifest_usage, snapshot_object_ref_payload
    from linktools.ai.snapshot._snapshot import _storage_snapshot_usage

    content, raw, row, bodies = await _content(external=external)
    manifest = {"kind": "runtime-storage-snapshot", "format_version": 1,
        "namespace": "budget", "tenant_id": "tenant", "domains": {}, "archives": [],
        "native_checkpoints": [], "native_projections": [],
        "native_messages": [{"group_id": "group", "domain": "execution", "sequences": [],
            "messages": [encode_native_message_row(row)]}],
        "objects": [{"domain": "execution", "source": snapshot_object_ref_payload(ref),
                     "content": snapshot_object_ref_payload(ref)} for ref in content.object_refs()]}
    entries, extra = snapshot_manifest_usage(manifest)
    assert (entries, extra) == _storage_snapshot_usage(manifest)
    assert entries == 1 + int(external)
    assert extra == len(raw) + sum(map(len, bodies.values()))
    total = len(canonical_json_bytes(manifest)) + extra
    exact = SnapshotLimits(entries, total)
    below = SnapshotLimits(entries, total - 1)
    assert len(canonical_json_bytes(manifest)) + _storage_snapshot_usage(manifest)[1] == exact.max_bytes
    assert len(canonical_json_bytes(manifest)) + snapshot_manifest_usage(manifest)[1] > below.max_bytes
