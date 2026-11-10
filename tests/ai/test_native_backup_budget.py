#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable capture charges decoded native bytes before hydration."""

import hashlib
import json

import pytest
from pydantic_ai.messages import ModelRequest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model.messages import encode_model_messages
from linktools.ai.persistence._archive import ContentRecord
from linktools.ai.persistence.snapshots import _SnapshotCaptureBudget, _capture_native_payloads, SnapshotLimits
from linktools.ai.storage import InMemoryObjectStore, ObjectRef, StoredPayload


@pytest.mark.asyncio
@pytest.mark.parametrize("external", (False, True))
async def test_native_raw_budget_rejects_before_object_open_or_decompression(monkeypatch, external: bool) -> None:
    from linktools.ai.persistence import snapshots

    async def forbidden_read(*args, **kwargs):
        raise AssertionError("object read preceded the decoded byte budget")

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("decompression preceded the decoded byte budget")

    monkeypatch.setattr(snapshots, "read_object", forbidden_read)
    monkeypatch.setattr(snapshots, "decode_payload_bytes", forbidden_decode)
    payload = (StoredPayload.object(ObjectRef("objects", "v1/archive/group/body", "a" * 64, 10))
               if external else StoredPayload.inline_bytes(b"compressed"))
    content = ContentRecord("model-message", "b" * 64, 10001, "zlib", payload)
    rows = [{"store_digest": "c" * 64, "sha256": "b" * 64, "message": json.dumps(content.to_json())}]
    with pytest.raises(AIError) as rejected:
        await _capture_native_payloads("group", [], rows,
            budget=_SnapshotCaptureBudget(SnapshotLimits(100, 10000)), objects=InMemoryObjectStore(), backup=None)
    assert rejected.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED


@pytest.mark.asyncio
async def test_native_capture_retains_authenticated_encoded_payload_for_publication(monkeypatch) -> None:
    from linktools.ai.persistence import snapshots

    raw = encode_model_messages((ModelRequest(parts=[]),))
    digest = hashlib.sha256(raw).hexdigest()
    content = ContentRecord("model-message", digest, len(raw), "raw",
        StoredPayload.object(ObjectRef("objects", "v1/archive/group/body", digest, len(raw))))
    reads = []

    async def read(*args, **kwargs):
        reads.append(args[1])
        return raw

    monkeypatch.setattr(snapshots, "read_object", read)
    result = await _capture_native_payloads("group", [], [
        {"store_digest": "c" * 64, "sha256": digest, "message": json.dumps(content.to_json())}],
        budget=_SnapshotCaptureBudget(SnapshotLimits(100, 10000)), objects=InMemoryObjectStore(), backup=None)
    assert result == {("c" * 64, digest): raw}
    assert reads == ["v1/archive/group/body"]


@pytest.mark.parametrize("external", (False, True))
def test_storage_and_outer_snapshot_share_exact_decoded_byte_boundary(external: bool) -> None:
    from linktools.ai.core import canonical_json_bytes
    from linktools.ai.persistence.snapshots import snapshot_manifest_usage, snapshot_object_ref_payload
    from linktools.ai.snapshot._snapshot import _storage_snapshot_usage

    raw = encode_model_messages((ModelRequest(parts=[]),))
    digest = hashlib.sha256(raw).hexdigest()
    reference = ObjectRef("objects", "v1/archive/group/body", digest, len(raw))
    payload = StoredPayload.object(reference) if external else StoredPayload.inline_bytes(raw)
    content = ContentRecord("model-message", digest, len(raw), "raw", payload)
    manifest = {"kind": "runtime-storage-snapshot", "format_version": 1,
        "namespace": "budget", "tenant_id": "tenant", "domains": {}, "archives": [],
        "native_checkpoints": [], "native_projections": [],
        "native_messages": [{"group_id": "group", "domain": "execution", "sequences": [],
            "messages": [{"store_digest": "c" * 64, "sha256": digest,
                          "message": json.dumps(content.to_json())}]}],
        "objects": ([{"domain": "execution", "source": snapshot_object_ref_payload(reference),
                      "content": snapshot_object_ref_payload(reference)}] if external else [])}
    entries, extra = snapshot_manifest_usage(manifest)
    assert (entries, extra) == _storage_snapshot_usage(manifest)
    assert entries == 1 + int(external)
    assert extra == len(raw) * (2 if external else 1)
    total = len(canonical_json_bytes(manifest)) + extra
    exact = SnapshotLimits(entries, total)
    below = SnapshotLimits(entries, total - 1)
    assert len(canonical_json_bytes(manifest)) + _storage_snapshot_usage(manifest)[1] == exact.max_bytes
    assert len(canonical_json_bytes(manifest)) + snapshot_manifest_usage(manifest)[1] > below.max_bytes
