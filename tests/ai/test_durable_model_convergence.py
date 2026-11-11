#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for minimized Runtime durable contracts."""

from datetime import datetime, timezone
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence.codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
    encode_envelope,
    wire_type_id,
)
from linktools.ai.persistence.contracts import (
    ArtifactRecord,
    ContextProjection,
    HistoryQuality,
    TranscriptHeadRecord,
    TranscriptOwnerDomain,
    TranscriptSpanRef,
)
from linktools.ai.storage import ObjectRef


def test_runtime_persisted_object_ref_omits_physical_store_identity() -> None:
    reference = ObjectRef("physical-store", "payload/key", "a" * 64, 7)

    payload = _encode_persisted_domain(reference)

    assert payload == _encode_persisted_domain(ObjectRef("another-store", "payload/key", "a" * 64, 7))
    restored = _decode_enveloped_domain(
        encode_envelope({"type": "object_ref", "payload": payload}),
        ObjectRef,
    )
    assert restored == ObjectRef("runtime", "payload/key", "a" * 64, 7)


def test_context_projection_digest_is_derived_from_items() -> None:
    projection = ContextProjection((TranscriptSpanRef(PersistenceDomain.CONVERSATION, "history", 0, 2),))

    payload = _encode_persisted_domain(projection)

    assert "digest" not in payload["fields"]
    restored = _decode_enveloped_domain(
        encode_envelope({"type": wire_type_id(projection), "payload": payload}),
        ContextProjection,
    )
    assert restored == projection and restored.digest == projection.digest
    assert ContextProjection(()).digest != projection.digest


def test_artifact_content_identity_is_derived_from_object_ref() -> None:
    reference = ObjectRef("runtime", "artifact/key", "b" * 64, 11)
    record = ArtifactRecord(
        artifact_id="artifact",
        execution_id="execution",
        producer="tool",
        mime_type="text/plain",
        object_ref=reference,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    payload = _encode_persisted_domain(record)

    assert not {"digest", "size"} & payload["fields"].keys()
    restored = _decode_enveloped_domain(
        encode_envelope({"type": wire_type_id(record), "payload": payload}),
        ArtifactRecord,
    )
    assert restored == record
    assert restored.object_ref.digest == reference.digest
    assert restored.size == reference.size


def test_transcript_head_uses_stored_record_as_its_only_revision_owner() -> None:
    head = TranscriptHeadRecord(
        TranscriptOwnerDomain.TRACE,
        "run",
        3,
        1,
        HistoryQuality.COMPLETE,
    )

    payload = _encode_persisted_domain(head)

    assert "revision" not in payload["fields"]
    assert _decode_enveloped_domain(
        encode_envelope({"type": wire_type_id(head), "payload": payload}),
        TranscriptHeadRecord,
    ) == head
