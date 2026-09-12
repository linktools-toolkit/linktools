#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for minimized Runtime durable contracts."""

from datetime import datetime, timezone

from linktools.ai.runtime.state import _codec as runtime_codec
from linktools.ai.runtime.state._codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
    encode_envelope,
)
from linktools.ai.runtime.state._contracts import ArtifactRecord, ContextProjection
from linktools.ai.storage import ObjectRef


def test_runtime_persisted_object_ref_omits_physical_store_identity() -> None:
    reference = ObjectRef("physical-store", "payload/key", "a" * 64, 7)

    payload = _encode_persisted_domain(reference)

    assert set(payload["fields"]) == {"key", "digest", "size"}
    restored = _decode_enveloped_domain(
        encode_envelope({"type": "object_ref", "payload": payload}),
        ObjectRef,
    )
    assert restored == ObjectRef("runtime", "payload/key", "a" * 64, 7)


def test_context_projection_digest_is_derived_from_items() -> None:
    projection = ContextProjection(())

    payload = _encode_persisted_domain(projection)

    assert set(payload["fields"]) == {"items"}
    assert projection.digest == ContextProjection(()).digest


def test_artifact_content_identity_is_derived_from_object_ref() -> None:
    reference = ObjectRef("runtime", "artifact/key", "b" * 64, 11)
    record = ArtifactRecord(
        artifact_id="artifact",
        execution_id="execution",
        tenant_id="tenant",
        producer="tool",
        media_type="text/plain",
        object_ref=reference,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    payload = _encode_persisted_domain(record)

    assert set(payload["fields"]) == {
        "artifact_id",
        "execution_id",
        "tenant_id",
        "producer",
        "media_type",
        "object_ref",
        "created_at",
    }
    assert record.digest == reference.digest
    assert record.size == reference.size


def test_session_fork_receipt_is_not_a_durable_wire_type() -> None:
    assert "session_fork_result" not in runtime_codec._V1_DOMAIN_TYPES
