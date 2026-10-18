#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MIME fields use the same names in public values, durable wire, and evidence identity."""

from datetime import datetime, timezone

import pytest

from linktools.ai.core import canonical_json_bytes, canonical_sha256
from linktools.ai.evaluation import (
    EvidenceAttachmentRef, EvidenceBundle, EvidenceRef,
    ExecutionSubjectRef, ExecutionTargetEvidence, TargetTrialRef,
)
from linktools.ai.runtime.state._codec import _CURRENT_CODEC, _decode_domain, _encode_domain
from linktools.ai.runtime.state._contracts import ArtifactRecord
from linktools.ai.storage import ObjectRef


@pytest.mark.parametrize("persisted", [False, True])
@pytest.mark.parametrize("kind", ["artifact_record", "evaluation_attachment_ref"])
def test_public_mime_fields_match_current_wire(kind: str, persisted: bool) -> None:
    ref = ObjectRef("runtime", "body", "a" * 64, 4)
    created_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
    if kind == "artifact_record":
        value = ArtifactRecord("artifact", "execution", "producer", mime_type="text/plain",
                               object_ref=ref, created_at=created_at)
        current_fields = {
            "artifact_id": "artifact", "execution_id": "execution", "producer": "producer",
            "mime_type": "text/plain", "object_ref": _encode_domain(ref, _CURRENT_CODEC, persisted=persisted),
            "created_at": {"$datetime": created_at.isoformat()},
        }
    else:
        value = EvidenceAttachmentRef("attachment", mime_type="text/plain", object_ref=ref)
        current_fields = {
            "attachment_id": "attachment", "mime_type": "text/plain",
            "object_ref": _encode_domain(ref, _CURRENT_CODEC, persisted=persisted),
        }
    current_wire = {"$dataclass": kind, "fields": current_fields}
    if persisted:
        current_wire["schema"] = 1
    encoded = _encode_domain(value, _CURRENT_CODEC, persisted=persisted)
    assert canonical_json_bytes(encoded) == canonical_json_bytes(current_wire)
    restored = _decode_domain(current_wire, type(value), _CURRENT_CODEC, persisted=persisted)
    assert restored == value
    assert restored.mime_type == "text/plain"
    assert not hasattr(restored, "media_type")


def test_evidence_public_mapping_defines_digest() -> None:
    attachment = EvidenceAttachmentRef("attachment", "text/plain", ObjectRef("runtime", "body", "a" * 64, 4))
    bundle = EvidenceBundle(
        EvidenceRef("ns", "tenant", "evidence", "0" * 64),
        TargetTrialRef("experiment", "trial"),
        ExecutionTargetEvidence(ExecutionSubjectRef("ns", "tenant", "execution"), "SUCCEEDED"),
        attachments=(attachment,),
    )
    current_content = {
        "trial": {"target_experiment_id": "experiment", "trial_id": "trial"},
        "target": {"kind": "execution", "subject": {"kind": "execution", "namespace": "ns",
                   "tenant_id": "tenant", "execution_id": "execution"},
                   "status": "SUCCEEDED", "output": None, "error_code": None},
        "input": None, "trace": [],
        "attachments": [{"kind": "runtime_content", "attachment_id": "attachment",
                         "mime_type": "text/plain", "digest": "a" * 64, "size": 4}],
        "usage": None, "usage_complete": False, "cutoff": {}, "source_ref": None, "model_usage": [],
    }
    assert attachment.to_mapping()["mime_type"] == "text/plain"
    assert "media_type" not in attachment.to_mapping()
    assert bundle.content_mapping()["attachments"] == [attachment.to_mapping()]
    assert bundle.digest == canonical_sha256(current_content)
    for persisted in (False, True):
        encoded = _encode_domain(bundle, _CURRENT_CODEC, persisted=persisted)
        restored = _decode_domain(encoded, EvidenceBundle, _CURRENT_CODEC, persisted=persisted)
        assert restored.digest == bundle.digest
        assert restored.attachments[0].mime_type == "text/plain"


def test_public_attachment_fact_uses_mime_type() -> None:
    from linktools.ai.runtime import AttachmentFact

    value = AttachmentFact(
        execution_id="execution", attachment_id="a" * 64, fact="accepted",
        source="input", mime_type="text/plain", size=4, digest="b" * 64, position=0,
    )
    assert value.mime_type == "text/plain"
    assert not hasattr(value, "media_type")
