#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for minimized Runtime durable contracts."""

from linktools.ai.runtime.state._codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
    encode_envelope,
)
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
