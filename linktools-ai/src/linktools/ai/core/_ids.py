#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable identifiers and digest helpers."""

import hashlib
import uuid

from ._json import JsonValue, canonical_json_bytes
from ._validation import validate_persistence_namespace, validate_resource_id, validate_tenant_id
from ._value import Principal


def canonical_sha256(value: JsonValue) -> str:
    """Return the SHA-256 digest of a canonical JSON value."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def idempotency_key_hash(value: str) -> str:
    if not value:
        raise ValueError("idempotency key must not be empty")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def deterministic_id(*parts: JsonValue) -> str:
    """Return a stable UUID derived from canonical values."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, canonical_sha256(parts)))


def step_conversation_id(*, namespace: str, tenant_id: str, execution_id: str) -> str:
    """Return the execution-scoped Harness conversation identity."""
    validate_persistence_namespace(namespace)
    validate_tenant_id(tenant_id)
    validate_resource_id(execution_id)
    return "c-" + canonical_sha256(["step-conversation", namespace, tenant_id, execution_id])


def step_run_id(*, namespace: str, tenant_id: str, execution_id: str, segment_sequence: int) -> str:
    """Return the deterministic Harness Step identity for one execution segment."""
    validate_persistence_namespace(namespace)
    validate_tenant_id(tenant_id)
    validate_resource_id(execution_id)
    if segment_sequence < 1:
        raise ValueError("segment_sequence must be positive")
    return "r-" + canonical_sha256(
        ["step-run", namespace, tenant_id, execution_id, str(segment_sequence)]
    )


def principal_identity_payload(principal: Principal) -> dict[str, str]:
    """Return the stable principal identity used by request digests."""
    return {
        "tenant_id": principal.tenant_id,
        "principal_id": principal.principal_id,
        "kind": principal.kind,
    }


__all__ = [
    "canonical_sha256",
    "deterministic_id",
    "idempotency_key_hash",
    "principal_identity_payload",
    "step_conversation_id",
    "step_run_id",
]
