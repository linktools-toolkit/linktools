#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable identifiers and digest helpers."""

import hashlib
import uuid

from ._json import JsonValue, canonical_json_bytes
from ._validation import (
    validate_persistence_namespace,
    validate_resource_id,
    validate_tenant_id,
)
from ._value import Principal


RUNTIME_OBJECT_STORE_ID = "runtime"


def canonical_sha256(value: JsonValue) -> str:
    """Return the SHA-256 digest of a canonical JSON value."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def idempotency_key_digest(value: str) -> str:
    if not value:
        raise ValueError("idempotency key must not be empty")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def deterministic_id(*parts: JsonValue) -> str:
    """Return a stable UUID derived from canonical values."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, canonical_sha256(parts)))


def agent_conversation_id(*, namespace: str, tenant_id: str, execution_id: str) -> str:
    """Return the execution-scoped conversation identity."""
    validate_persistence_namespace(namespace)
    validate_tenant_id(tenant_id)
    validate_resource_id(execution_id)
    return "c-" + canonical_sha256(["agent-conversation", namespace, tenant_id, execution_id])


def agent_run_id(
    *,
    namespace: str,
    tenant_id: str,
    execution_id: str,
    agent_run_sequence: int,
) -> str:
    """Return the deterministic AgentRun identity for one execution."""
    validate_persistence_namespace(namespace)
    validate_tenant_id(tenant_id)
    validate_resource_id(execution_id)
    if agent_run_sequence < 1:
        raise ValueError("agent_run_sequence must be positive")
    return "r-" + canonical_sha256(
        ["agent-run", namespace, tenant_id, execution_id, str(agent_run_sequence)]
    )


def principal_identity_payload(principal: Principal) -> dict[str, str]:
    """Return the stable principal identity used by request digests."""
    return {
        "tenant_id": principal.tenant_id,
        "principal_id": principal.principal_id,
        "kind": principal.kind,
    }


__all__ = [
    "RUNTIME_OBJECT_STORE_ID",
    "canonical_sha256",
    "deterministic_id",
    "idempotency_key_digest",
    "principal_identity_payload",
    "agent_conversation_id",
    "agent_run_id",
]
