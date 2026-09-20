#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable identities shared by Runtime composition and history."""

import hashlib

from ..core import canonical_sha256


def grant_key(namespace: str) -> bytes:
    """Return the namespace-scoped key used for local capability grants."""
    return hashlib.sha256(f"workspace:{namespace}".encode()).digest()


def task_capability_snapshot_key(
    namespace: str,
    tenant_id: str,
    graph_id: str,
    request_digest: str,
) -> str:
    """Return the deterministic object key for one Task capability snapshot."""
    digest = canonical_sha256(
        {
            "version": 1,
            "namespace": namespace,
            "tenant_id": tenant_id,
            "graph_id": graph_id,
            "request_digest": request_digest,
        }
    )
    return f"v1/task-capability-snapshot/{digest}"


__all__ = ["grant_key", "task_capability_snapshot_key"]
