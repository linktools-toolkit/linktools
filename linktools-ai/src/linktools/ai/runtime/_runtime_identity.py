#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable identities shared by Runtime composition and history."""

import hashlib

from ..core import canonical_sha256


def token_seed(namespace: str) -> bytes:
    """Return a deterministic namespace-scoped seed for opaque Runtime tokens."""
    return hashlib.sha256(f"runtime-token-v1:{namespace}".encode()).digest()


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


__all__ = ["task_capability_snapshot_key", "token_seed"]
