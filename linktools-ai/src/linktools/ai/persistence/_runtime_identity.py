#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable identities shared by Runtime composition and history."""

import hashlib

from linktools.ai.core import canonical_sha256



def task_graph_binding_capture_key(
    namespace: str,
    tenant_id: str,
    graph_id: str,
    request_digest: str,
) -> str:
    """Return the key for one graph's admitted Task and expander declarations."""
    digest = canonical_sha256(
        {
            "version": 1,
            "namespace": namespace,
            "tenant_id": tenant_id,
            "graph_id": graph_id,
            "request_digest": request_digest,
        }
    )
    return f"v1/task-graph-binding-capture/{digest}"


__all__ = ["task_graph_binding_capture_key"]
