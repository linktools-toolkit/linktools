#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Deterministic identifiers for tenant-scoped operations."""

import uuid

from .json import canonical_json_bytes
from .digest import hmac_digest


def deterministic_id(key: bytes, *parts: object) -> str:
    """Create a deterministic UUID from canonical parts and a secret key."""
    digest = hmac_digest(key, canonical_json_bytes(parts))
    return str(uuid.UUID(digest[:32]))


def workflow_id(tenant_id: str, execution_id: str) -> str:
    """Return the stable Temporal workflow identifier."""
    return f"lt/{hmac_digest(b'workflow-tenant', tenant_id.encode())[:16]}/{execution_id}"
