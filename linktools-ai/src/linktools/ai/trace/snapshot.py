#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Immutable RunSnapshot construction and digest verification."""

from typing import Any

from ..domain.trace import RunSnapshot
from ..foundation.digest import sha256_digest
from ..foundation.json import canonical_json_bytes


def snapshot_digest(values: "dict[str, Any]") -> str:
    """Compute the digest over canonical snapshot content."""
    return sha256_digest(canonical_json_bytes(values))


def verify_snapshot(snapshot: RunSnapshot) -> bool:
    """Verify the stored immutable snapshot digest."""
    return snapshot.verify()


__all__ = ["snapshot_digest", "verify_snapshot"]
