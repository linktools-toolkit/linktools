#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable identifiers and digest helpers."""

import hashlib
import uuid

from .json import JsonValue, canonical_json_bytes


def canonical_sha256(value: JsonValue) -> str:
    """Return the SHA-256 digest of a canonical JSON value."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def deterministic_id(*parts: JsonValue) -> str:
    """Return a stable UUID derived from canonical values."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, canonical_sha256(parts)))


__all__ = ["canonical_sha256", "deterministic_id"]
