#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Final result contract helpers."""

from ..foundation.digest import sha256_digest
from ..foundation.json import canonical_json_bytes
from .execution import ExecutionResult


def result_digest(value: object) -> str:
    """Compute the canonical digest used for result idempotency."""
    return sha256_digest(canonical_json_bytes(value))


__all__ = ["ExecutionResult", "result_digest"]
