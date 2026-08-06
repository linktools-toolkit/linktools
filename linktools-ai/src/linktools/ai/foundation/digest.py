#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Digest helpers with strict validation."""

import hashlib
import hmac


def sha256_digest(value: bytes) -> str:
    """Return the lowercase SHA-256 digest of ``value``."""
    return hashlib.sha256(value).hexdigest()


def hmac_digest(key: bytes, value: bytes) -> str:
    """Return a lowercase HMAC-SHA-256 digest."""
    return hmac.new(key, value, hashlib.sha256).hexdigest()


def verify_digest(value: bytes, digest: str) -> None:
    """Raise ``ValueError`` when ``value`` does not match ``digest``."""
    expected = sha256_digest(value)
    if not hmac.compare_digest(expected, digest):
        raise ValueError("digest verification failed")
