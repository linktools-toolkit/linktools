#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared names and opaque physical partition keys."""

import hashlib

TABLE_PREFIX = "ai_"


def storage_name(name: str) -> str:
    """Prefix a storage object name with the shared namespace."""
    if not name or name.startswith(TABLE_PREFIX):
        raise ValueError("storage object name must be unprefixed and non-empty")
    return f"{TABLE_PREFIX}{name}"


def namespace_key(namespace: str) -> str:
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a non-empty string")
    return hashlib.sha256(namespace.encode("utf-8")).hexdigest()


__all__ = ["TABLE_PREFIX", "namespace_key", "storage_name"]
