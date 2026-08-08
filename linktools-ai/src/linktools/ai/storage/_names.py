#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared names for persistent storage objects."""

TABLE_PREFIX = "ai_"


def storage_name(name: str) -> str:
    """Prefix a storage object name with the shared namespace."""
    if not name or name.startswith(TABLE_PREFIX):
        raise ValueError("storage object name must be unprefixed and non-empty")
    return f"{TABLE_PREFIX}{name}"


__all__ = ["TABLE_PREFIX", "storage_name"]
