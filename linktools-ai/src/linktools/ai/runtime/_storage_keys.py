#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physical addresses for Runtime-owned immutable bytes."""


def runtime_object_key(
    *,
    namespace_digest: str,
    tenant_digest: str,
    stored_digest: str,
) -> str:
    """Build the tenant-scoped physical key for immutable Runtime bytes."""
    for value in (namespace_digest, tenant_digest, stored_digest):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("object digest is invalid")
    return f"v1/runtime/{namespace_digest}/{tenant_digest}/{stored_digest}"


__all__ = ["runtime_object_key"]
