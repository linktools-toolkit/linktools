#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Runtime object addressing, independent of object I/O and maintenance."""

import re
from dataclasses import dataclass

from ..core import validate_persistence_namespace, validate_tenant_id
from ._plan import PersistenceDomain

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class ObjectKeyFactory:
    namespace: str

    def __post_init__(self) -> None:
        validate_persistence_namespace(self.namespace)

    def key(self, runtime_domain: PersistenceDomain, tenant_id: str, digest: str) -> str:
        if not isinstance(runtime_domain, PersistenceDomain):
            raise TypeError("domain must be PersistenceDomain")
        tenant_id = validate_tenant_id(tenant_id)
        if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
            raise ValueError("Runtime object digest is invalid")
        # Character lengths frame opaque components, including slashes and colons.
        return f"v1/runtime/{len(self.namespace)}:{self.namespace}/{len(tenant_id)}:{tenant_id}/{digest}"
