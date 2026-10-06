#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime object addresses retain their durable namespace and tenant scope."""

import hashlib

import pytest

from linktools.ai.runtime._object import RuntimeObjectKeyFactory
from linktools.ai.runtime._storage_keys import runtime_object_key
from linktools.ai.runtime.state import RuntimeDomain


def test_runtime_object_keys_keep_the_existing_durable_address() -> None:
    domain = RuntimeDomain.EXECUTION
    namespace = "runtime-namespace"
    tenant = "tenant"
    stored_digest = hashlib.sha256(b"immutable payload").hexdigest()
    namespace_digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
    tenant_digest = hashlib.sha256(tenant.encode("utf-8")).hexdigest()
    expected = f"v1/runtime/{namespace_digest}/{tenant_digest}/{stored_digest}"

    assert RuntimeObjectKeyFactory(namespace).key(domain, tenant, stored_digest) == expected
    assert runtime_object_key(
        namespace_digest=namespace_digest,
        tenant_digest=tenant_digest,
        stored_digest=stored_digest,
    ) == expected
    assert RuntimeObjectKeyFactory("other").key(domain, tenant, stored_digest) != expected
    assert RuntimeObjectKeyFactory(namespace).key(domain, "other", stored_digest) != expected


@pytest.mark.parametrize(("field", "value"), [
    ("namespace_digest", None),
    ("tenant_digest", "a" * 63),
    ("stored_digest", "A" * 64),
])
def test_runtime_object_address_rejects_invalid_digests(field: str, value: object) -> None:
    values = {"namespace_digest": "a" * 64, "tenant_digest": "b" * 64, "stored_digest": "c" * 64}
    values[field] = value

    with pytest.raises(ValueError):
        runtime_object_key(**values)
