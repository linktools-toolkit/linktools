#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime object addresses retain their logical namespace and tenant scope."""

import hashlib

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence.objects import ObjectKeyFactory
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.storage import ObjectRef


def test_runtime_object_keys_preserve_unambiguous_logical_scope() -> None:
    domain = PersistenceDomain.EXECUTION
    namespace = "runtime/命名:空间"
    tenant = "tenant/租户:2"
    stored_digest = hashlib.sha256(b"immutable payload").hexdigest()
    factory = ObjectKeyFactory(namespace)
    key = factory.key(domain, tenant, stored_digest)

    assert namespace in key and tenant in key and key.endswith(stored_digest)
    assert factory.key(PersistenceDomain.TASK, tenant, stored_digest) == key
    scopes = ((namespace, tenant), ("other", tenant), (namespace, "other"),
              ("a/b", "c"), ("a", "b/c"), ("a:1", "b"), ("a", "1:b"))
    assert len({ObjectKeyFactory(ns).key(domain, owner, stored_digest) for ns, owner in scopes}) == len(scopes)


def test_runtime_object_key_accepts_maximum_utf8_scope_sizes() -> None:
    namespace = "界" * 85 + "a"
    tenant = "户" * 42 + "aa"
    digest = "a" * 64
    key = ObjectKeyFactory(namespace).key(PersistenceDomain.EXECUTION, tenant, digest)

    assert namespace in key and tenant in key
    assert len(key.encode("utf-8")) <= 1024
    assert ObjectRef("runtime", key, digest, 0).key == key


@pytest.mark.parametrize(("namespace", "tenant"), [
    ("界" * 85 + "ab", "tenant"),
    ("namespace", "户" * 43),
])
def test_runtime_object_key_rejects_scope_beyond_utf8_limits(namespace: str, tenant: str) -> None:
    with pytest.raises(AIError) as error:
        ObjectKeyFactory(namespace).key(PersistenceDomain.EXECUTION, tenant, "a" * 64)
    assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID


@pytest.mark.parametrize("digest", [None, "a" * 63, "A" * 64])
def test_runtime_object_key_rejects_invalid_content_digest(digest: object) -> None:
    with pytest.raises(ValueError):
        ObjectKeyFactory("namespace").key(PersistenceDomain.EXECUTION, "tenant", digest)
