#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retained payload reuse requires physical identity and durable route ownership."""

from collections.abc import AsyncIterator
from functools import partial
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence._history import TranscriptRepository
from linktools.ai.persistence._memory import InMemoryStateStore
from linktools.ai.persistence._object_router import _RuntimeObjectRouter
from linktools.ai.persistence.objects import ObjectKeyFactory, put_runtime_object, read_runtime_object
from linktools.ai.storage import InMemoryObjectStore, ObjectRef, ObjectStat, StoredPayload


class CountingObjects(InMemoryObjectStore):
    def __init__(self) -> None:
        super().__init__("same-logical-store")
        self.puts = 0

    async def put(
        self, key: str, chunks: AsyncIterator[bytes], *, expected_size: int, expected_digest: str,
    ) -> ObjectStat:
        self.puts += 1
        return await super().put(key, chunks, expected_size=expected_size, expected_digest=expected_digest)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ("shared", "distinct", "volatile_source", "volatile_target", "foreign_key", "missing", "unproven"))
async def test_retained_payload_reuse_requires_present_shared_durable_canonical_object(case: str) -> None:
    source = CountingObjects()
    target = CountingObjects() if case == "distinct" else source
    domains = (PersistenceDomain.EXECUTION, PersistenceDomain.CONVERSATION)
    durable = tuple(domain for domain in domains if domain is not (
        PersistenceDomain.EXECUTION if case == "volatile_source"
        else PersistenceDomain.CONVERSATION if case == "volatile_target" else None
    ))
    if case == "unproven":
        durable = ()
    router = _RuntimeObjectRouter(
        dict(zip(domains, (source, target))), close_guard_stores=(),
        durable_domains=durable, cache_domains=domains,
    )
    value = b"retained input" * 100
    reference = await put_runtime_object(
        source, ObjectKeyFactory("foreign" if case == "foreign_key" else "retention"),
        PersistenceDomain.EXECUTION, "tenant", value,
    )
    reads = []

    async def read(domain: PersistenceDomain, ref: ObjectRef) -> bytes:
        reads.append((domain, ref))
        if case == "missing":
            # A hot read can outlive source membership; retention must recreate the object.
            return value
        return await router.read_object(domain, ref)

    if case == "missing":
        await source.delete_object(reference.key, expected_digest=reference.digest)
    history = TranscriptRepository(
        InMemoryStateStore(), object_store=target, namespace="retention", tenant_id="tenant",
        runtime_domain=PersistenceDomain.CONVERSATION, object_reader=read,
        retained_object_reuse=partial(router.can_reuse_retained_object, PersistenceDomain.CONVERSATION),
    )
    before = target.puts
    retained = await history.retain_payload(StoredPayload.object(reference), source_domain=PersistenceDomain.EXECUTION)
    assert target.puts - before == (0 if case == "shared" else 1)
    assert reads == [(PersistenceDomain.EXECUTION, reference)]
    assert await read_runtime_object(target, retained.ref) == value
    if case == "distinct":
        await source.delete_object(reference.key, expected_digest=reference.digest)
        assert await read_runtime_object(target, retained.ref) == value
    await router.drain_reads()


@pytest.mark.asyncio
@pytest.mark.parametrize("corrupt", (b"bad", b"wrong contents"))
async def test_reuse_does_not_bypass_full_payload_integrity(corrupt: bytes) -> None:
    objects = CountingObjects()
    reference = await put_runtime_object(
        objects, ObjectKeyFactory("retention"), PersistenceDomain.EXECUTION, "tenant", b"valid contents",
    )
    router = _RuntimeObjectRouter(
        {domain: objects for domain in (PersistenceDomain.EXECUTION, PersistenceDomain.CONVERSATION)},
        close_guard_stores=(), durable_domains=(PersistenceDomain.EXECUTION, PersistenceDomain.CONVERSATION),
    )

    async def read(domain: PersistenceDomain, ref: ObjectRef) -> bytes:
        return corrupt

    history = TranscriptRepository(
        InMemoryStateStore(), object_store=objects, namespace="retention", tenant_id="tenant",
        runtime_domain=PersistenceDomain.CONVERSATION, object_reader=read,
        retained_object_reuse=partial(router.can_reuse_retained_object, PersistenceDomain.CONVERSATION),
    )
    with pytest.raises(AIError) as raised:
        await history.retain_payload(StoredPayload.object(reference), source_domain=PersistenceDomain.EXECUTION)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert objects.puts == 1
    await router.drain_reads()


@pytest.mark.asyncio
async def test_materialized_durable_routes_reuse_retained_payload(tmp_path: Path) -> None:
    objects = CountingObjects()
    persistence = Persistence(PersistenceStorage.sqlite(tmp_path / "state.db", object_store=objects))
    await persistence.initialize(namespace="retention", tenant_id="tenant")
    try:
        reference = await put_runtime_object(
            objects, ObjectKeyFactory("retention"), PersistenceDomain.EXECUTION, "tenant", b"retained input",
        )
        history = persistence.run_store.read_store(PersistenceDomain.CONVERSATION).transcript_repository
        before = objects.puts
        retained = await history.retain_payload(StoredPayload.object(reference), source_domain=PersistenceDomain.EXECUTION)
        assert objects.puts == before
        assert retained == StoredPayload.object(reference)
        assert await history.read_payload(retained) == b"retained input"
    finally:
        await persistence.close()
