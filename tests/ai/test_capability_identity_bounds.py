#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability-owned identities are bounded before admission and on durable reads."""

from collections.abc import Mapping
from copy import copy
from pathlib import Path

import pytest
from pydantic_ai.capabilities import AbstractCapability
from sqlalchemy import event

from linktools.ai.core import CommitPolicy, JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistencePlan, PersistenceStorage
from linktools.ai.persistence._materializer import materialize_runtime_storage
from linktools.ai.persistence._sql import SqlStateStore
from linktools.ai.persistence.capabilities import CapabilityWriter, RuntimeCapabilityWrites
from linktools.ai.persistence.state import StateTransaction
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.runtime._persistence_configuration import capability_persistence_ids
from linktools.ai.spec import CapabilityRef
from linktools.ai.storage import StorageRoute

_CAPABILITY_ID = "界" * 42 + "ab"
_RUN_ID = "界" * 42 + "ab"
_WRITE_KEY = "界" * 42 + "ab"
_REVISION = 9223372036854775807


class _Capability(AbstractCapability[object]):
    commit = CommitPolicy.sync()

    def __init__(self, identity: str, persisted: list[dict[str, JsonValue]]) -> None:
        self.id = identity
        self.persisted = persisted
        self.writer: CapabilityWriter | None = None

    def bind_persistence(self, writer: CapabilityWriter) -> "_Capability":
        bound = copy(self)
        bound.writer = writer
        return bound

    async def persist_in_transaction(
        self, transaction: StateTransaction, payload: Mapping[str, JsonValue],
    ) -> None:
        self.persisted.append(dict(payload))


def test_capability_declarations_and_references_preserve_exact_maximum_identity() -> None:
    features = RuntimeFeatures()
    capability = _Capability(_CAPABILITY_ID, [])
    assert features.capability(capability, revision=_REVISION) is capability
    reference = CapabilityRef("capability", _CAPABILITY_ID, _REVISION)
    assert CapabilityRef.from_payload(reference.to_payload()) == reference
    assert capability_persistence_ids(((_CAPABILITY_ID, capability),)) == (_CAPABILITY_ID,)
    # Capability IDs remain opaque; bounding them does not normalize their text.
    spaced = _Capability(" capability/name ", [])
    assert features.capability(spaced) is spaced
    assert CapabilityRef.from_payload({"kind": "capability", "id": spaced.id, "revision": 1}).id == spaced.id


@pytest.mark.parametrize(("identity", "revision"), (
    (_CAPABILITY_ID + "b", 1), ("\ud800", 1), (" ", 1),
    ("capability", True), ("capability", 0), ("capability", 1.0), ("capability", _REVISION + 1),
))
def test_invalid_capability_declarations_and_durable_references_use_owner_errors(
    identity: str, revision: int,
) -> None:
    with pytest.raises(AIError) as declaration:
        RuntimeFeatures().capability(_Capability(identity, []), revision=revision)
    assert declaration.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
    with pytest.raises(AIError) as durable:
        CapabilityRef.from_payload({"kind": "capability", "id": identity, "revision": revision})
    assert durable.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert repr(identity) not in str(declaration.value)
    assert repr(identity) not in str(durable.value)


@pytest.mark.parametrize("kind", ("agent", "tool", "skill", "mcp"))
def test_capability_bounds_do_not_redefine_other_named_reference_contracts(kind: str) -> None:
    reference = CapabilityRef(kind, "x" * 257, _REVISION + 1)
    assert CapabilityRef.from_payload(reference.to_payload()) == reference


@pytest.mark.asyncio
async def test_capability_storage_routes_and_dynamic_partitions_validate_owner_ids() -> None:
    oversized = _CAPABILITY_ID + "b"
    with pytest.raises(AIError) as configured:
        PersistencePlan(capability_routes={oversized: StorageRoute.memory()})
    assert configured.value.code is ErrorCode.REQUEST_FIELD_INVALID
    cleanups = []
    with pytest.raises(AIError) as materialized:
        await materialize_runtime_storage(
            PersistencePlan(), namespace="namespace", tenant_id="tenant", object_store=None,
            capability_routes={oversized: StorageRoute.memory()}, register_cleanup=cleanups.extend,
        )
    assert materialized.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert cleanups == []
    state = Persistence(PersistenceStorage.from_plan(PersistencePlan(
        capability_routes={_CAPABILITY_ID: StorageRoute.memory()},
    )))
    with pytest.raises(AIError) as dynamic:
        await state.capability_store(oversized)
    assert dynamic.value.code is ErrorCode.REQUEST_FIELD_INVALID
    await state.initialize(namespace="namespace", tenant_id="tenant")
    try:
        assert await state.capability_store(_CAPABILITY_ID) is state.capability_stores[_CAPABILITY_ID]
        other = "a" + _CAPABILITY_ID[:-1]
        assert await state.capability_store(other) is await state.capability_store(other)
    finally:
        await state.close()
    with pytest.raises(AIError) as participant:
        capability_persistence_ids(((oversized, _Capability(oversized, [])),))
    assert participant.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_capability_writer_rejects_invalid_identity_before_admission_or_storage(
    backend: str, tmp_path: Path,
) -> None:
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "runtime.db")
    state = Persistence(storage)
    await state.initialize(namespace="namespace", tenant_id="tenant")
    store = state.execution.executions.state_store
    admitted = []
    writes = RuntimeCapabilityWrites(
        {_CAPABILITY_ID: store}, namespace="namespace", tenant_id="tenant",
        durable_capability_ids=frozenset(), require_active_run=lambda *args: admitted.append(args),
        has_active_execution=lambda _: True,
    )
    capability = _Capability(_CAPABILITY_ID, [])
    statements = []

    def capture(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement)

    if isinstance(store, SqlStateStore):
        event.listen(store.context.engine.sync_engine, "before_cursor_execute", capture)
    try:
        for identity, run_id, revision in (
            (_CAPABILITY_ID + "b", _RUN_ID, 1), (_CAPABILITY_ID, _RUN_ID + "c", 1),
            (_CAPABILITY_ID, _RUN_ID, True), (_CAPABILITY_ID, _RUN_ID, 0),
            (_CAPABILITY_ID, _RUN_ID, _REVISION + 1),
        ):
            with pytest.raises(AIError) as bound:
                writes.bind(identity, run_id, "execution", capability, revision=revision)
            assert bound.value.code is ErrorCode.REQUEST_FIELD_INVALID
        assert admitted == []
        with pytest.raises(AIError) as registered:
            writes.register_store(_CAPABILITY_ID + "b", store, durable=False)
        assert registered.value.code is ErrorCode.REQUEST_FIELD_INVALID
        with pytest.raises(AIError) as initial:
            RuntimeCapabilityWrites(
                {_CAPABILITY_ID + "b": store}, namespace="namespace", tenant_id="tenant",
                durable_capability_ids=frozenset(), require_active_run=lambda *_: None,
                has_active_execution=lambda _: True,
            )
        assert initial.value.code is ErrorCode.REQUEST_FIELD_INVALID
        bound = writes.bind(_CAPABILITY_ID, _RUN_ID, "execution", capability, revision=_REVISION)
        for key in ("", _WRITE_KEY + "b", "\ud800"):
            with pytest.raises(AIError) as submitted:
                await bound.writer.submit(key=key, payload={"value": 1})
            assert submitted.value.code is ErrorCode.IDEMPOTENCY_KEY_INVALID
        assert not capability.persisted
        assert statements == []
    finally:
        if isinstance(store, SqlStateStore):
            event.remove(store.context.engine.sync_engine, "before_cursor_execute", capture)
        await writes.close()
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_maximum_capability_receipts_replay_only_within_their_exact_scope(
    backend: str, tmp_path: Path,
) -> None:
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "runtime.db")
    state = Persistence(storage)
    await state.initialize(namespace="namespace", tenant_id="tenant")
    store = state.execution.executions.state_store
    persisted = []
    original = ("界" * 42 + "ab", "界" * 42 + "ab", _CAPABILITY_ID, _REVISION, _RUN_ID, _WRITE_KEY)
    scopes = [original]
    for index in range(len(original)):
        scope = list(original)
        scope[index] = _REVISION - 1 if index == 3 else "other"
        scopes.append(tuple(scope))
    try:
        for namespace, tenant, identity, revision, run_id, key in scopes:
            # Recreate the writer to exercise persisted, rather than in-memory, replay.
            for _ in range(2):
                writes = RuntimeCapabilityWrites(
                    {identity: store}, namespace=namespace, tenant_id=tenant,
                    durable_capability_ids=frozenset(), require_active_run=lambda *_: None,
                    has_active_execution=lambda _: True,
                )
                try:
                    bound = writes.bind(identity, run_id, "execution", _Capability(identity, persisted), revision=revision)
                    await bound.writer.submit(key=key, payload={"value": 1})
                finally:
                    await writes.close()
        assert persisted == [{"value": 1}] * len(scopes)
        writes = RuntimeCapabilityWrites(
            {_CAPABILITY_ID: store}, namespace=original[0], tenant_id=original[1],
            durable_capability_ids=frozenset(), require_active_run=lambda *_: None,
            has_active_execution=lambda _: True,
        )
        try:
            bound = writes.bind(_CAPABILITY_ID, _RUN_ID, "execution", _Capability(_CAPABILITY_ID, persisted), revision=_REVISION)
            # Whitespace and control characters remain valid opaque write-key bytes.
            for key in (" " * 128, "\n\x00", "e\u0301", "é"):
                await bound.writer.submit(key=key, payload={"value": 2})
            assert persisted[-4:] == [{"value": 2}] * 4
        finally:
            await writes.close()
        conflict = RuntimeCapabilityWrites(
            {_CAPABILITY_ID: store}, namespace=original[0], tenant_id=original[1],
            durable_capability_ids=frozenset(), require_active_run=lambda *_: None,
            has_active_execution=lambda _: True,
        )
        bound = conflict.bind(_CAPABILITY_ID, _RUN_ID, "execution", _Capability(_CAPABILITY_ID, persisted), revision=_REVISION)
        with pytest.raises(AIError) as replay:
            await bound.writer.submit(key=_WRITE_KEY, payload={"value": 99})
        assert replay.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        with pytest.raises(AIError):
            await conflict.close()
        assert {"value": 99} not in persisted
    finally:
        await state.close()
