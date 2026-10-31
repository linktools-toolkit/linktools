#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Source authority is selected, owned and snapshotted through public Persistence."""

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus, PersistenceDomain
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._snapshot_validation import validate_snapshot_domain
from linktools.ai.persistence.turns import StateTurnCoordination
from linktools.ai.persistence.state import StateTransactionNestingError
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.snapshot import Snapshot, SnapshotLimits
from linktools.ai.storage import (
    CacheBackend, InMemoryCacheBackend, InMemoryObjectStore, read_object,
    TurnAdmissionReader, TurnCoordinationBackend, TurnSessionReader,
)

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_runtime_snapshot_operations import Offline

pytestmark = pytest.mark.asyncio
_LIMITS = SnapshotLimits(max_entries=1000, max_bytes=1_000_000)


class _BorrowedCache(InMemoryCacheBackend):
    def __init__(self) -> None:
        super().__init__(max_bytes=2**24)
        self.closed = False

    async def claim_turn(self, *args, **kwargs):
        raise AssertionError("borrowed hot content must not remain an authority")

    async def read_turn(self, *args, **kwargs):
        raise AssertionError("source authority must not consult cache ownership")

    async def close(self) -> None:
        self.closed = True


class _HostCoordination:
    """A host's explicit once-bound cache/turn composition, not an SDK factory."""

    def __init__(self, cache: _BorrowedCache) -> None:
        self.cache = cache
        self.authority = None

    def bind(self, state: Persistence) -> None:
        if self.authority is not None:
            raise ValueError("authority is already bound")
        if not state.ready:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

        def require_ready():
            if not state.ready:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

        self.authority = StateTurnCoordination(
            state.execution.executions.state_store,
            namespace=state.namespace, tenant_id=state.tenant_id,
            read_only=state.read_only, require_ready=require_ready,
        )

    def _bound(self):
        if self.authority is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self.authority

    async def get(self, key):
        return await self.cache.get(key)

    async def put(self, key, content):
        await self.cache.put(key, content)

    async def delete(self, key):
        await self.cache.delete(key)

    async def claim_turn(self, scope, **kwargs):
        return await self._bound().claim_turn(scope, **kwargs)

    async def read_turn(self, scope, execution_id):
        return await self._bound().read_turn(scope, execution_id)

    async def renew_turn(self, lease, *, lease_seconds):
        return await self._bound().renew_turn(lease, lease_seconds=lease_seconds)

    async def release_turn(self, lease):
        return await self._bound().release_turn(lease)

    async def cancel_turn(self, scope, execution_id):
        return await self._bound().cancel_turn(scope, execution_id)

    async def read_turn_by_request(self, scope, request_key):
        return await self._bound().read_turn_by_request(scope, request_key)

    async def read_turn_by_session(self, scope, session_key):
        return await self._bound().read_turn_by_session(scope, session_key)


def _storage(backend: str, path: Path, cache: _BorrowedCache) -> PersistenceStorage:
    coordination = _HostCoordination(cache)
    if backend == "memory":
        return PersistenceStorage.in_memory(coordination=coordination)
    if backend == "filesystem":
        return PersistenceStorage.filesystem(path, coordination=coordination)
    return PersistenceStorage.sqlite(path / "state.db", coordination=coordination)


@asynccontextmanager
async def _opened(backend, path, cache, *, namespace="source-turns", read_only=False):
    async with Persistence.open(namespace, storage=_storage(backend, path, cache),
                                tenant_id="tenant", read_only=read_only) as state:
        state.coordination_backend.bind(state)
        yield state


async def _claim(backend: _HostCoordination, *, execution: str = "execution"):
    return await backend.claim_turn(
        "scope", session_key="session", request_key="request",
        request_digest="request-digest", execution_id=execution,
        owner_id="owner-" + execution, lease_seconds=600,
    )


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_host_binds_source_once_and_borrows_cache(
    tmp_path: Path, backend: str,
) -> None:
    cache = _BorrowedCache()
    configuration = _storage(backend, tmp_path / backend, cache)
    state = Persistence(configuration)
    await state.initialize(namespace="source-turns", tenant_id="tenant")
    bound = state.coordination_backend
    try:
        for observe in (
            lambda: _claim(bound), lambda: bound.read_turn("scope", "execution"),
            lambda: bound.read_turn_by_request("scope", "request"),
            lambda: bound.read_turn_by_session("scope", "session"),
            lambda: bound.renew_turn(None, lease_seconds=600),
            lambda: bound.cancel_turn("scope", "execution"), lambda: bound.release_turn(None),
        ):
            with pytest.raises(AIError) as unbound:
                await observe()
            assert unbound.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
        bound.bind(state)
        assert isinstance(bound.authority, StateTurnCoordination)
        assert not isinstance(bound.authority, CacheBackend)
        assert isinstance(bound, CacheBackend)
        assert isinstance(bound, TurnCoordinationBackend)
        assert isinstance(bound, TurnAdmissionReader)
        assert isinstance(bound, TurnSessionReader)
        assert bound is configuration.coordination
        with pytest.raises(ValueError, match="already bound"):
            bound.bind(state)
        await bound.put("display", b"content")
        assert await cache.get("display") == b"content"
        claimed = await _claim(bound)
        assert claimed is not None and claimed.lease is not None
        rows = await state.execution.executions.state_store.read(lambda tx: tx.scan_records())
        assert {row.kind for row in rows} == {"turn_session", "turn_request", "turn_execution"}
        assert not state.capability_stores and not state.durable_capability_ids
        assert await bound.read_turn_by_session("scope", "session") == replace(claimed, lease=None)
        assert await bound.release_turn(claimed.lease)
    finally:
        await state.close()
    assert not cache.closed
    with pytest.raises(AIError) as closed:
        await bound.read_turn_by_session("scope", "session")
    assert closed.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
    with pytest.raises(AIError) as closed:
        await _claim(bound, execution="after-close")
    assert closed.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY


async def test_separate_host_bindings_do_not_rebind_existing_owner(tmp_path: Path) -> None:
    cache = _BorrowedCache()
    async with _opened("sqlite", tmp_path, cache, namespace="first") as first:
        original = first.coordination_backend
        claimed = await _claim(original)
        async with _opened("sqlite", tmp_path, cache, namespace="second") as second:
            assert second.coordination_backend is not original
            assert first.coordination_backend is original
            assert await second.coordination_backend.read_turn_by_session("scope", "session") is None
            assert await original.read_turn_by_session("scope", "session") == replace(claimed, lease=None)
            other = await _claim(second.coordination_backend)
            assert other.lease != claimed.lease
            assert await second.coordination_backend.release_turn(other.lease)
        assert await original.release_turn(claimed.lease)
    assert not cache.closed


@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
async def test_readonly_source_turn_binding_never_mutates(tmp_path: Path, backend: str) -> None:
    cache = _BorrowedCache()
    async with _opened(backend, tmp_path / backend, cache) as state:
        claimed = await _claim(state.coordination_backend)
        original = await state.execution.executions.state_store.read(lambda tx: tx.scan_records())
    async with _opened(backend, tmp_path / backend, cache, read_only=True) as state:
        reader = state.coordination_backend
        assert await reader.read_turn("scope", "execution") == replace(claimed, lease=None)
        assert await reader.read_turn_by_request("scope", "request") == replace(claimed, lease=None)
        assert await reader.read_turn_by_session("scope", "session") == replace(claimed, lease=None)
        actions = (
            lambda: _claim(reader, execution="other"),
            lambda: reader.renew_turn(claimed.lease, lease_seconds=600),
            lambda: reader.cancel_turn("scope", "execution"),
            lambda: reader.release_turn(claimed.lease),
        )
        for action in actions:
            with pytest.raises(StateTransactionNestingError, match="read-only"):
                await action()
        actual = await state.execution.executions.state_store.read(lambda tx: tx.scan_records())
        assert {record.key_digest: record for record in actual} == {record.key_digest: record for record in original}
    assert not cache.closed


@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
@pytest.mark.parametrize("active", (False, True))
@pytest.mark.parametrize("exclusive_writer", (False, True))
async def test_source_turn_snapshot_roundtrips_complete_graph_and_receipts(
    tmp_path: Path, backend: str, active: bool, exclusive_writer: bool,
) -> None:
    cache = _BorrowedCache()
    async with _opened(backend, tmp_path / "source", cache) as state:
        authority = state.coordination_backend
        first = await _claim(authority)
        assert await authority.cancel_turn("scope", "execution")
        assert await authority.release_turn(first.lease)
        current = await _claim(authority, execution="next")
        current = await authority.renew_turn(current.lease, lease_seconds=600)
        if not active:
            assert await authority.release_turn(current.lease)
        expected = await state.execution.executions.state_store.read(lambda tx: tx.scan_records())
    archive = InMemoryObjectStore()
    async with _opened(backend, tmp_path / "source", cache, read_only=True) as state:
        # The public wrapper retains caller-owned offline exclusion.
        assert await Snapshot.create(persistence=state, object_store=archive, exclusive=Offline(), limits=_LIMITS)
        reference = await state.export_snapshot(object_store=archive, limits=_LIMITS)
    destination = tmp_path / "restored"
    await Persistence.restore_snapshot(reference, object_store=archive, root=destination,
                                       limits=_LIMITS, exclusive_writer=exclusive_writer)
    restored_configuration = PersistenceStorage.from_root(
        destination, coordination=_HostCoordination(cache),
    )
    async with Persistence.open("source-turns", storage=restored_configuration, tenant_id="tenant", read_only=True) as restored:
        assert not restored.exclusive_execution_writer
        restored.coordination_backend.bind(restored)
        actual = await restored.execution.executions.state_store.read(lambda tx: tx.scan_records())
        assert {record.key_digest: record for record in actual} == {record.key_digest: record for record in expected}
        roundtrip = await restored.export_snapshot(object_store=archive, limits=_LIMITS)
        original_payload = json.loads(await read_object(archive, reference.key,
            expected_digest=reference.digest, expected_size=reference.size))
        restored_payload = json.loads(await read_object(archive, roundtrip.key,
            expected_digest=roundtrip.digest, expected_size=roundtrip.size))
        for payload in (original_payload, restored_payload):
            for domain in payload["domains"].values():
                records = domain["records"]
                assert len(records) == len({record["key"] for record in records})
                # StateStore traversal order is not logical record identity.
                records.sort(key=lambda record: record["key"])
        assert restored_payload == original_payload
        if backend == "sqlite":
            assert roundtrip == reference
        assert await restored.export_snapshot(object_store=archive, limits=_LIMITS) == roundtrip
        assert await restored.coordination_backend.read_turn_by_session("scope", "session") == (
            replace(current, lease=None) if active else None
        )
    writable_configuration = PersistenceStorage.from_root(
        destination, coordination=_HostCoordination(cache), exclusive_writer=exclusive_writer,
    )
    async with Persistence.open("source-turns", storage=writable_configuration, tenant_id="tenant") as restored:
        restored.coordination_backend.bind(restored)
        assert restored.exclusive_execution_writer is exclusive_writer
        if exclusive_writer:
            contender = Persistence(PersistenceStorage.from_root(destination, exclusive_writer=True))
            try:
                with pytest.raises(AIError) as rejected:
                    await contender.initialize(namespace="source-turns", tenant_id="tenant")
                assert rejected.value.code is ErrorCode.STORAGE_CONFLICT
            finally:
                await contender.close()
        assert await restored.coordination_backend.read_turn_by_request("scope", "request") == (
            replace(current, lease=None) if active else None
        )
        if active:
            assert await _claim(restored.coordination_backend, execution="ignored") == replace(current, lease=None)
        assert not await restored.coordination_backend.release_turn(first.lease)
        assert await restored.coordination_backend.release_turn(current.lease) is active
    assert not cache.closed


@pytest.mark.parametrize("defect", ("missing_execution", "foreign_namespace", "wrong_domain", "bad_physical_key"))
async def test_snapshot_rejects_broken_source_turn_graph(tmp_path: Path, defect: str) -> None:
    cache = _BorrowedCache()
    async with _opened("sqlite", tmp_path, cache) as state:
        await _claim(state.coordination_backend)
        records = await state.execution.executions.state_store.read(lambda tx: tx.scan_records())
    namespace = "source-turns"
    domain = PersistenceDomain.EXECUTION
    if defect == "missing_execution":
        records = tuple(record for record in records if record.kind != "turn_execution")
    elif defect == "foreign_namespace":
        namespace = "wrong"
    elif defect == "wrong_domain":
        domain = PersistenceDomain.MEMORY
    else:
        records = (replace(records[0], key_digest=b"x" * 32), *records[1:])
    with pytest.raises(AIError) as rejected:
        validate_snapshot_domain(
            namespace=namespace, tenant_id="tenant", domain=domain,
            records=records, aliases=(), facts=(), operations=(), sequences={},
        )
    assert rejected.value.code is (
        ErrorCode.STORAGE_VERSION_UNSUPPORTED if defect == "wrong_domain" else ErrorCode.STORAGE_INTEGRITY_ERROR
    )


async def test_runtime_selects_source_authority_before_sync_execution_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    claim = StateTurnCoordination.claim_turn
    observed = {}
    provider_calls = 0

    async def blocked_claim(backend, scope, **kwargs):
        result = await claim(backend, scope, **kwargs)
        observed.update(scope=scope, session_key=kwargs["session_key"], result=result)
        entered.set()
        await release.wait()
        return result

    async def model(messages, info):
        nonlocal provider_calls
        provider_calls += 1
        return ModelResponse(parts=[TextPart("answer")])

    monkeypatch.setattr(StateTurnCoordination, "claim_turn", blocked_claim)
    features = RuntimeFeatures("source-turns")
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.sync()), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    features = model_features(_Models(_UsageFunctionModel(model)), features)
    cache = _BorrowedCache()
    async with _opened("sqlite", tmp_path, cache) as state:
        async with Runtime.attach(persistence=state, features=features) as runtime:
            session = await runtime.agents.get().create_session("session")
            starting = asyncio.create_task(session.start("question", idempotency_key="request"))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                admitted = observed["result"]
                assert admitted is not None and admitted.lease is not None
                assert provider_calls == 0
                assert await state.execution.executions.get(admitted.execution_id, tenant_id=runtime.tenant_id) is None
                assert await state.coordination_backend.read_turn_by_session(
                    observed["scope"], observed["session_key"],
                ) == replace(admitted, lease=None)
                release.set()
                execution = await asyncio.wait_for(starting, 5)
                result = await execution.wait()
                assert result.result.status is ExecutionStatus.SUCCEEDED
                assert provider_calls == 1
                assert await state.coordination_backend.read_turn_by_session(
                    observed["scope"], observed["session_key"],
                ) is None
            finally:
                release.set()
                if not starting.done():
                    starting.cancel()
                await asyncio.gather(starting, return_exceptions=True)
    assert not cache.closed


async def test_failed_host_binding_does_not_admit_runtime_or_close_borrowed_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _BorrowedCache()
    configuration = _storage("sqlite", tmp_path, cache)
    failed = OSError("source binding unavailable")
    attempted = []

    def reject(*args, **kwargs):
        raise failed

    def forbidden_attach(*args, **kwargs):
        attempted.append("runtime")
        raise AssertionError("a failed binding must not admit Runtime")

    monkeypatch.setattr(StateTurnCoordination, "__init__", reject)
    monkeypatch.setattr(Runtime, "attach", forbidden_attach)
    with pytest.raises(OSError) as caught:
        async with Persistence.open("source-turns", storage=configuration) as state:
            source = state.execution.executions.state_store
            configuration.coordination.bind(state)
            Runtime.attach(persistence=state, features=RuntimeFeatures("not-admitted"))
    assert caught.value is failed
    assert attempted == []
    assert not state.ready
    with pytest.raises(AIError) as closed:
        await source.read(lambda tx: tx.scan_records())
    assert closed.value.code is ErrorCode.STORAGE_CLOSED
    assert configuration.coordination.authority is None
    assert not cache.closed
    with pytest.raises(AIError) as unbound:
        await configuration.coordination.read_turn_by_session("scope", "session")
    assert unbound.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
