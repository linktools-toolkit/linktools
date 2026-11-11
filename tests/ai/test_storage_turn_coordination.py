#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live turn admission, ownership, replay, and expiration contracts."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, replace

import pytest

from linktools.ai.storage import TurnScopeIdentity
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import (
    InMemoryCacheBackend, TurnAdmissionReader, TurnCoordination,
    TurnCoordinationBackend, TurnSessionReader,
)
from linktools.ai.storage import _cache as cache_module

_SCOPE = TurnScopeIdentity("runtime", "tenant", "tenant:runtime")


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [100.0]
    monkeypatch.setattr(cache_module, "monotonic", lambda: now[0])
    return now


async def _claim(
    cache: TurnCoordinationBackend,
    *,
    scope: TurnScopeIdentity = _SCOPE,
    session_key: str = "session",
    request_key: str = "request",
    request_digest: str = "digest",
    execution_id: str = "execution",
    owner_id: str = "owner",
    lease_seconds: float = 10.0,
) -> TurnCoordination | None:
    return await cache.claim_turn(
        scope, session_key=session_key, request_key=request_key,
        request_digest=request_digest, execution_id=execution_id,
        owner_id=owner_id, lease_seconds=lease_seconds,
    )


@pytest.mark.asyncio
async def test_duplicate_claims_share_execution_without_transferring_ownership() -> None:
    cache = InMemoryCacheBackend(max_bytes=1)
    assert isinstance(cache, TurnCoordinationBackend)
    results = await asyncio.gather(*(
        _claim(cache, execution_id=f"execution-{index}", owner_id=f"owner-{index}")
        for index in range(20)
    ))
    assert all(result is not None for result in results)
    owners = [result for result in results if result is not None and result.lease is not None]
    assert len(owners) == 1
    owner = owners[0]
    assert all(
        result is not None and result.execution_id == owner.execution_id
        and result.owner_id == owner.owner_id
        for result in results
    )
    assert await cache.read_turn(_SCOPE, owner.execution_id) == replace(owner, lease=None)
    with pytest.raises(FrozenInstanceError):
        owner.owner_id = "other"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_same_session_claims_exclude_each_other_without_blocking_other_sessions() -> None:
    cache = InMemoryCacheBackend(max_bytes=1)
    results = await asyncio.gather(*(
        _claim(cache, request_key=f"request-{index}", execution_id=f"execution-{index}")
        for index in range(20)
    ))
    assert sum(result is not None for result in results) == 1
    independent = await _claim(cache, session_key="other", request_key="other", execution_id="other")
    assert independent is not None and independent.lease is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ({"request_digest": "changed"}, {"session_key": "other"}))
async def test_request_reuse_rejects_changed_digest_or_session(changed: dict[str, str]) -> None:
    cache = InMemoryCacheBackend(max_bytes=1)
    owner = await _claim(cache)
    with pytest.raises(AIError) as raised:
        await _claim(cache, **changed)
    assert raised.value.code == ErrorCode.IDEMPOTENCY_CONFLICT
    assert owner is not None
    assert await cache.read_turn(_SCOPE, "execution") == replace(owner, lease=None)


@pytest.mark.asyncio
async def test_scopes_isolate_request_session_execution_and_cancellation() -> None:
    cache = InMemoryCacheBackend(max_bytes=1)
    first = await _claim(cache, scope=TurnScopeIdentity('runtime', "tenant", 'tenant:a\0runtime:"b"'))
    second = await _claim(cache, scope=TurnScopeIdentity('runtime', "tenant", 'tenant:a\0runtime:"c"'), request_digest="other")
    assert first is not None and second is not None
    assert first.lease is not None and second.lease is not None
    assert first.lease.key != second.lease.key
    assert await cache.cancel_turn(TurnScopeIdentity('runtime', "tenant", 'tenant:a\0runtime:"b"'), "execution")
    assert await cache.read_turn(TurnScopeIdentity('runtime', "tenant", 'tenant:a\0runtime:"c"'), "execution") == replace(second, lease=None)
    assert await cache.read_turn(TurnScopeIdentity('runtime', "tenant", "unknown"), "execution") is None
    assert not await cache.cancel_turn(TurnScopeIdentity('runtime', "tenant", "unknown"), "execution")


@pytest.mark.asyncio
async def test_live_execution_identity_cannot_be_reassigned_to_another_request() -> None:
    cache = InMemoryCacheBackend(max_bytes=1)
    await _claim(cache)
    with pytest.raises(AIError) as raised:
        await _claim(cache, session_key="other", request_key="other")
    assert raised.value.code == ErrorCode.STORAGE_CONFLICT
    assert await _claim(cache, session_key="other", request_key="other", execution_id="new") is not None


@pytest.mark.asyncio
async def test_reads_replays_and_cancellation_never_extend_expiry(clock: list[float]) -> None:
    cache = InMemoryCacheBackend(max_bytes=1)
    owner = await _claim(cache)
    assert owner is not None and owner.lease is not None
    clock[0] = 109.0
    assert await cache.read_turn(_SCOPE, "execution") == replace(owner, lease=None)
    assert await _claim(cache, lease_seconds=1000.0) == replace(owner, lease=None)
    assert await cache.cancel_turn(_SCOPE, "execution")
    assert await cache.cancel_turn(_SCOPE, "execution")
    clock[0] = 110.0
    assert await cache.read_turn(_SCOPE, "execution") is None
    assert not await cache.cancel_turn(_SCOPE, "execution")
    assert await cache.renew_turn(owner.lease, lease_seconds=10.0) is None
    assert not await cache.release_turn(owner.lease)
    replacement = await _claim(cache, execution_id="replacement", owner_id="replacement")
    assert replacement is not None and replacement.lease is not None
    assert replacement.lease.fence > owner.lease.fence


@pytest.mark.asyncio
async def test_renewal_extends_only_its_live_generation_and_returns_cancellation(clock: list[float]) -> None:
    cache = InMemoryCacheBackend(max_bytes=1)
    owner = await _claim(cache)
    assert owner is not None and owner.lease is not None
    assert await cache.cancel_turn(_SCOPE, "execution")
    clock[0] = 105.0
    renewed = await cache.renew_turn(owner.lease, lease_seconds=10.0)
    assert renewed == replace(owner, cancel_requested=True)
    clock[0] = 110.0
    assert await cache.read_turn(_SCOPE, "execution") == replace(renewed, lease=None)
    assert await _claim(cache, request_key="other", execution_id="other") is None
    clock[0] = 115.0
    assert await cache.read_turn(_SCOPE, "execution") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("expire", (False, True))
async def test_stale_owner_and_old_execution_cannot_affect_new_generation(
    clock: list[float], expire: bool,
) -> None:
    cache = InMemoryCacheBackend(max_bytes=1)
    old = await _claim(cache)
    assert old is not None and old.lease is not None
    if expire:
        clock[0] += 10.0
    else:
        assert await cache.release_turn(old.lease)
        assert not await cache.release_turn(old.lease)
    new = await _claim(cache, execution_id="new-execution", owner_id="new-owner")
    assert new is not None and new.lease is not None
    assert old.lease.key == new.lease.key
    assert old.lease.fence < new.lease.fence
    assert old.lease.token != new.lease.token
    assert await cache.renew_turn(old.lease, lease_seconds=100.0) is None
    assert not await cache.release_turn(old.lease)
    assert not await cache.cancel_turn(_SCOPE, "execution")
    assert await cache.read_turn(_SCOPE, "new-execution") == replace(new, lease=None)
    assert await cache.read_turn_by_session(_SCOPE, "session") == replace(new, lease=None)
    for forged in (
        replace(new.lease, token=old.lease.token),
        replace(new.lease, fence=old.lease.fence),
    ):
        assert await cache.renew_turn(forged, lease_seconds=100.0) is None
        assert not await cache.release_turn(forged)
    assert await cache.release_turn(new.lease)
    assert await cache.read_turn(_SCOPE, "new-execution") is None
    assert await cache.read_turn_by_session(_SCOPE, "session") is None


@pytest.mark.asyncio
async def test_cache_capacity_pressure_cannot_evict_live_turn_ownership() -> None:
    cache = InMemoryCacheBackend(max_bytes=1, max_size=1)
    owner = await _claim(cache)
    await cache.put("hot", b"x")
    assert await cache.append("writes", "id", "key", b"y", wait=False)
    assert owner is not None
    assert await cache.read_turn(_SCOPE, "execution") == replace(owner, lease=None)
    assert await cache.read_turn_by_session(_SCOPE, "session") == replace(owner, lease=None)
    assert await _claim(cache, request_key="other", execution_id="other") is None
    independent = InMemoryCacheBackend(max_bytes=0, max_size=0)
    assert await _claim(independent) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("seconds", (True, False, 0, -1, float("nan"), float("inf"), "10", None))
async def test_invalid_durations_do_not_claim_or_renew(seconds: float) -> None:
    cache = InMemoryCacheBackend(max_bytes=1)
    with pytest.raises(ValueError):
        await _claim(cache, lease_seconds=seconds)
    assert await cache.read_turn(_SCOPE, "execution") is None
    owner = await _claim(cache)
    assert owner is not None and owner.lease is not None
    with pytest.raises(ValueError):
        await cache.renew_turn(owner.lease, lease_seconds=seconds)
    assert await cache.read_turn(_SCOPE, "execution") == replace(owner, lease=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", (
    "scope", "session_key", "request_key", "request_digest", "execution_id", "owner_id",
))
async def test_empty_claim_identity_is_rejected(field: str) -> None:
    cache = InMemoryCacheBackend(max_bytes=1)
    with pytest.raises(TypeError if field == "scope" else ValueError):
        await _claim(cache, **{field: ""})
    assert await cache.read_turn(_SCOPE, "execution") is None


@pytest.mark.asyncio
async def test_invalid_lookup_identity_is_rejected() -> None:
    cache = InMemoryCacheBackend(max_bytes=1)
    for scope, execution_id in ((None, "execution"), (TurnScopeIdentity("runtime", "tenant", "scope"), "")):
        with pytest.raises((TypeError, ValueError)):
            await cache.read_turn(scope, execution_id)
        with pytest.raises((TypeError, ValueError)):
            await cache.cancel_turn(scope, execution_id)


@pytest.mark.asyncio
async def test_request_lookup_is_scoped_token_free_and_does_not_extend_expiry(clock: list[float]) -> None:
    cache = InMemoryCacheBackend(max_bytes=1)
    owner = await _claim(cache)
    assert owner is not None and owner.lease is not None
    clock[0] += 9
    assert await cache.read_turn_by_request(_SCOPE, "request") == replace(owner, lease=None)
    assert await cache.read_turn_by_request(TurnScopeIdentity('runtime', "tenant", "other"), "request") is None
    assert await cache.read_turn_by_request(_SCOPE, "missing") is None
    clock[0] += 2
    assert await cache.read_turn_by_request(_SCOPE, "request") is None
    assert await cache.read_turn(_SCOPE, "execution") is None


@pytest.mark.asyncio
async def test_released_request_lookup_does_not_retain_old_owner() -> None:
    cache = InMemoryCacheBackend(max_bytes=1)
    owner = await _claim(cache)
    assert owner is not None and owner.lease is not None
    assert await cache.release_turn(owner.lease)
    assert await cache.read_turn_by_request(_SCOPE, "request") is None
    replacement = await _claim(cache, execution_id="next", owner_id="next-owner")
    assert replacement is not None
    assert await cache.read_turn_by_request(_SCOPE, "request") == replace(replacement, lease=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope,request_key", (("", "request"), ("scope", "")))
async def test_request_lookup_rejects_invalid_identity(scope: str, request_key: str) -> None:
    cache = InMemoryCacheBackend(max_bytes=1)
    if scope:
        scope = TurnScopeIdentity("runtime", "tenant", scope)
    with pytest.raises((TypeError, ValueError)):
        await cache.read_turn_by_request(scope, request_key)


@pytest.mark.asyncio
async def test_session_lookup_is_scoped_token_free_and_never_renews(clock: list[float]) -> None:
    cache = InMemoryCacheBackend(max_bytes=1)
    assert isinstance(cache, TurnSessionReader)
    owner = await _claim(cache)
    assert owner is not None and owner.lease is not None
    clock[0] = 109.0
    assert await cache.read_turn_by_session(_SCOPE, "session") == replace(owner, lease=None)
    assert await cache.read_turn_by_session(TurnScopeIdentity('runtime', "tenant", "other"), "session") is None
    assert await cache.read_turn_by_session(_SCOPE, "other") is None
    assert await cache.cancel_turn(_SCOPE, "execution")
    assert await cache.read_turn_by_session(_SCOPE, "session") == replace(
        owner, lease=None, cancel_requested=True,
    )
    clock[0] = 110.0
    assert await cache.read_turn_by_session(_SCOPE, "session") is None
    assert await cache.read_turn(_SCOPE, "execution") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("scope,session_key", (("", "session"), ("scope", ""), (None, "session"), ("scope", 1)))
async def test_session_lookup_rejects_invalid_identity(scope: str, session_key: str) -> None:
    cache = InMemoryCacheBackend(max_bytes=1)
    if scope:
        scope = TurnScopeIdentity("runtime", "tenant", scope)
    with pytest.raises((TypeError, ValueError)):
        await cache.read_turn_by_session(scope, session_key)


@pytest.mark.asyncio
async def test_session_lookup_is_independent_of_coordination_and_request_readers() -> None:
    cache = InMemoryCacheBackend(max_bytes=1)

    class CoordinationBackend:
        enter = cache.enter
        claim_turn = cache.claim_turn
        read_turn = cache.read_turn
        renew_turn = cache.renew_turn
        release_turn = cache.release_turn
        cancel_turn = cache.cancel_turn
        read_turn_by_request = cache.read_turn_by_request

    class RequestReader:
        read_turn_by_request = cache.read_turn_by_request

    reader = RequestReader()
    assert isinstance(reader, TurnAdmissionReader)
    assert not isinstance(reader, TurnSessionReader)
    assert not isinstance(reader, TurnCoordinationBackend)
    backend = CoordinationBackend()
    assert isinstance(backend, TurnCoordinationBackend)
    assert isinstance(backend, TurnAdmissionReader)
    assert not isinstance(backend, TurnSessionReader)
    owner = await _claim(backend)
    assert owner is not None
    assert await reader.read_turn_by_request(_SCOPE, "request") == replace(owner, lease=None)
    assert await backend.read_turn_by_request(_SCOPE, "request") == replace(owner, lease=None)
    assert await backend.read_turn(_SCOPE, "execution") == replace(owner, lease=None)


@pytest.mark.asyncio
async def test_session_lookup_does_not_turn_backend_failure_into_absence(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = InMemoryCacheBackend(max_bytes=1)
    owner = await _claim(cache)
    assert owner is not None
    failure = AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    def unavailable_clock() -> float:
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(cache_module, "monotonic", unavailable_clock)
        with pytest.raises(AIError) as raised:
            await cache.read_turn_by_session(_SCOPE, "session")
        assert raised.value is failure
    assert await cache.read_turn_by_session(_SCOPE, "session") == replace(owner, lease=None)


def test_scope_identity_is_exact_bounded_frozen_and_private() -> None:
    scope = TurnScopeIdentity('"' * 128, '"' * 128, "é" * 64)
    assert scope.as_tuple() == ('"' * 128, '"' * 128, "é" * 64)
    assert scope != TurnScopeIdentity(scope.namespace, scope.tenant_id, "e\u0301" * 42)
    assert all(component not in repr(scope) for component in scope.as_tuple())
    with pytest.raises(FrozenInstanceError):
        scope.scope = "changed"
    for invalid in ("", "é" * 64 + "a", "\ud800", None):
        with pytest.raises((TypeError, ValueError)):
            TurnScopeIdentity("runtime", "tenant", invalid)
    for namespace, tenant in (("n" * 129, "tenant"), ("runtime", "t" * 129)):
        with pytest.raises(AIError):
            TurnScopeIdentity(namespace, tenant, "scope")


@pytest.mark.asyncio
async def test_shared_cache_turns_isolate_each_scope_component() -> None:
    cache = InMemoryCacheBackend(max_bytes=1)
    scopes = (
        TurnScopeIdentity("runtime", "tenant", "purpose"),
        TurnScopeIdentity("other-runtime", "tenant", "purpose"),
        TurnScopeIdentity("runtime", "other-tenant", "purpose"),
        TurnScopeIdentity("runtime", "tenant", "other-purpose"),
        TurnScopeIdentity("runtime", "tenant", "é"),
        TurnScopeIdentity("runtime", "tenant", "e\u0301"),
    )
    owners = [await _claim(cache, scope=scope) for scope in scopes]
    assert len({owner.lease.key for owner in owners}) == len(scopes)
    for scope, owner in zip(scopes, owners):
        assert await cache.read_turn_by_request(scope, "request") == replace(owner, lease=None)
        assert await cache.read_turn_by_session(scope, "session") == replace(owner, lease=None)
    assert await cache.cancel_turn(scopes[0], "execution")
    assert not (await cache.read_turn(scopes[1], "execution")).cancel_requested
    assert await cache.release_turn(owners[0].lease)
    assert await cache.read_turn(scopes[0], "execution") is None
    assert await cache.read_turn(scopes[1], "execution") == replace(owners[1], lease=None)


@pytest.mark.asyncio
async def test_turn_scope_requires_explicit_identity() -> None:
    cache = InMemoryCacheBackend(max_bytes=1)
    with pytest.raises(TypeError):
        await _claim(cache, scope='["runtime-pending-turn","runtime","tenant"]')
