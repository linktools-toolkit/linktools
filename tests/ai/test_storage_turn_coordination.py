#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live turn admission, ownership, replay, and expiration contracts."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, replace

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import InMemoryCacheBackend, TurnCoordination, TurnCoordinationBackend
from linktools.ai.storage import _cache as cache_module


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [100.0]
    monkeypatch.setattr(cache_module, "monotonic", lambda: now[0])
    return now


async def _claim(
    cache: TurnCoordinationBackend,
    *,
    scope: str = "tenant:runtime",
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
    assert await cache.read_turn("tenant:runtime", owner.execution_id) == replace(owner, lease=None)
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
    assert await cache.read_turn("tenant:runtime", "execution") == replace(owner, lease=None)


@pytest.mark.asyncio
async def test_scopes_isolate_request_session_execution_and_cancellation() -> None:
    cache = InMemoryCacheBackend(max_bytes=1)
    first = await _claim(cache, scope='tenant:a\0runtime:"b"')
    second = await _claim(cache, scope='tenant:a\0runtime:"c"', request_digest="other")
    assert first is not None and second is not None
    assert first.lease is not None and second.lease is not None
    assert first.lease.key != second.lease.key
    assert await cache.cancel_turn('tenant:a\0runtime:"b"', "execution")
    assert await cache.read_turn('tenant:a\0runtime:"c"', "execution") == replace(second, lease=None)
    assert await cache.read_turn("unknown", "execution") is None
    assert not await cache.cancel_turn("unknown", "execution")


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
    assert await cache.read_turn("tenant:runtime", "execution") == replace(owner, lease=None)
    assert await _claim(cache, lease_seconds=1000.0) == replace(owner, lease=None)
    assert await cache.cancel_turn("tenant:runtime", "execution")
    assert await cache.cancel_turn("tenant:runtime", "execution")
    clock[0] = 110.0
    assert await cache.read_turn("tenant:runtime", "execution") is None
    assert not await cache.cancel_turn("tenant:runtime", "execution")
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
    assert await cache.cancel_turn("tenant:runtime", "execution")
    clock[0] = 105.0
    renewed = await cache.renew_turn(owner.lease, lease_seconds=10.0)
    assert renewed == replace(owner, cancel_requested=True)
    clock[0] = 110.0
    assert await cache.read_turn("tenant:runtime", "execution") == replace(renewed, lease=None)
    assert await _claim(cache, request_key="other", execution_id="other") is None
    clock[0] = 115.0
    assert await cache.read_turn("tenant:runtime", "execution") is None


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
    assert not await cache.cancel_turn("tenant:runtime", "execution")
    assert await cache.read_turn("tenant:runtime", "new-execution") == replace(new, lease=None)
    for forged in (
        replace(new.lease, token=old.lease.token),
        replace(new.lease, fence=old.lease.fence),
    ):
        assert await cache.renew_turn(forged, lease_seconds=100.0) is None
        assert not await cache.release_turn(forged)
    assert await cache.release_turn(new.lease)
    assert await cache.read_turn("tenant:runtime", "new-execution") is None


@pytest.mark.asyncio
async def test_cache_capacity_pressure_cannot_evict_live_turn_ownership() -> None:
    cache = InMemoryCacheBackend(max_bytes=1, max_size=1)
    owner = await _claim(cache)
    await cache.put("hot", b"x")
    assert await cache.append("writes", "id", "key", b"y", wait=False)
    assert owner is not None
    assert await cache.read_turn("tenant:runtime", "execution") == replace(owner, lease=None)
    assert await _claim(cache, request_key="other", execution_id="other") is None
    independent = InMemoryCacheBackend(max_bytes=0, max_size=0)
    assert await _claim(independent) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("seconds", (True, False, 0, -1, float("nan"), float("inf"), "10", None))
async def test_invalid_durations_do_not_claim_or_renew(seconds: float) -> None:
    cache = InMemoryCacheBackend(max_bytes=1)
    with pytest.raises(ValueError):
        await _claim(cache, lease_seconds=seconds)
    assert await cache.read_turn("tenant:runtime", "execution") is None
    owner = await _claim(cache)
    assert owner is not None and owner.lease is not None
    with pytest.raises(ValueError):
        await cache.renew_turn(owner.lease, lease_seconds=seconds)
    assert await cache.read_turn("tenant:runtime", "execution") == replace(owner, lease=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", (
    "scope", "session_key", "request_key", "request_digest", "execution_id", "owner_id",
))
async def test_empty_claim_identity_is_rejected(field: str) -> None:
    cache = InMemoryCacheBackend(max_bytes=1)
    with pytest.raises(ValueError):
        await _claim(cache, **{field: ""})
    assert await cache.read_turn("tenant:runtime", "execution") is None


@pytest.mark.asyncio
async def test_invalid_lookup_identity_is_rejected() -> None:
    cache = InMemoryCacheBackend(max_bytes=1)
    for scope, execution_id in (("", "execution"), ("scope", "")):
        with pytest.raises(ValueError):
            await cache.read_turn(scope, execution_id)
        with pytest.raises(ValueError):
            await cache.cancel_turn(scope, execution_id)
