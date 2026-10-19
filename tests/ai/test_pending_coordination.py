#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pending views require current ownership, identity, and ordinary authorization."""

import asyncio
import json
from dataclasses import replace

import pytest

from linktools.ai.core import ExecutionStatus, canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._pending_coordination import _PendingTurnCoordination
from linktools.ai.runtime.state._codec import encode_domain
from linktools.ai.storage import InMemoryCacheBackend, TurnCoordination
from linktools.ai.storage import _cache as cache_module

from .test_history_projection_conformance import _record


class _ObservedCache(InMemoryCacheBackend):
    def __init__(self) -> None:
        super().__init__(max_bytes=1024 * 1024)
        self.writes: dict[str, bytes] = {}
        self.paused_key: str | None = None
        self.read_started = asyncio.Event()
        self.read_continue = asyncio.Event()
        self.pause_writes = False
        self.write_started = asyncio.Event()
        self.write_continue = asyncio.Event()

    async def put(self, key: str, content: bytes) -> None:
        if self.pause_writes:
            self.write_started.set()
            await self.write_continue.wait()
        self.writes[key] = content
        await super().put(key, content)

    async def get(self, key: str) -> bytes | None:
        content = await super().get(key)
        if key == self.paused_key:
            self.read_started.set()
            await self.read_continue.wait()
        return content

    def record_key(self) -> str:
        return next(key for key, value in self.writes.items() if "record" in json.loads(value))

    def session_key(self) -> str:
        return next(key for key, value in self.writes.items() if "session_id" in json.loads(value))


def _helper(cache: object, *, namespace: str = "namespace", tenant_id: str = "tenant") -> _PendingTurnCoordination:
    return _PendingTurnCoordination(cache, cache, namespace=namespace, tenant_id=tenant_id)


async def _claim(helper: _PendingTurnCoordination, *, execution_id: str = "execution", owner_id: str = "owner") -> TurnCoordination:
    claim = await helper.claim(
        session_key="session", request_key="request", request_digest="digest",
        execution_id=execution_id, owner_id=owner_id, lease_seconds=30,
    )
    assert claim is not None
    return claim


class _ContentOnly:
    async def get(self, key: str) -> bytes | None:
        return None

    async def put(self, key: str, content: bytes) -> None:
        pass

    async def delete(self, key: str) -> None:
        pass


def test_incomplete_borrowed_cache_never_gets_implicit_local_authority() -> None:
    for candidate in (None, object(), _ContentOnly()):
        assert _PendingTurnCoordination.select(candidate, namespace="runtime", tenant_id="tenant") is None
    cache = InMemoryCacheBackend(max_bytes=1024)
    assert _PendingTurnCoordination.select(cache, namespace="runtime", tenant_id="tenant") is not None


@pytest.mark.asyncio
async def test_shared_pending_view_replays_without_leaking_owner_token_or_persisting_cancellation() -> None:
    cache = _ObservedCache()
    writer, reader = _helper(cache), _helper(cache)
    record = replace(_record(ExecutionStatus.STARTED, 1), session_id="session")
    owned = await _claim(writer)
    duplicate = await _claim(reader, execution_id="ignored", owner_id="ignored")
    assert owned.lease is not None and duplicate.lease is None
    assert duplicate.execution_id == owned.execution_id
    with pytest.raises(AIError) as raised:
        await reader.publish(record, duplicate)
    assert raised.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
    await writer.publish(record, owned)
    assert await reader.read_execution(record.execution_id, tenant_id="tenant") == record
    assert await reader.read_session_execution("session", tenant_id="tenant") == record
    assert await reader.cancel(record.execution_id, tenant_id="tenant")
    renewed = await writer.renew(owned.lease, lease_seconds=30)
    assert renewed is not None and renewed.lease == owned.lease and renewed.cancel_requested
    assert await reader.read_execution(record.execution_id, tenant_id="tenant") == record
    assert await writer.release(owned.lease)
    assert await reader.read_execution(record.execution_id, tenant_id="tenant") is None
    assert await reader.read_session_execution("session", tenant_id="tenant") is None
    assert await writer.renew(owned.lease, lease_seconds=30) is None
    assert not await writer.release(owned.lease)


@pytest.mark.asyncio
async def test_scope_isolates_namespace_and_tenant_even_for_delimiter_like_names() -> None:
    cache = _ObservedCache()
    left = _helper(cache, namespace="a:b", tenant_id="c")
    right = _helper(cache, namespace="a", tenant_id="b:c")
    left_claim, right_claim = await _claim(left), await _claim(right)
    record = replace(_record(ExecutionStatus.STARTED, 1), session_id="session")
    other = replace(record, principal_id="other-principal")
    await left.publish(record, left_claim)
    await right.publish(other, right_claim)
    assert await left.read_execution("execution", tenant_id="c") == record
    assert await right.read_execution("execution", tenant_id="b:c") == other
    assert await left.read_execution("execution", tenant_id="b:c") is None
    assert await left.read_session_execution("session", tenant_id="b:c") is None
    assert not await left.cancel("execution", tenant_id="b:c")
    assert right_claim.lease is not None
    renewed = await right.renew(right_claim.lease, lease_seconds=30)
    assert renewed is not None and not renewed.cancel_requested


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ("scope", "execution_id", "owner_id", "principal_id", "principal_kind", "record"))
async def test_cached_view_identity_must_match_current_coordination_and_record(field: str) -> None:
    cache = _ObservedCache()
    helper = _helper(cache)
    owned = await _claim(helper)
    record = _record(ExecutionStatus.STARTED, 1)
    await helper.publish(record, owned)
    key = cache.record_key()
    envelope = json.loads(cache.writes[key])
    envelope[field] = encode_domain(replace(record, execution_id="different")) if field == "record" else "different"
    await cache.put(key, canonical_json_bytes(envelope))
    assert await helper.read_execution(record.execution_id, tenant_id="tenant") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("content", (b"not-json", b"[]", b"\xff"))
async def test_corrupt_observation_is_never_an_execution_source(content: bytes) -> None:
    cache = _ObservedCache()
    helper = _helper(cache)
    owned = await _claim(helper)
    await helper.publish(_record(ExecutionStatus.STARTED, 1), owned)
    await cache.put(cache.record_key(), content)
    assert await helper.read_execution("execution", tenant_id="tenant") is None


@pytest.mark.asyncio
async def test_terminal_bytes_cannot_replace_the_durable_terminal_winner() -> None:
    cache = _ObservedCache()
    helper = _helper(cache)
    owned = await _claim(helper)
    record = _record(ExecutionStatus.STARTED, 1)
    await helper.publish(record, owned)
    terminal = replace(record, status=ExecutionStatus.SUCCEEDED)
    with pytest.raises(AIError) as raised:
        await helper.publish(terminal, owned)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    key = cache.record_key()
    envelope = json.loads(cache.writes[key])
    envelope["record"] = encode_domain(terminal)
    await cache.put(key, canonical_json_bytes(envelope))
    assert await helper.read_execution("execution", tenant_id="tenant") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("observation", ("execution", "stream"))
async def test_owner_release_during_suspended_byte_read_invalidates_captured_view(observation: str) -> None:
    cache = _ObservedCache()
    helper = _helper(cache)
    owned = await _claim(helper)
    record = replace(_record(ExecutionStatus.STARTED, 1), session_id="session")
    await helper.publish(record, owned)
    cache.paused_key = cache.record_key()
    read = helper.read_execution if observation == "execution" else helper.read_stream
    pending = asyncio.create_task(read("execution", tenant_id="tenant"))
    await cache.read_started.wait()
    assert owned.lease is not None and await helper.release(owned.lease)
    replacement = await _claim(helper, execution_id="new-execution", owner_id="new-owner")
    await helper.publish(replace(record, execution_id="new-execution"), replacement)
    cache.read_continue.set()
    assert await pending is None
    await helper.publish(record, owned)
    assert await helper.read_session_execution("session", tenant_id="tenant") is None
    await helper.publish(replace(record, execution_id="new-execution"), replacement)
    latest = await helper.read_session_execution("session", tenant_id="tenant")
    assert latest is not None and latest.execution_id == "new-execution"


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_run_seq", (0, 1, 2))
async def test_pending_stream_binding_requires_the_single_activated_run(agent_run_seq: int) -> None:
    cache = _ObservedCache()
    writer, reader = _helper(cache), _helper(cache)
    owned = await _claim(writer)
    record = _record(ExecutionStatus.STARTED, agent_run_seq)
    await writer.publish(record, owned)
    stream = await reader.read_stream("execution", tenant_id="tenant")
    assert stream == ((record, 2, owned.owner_id) if agent_run_seq == 1 else None)
    assert await reader.read_execution("execution", tenant_id="tenant") == record
    assert await reader.read_stream("execution", tenant_id="other-tenant") is None


@pytest.mark.asyncio
async def test_lost_observation_never_releases_or_reacquires_the_live_turn() -> None:
    cache = InMemoryCacheBackend(max_bytes=0)
    helper = _helper(cache)
    owned = await _claim(helper)
    await helper.publish(_record(ExecutionStatus.STARTED, 1), owned)
    assert await helper.read_execution("execution", tenant_id="tenant") is None
    duplicate = await _claim(helper, execution_id="not-admitted", owner_id="not-owner")
    assert duplicate.execution_id == "execution" and duplicate.lease is None


@pytest.mark.asyncio
async def test_failed_authority_recheck_after_byte_read_is_not_a_cache_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _ObservedCache()
    helper = _helper(cache)
    owned = await _claim(helper)
    await helper.publish(_record(ExecutionStatus.STARTED, 1), owned)
    cache.paused_key = cache.record_key()
    pending = asyncio.create_task(helper.read_execution("execution", tenant_id="tenant"))
    await cache.read_started.wait()

    async def fail(scope: str, execution_id: str) -> TurnCoordination | None:
        raise OSError("authority unavailable")

    monkeypatch.setattr(cache, "read_turn", fail)
    cache.read_continue.set()
    with pytest.raises(OSError, match="authority unavailable"):
        await pending


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ("scope", "session_id", "execution_id", "owner_id"))
async def test_session_pointer_never_grants_execution_identity(field: str) -> None:
    cache = _ObservedCache()
    helper = _helper(cache)
    owned = await _claim(helper)
    record = replace(_record(ExecutionStatus.STARTED, 1), session_id="session")
    await helper.publish(record, owned)
    key = cache.session_key()
    pointer = json.loads(cache.writes[key])
    pointer[field] = "different"
    await cache.put(key, canonical_json_bytes(pointer))
    assert await helper.read_session_execution("session", tenant_id="tenant") is None
    assert await helper.read_execution("execution", tenant_id="tenant") == record


@pytest.mark.asyncio
async def test_wrong_session_in_valid_record_does_not_satisfy_session_pointer() -> None:
    cache = _ObservedCache()
    helper = _helper(cache)
    owned = await _claim(helper)
    record = replace(_record(ExecutionStatus.STARTED, 1), session_id="session")
    await helper.publish(record, owned)
    key = cache.record_key()
    envelope = json.loads(cache.writes[key])
    envelope["record"] = encode_domain(replace(record, session_id="different-session"))
    await cache.put(key, canonical_json_bytes(envelope))
    assert await helper.read_session_execution("session", tenant_id="tenant") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("claim", "read", "renew", "release", "cancel", "get", "put"))
async def test_backend_failure_never_becomes_absence_or_permission_to_execute(
    monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    cache = _ObservedCache()
    helper = _helper(cache)
    owned = await _claim(helper)
    record = _record(ExecutionStatus.STARTED, 1)
    await helper.publish(record, owned)

    async def fail(*args: object, **kwargs: object) -> None:
        raise OSError("backend unavailable")

    monkeypatch.setattr(cache, operation if operation in ("get", "put") else f"{operation}_turn", fail)
    with pytest.raises(OSError, match="backend unavailable"):
        if operation == "claim":
            await _claim(helper)
        elif operation in ("read", "get"):
            await helper.read_execution("execution", tenant_id="tenant")
        elif operation == "put":
            await helper.publish(record, owned)
        elif operation == "renew":
            assert owned.lease is not None
            await helper.renew(owned.lease, lease_seconds=30)
        elif operation == "release":
            assert owned.lease is not None
            await helper.release(owned.lease)
        else:
            await helper.cancel("execution", tenant_id="tenant")


@pytest.mark.asyncio
async def test_promotion_hint_is_scope_bound_and_does_not_extend_owner_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    monkeypatch.setattr(cache_module, "monotonic", lambda: now[0])
    cache = _ObservedCache()
    owner, remote = _helper(cache), _helper(cache)
    owned = await _claim(owner)
    assert await remote.read_owner("execution", tenant_id="tenant") == replace(owned, lease=None)
    assert await remote.read_execution("execution", tenant_id="tenant") is None
    assert await remote.read_owner("execution", tenant_id="other") is None
    other_namespace = _helper(cache, namespace="other")
    other_tenant = _helper(cache, tenant_id="other")
    namespace_owner, tenant_owner = await _claim(other_namespace), await _claim(other_tenant)
    assert not await owner.promotion_requested(owned)
    now[0] = 129.0
    assert await remote.request_promotion("execution", tenant_id="tenant") == owned.owner_id
    assert await owner.promotion_requested(owned)
    assert await remote.remote_promotion_pending("execution", owned.owner_id, tenant_id="tenant")
    assert not await other_namespace.promotion_requested(namespace_owner)
    assert not await other_tenant.promotion_requested(tenant_owner)
    assert await remote.request_promotion("execution", tenant_id="other") is None
    assert not await remote.remote_promotion_pending("execution", owned.owner_id, tenant_id="other")
    assert not await remote.remote_promotion_pending("execution", "other-owner", tenant_id="tenant")
    now[0] = 130.0
    assert await remote.read_owner("execution", tenant_id="tenant") is None
    assert await remote.request_promotion("execution", tenant_id="tenant") is None
    assert not await remote.remote_promotion_pending("execution", owned.owner_id, tenant_id="tenant")
    assert owned.lease is not None
    assert await owner.renew(owned.lease, lease_seconds=30) is None


@pytest.mark.asyncio
async def test_lost_or_corrupt_promotion_hint_is_not_a_successful_request() -> None:
    helper = _helper(InMemoryCacheBackend(max_bytes=0))
    owned = await _claim(helper)
    assert await helper.request_promotion("execution", tenant_id="tenant") is None
    assert not await helper.promotion_requested(owned)
    assert not await helper.remote_promotion_pending("execution", owned.owner_id, tenant_id="tenant")

    cache = _ObservedCache()
    helper = _helper(cache)
    owned = await _claim(helper)
    assert await helper.request_promotion("execution", tenant_id="tenant") == owned.owner_id
    key = next(iter(cache.writes))
    await cache.put(key, b"unrecognized")
    assert not await helper.promotion_requested(owned)
    assert not await helper.remote_promotion_pending("execution", owned.owner_id, tenant_id="tenant")


@pytest.mark.asyncio
async def test_late_promotion_hint_is_never_transferred_to_replacement_owner() -> None:
    cache = _ObservedCache()
    helper = _helper(cache)
    owned = await _claim(helper)
    cache.pause_writes = True
    pending = asyncio.create_task(helper.request_promotion("execution", tenant_id="tenant"))
    await cache.write_started.wait()
    assert owned.lease is not None and await helper.release(owned.lease)
    replacement = await _claim(helper, execution_id="replacement", owner_id="replacement-owner")
    cache.write_continue.set()
    assert await pending is None
    # The old hint can remain cached, but cannot prove current ownership or be
    # delivered to another generation just because the same session was reused.
    assert await helper.promotion_requested(owned)
    assert not await helper.remote_promotion_pending("execution", owned.owner_id, tenant_id="tenant")
    assert not await helper.promotion_requested(replacement)
    assert not await helper.remote_promotion_pending("replacement", replacement.owner_id, tenant_id="tenant")
    assert await helper.request_promotion("replacement", tenant_id="tenant") == replacement.owner_id
    assert await helper.promotion_requested(replacement)


@pytest.mark.asyncio
async def test_remote_promotion_observation_rechecks_owner_after_suspended_hint_read() -> None:
    cache = _ObservedCache()
    helper = _helper(cache)
    owned = await _claim(helper)
    assert await helper.request_promotion("execution", tenant_id="tenant") == owned.owner_id
    cache.paused_key = next(iter(cache.writes))
    pending = asyncio.create_task(helper.remote_promotion_pending("execution", owned.owner_id, tenant_id="tenant"))
    await cache.read_started.wait()
    assert owned.lease is not None and await helper.release(owned.lease)
    replacement = await _claim(helper, execution_id="replacement", owner_id="replacement-owner")
    cache.read_continue.set()
    assert not await pending
    assert not await helper.promotion_requested(replacement)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation, failure", (
    ("request", "read_turn"), ("request", "put"), ("request", "get"),
    ("owner_read", "get"), ("remote_read", "read_turn"), ("remote_read", "get"),
    ("read_owner", "read_turn"),
))
async def test_promotion_backend_errors_propagate(
    monkeypatch: pytest.MonkeyPatch, operation: str, failure: str,
) -> None:
    cache = _ObservedCache()
    helper = _helper(cache)
    owned = await _claim(helper)
    assert await helper.request_promotion("execution", tenant_id="tenant") == owned.owner_id

    async def fail(*args: object, **kwargs: object) -> None:
        raise OSError("promotion backend unavailable")

    monkeypatch.setattr(cache, failure, fail)
    with pytest.raises(OSError, match="promotion backend unavailable"):
        if operation == "request":
            await helper.request_promotion("execution", tenant_id="tenant")
        elif operation == "owner_read":
            await helper.promotion_requested(owned)
        elif operation == "read_owner":
            await helper.read_owner("execution", tenant_id="tenant")
        else:
            await helper.remote_promotion_pending("execution", owned.owner_id, tenant_id="tenant")


@pytest.mark.asyncio
@pytest.mark.parametrize("issue_request", (False, True))
async def test_promotion_authority_recheck_failure_cannot_confirm_the_hint(
    monkeypatch: pytest.MonkeyPatch, issue_request: bool,
) -> None:
    cache = _ObservedCache()
    helper = _helper(cache)
    owned = await _claim(helper)
    assert await helper.request_promotion("execution", tenant_id="tenant") == owned.owner_id
    cache.paused_key = next(iter(cache.writes))
    pending = asyncio.create_task(
        helper.request_promotion("execution", tenant_id="tenant") if issue_request
        else helper.remote_promotion_pending("execution", owned.owner_id, tenant_id="tenant")
    )
    await cache.read_started.wait()

    async def fail(scope: str, execution_id: str) -> TurnCoordination | None:
        raise OSError("authority unavailable")

    monkeypatch.setattr(cache, "read_turn", fail)
    cache.read_continue.set()
    with pytest.raises(OSError, match="authority unavailable"):
        await pending
