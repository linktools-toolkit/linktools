#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pending and source-first writers share one live admission owner."""

from linktools.ai.persistence import Persistence
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
from linktools.ai.core import CommitPolicy, Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence.coordination import PendingTurnCoordination
from linktools.ai.storage import InMemoryCacheBackend

from .persistence.test_execution_start_service import (
    _binding,
    _Launcher,
    _request,
    _service,
)
from .test_remote_pending_execution_consumers import _RemotePendingTurns
from .test_session_admission_conformance import _session

_PRINCIPAL = Principal("owner", "tenant", "service")


class _Coordinator(PendingTurnCoordination):
    def __init__(self, cache):
        super().__init__(cache, cache, namespace="mixed-admission", tenant_id="tenant")
        self.claims = []

    async def claim(self, **kwargs):
        result = await super().claim(**kwargs)
        self.claims.append(result)
        return result

    async def read_owner(self, execution_id, *, tenant_id):
        if tenant_id != self._tenant_id:
            return None
        return await self._backend.read_turn(self._scope, execution_id)


@asynccontextmanager
async def _writers():
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace="mixed-admission", tenant_id="tenant")
    cache = InMemoryCacheBackend(max_bytes=1024 * 1024)
    services = []
    # Both writers support sessions; only the pending writer binds pending_start.
    for commit in (CommitPolicy.terminal(), CommitPolicy.sync()):
        coordinator = _Coordinator(cache)
        service = _service(state, backend=_Launcher(state.execution.executions), commit=commit)
        service._conversation = True
        service._runtime_bridge.pending_turns = _RemotePendingTurns(coordinator)
        services.append(service)
    try:
        yield state, *services
    finally:
        for service in services:
            coordinator = service._runtime_bridge.pending_turns.coordination
            for claim in coordinator.claims:
                if claim is not None and claim.lease is not None:
                    await coordinator.release(claim.lease)
        await state.close()


def _pending_callback(service, *, entered=None, release=None, claims=None):
    async def start(request, reservation, session, coordination):
        assert coordination is not None and coordination.lease is not None
        if claims is not None:
            claims.append(coordination)
        if entered is not None:
            entered.set()
        if release is not None:
            await release.wait()
        pending = service._runtime_bridge.pending_turns
        turn, created = pending.admit("tenant", reservation, session, now=reservation.execution.created_at)
        assert created
        await pending.coordination.publish(turn.execution, coordination)
        return turn.execution.execution_id

    service._runtime_bridge.pending_start = start


async def _start(service, *, prompt="question", key="request", session=False):
    request = _request(prompt, _PRINCIPAL, key, memory_scope=None)
    if session:
        return await service._start(
            _binding().binding_digest, request, session_id="session",
            session_agent_id="agent", scope="session.resume",
        )
    return await service.start(_binding().binding_digest, request)


@pytest.mark.asyncio
@pytest.mark.parametrize("session", (False, True))
async def test_source_first_writer_replays_pending_owner_without_new_worker(session) -> None:
    async with _writers() as (state, pending, legacy):
        if session:
            await state.conversation.sessions.create(replace(_session(), agent_id="agent"))
        claims = []
        _pending_callback(pending, claims=claims)
        first = await _start(pending, session=session)
        same = await _start(legacy, session=session)
        assert first.execution_id == same.execution_id
        assert len(claims) == 1
        assert legacy.runtime_backend().calls == 0
        assert await state.execution.executions.get(first.execution_id, tenant_id="tenant") is None
        coordinator = pending._runtime_bridge.pending_turns.coordination
        assert await coordinator.read_owner(first.execution_id, tenant_id="tenant") is not None
        with pytest.raises(AIError) as raised:
            await _start(legacy, prompt="different", session=session)
        assert raised.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        if session:
            with pytest.raises(AIError) as raised:
                await _start(legacy, key="next", session=True)
            assert raised.value.code is ErrorCode.SESSION_BUSY
        assert await coordinator.read_owner(first.execution_id, tenant_id="tenant") is not None


@pytest.mark.asyncio
async def test_pending_writer_joins_source_first_admission_before_source_is_visible(monkeypatch) -> None:
    async with _writers() as (state, pending, legacy):
        entered, release = asyncio.Event(), asyncio.Event()
        claims = []
        _pending_callback(pending, claims=claims)
        original = state.execution.executions.reserve_start

        async def pause(reservation):
            entered.set()
            await release.wait()
            return await original(reservation)

        monkeypatch.setattr(state.execution.executions, "reserve_start", pause)
        first = asyncio.create_task(_start(legacy))
        await asyncio.wait_for(entered.wait(), 2)
        duplicate = asyncio.create_task(_start(pending))
        try:
            await asyncio.sleep(0.01)
            assert claims == [] and not duplicate.done()
        finally:
            release.set()
        initial, replay = await asyncio.gather(first, duplicate)
        assert initial.execution_id == replay.execution_id
        assert claims == [] and legacy.runtime_backend().calls == 1
        assert await legacy._runtime_bridge.pending_turns.coordination.read_owner(initial.execution_id, tenant_id="tenant") is None


@pytest.mark.asyncio
async def test_ineligible_pending_writer_keeps_claim_through_legacy_launch_then_releases(monkeypatch) -> None:
    async with _writers() as (_state, pending, _legacy):
        coordinator = pending._runtime_bridge.pending_turns.coordination
        claims = []

        async def decline(request, reservation, session, coordination):
            claims.append(coordination)
            return None

        pending._runtime_bridge.pending_start = decline
        backend = pending.runtime_backend()
        launch = backend.launch

        async def assert_owned(request, execution):
            assert await coordinator.read_owner(execution.execution_id, tenant_id="tenant") is not None
            await launch(request, execution)

        monkeypatch.setattr(backend, "launch", assert_owned)
        result = await _start(pending)
        assert len(claims) == 1 and claims[0].lease is not None
        assert backend.calls == 1
        assert await coordinator.read_owner(result.execution_id, tenant_id="tenant") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ("pending", "legacy"))
@pytest.mark.parametrize("code", (ErrorCode.SESSION_BUSY, ErrorCode.EXECUTION_START_UNKNOWN))
async def test_admission_errors_release_only_the_acquired_lease(monkeypatch, path, code) -> None:
    async with _writers() as (state, pending, legacy):
        async def fail(*args):
            raise AIError(code)

        service = pending if path == "pending" else legacy
        if path == "pending":
            service._runtime_bridge.pending_start = fail
        else:
            monkeypatch.setattr(state.execution.executions, "reserve_start", fail)
        with pytest.raises(AIError) as raised:
            await _start(service)
        assert raised.value.code is code
        coordinator = service._runtime_bridge.pending_turns.coordination
        owned, = coordinator.claims
        assert owned.lease is not None
        assert await coordinator.read_owner(owned.execution_id, tenant_id="tenant") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ("pending", "legacy"))
async def test_caller_cancellation_does_not_cancel_owned_admission_or_release_transferred_claim(monkeypatch, path) -> None:
    async with _writers() as (state, pending, legacy):
        entered, release = asyncio.Event(), asyncio.Event()
        service = pending if path == "pending" else legacy
        other = legacy if path == "pending" else pending
        if path == "pending":
            _pending_callback(service, entered=entered, release=release)
        else:
            original = state.execution.executions.reserve_start

            async def pause(reservation):
                entered.set()
                await release.wait()
                return await original(reservation)

            monkeypatch.setattr(state.execution.executions, "reserve_start", pause)
        first = asyncio.create_task(_start(service))
        await asyncio.wait_for(entered.wait(), 2)
        first.cancel()
        await asyncio.sleep(0)
        assert not first.done()
        duplicate = asyncio.create_task(_start(other))
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        replay = await asyncio.wait_for(duplicate, 2)
        coordinator = service._runtime_bridge.pending_turns.coordination
        owned, = coordinator.claims
        assert replay.execution_id == owned.execution_id
        owner = await coordinator.read_owner(replay.execution_id, tenant_id="tenant")
        assert (owner is not None) is (path == "pending")
        assert other.runtime_backend().calls == 0
        assert service.runtime_backend().calls == (0 if path == "pending" else 1)


@pytest.mark.asyncio
async def test_duplicate_admission_reports_unknown_if_owner_fails_before_publication() -> None:
    async with _writers() as (_state, pending, legacy):
        entered, release = asyncio.Event(), asyncio.Event()

        async def fail(request, reservation, session, coordination):
            entered.set()
            await release.wait()
            raise AIError(ErrorCode.SESSION_BUSY)

        pending._runtime_bridge.pending_start = fail
        first = asyncio.create_task(_start(pending))
        await asyncio.wait_for(entered.wait(), 2)
        duplicate = asyncio.create_task(_start(legacy))
        coordinator = legacy._runtime_bridge.pending_turns.coordination
        while not coordinator.claims:
            await asyncio.sleep(0)
        assert coordinator.claims[0].lease is None
        release.set()
        with pytest.raises(AIError) as raised:
            await first
        assert raised.value.code is ErrorCode.SESSION_BUSY
        with pytest.raises(AIError) as raised:
            await asyncio.wait_for(duplicate, 2)
        assert raised.value.code is ErrorCode.EXECUTION_START_UNKNOWN
        assert legacy.runtime_backend().calls == 0


@pytest.mark.asyncio
async def test_coordination_failure_does_not_fall_through_to_source_admission(monkeypatch) -> None:
    async with _writers() as (_state, _pending, legacy):
        coordinator = legacy._runtime_bridge.pending_turns.coordination

        async def unavailable(**kwargs):
            raise OSError("claim outcome unavailable")

        monkeypatch.setattr(coordinator, "claim", unavailable)
        with pytest.raises(OSError, match="claim outcome unavailable"):
            await _start(legacy)
        assert legacy.runtime_backend().calls == 0


@pytest.mark.asyncio
async def test_release_failure_preserves_successful_source_admission(monkeypatch) -> None:
    async with _writers() as (state, _pending, legacy):
        coordinator = legacy._runtime_bridge.pending_turns.coordination

        async def unavailable(lease):
            raise OSError("release unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(coordinator, "release", unavailable)
            result = await _start(legacy)
        assert legacy.runtime_backend().calls == 1
        assert await state.execution.executions.get(result.execution_id, tenant_id="tenant") is not None
        assert await coordinator.read_owner(result.execution_id, tenant_id="tenant") is not None


@pytest.mark.asyncio
async def test_release_failure_preserves_original_admission_error(monkeypatch) -> None:
    async with _writers() as (state, _pending, legacy):
        coordinator = legacy._runtime_bridge.pending_turns.coordination
        original = AIError(ErrorCode.SESSION_BUSY)

        async def reject(reservation):
            raise original

        async def unavailable(lease):
            raise OSError("release unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(state.execution.executions, "reserve_start", reject)
            patch.setattr(coordinator, "release", unavailable)
            with pytest.raises(AIError) as raised:
                await _start(legacy)
        assert raised.value is original
        assert legacy.runtime_backend().calls == 0
        owned, = coordinator.claims
        assert await coordinator.read_owner(owned.execution_id, tenant_id="tenant") is not None
