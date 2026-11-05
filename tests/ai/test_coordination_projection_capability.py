#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live ownership does not depend on disposable pending projections."""

from dataclasses import replace

import pytest

from linktools.ai.core import ExecutionStatus, Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.coordination import PendingTurnCoordination
from linktools.ai.storage import InMemoryCacheBackend

from .test_history_projection_conformance import _record


class _OwnershipOnly:
    def __init__(self) -> None:
        self.backend = InMemoryCacheBackend(max_bytes=1024)
        self.enter = self.backend.enter
        self.claim_turn = self.backend.claim_turn
        self.read_turn = self.backend.read_turn
        self.renew_turn = self.backend.renew_turn
        self.release_turn = self.backend.release_turn
        self.cancel_turn = self.backend.cancel_turn
        self.read_turn_by_session = self.backend.read_turn_by_session
        self.read_turn_by_request = self.backend.read_turn_by_request


def test_ownership_without_lifetime_is_not_a_complete_coordination_backend() -> None:
    backend = _OwnershipOnly()
    del backend.enter
    assert PendingTurnCoordination.select(
        backend, namespace="namespace", tenant_id="tenant",
    ) is None


@pytest.mark.asyncio
async def test_ownership_only_backend_retains_admission_without_pending_views() -> None:
    backend = _OwnershipOnly()
    coordinator = PendingTurnCoordination.select(backend, namespace="namespace", tenant_id="tenant")
    assert coordinator is not None and not coordinator.supports_pending_views
    scope = coordinator.enter(
        session_key="session:session", request_key="request", request_digest="digest",
        execution_id="execution", owner_id="owner",
    )
    async with scope as owned:
        assert owned is not None and owned.lease is not None
        assert await coordinator.read_owner("execution", tenant_id="tenant") == replace(owned, lease=None)
        assert await coordinator.read_session_owner("session", tenant_id="tenant") == replace(owned, lease=None)
        assert await coordinator.read_execution("execution", tenant_id="tenant") is None
        assert await coordinator.read_session_execution("session", tenant_id="tenant") is None
        assert await coordinator.read_stream("execution", tenant_id="tenant") is None
        with pytest.raises(AIError) as unknown:
            await coordinator.read_request_execution("request", tenant_id="tenant")
        assert unknown.value.code is ErrorCode.EXECUTION_START_UNKNOWN
        assert await coordinator.read_request_execution("missing", tenant_id="tenant") is None
        assert await coordinator.request_promotion("execution", tenant_id="tenant") is None
        assert not await coordinator.promotion_requested(owned)
        assert not await coordinator.remote_promotion_pending("execution", "owner", tenant_id="tenant")
        with pytest.raises(AIError) as unsupported:
            await coordinator.publish(_record(ExecutionStatus.STARTED, 1), owned)
        assert unsupported.value.safe_details == {"reason": "pending_views_not_supported"}
        async with coordinator.enter(
            session_key="session:session", request_key="request", request_digest="digest",
            execution_id="duplicate", owner_id="duplicate-owner",
        ) as duplicate:
            assert duplicate == replace(owned, lease=None)
        assert await coordinator.cancel("execution", tenant_id="tenant")
        assert (await coordinator.read_owner("execution", tenant_id="tenant")).cancel_requested
        scope.require_active()
    assert await coordinator.read_owner("execution", tenant_id="tenant") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("with_cache", (False, True))
async def test_history_never_enters_or_mutates_live_turn_ownership(monkeypatch, with_cache: bool) -> None:
    backend = InMemoryCacheBackend(max_bytes=1024 * 1024) if with_cache else _OwnershipOnly()
    coordinator = PendingTurnCoordination.select(backend, namespace="namespace", tenant_id="tenant")
    assert coordinator is not None
    scope = coordinator.enter(
        session_key="session:session", request_key="request", request_digest="digest",
        execution_id="execution", owner_id="owner",
    )
    async with scope as owned:
        if with_cache:
            await coordinator.publish(_record(ExecutionStatus.STARTED, 1), owned)

        def forbidden(*args: object, **kwargs: object) -> None:
            pytest.fail("History must not enter, claim, renew, release or cancel ownership")

        with monkeypatch.context() as guard:
            for method in ("enter", "claim_turn", "renew_turn", "release_turn", "cancel_turn"):
                guard.setattr(backend, method, forbidden)
            storage = PersistenceStorage.in_memory(coordination=backend)
            async with Persistence.open(
                "namespace", storage=storage, tenant_id="tenant", read_only=True,
            ) as persistence, History.open(persistence=persistence) as history:
                if with_cache:
                    result = await history.inspect_execution(
                        "execution", principal=Principal("principal", "tenant", "service"), include_usage=False,
                    )
                    assert result.execution_id == "execution"
                else:
                    with pytest.raises(AIError) as missing:
                        await history.inspect_execution(
                            "execution", principal=Principal("principal", "tenant", "service"), include_usage=False,
                        )
                    assert missing.value.code is ErrorCode.AUTHORIZATION_DENIED
        scope.require_active()
