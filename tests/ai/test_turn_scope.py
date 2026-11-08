#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local authority and resource lifetime of the reference turn scope."""

import asyncio
from dataclasses import replace

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.storage import InMemoryCacheBackend, RenewingTurnScope, TurnCoordination, TurnScope


def scope(backend, **kwargs) -> RenewingTurnScope:
    return RenewingTurnScope(
        backend, "scope", session_key="session", request_key="request", request_digest="digest",
        execution_id="execution", owner_id="owner", lease_seconds=.15, renew_seconds=.01, **kwargs,
    )


@pytest.mark.asyncio
async def test_owned_scope_releases_and_rejects_reentry() -> None:
    backend = InMemoryCacheBackend(max_bytes=1)
    lifetime = backend.enter("scope", session_key="session", request_key="request", request_digest="digest",
                             execution_id="execution", owner_id="owner")
    assert isinstance(lifetime, TurnScope)
    async with lifetime as owned:
        assert owned.lease is not None
        lifetime.require_active()
    assert await backend.read_turn("scope", "execution") is None
    with pytest.raises(RuntimeError):
        lifetime.require_active()
    with pytest.raises(RuntimeError):
        await lifetime.__aenter__()


@pytest.mark.asyncio
async def test_observation_and_busy_scopes_do_not_release_owner() -> None:
    backend = InMemoryCacheBackend(max_bytes=1)
    async with scope(backend) as owned:
        async with scope(backend) as replay:
            assert replay.lease is None
        busy = backend.enter("scope", session_key="session", request_key="other", request_digest="digest",
                             execution_id="other", owner_id="other")
        async with busy as result:
            assert result is None
            with pytest.raises(RuntimeError):
                busy.require_active()
            with pytest.raises(RuntimeError):
                await busy.wait_signal()
        assert (await backend.read_turn("scope", owned.execution_id)).owner_id == owned.owner_id


@pytest.mark.asyncio
async def test_cancel_signal_is_once_and_renewal_continues_until_owner_lost() -> None:
    backend = InMemoryCacheBackend(max_bytes=1)
    lifetime = scope(backend)
    async with lifetime as owned:
        await backend.cancel_turn("scope", "execution")
        assert await asyncio.wait_for(lifetime.wait_signal(), 1) == "cancel_requested"
        await asyncio.sleep(.18)
        lifetime.require_active()
        waiter = asyncio.create_task(lifetime.wait_signal())
        await asyncio.sleep(.025)
        assert not waiter.done()
        await backend.release_turn(owned.lease)
        assert await asyncio.wait_for(waiter, 1) == "owner_lost"
        with pytest.raises(AIError) as error:
            lifetime.require_active()
        assert error.value.code == ErrorCode.STORAGE_CONFLICT


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["none", "owner", "execution", "lease", "cancel"])
async def test_invalid_or_failed_renewal_permanently_loses_owner(failure: str) -> None:
    class Backend(InMemoryCacheBackend):
        async def renew_turn(self, lease, *, lease_seconds):
            result = await super().renew_turn(lease, lease_seconds=lease_seconds)
            if failure == "none":
                return None
            if failure in {"owner", "execution"}:
                return replace(result, **{failure + "_id": "other"})
            if failure == "lease":
                return replace(result, lease=replace(result.lease, token="other"))
            raise asyncio.CancelledError
    lifetime = scope(Backend(max_bytes=1))
    async with lifetime:
        assert await asyncio.wait_for(lifetime.wait_signal(), 1) == "owner_lost"
        with pytest.raises(AIError):
            lifetime.require_active()


@pytest.mark.asyncio
async def test_late_renewal_cannot_restore_expired_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    started, finish, returned = asyncio.Event(), asyncio.Event(), asyncio.Event()
    class Backend(InMemoryCacheBackend):
        async def renew_turn(self, lease, *, lease_seconds):
            result = await super().renew_turn(lease, lease_seconds=lease_seconds)
            assert result is not None and result.lease == lease
            started.set()
            await finish.wait()
            returned.set()
            return result
    lifetime = RenewingTurnScope(
        Backend(max_bytes=1), "scope", session_key="session", request_key="request", request_digest="digest",
        execution_id="execution", owner_id="owner", lease_seconds=60, renew_seconds=.01,
    )
    async with lifetime:
        try:
            await asyncio.wait_for(started.wait(), 1)
            loop = asyncio.get_running_loop()
            expired = loop.time() + 60
            # Advance only the authority check, not the event loop's scheduler.
            with monkeypatch.context() as clock:
                clock.setattr(loop, "time", lambda: expired)
                with pytest.raises(AIError) as error:
                    lifetime.require_active()
                assert error.value.code is ErrorCode.STORAGE_CONFLICT
            assert await asyncio.wait_for(lifetime.wait_signal(), 1) == "owner_lost"
        finally:
            finish.set()
        await asyncio.wait_for(returned.wait(), 1)
        with pytest.raises(AIError) as error:
            lifetime.require_active()
        assert error.value.code is ErrorCode.STORAGE_CONFLICT


@pytest.mark.asyncio
async def test_slow_claim_has_no_extra_local_lease_time() -> None:
    class Backend(InMemoryCacheBackend):
        async def claim_turn(self, *args, **kwargs):
            result = await super().claim_turn(*args, **kwargs)
            await asyncio.sleep(.16)
            return result
    lifetime = scope(Backend(max_bytes=1))
    async with lifetime as owned:
        assert owned.lease is not None
        assert await lifetime.wait_signal() == "owner_lost"
        with pytest.raises(AIError):
            lifetime.require_active()


@pytest.mark.asyncio
async def test_uncertain_claim_propagates_without_becoming_contention() -> None:
    class Backend(InMemoryCacheBackend):
        async def claim_turn(self, *args, **kwargs):
            await super().claim_turn(*args, **kwargs)
            raise OSError("unknown mutation outcome")
    with pytest.raises(OSError, match="unknown mutation"):
        async with scope(Backend(max_bytes=1)):
            pytest.fail("must not execute")


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_replace_business_error() -> None:
    class Backend(InMemoryCacheBackend):
        async def release_turn(self, lease):
            raise OSError("release unavailable")
    with pytest.raises(ValueError, match="business"):
        async with scope(Backend(max_bytes=1)):
            raise ValueError("business")


@pytest.mark.asyncio
async def test_exit_settles_release_even_when_caller_is_cancelled_again() -> None:
    releasing, finish = asyncio.Event(), asyncio.Event()
    class Backend(InMemoryCacheBackend):
        async def release_turn(self, lease):
            releasing.set()
            await finish.wait()
            return await super().release_turn(lease)
    backend = Backend(max_bytes=1)
    async def run():
        async with scope(backend):
            pass
    task = asyncio.create_task(run())
    await releasing.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await backend.read_turn("scope", "execution") is None


def test_missing_enter_is_configuration_error_even_with_cache_capability() -> None:
    class Incomplete(InMemoryCacheBackend):
        enter = None
    with pytest.raises(TypeError, match="enter"):
        PersistenceStorage.in_memory(coordination=Incomplete(max_bytes=1))


@pytest.mark.asyncio
async def test_cancelled_entry_waits_for_claim_and_releases_known_owner() -> None:
    claimed, finish = asyncio.Event(), asyncio.Event()
    class Backend(InMemoryCacheBackend):
        async def claim_turn(self, *args, **kwargs):
            result = await super().claim_turn(*args, **kwargs)
            claimed.set()
            await finish.wait()
            return result
    backend = Backend(max_bytes=1)
    async def run():
        async with scope(backend):
            pytest.fail("cancelled entry must not execute")
    task = asyncio.create_task(run())
    await claimed.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    task.cancel()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await backend.read_turn("scope", "execution") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["memory", "sql"])
async def test_state_scope_owns_and_releases_exact_admission(tmp_path, kind: str) -> None:
    from linktools.ai.persistence.turns import StateTurnCoordination
    from .test_state_batch_delete_backends import _open_store
    async with _open_store(kind, tmp_path / kind) as store:
        backend = StateTurnCoordination(store, namespace="turn-scope", tenant_id="tenant")
        lifetime = backend.enter("scope", session_key="session", request_key="request", request_digest="digest",
                                 execution_id="execution", owner_id="owner")
        async with lifetime as owned:
            lifetime.require_active()
            assert owned.lease is not None
            async with backend.enter("scope", session_key="session", request_key="request", request_digest="digest",
                                     execution_id="ignored", owner_id="ignored") as replay:
                assert replay == replace(owned, lease=None)
        assert await backend.read_turn("scope", "execution") is None


@pytest.mark.asyncio
async def test_scope_exit_rejects_a_different_task_without_closing_owner() -> None:
    lifetime = scope(InMemoryCacheBackend(max_bytes=1))
    async with lifetime:
        task = asyncio.create_task(lifetime.__aexit__(None, None, None))
        with pytest.raises(RuntimeError, match="entering task"):
            await task
        lifetime.require_active()


@pytest.mark.asyncio
async def test_cancelled_claim_with_unknown_outcome_still_reports_storage_error() -> None:
    claimed, finish = asyncio.Event(), asyncio.Event()
    class Backend(InMemoryCacheBackend):
        async def claim_turn(self, *args, **kwargs):
            await super().claim_turn(*args, **kwargs)
            claimed.set()
            await finish.wait()
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
    async def run():
        async with scope(Backend(max_bytes=1)):
            pytest.fail("unknown claim must not execute")
    task = asyncio.create_task(run())
    await claimed.wait()
    task.cancel()
    finish.set()
    with pytest.raises(AIError) as error:
        await task
    assert error.value.code == ErrorCode.STORAGE_COMMIT_UNKNOWN


@pytest.mark.asyncio
async def test_renewal_latency_counts_against_the_new_local_deadline() -> None:
    responded = asyncio.Event()
    calls = 0
    class Backend(InMemoryCacheBackend):
        async def renew_turn(self, lease, *, lease_seconds):
            nonlocal calls
            calls += 1
            if calls == 1:
                result = await super().renew_turn(lease, lease_seconds=lease_seconds)
                await asyncio.sleep(.12)
                responded.set()
                return result
            await asyncio.Event().wait()
    lifetime = RenewingTurnScope(
        Backend(max_bytes=1), "scope", session_key="session", request_key="request", request_digest="digest",
        execution_id="execution", owner_id="owner", lease_seconds=.25, renew_seconds=.03,
    )
    async with lifetime:
        await responded.wait()
        lifetime.require_active()
        # The renewed deadline is .28 from entry, not .40 from its response.
        assert await asyncio.wait_for(lifetime.wait_signal(), .20) == "owner_lost"


@pytest.mark.asyncio
async def test_old_scope_exit_cannot_release_replacement_owner() -> None:
    backend = InMemoryCacheBackend(max_bytes=1)
    lifetime = scope(backend)
    async with lifetime as owned:
        await backend.release_turn(owned.lease)
        successor = await backend.claim_turn(
            "scope", session_key="session", request_key="request", request_digest="digest",
            execution_id="execution", owner_id="successor", lease_seconds=30,
        )
        assert await asyncio.wait_for(lifetime.wait_signal(), 1) == "owner_lost"
    assert await backend.read_turn("scope", "execution") == replace(successor, lease=None)
    await backend.release_turn(successor.lease)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [OSError("transport failed"), AIError(ErrorCode.STORAGE_UNAVAILABLE)])
async def test_renewal_failure_preserves_backend_error(failure: Exception) -> None:
    class Backend(InMemoryCacheBackend):
        async def renew_turn(self, lease, *, lease_seconds):
            raise failure
    lifetime = scope(Backend(max_bytes=1))
    async with lifetime:
        with pytest.raises(type(failure)) as signal:
            await asyncio.wait_for(lifetime.wait_signal(), 1)
        assert signal.value is failure
        with pytest.raises(type(failure)) as validity:
            lifetime.require_active()
        assert validity.value is failure
