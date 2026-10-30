#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inactivity is an authorized, non-consuming observation of current admission."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from linktools.ai.core import (
    AuthorizationAction, ExecutionStatus, HmacCursorSigner, Principal, TenantAuthorizationPolicy,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence
from linktools.ai.persistence.coordination import PendingTurnCoordination
from linktools.ai.runtime._session import RuntimeSessionService
import linktools.ai.runtime._session as session_module
from linktools.ai.storage import InMemoryCacheBackend

from .test_incremental_projection_boundaries import _storage
from .test_session_timeline import _commit_empty_turn, _execution, _session

pytestmark = pytest.mark.asyncio
_OWNER = Principal("owner", "tenant")


@asynccontextmanager
async def _state(tmp_path: Path, backend: str = "memory"):
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="session-inactivity", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        yield state
    finally:
        await state.close()


def _service(state, *, observe=None, owner=None, waiter=None, authorization=None):
    return RuntimeSessionService(
        state.conversation, state.execution.executions,
        authorization or TenantAuthorizationPolicy(), object(),
        HmacCursorSigner("inactivity", b"session-inactivity-key"), history_reader=object(),
        observe_local_session=observe or (lambda session_id, *, tenant_id: (None, 0)),
        read_session_owner=owner, local_waiter=waiter,
    )


async def _admit(state, execution_id="execution", status=ExecutionStatus.STARTED):
    record = _execution(execution_id, status=status, prompt="question")
    await state.execution.executions.create_with_history_head(record)
    await state.conversation.sessions.admit_execution(
        "session", tenant_id="tenant", execution_id=execution_id, expected=None,
    )
    if status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
        await _commit_empty_turn(state, record)
    return record


class _BlockingWaiter:
    """Worker signal has no result-consumption or execution-cancellation API."""

    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.execution_ids = []

    def owns_execution(self, execution_id, *, tenant_id):
        return True

    async def wait_terminal(self, execution_id, *, tenant_id):
        self.execution_ids.append(execution_id)
        self.entered.set()
        await self.release.wait()


async def _cancel_observer(task):
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("terminal", (False, True))
async def test_inactive_observation_does_not_write_or_release_retained_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, terminal: bool,
) -> None:
    async with _state(tmp_path, backend) as state:
        if terminal:
            await _admit(state, status=ExecutionStatus.SUCCEEDED)
        sessions = state.conversation.sessions
        before = await sessions.get("session", tenant_id="tenant")

        async def forbidden_mutation(*args, **kwargs):
            pytest.fail("observing inactivity must not mutate authoritative state")

        with monkeypatch.context() as patch:
            patch.setattr(sessions.state_store, "mutate", forbidden_mutation)
            patch.setattr(state.execution.executions.state_store, "mutate", forbidden_mutation)
            view = await _service(state).wait_inactive(
                "session", principal=_OWNER, timeout_seconds=5, observation_timeout_seconds=5,
            )
        assert view.active_execution_id is None
        assert await sessions.get("session", tenant_id="tenant") == before
        if terminal:
            assert before.active_execution_id == "execution"
            assert (await state.execution.executions.get("execution", tenant_id="tenant")).result is not None


@pytest.mark.parametrize("status", (
    ExecutionStatus.RECOVERY_REQUIRED, ExecutionStatus.PENDING_START, ExecutionStatus.WAITING_DEFERRED,
))
async def test_nonterminal_admissions_remain_busy_and_cancellation_only_stops_observer(
    tmp_path: Path, status: ExecutionStatus,
) -> None:
    async with _state(tmp_path) as state:
        record = await _admit(state, status=status)
        waiter = _BlockingWaiter()
        task = asyncio.create_task(_service(state, waiter=waiter).wait_inactive(
            "session", principal=_OWNER, timeout_seconds=5, observation_timeout_seconds=5,
        ))
        try:
            await asyncio.wait_for(waiter.entered.wait(), 5)
            assert not task.done()
            assert waiter.execution_ids == [record.execution_id]
        finally:
            await _cancel_observer(task)
        assert await state.execution.executions.get(record.execution_id, tenant_id="tenant") == record
        assert (await state.conversation.sessions.get("session", tenant_id="tenant")).active_execution_id == record.execution_id


async def test_live_coordination_without_display_record_stays_busy(tmp_path: Path) -> None:
    async with _state(tmp_path) as state:
        cache = InMemoryCacheBackend(max_bytes=2**20)
        coordination = PendingTurnCoordination(cache, cache, namespace="inactivity", tenant_id="tenant")
        claim = await coordination.claim(
            session_key="session:session", request_key="request", request_digest="digest",
            execution_id="remote", owner_id="owner", lease_seconds=60,
        )
        assert claim is not None and claim.lease is not None
        assert await coordination.read_session_execution("session", tenant_id="tenant") is None

        async def owner(session_id, *, tenant_id):
            return await coordination.read_session_owner(session_id, tenant_id=tenant_id)

        waiter = _BlockingWaiter()
        task = asyncio.create_task(_service(state, owner=owner, waiter=waiter).wait_inactive(
            "session", principal=_OWNER, timeout_seconds=5,
        ))
        try:
            await asyncio.wait_for(waiter.entered.wait(), 5)
            assert not task.done()
            assert waiter.execution_ids == ["remote"]
            assert await state.execution.executions.get("remote", tenant_id="tenant") is None
        finally:
            await _cancel_observer(task)
        assert await coordination.read_session_owner("session", tenant_id="tenant") is not None


@pytest.mark.parametrize("source", ("local", "coordination"))
async def test_unknown_admission_is_not_reported_as_inactive(tmp_path: Path, source: str) -> None:
    async with _state(tmp_path) as state:
        failure = AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        def observe(session_id, *, tenant_id):
            raise failure

        async def owner(session_id, *, tenant_id):
            raise failure

        service = _service(state, observe=observe if source == "local" else None,
                           owner=owner if source == "coordination" else None)
        with pytest.raises(AIError) as error:
            await service.wait_inactive("session", principal=_OWNER, timeout_seconds=5)
        assert error.value is failure


async def test_observer_deadline_is_distinct_from_dependency_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _state(tmp_path) as state:
        entered = asyncio.Event()
        deadlines = []

        def timeout(delay):
            deadline = asyncio.timeout(delay)
            deadlines.append(deadline)
            return deadline

        async def blocked_owner(session_id, *, tenant_id):
            entered.set()
            await asyncio.Event().wait()

        with monkeypatch.context() as patch:
            patch.setattr(session_module, "asyncio", SimpleNamespace(timeout=timeout, sleep=asyncio.sleep))
            task = asyncio.create_task(_service(state, owner=blocked_owner).wait_inactive(
                "session", principal=_OWNER, timeout_seconds=None,
            ))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                deadlines[0].reschedule(asyncio.get_running_loop().time())
                with pytest.raises(AIError) as error:
                    await asyncio.wait_for(task, 5)
            finally:
                if not task.done():
                    await _cancel_observer(task)
        assert error.value.code is ErrorCode.WAIT_TIMEOUT
        failure = TimeoutError("backend read failed")

        async def broken_owner(session_id, *, tenant_id):
            raise failure

        with pytest.raises(TimeoutError) as dependency:
            await _service(state, owner=broken_owner).wait_inactive(
                "session", principal=_OWNER, timeout_seconds=5,
            )
        assert dependency.value is failure


@pytest.mark.parametrize("principal", (Principal("other", "tenant"), Principal("owner", "other")))
async def test_inactivity_does_not_reveal_admission_before_session_permission(
    tmp_path: Path, principal: Principal,
) -> None:
    async with _state(tmp_path) as state:
        async def forbidden_owner(session_id, *, tenant_id):
            pytest.fail("unauthorized caller reached admission lookup")

        with pytest.raises(AIError) as error:
            await _service(state, owner=forbidden_owner).wait_inactive(
                "session", principal=principal, timeout_seconds=5,
            )
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED


async def test_current_execution_permission_is_required(tmp_path: Path) -> None:
    async with _state(tmp_path) as state:
        await _admit(state, status=ExecutionStatus.SUCCEEDED)

        class DenyExecution(TenantAuthorizationPolicy):
            async def authorize(self, principal, action, resource):
                await super().authorize(principal, action, resource)
                if action is AuthorizationAction.EXECUTION_READ:
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)

        with pytest.raises(AIError) as error:
            await _service(state, authorization=DenyExecution()).wait_inactive(
                "session", principal=_OWNER, timeout_seconds=5,
            )
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED


@pytest.mark.parametrize("change", ("sql_admit_release", "local_membership"))
async def test_absent_admission_aba_retries_before_reporting_inactive(
    tmp_path: Path, change: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _state(tmp_path, "sqlite") as state:
        epoch = 0
        observed_change = False
        retry_reached = AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        initial = await state.conversation.sessions.get("session", tenant_id="tenant")

        def observe(session_id, *, tenant_id):
            return None, epoch

        async def owner(session_id, *, tenant_id):
            nonlocal epoch, observed_change
            if observed_change:
                raise retry_reached
            observed_change = True
            if change == "sql_admit_release":
                await _admit(state, status=ExecutionStatus.SUCCEEDED)
                await state.conversation.sessions.release_execution(
                    "session", tenant_id="tenant", execution_id="execution",
                )
                current = await state.conversation.sessions.get("session", tenant_id="tenant")
                assert current.revision == initial.revision
                assert current.active_execution_id is None
            else:
                # Two absent observations must not conceal an intervening local admission/removal.
                epoch += 2
            return None

        async def yield_retry(delay):
            # Keep a scheduling boundary without waiting for the production backoff.
            await asyncio.sleep(0)

        with monkeypatch.context() as patch:
            patch.setattr(session_module, "asyncio", SimpleNamespace(timeout=asyncio.timeout, sleep=yield_retry))
            with pytest.raises(AIError) as error:
                await _service(state, observe=observe, owner=owner).wait_inactive(
                    "session", principal=_OWNER, timeout_seconds=5,
                )
        assert error.value is retry_reached


async def test_completed_old_execution_does_not_hide_replacement(tmp_path: Path) -> None:
    async with _state(tmp_path) as state:
        await _admit(state, "old", ExecutionStatus.SUCCEEDED)
        replaced = False
        waiter = _BlockingWaiter()

        async def owner(session_id, *, tenant_id):
            nonlocal replaced
            if not replaced:
                replaced = True
                await state.conversation.sessions.release_execution(
                    "session", tenant_id="tenant", execution_id="old",
                )
                await _admit(state, "replacement", ExecutionStatus.STARTED)
            return None

        task = asyncio.create_task(_service(state, owner=owner, waiter=waiter).wait_inactive(
            "session", principal=_OWNER, timeout_seconds=5,
        ))
        try:
            await asyncio.wait_for(waiter.entered.wait(), 5)
            assert not task.done()
            assert waiter.execution_ids == ["replacement"]
        finally:
            await _cancel_observer(task)


async def test_optional_session_lookup_failure_preserves_old_request_lookup(tmp_path: Path) -> None:
    async with _state(tmp_path) as state:
        cache = InMemoryCacheBackend(max_bytes=2**20)

        class RequestOnlyBackend:
            claim_turn = cache.claim_turn
            read_turn = cache.read_turn
            read_turn_by_request = cache.read_turn_by_request
            renew_turn = cache.renew_turn
            release_turn = cache.release_turn
            cancel_turn = cache.cancel_turn

        coordination = PendingTurnCoordination(
            cache, RequestOnlyBackend(), namespace="inactivity", tenant_id="tenant",
        )
        assert await coordination.read_request_execution("absent", tenant_id="tenant") is None
        with pytest.raises(AIError) as error:
            await _service(state, owner=coordination.read_session_owner).wait_inactive(
                "session", principal=_OWNER, timeout_seconds=5,
            )
        assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID
        assert await coordination.read_request_execution("absent", tenant_id="tenant") is None


async def test_worker_wakeup_rechecks_authority_without_consuming_result(tmp_path: Path) -> None:
    async with _state(tmp_path) as state:
        original = await _admit(state)
        waiter = _BlockingWaiter()
        task = asyncio.create_task(_service(state, waiter=waiter).wait_inactive(
            "session", principal=_OWNER, timeout_seconds=5,
        ))
        try:
            await asyncio.wait_for(waiter.entered.wait(), 5)
            terminal = replace(
                original, status=ExecutionStatus.SUCCEEDED, revision=original.revision + 1,
                agent_run_seq=1,
                result=_execution("execution", status=ExecutionStatus.SUCCEEDED, prompt="question").result,
            )
            await _commit_empty_turn(state, terminal)
            await state.execution.executions.compare_and_swap(
                "execution", tenant_id="tenant", expected_revision=original.revision,
                next_record=terminal,
            )
            waiter.release.set()
            view = await asyncio.wait_for(task, 5)
            assert view.active_execution_id is None
            assert await state.execution.executions.get("execution", tenant_id="tenant") == terminal
            assert (await state.conversation.sessions.get("session", tenant_id="tenant")).active_execution_id == "execution"
        finally:
            if not task.done():
                await _cancel_observer(task)


@pytest.mark.parametrize("phase", ("source", "authorization", "owner", "confirmation", "projection"))
async def test_observation_budget_covers_complete_authoritative_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    async with _state(tmp_path) as state:
        entered = asyncio.Event()
        budgets = []

        def timeout(delay):
            deadline = asyncio.timeout(delay)
            if delay == 7:
                budgets.append(deadline)
            return deadline

        async def block():
            entered.set()
            await asyncio.Event().wait()

        class Authorization(TenantAuthorizationPolicy):
            async def authorize(self, principal, action, resource):
                await super().authorize(principal, action, resource)
                if phase == "authorization":
                    await block()

        async def owner(session_id, *, tenant_id):
            if phase == "owner":
                await block()
            return None

        service = _service(state, owner=owner, authorization=Authorization())
        read = state.conversation.sessions.state_store.read
        reads = 0

        async def source(callback):
            nonlocal reads
            reads += 1
            if (phase == "source" and reads == 1) or (phase == "confirmation" and reads == 2):
                await block()
            return await read(callback)

        with monkeypatch.context() as patch:
            patch.setattr(session_module, "asyncio", SimpleNamespace(timeout=timeout, sleep=asyncio.sleep))
            patch.setattr(state.conversation.sessions.state_store, "read", source)
            if phase == "projection":
                async def project(*args, **kwargs):
                    await block()
                patch.setattr(service, "project_view", project)
            task = asyncio.create_task(service.wait_inactive(
                "session", principal=_OWNER, timeout_seconds=60, observation_timeout_seconds=7,
            ))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                budgets[0].reschedule(asyncio.get_running_loop().time())
                with pytest.raises(TimeoutError, match="session inactivity observation timed out"):
                    await asyncio.wait_for(task, 5)
            finally:
                if not task.done():
                    await _cancel_observer(task)
        assert (await state.conversation.sessions.get("session", tenant_id="tenant")).active_execution_id is None


async def test_observation_budget_is_shared_across_source_and_owner_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _state(tmp_path) as state:
        first_read, continue_read, owner_read = asyncio.Event(), asyncio.Event(), asyncio.Event()
        budgets = []
        read = state.conversation.sessions.state_store.read

        def timeout(delay):
            deadline = asyncio.timeout(delay)
            if delay == 7:
                budgets.append(deadline)
            return deadline

        async def source(callback):
            first_read.set()
            await continue_read.wait()
            return await read(callback)

        async def owner(session_id, *, tenant_id):
            owner_read.set()
            await asyncio.Event().wait()

        with monkeypatch.context() as patch:
            patch.setattr(session_module, "asyncio", SimpleNamespace(timeout=timeout, sleep=asyncio.sleep))
            patch.setattr(state.conversation.sessions.state_store, "read", source)
            task = asyncio.create_task(_service(state, owner=owner).wait_inactive(
                "session", principal=_OWNER, timeout_seconds=60, observation_timeout_seconds=7,
            ))
            try:
                await asyncio.wait_for(first_read.wait(), 5)
                first_budget = budgets[0]
                continue_read.set()
                await asyncio.wait_for(owner_read.wait(), 5)
                # The original source-read deadline still controls this later I/O.
                first_budget.reschedule(asyncio.get_running_loop().time())
                with pytest.raises(TimeoutError, match="session inactivity observation timed out"):
                    await asyncio.wait_for(task, 5)
            finally:
                continue_read.set()
                if not task.done():
                    await _cancel_observer(task)


@pytest.mark.parametrize("passive", ("local", "remote", "retry"))
async def test_observation_budget_excludes_passive_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passive: str,
) -> None:
    async with _state(tmp_path) as state:
        waiting = asyncio.Event()
        deadlines = {}
        epoch = 0
        if passive != "retry":
            await _admit(state)

        def timeout(delay):
            deadline = asyncio.timeout(delay)
            if delay in (7, 60):
                deadlines[delay] = deadline
            return deadline

        async def sleep(delay):
            waiting.set()
            await asyncio.Event().wait()

        def observe(session_id, *, tenant_id):
            return None, epoch

        async def owner(session_id, *, tenant_id):
            nonlocal epoch
            if passive == "retry":
                epoch += 1
            return None

        waiter = _BlockingWaiter() if passive == "local" else None
        with monkeypatch.context() as patch:
            patch.setattr(session_module, "asyncio", SimpleNamespace(timeout=timeout, sleep=sleep))
            task = asyncio.create_task(_service(state, observe=observe, owner=owner, waiter=waiter).wait_inactive(
                "session", principal=_OWNER, timeout_seconds=60, observation_timeout_seconds=7,
            ))
            try:
                await asyncio.wait_for((waiter.entered if waiter else waiting).wait(), 5)
                # A finished observation budget cannot cancel a passive wait.
                with pytest.raises(RuntimeError):
                    deadlines[7].reschedule(asyncio.get_running_loop().time())
                deadlines[60].reschedule(asyncio.get_running_loop().time())
                with pytest.raises(AIError) as caught:
                    await asyncio.wait_for(task, 5)
                assert caught.value.code is ErrorCode.WAIT_TIMEOUT
            finally:
                if not task.done():
                    await _cancel_observer(task)


@pytest.mark.parametrize("value", (True, -1, float("nan"), float("inf"), "1"))
async def test_invalid_observation_budget_is_rejected(tmp_path: Path, value: object) -> None:
    async with _state(tmp_path) as state:
        with pytest.raises(AIError) as caught:
            await _service(state).wait_inactive("session", principal=_OWNER, observation_timeout_seconds=value)
        assert caught.value.code is ErrorCode.REQUEST_FIELD_INVALID


async def test_zero_observation_budget_is_an_immediate_deadline(tmp_path: Path) -> None:
    async with _state(tmp_path) as state:
        async def owner(session_id, *, tenant_id):
            await asyncio.Event().wait()
        with pytest.raises(TimeoutError, match="session inactivity observation timed out"):
            await _service(state, owner=owner).wait_inactive(
                "session", principal=_OWNER, observation_timeout_seconds=0,
            )


@pytest.mark.parametrize("simultaneous", (False, True))
async def test_overall_deadline_keeps_wait_timeout_during_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, simultaneous: bool,
) -> None:
    async with _state(tmp_path) as state:
        entered = asyncio.Event()
        deadlines = {}

        def timeout(delay):
            deadline = asyncio.timeout(delay)
            deadlines[delay] = deadline
            return deadline

        async def owner(session_id, *, tenant_id):
            entered.set()
            await asyncio.Event().wait()

        with monkeypatch.context() as patch:
            patch.setattr(session_module, "asyncio", SimpleNamespace(timeout=timeout, sleep=asyncio.sleep))
            task = asyncio.create_task(_service(state, owner=owner).wait_inactive(
                "session", principal=_OWNER, timeout_seconds=60, observation_timeout_seconds=7,
            ))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                now = asyncio.get_running_loop().time()
                deadlines[60].reschedule(now)
                if simultaneous:
                    deadlines[7].reschedule(now)
                with pytest.raises(AIError) as caught:
                    await asyncio.wait_for(task, 5)
                assert caught.value.code is ErrorCode.WAIT_TIMEOUT
            finally:
                if not task.done():
                    await _cancel_observer(task)


@pytest.mark.parametrize("overall_during_settlement", (False, True))
async def test_budget_does_not_hide_backend_timeout_or_cancellation_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overall_during_settlement: bool,
) -> None:
    async with _state(tmp_path) as state:
        failure = TimeoutError("backend timed out")

        async def broken(session_id, *, tenant_id):
            raise failure

        with pytest.raises(TimeoutError) as caught:
            await _service(state, owner=broken).wait_inactive(
                "session", principal=_OWNER, timeout_seconds=60, observation_timeout_seconds=7,
            )
        assert caught.value is failure

        entered, settling, settled = asyncio.Event(), asyncio.Event(), asyncio.Event()
        budgets = []
        overall = []

        def timeout(delay):
            deadline = asyncio.timeout(delay)
            if delay == 7:
                budgets.append(deadline)
            elif delay == 60:
                overall.append(deadline)
            return deadline

        async def slow_cancel(session_id, *, tenant_id):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                settling.set()
                await settled.wait()
                raise

        with monkeypatch.context() as patch:
            patch.setattr(session_module, "asyncio", SimpleNamespace(timeout=timeout, sleep=asyncio.sleep))
            task = asyncio.create_task(_service(state, owner=slow_cancel).wait_inactive(
                "session", principal=_OWNER, timeout_seconds=60, observation_timeout_seconds=7,
            ))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                budgets[0].reschedule(asyncio.get_running_loop().time())
                await asyncio.wait_for(settling.wait(), 5)
                assert not task.done()
                if overall_during_settlement:
                    overall[0].reschedule(asyncio.get_running_loop().time())
                    with pytest.raises(AIError) as caught:
                        await asyncio.wait_for(task, 5)
                    assert caught.value.code is ErrorCode.WAIT_TIMEOUT
                else:
                    settled.set()
                    with pytest.raises(TimeoutError, match="session inactivity observation timed out"):
                        await asyncio.wait_for(task, 5)
            finally:
                settled.set()
                if not task.done():
                    await _cancel_observer(task)


@pytest.mark.parametrize("expired", ("observation", "overall"))
async def test_dependency_suppressing_cancellation_cannot_return_idle_after_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expired: str,
) -> None:
    async with _state(tmp_path) as state:
        entered = asyncio.Event()
        deadlines = {}

        def timeout(delay):
            deadline = asyncio.timeout(delay)
            deadlines[delay] = deadline
            return deadline

        async def suppress_cancel(session_id, *, tenant_id):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return None

        with monkeypatch.context() as patch:
            patch.setattr(session_module, "asyncio", SimpleNamespace(timeout=timeout, sleep=asyncio.sleep))
            task = asyncio.create_task(_service(state, owner=suppress_cancel).wait_inactive(
                "session", principal=_OWNER, timeout_seconds=60, observation_timeout_seconds=7,
            ))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                deadlines[7 if expired == "observation" else 60].reschedule(asyncio.get_running_loop().time())
                if expired == "observation":
                    with pytest.raises(TimeoutError, match="session inactivity observation timed out"):
                        await asyncio.wait_for(task, 5)
                else:
                    with pytest.raises(AIError) as caught:
                        await asyncio.wait_for(task, 5)
                    assert caught.value.code is ErrorCode.WAIT_TIMEOUT
            finally:
                if not task.done():
                    await _cancel_observer(task)
