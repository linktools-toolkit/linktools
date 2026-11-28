#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live cached Conversation turns share Session display paging without durable writes."""
from ._runtime_test_helpers import model_features

from linktools.ai.persistence import Persistence
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceDomain, PersistenceStorage, StateTurnResources
from linktools.ai.execution import CancelExecutionRequest
from linktools.ai.session import ListSessionRequest
from linktools.ai.persistence.state import active_state_scope, record_key
from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.core import CommitPolicy
from linktools.ai.storage._database import SqlStorageContext

from ._runtime_test_helpers import _Models, _UsageFunctionModel, _wait_for_committed
from .test_session_refresh import _service
from .test_session_timeline import _commit_empty_turn, _execution, _session

pytestmark = pytest.mark.asyncio


class _SessionAuthorization(TenantAuthorizationPolicy):
    """Explicit host policy for canonical SQL sessions."""


@asynccontextmanager
async def _runtimes(tmp_path, *, cache, shared=True, native=False, source_authority=False, fail_next=False):
    entered, finish = asyncio.Event(), asyncio.Event()
    pause = False

    async def model(messages, info):
        nonlocal fail_next
        if pause:
            entered.set()
            await finish.wait()
            if fail_next:
                fail_next = False
                raise RuntimeError("controlled model failure")
        return ModelResponse(parts=[TextPart("answer")])

    def group():
        value = RuntimeFeatures("pending-session-display", defaults={"execution": {"retain_history": True}}, execution_commit=CommitPolicy.terminal())
        value.agent("default", model="default", allow_tools=())
        return value

    def block_next():
        nonlocal pause
        pause = True

    models = _Models(_UsageFunctionModel(model))
    async with AsyncExitStack() as stack:
        values = []
        coordination = StateTurnResources(pending_cache=cache) if source_authority else cache
        if not native:
            # Legacy pending-cache observation is still supported on non-SQL routes.
            persistence = await stack.enter_async_context(Persistence.open(
                "pending-session-display",
                storage=PersistenceStorage.in_memory(cache=cache, coordination=coordination),
            ))
        for _ in range(2 if shared else 1):
            if native:
                persistence = await stack.enter_async_context(Persistence.open(
                    "pending-session-display",
                    storage=PersistenceStorage.sqlite(tmp_path / "state.db", cache=cache, coordination=coordination),
                ))
            runtime = await stack.enter_async_context(Runtime.open(
                persistence=persistence, features=model_features(models, group()),
                authorization=_SessionAuthorization() if native else None,
            ))
            assert runtime.sessions.supports_turn_sessions is native
            values.append((runtime, runtime._state))
        try:
            yield values, entered, finish, block_next
        finally:
            finish.set()


@pytest.mark.parametrize("history_count,shared,cache_enabled", (
    (0, False, False), (0, True, True), (3, True, False), (3, True, True),
))
async def test_pending_session_projection_paging_and_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    history_count: int, shared: bool, cache_enabled: bool,
) -> None:
    cache = InMemoryCacheBackend(max_bytes=2**24) if cache_enabled else None
    async with _runtimes(tmp_path, cache=cache, shared=shared, native=True) as (runtimes, entered, finish, block):
        owner, storage = runtimes[0]
        principal = owner.default_principal
        await owner.sessions.create_turn_session("session", principal=principal)

        async def start(prompt, key):
            return (await owner.sessions.start_turn(
                "session", prompt, agent="default", principal=principal, idempotency_key=key,
            )).key.turn_id

        async def committed(turn_id):
            return await _wait_for_committed(
                lambda: owner.history.turn_result("session", turn_id, principal=principal),
                lambda turn: turn["conversation_committed"],
            )

        completed = []
        for index in range(history_count):
            turn_id = await start(f"question {index}", str(index))
            assert (await committed(turn_id))["status"] == "SUCCEEDED"
            completed.append(turn_id)
        for reader, _ in runtimes:
            warm = await reader.history.turn_session("session", principal=principal, limit=2)
            assert warm["session"]["active_turn_id"] is None
            assert [turn["turn_id"] for turn in warm["timeline"]["items"]] == completed[-2:]

        mutations = []
        publications = []
        finish_committed = asyncio.Event()
        original_mutation = SqlStorageContext.run_mutation
        turns = storage.session_turns
        assert turns is not None
        original_finish = turns.finish

        async def count_mutation(context, callback, **kwargs):
            result = await original_mutation(context, callback, **kwargs)
            mutations.append(kwargs.get("domain"))
            return result

        async def track_finish(key, **kwargs):
            result = await original_finish(key, **kwargs)
            publications.append((key.turn_id, kwargs["generation"], kwargs["state"], result))
            if result:
                finish_committed.set()
            return result

        monkeypatch.setattr(SqlStorageContext, "run_mutation", count_mutation)
        monkeypatch.setattr(turns, "finish", track_finish)
        block()
        turn_id = await start("pending question", "pending")
        cursors = []
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert await storage.execution.executions.get(turn_id, tenant_id=owner.tenant_id) is None
            admitted_mutations = tuple(mutations)
            for reader, _ in runtimes:
                state = await reader.history.turn_session_state("session", principal=principal)
                assert state["active_turn_id"] == turn_id
                listed = await reader.history.turn_sessions(principal=principal, session_ids=("session",))
                assert [entry["active_turn_id"] for entry in listed["items"]] == [turn_id]
                pending = await reader.history.turn_result("session", turn_id, principal=principal)
                assert pending["status"] == "PENDING_START" and not pending["conversation_committed"]
                assert pending["user_input"]["prompt"] == "pending question"
                assert pending["items"] == [] and pending["process"] == [] and pending["output"] is None

                detail = await reader.history.turn_session("session", principal=principal, limit=2)
                assert detail["session"] == state
                page = detail["timeline"]
                assert [turn["turn_id"] for turn in page["items"]] == completed[-2:]
                if history_count:
                    assert page["next_cursor"] is not None
                    cursors.append((reader, page["next_cursor"]))
                else:
                    assert page["next_cursor"] is None

                seen = []
                cursor = None
                while True:
                    detail = await reader.history.turn_session("session", principal=principal, cursor=cursor, limit=1)
                    assert detail["session"]["active_turn_id"] == turn_id
                    page = detail["timeline"]
                    seen.extend(turn["turn_id"] for turn in page["items"])
                    cursor = page["next_cursor"]
                    if cursor is None:
                        break
                assert seen == list(reversed(completed))
            assert tuple(mutations) == admitted_mutations
        finally:
            finish.set()
        await asyncio.wait_for(finish_committed.wait(), 5)
        terminal = await committed(turn_id)
        assert terminal["status"] == "SUCCEEDED" and terminal["output"] == {"text": "answer"}
        assert publications == [(turn_id, 1, "succeeded", True)]
        assert len(mutations) > len(admitted_mutations)
        completed_mutations = tuple(mutations)
        for reader, _ in runtimes:
            detail = await reader.history.turn_session("session", principal=principal, limit=2)
            assert detail["session"]["active_turn_id"] is None
            assert [turn["turn_id"] for turn in detail["timeline"]["items"]] == (completed + [turn_id])[-2:]
            assert detail["timeline"]["items"][-1] == terminal
            assert await reader.history.turn_result("session", turn_id, principal=principal) == terminal
        for reader, cursor in cursors:
            detail = await reader.history.turn_session("session", principal=principal, cursor=cursor, limit=2)
            assert [turn["turn_id"] for turn in detail["timeline"]["items"]] == completed[:-2]
            assert detail["timeline"]["next_cursor"] is None
        assert tuple(mutations) == completed_mutations


@pytest.mark.parametrize("transition", ("complete", "cancel", "promote"))
@pytest.mark.parametrize("remote", (False, True))
async def test_pending_session_projection_source_wins_observation_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transition: str, remote: bool,
) -> None:
    cache = InMemoryCacheBackend(max_bytes=2**24)
    async with _runtimes(tmp_path, cache=cache) as (runtimes, entered, finish, block):
        owner, _ = runtimes[0]
        reader, reader_storage = runtimes[int(remote)]
        session = await owner.agents.get().create_session("session")
        block()
        execution = await session.start("question")
        await asyncio.wait_for(entered.wait(), 5)
        # This non-SQL runtime durably exposes admission before invocation.
        # Feed that exact, now-stale snapshot in through the advisory port.
        reader_principal = reader.default_principal
        pending = await reader_storage.execution.executions.get(execution.execution_id, tenant_id=reader.tenant_id)
        assert pending is not None and pending.status is ExecutionStatus.STARTED and pending.result is None
        service = reader.sessions._service
        crossed = False

        async def cross_boundary(session_id, *, tenant_id):
            nonlocal crossed
            assert active_state_scope() is None
            assert (session_id, tenant_id) == ("session", reader_principal.tenant_id)
            if not crossed:
                crossed = True
                if transition == "complete":
                    finish.set()
                    await execution.wait(timeout_seconds=5)
                elif transition == "cancel":
                    await owner.executions.cancel(execution.execution_id, CancelExecutionRequest(owner.default_principal, "cancel"))
                else:
                    assert await owner.executions._service.acquire_dependency_hold(execution.execution_id, tenant_id=owner.tenant_id, hold_id="hold")
            return pending

        monkeypatch.setattr(service, "_runtime_pending_session_execution", cross_boundary)
        detail = await reader.sessions.refresh("session", principal=reader.default_principal, limit=1, max_age=60)
        assert crossed
        assert [turn.execution_id for turn in detail.timeline.items] == [execution.execution_id]
        assert detail.timeline.next_cursor is None
        canonical = await reader_storage.execution.executions.get(execution.execution_id, tenant_id=reader.tenant_id)
        assert canonical is not None and detail.timeline.items[0].status is canonical.status
        assert detail.session.active_execution_id == (execution.execution_id if transition == "promote" else None)
        assert detail.timeline.items[0].conversation_committed is (transition in {"complete", "cancel"})
        if transition == "cancel":
            assert detail.timeline.items[0].status is ExecutionStatus.CANCELLED
            assert detail.timeline.items[0].output is None and detail.timeline.items[0].items == ()
        if transition == "promote":
            await owner.executions._service.release_dependency_hold(execution.execution_id, tenant_id=owner.tenant_id, hold_id="hold")
        finish.set()
        await execution.wait(timeout_seconds=5)


@pytest.mark.parametrize("principal", (Principal("other", "tenant"), Principal("owner", "other")))
@pytest.mark.parametrize("method", ("get", "refresh", "timeline"))
async def test_pending_session_projection_denial_precedes_observation(principal: Principal, method: str) -> None:
    storage = PersistenceStorage.in_memory()
    storage = Persistence(storage)
    await storage.initialize(namespace="session-refresh", tenant_id="tenant")
    try:
        await storage.conversation.sessions.create(_session())
        service = _service(storage)

        async def forbidden(*args, **kwargs):
            raise AssertionError("unauthorized caller reached a live observation")

        service._runtime_pending_session_execution = forbidden
        with pytest.raises(AIError) as denied:
            await getattr(service, method)("session", principal=principal)
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        await storage.close()


@pytest.mark.parametrize("fault", ("session", "child", "terminal", "older_session"))
async def test_pending_session_projection_rejects_unbound_observations(fault: str) -> None:
    storage = PersistenceStorage.in_memory()
    storage = Persistence(storage)
    await storage.initialize(namespace="session-refresh", tenant_id="tenant")
    try:
        record = await storage.conversation.sessions.create(_session())
        pending = _execution("pending", status=ExecutionStatus.STARTED, prompt="question")
        changes = {
            "session": {"session_id": "other"},
            "child": {"parent_execution_id": "parent", "parent_invocation_id": "invocation",
                      "lineage_kind": ExecutionLineageKind.SUBAGENT},
            "terminal": {"status": ExecutionStatus.CANCELLED},
            "older_session": {"created_at": record.created_at - timedelta(seconds=1)},
        }
        pending = replace(pending, **changes[fault])
        service = _service(storage)

        async def observe(session_id, *, tenant_id):
            assert (session_id, tenant_id) == ("session", "tenant")
            return pending

        service._runtime_pending_session_execution = observe
        principal = Principal("owner", "tenant")
        detail = await service.refresh("session", principal=principal)
        assert detail.session.active_execution_id is None and detail.timeline.items == ()
        assert (await service.get("session", principal=principal)).active_execution_id is None
        assert (await service.timeline("session", principal=principal)).items == ()
    finally:
        await storage.close()


async def test_pending_session_projection_peer_retains_source_when_live_coordination_is_hidden(tmp_path: Path) -> None:
    class HiddenOwnerCache(InMemoryCacheBackend):
        hidden = False

        async def read_turn(self, scope, execution_id):
            return None if self.hidden else await super().read_turn(scope, execution_id)

    cache = HiddenOwnerCache(max_bytes=2**24)
    async with _runtimes(tmp_path, cache=cache) as (runtimes, entered, finish, block):
        owner, _ = runtimes[0]
        reader, _ = runtimes[1]
        session = await owner.agents.get().create_session("session")
        block()
        execution = await session.start("question")
        await asyncio.wait_for(entered.wait(), 5)
        assert (await reader.sessions.refresh("session", principal=reader.default_principal)).session.active_execution_id == execution.execution_id
        cache.hidden = True
        try:
            detail = await reader.sessions.refresh("session", principal=reader.default_principal, max_age=60)
            assert detail.session.active_execution_id == execution.execution_id
            [turn] = detail.timeline.items
            assert turn.execution_id == execution.execution_id and not turn.conversation_committed
        finally:
            cache.hidden = False
            finish.set()
        await execution.wait(timeout_seconds=5)


async def test_pending_session_projection_no_cache_keeps_source_admission(tmp_path: Path) -> None:
    async with _runtimes(tmp_path, cache=None, shared=False) as (runtimes, entered, finish, block):
        runtime, storage = runtimes[0]
        session = await runtime.agents.get().create_session("session")
        block()
        execution = await session.start("question")
        await asyncio.wait_for(entered.wait(), 5)
        source = await storage.conversation.sessions.get("session", tenant_id=runtime.tenant_id)
        assert source.active_execution_id == execution.execution_id
        detail = await runtime.sessions.refresh("session", principal=runtime.default_principal, max_age=60)
        assert detail.session.active_execution_id == execution.execution_id
        assert [turn.execution_id for turn in detail.timeline.items] == [execution.execution_id]
        finish.set()
        await execution.wait(timeout_seconds=5)


@pytest.mark.parametrize("limit", (1, 10))
async def test_pending_session_projection_cannot_fill_missing_source_execution(limit: int) -> None:
    storage = PersistenceStorage.in_memory()
    storage = Persistence(storage)
    await storage.initialize(namespace="session-refresh", tenant_id="tenant")
    try:
        await storage.conversation.sessions.create(_session())
        pending = _execution("pending", status=ExecutionStatus.STARTED, prompt="question")
        sessions = storage.conversation.sessions
        await sessions.admit_execution("session", tenant_id="tenant", execution_id="pending", expected=None)
        await sessions.release_execution("session", tenant_id="tenant", execution_id="pending")
        service = _service(storage)

        async def observe(session_id, *, tenant_id):
            return pending

        service._runtime_pending_session_execution = observe
        with pytest.raises(AIError) as unavailable:
            await service.refresh("session", principal=Principal("owner", "tenant"), limit=limit)
        assert unavailable.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await storage.close()


@pytest.mark.parametrize("method", ("timeline", "refresh"))
async def test_pending_session_projection_older_page_cannot_fill_missing_source_execution(method: str) -> None:
    storage = PersistenceStorage.in_memory()
    storage = Persistence(storage)
    await storage.initialize(namespace="session-refresh", tenant_id="tenant")
    try:
        sessions = storage.conversation.sessions
        await sessions.create(_session())
        pending = _execution("missing", status=ExecutionStatus.STARTED, prompt="cache observation")
        for execution_id in ("missing", "newer"):
            if execution_id == "newer":
                await storage.execution.executions.create(
                    _execution(execution_id, status=ExecutionStatus.FAILED, prompt="canonical"),
                )
            await sessions.admit_execution("session", tenant_id="tenant", execution_id=execution_id, expected=None)
            if execution_id == "newer":
                await _commit_empty_turn(
                    storage, await storage.execution.executions.get(execution_id, tenant_id="tenant"),
                )
            await sessions.release_execution("session", tenant_id="tenant", execution_id=execution_id)
        service = _service(storage)
        principal = Principal("owner", "tenant")
        newest = await service.timeline("session", principal=principal, limit=1)
        assert [turn.execution_id for turn in newest.items] == ["newer"]
        assert newest.next_cursor is not None
        read = getattr(service, method)
        with pytest.raises(AIError) as unavailable:
            await read("session", principal=principal, limit=1, cursor=newest.next_cursor)
        assert unavailable.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

        async def observe(session_id, *, tenant_id):
            return pending

        service._runtime_pending_session_execution = observe
        with pytest.raises(AIError) as still_unavailable:
            await read("session", principal=principal, limit=1, cursor=newest.next_cursor)
        assert still_unavailable.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await storage.close()


@pytest.mark.parametrize("observation", ("absent", "rejected"))
@pytest.mark.parametrize("method,max_age,keep_active", (
    ("timeline", 0, False), ("refresh", 0, False), ("refresh", 60, False), ("refresh", 0, True),
))
async def test_pending_session_projection_rechecks_owner_after_empty_observation(
    method: str, max_age: float, keep_active: bool, observation: str,
) -> None:
    storage = PersistenceStorage.in_memory()
    storage = Persistence(storage)
    await storage.initialize(namespace="session-refresh", tenant_id="tenant")
    try:
        sessions = storage.conversation.sessions
        record = await sessions.create(_session())
        service = _service(storage)
        service._cache = InMemoryCacheBackend(max_bytes=2**20)
        principal = Principal("owner", "tenant")
        if max_age:
            await service.refresh("session", principal=principal, limit=1, max_age=max_age)

        async def observe(session_id, *, tenant_id):
            kinds = (("session", "session"), ("conversation_history", record.history_id),
                     ("transcript_head", record.history_id))
            keys = tuple(record_key(
                sessions.namespace, sessions.tenant_id, PersistenceDomain.CONVERSATION.value, kind, (identity,),
            ) for kind, identity in kinds)
            await sessions.state_store.mutate(lambda transaction: transaction.delete_records(keys))
            await sessions.create(replace(_session(), owner_principal_id="other", metadata={"secret": "other owner"}))
            await storage.execution.executions.create(replace(
                _execution("other-execution", status=ExecutionStatus.FAILED, prompt="OTHER OWNER INPUT"),
                principal_id="other",
            ))
            await sessions.admit_execution("session", tenant_id="tenant", execution_id="other-execution", expected=None)
            if not keep_active:
                await sessions.release_execution("session", tenant_id="tenant", execution_id="other-execution")
            if observation == "absent":
                return None
            return replace(_execution("rejected", status=ExecutionStatus.STARTED, prompt="observation"), principal_id="other")

        service._runtime_pending_session_execution = observe
        options = {"max_age": max_age} if method == "refresh" else {}
        with pytest.raises(AIError) as denied:
            await getattr(service, method)("session", principal=principal, limit=1, **options)
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        current = await sessions.get("session", tenant_id="tenant")
        assert current.active_execution_id == ("other-execution" if keep_active else None)
    finally:
        await storage.close()


@pytest.mark.parametrize("transition", ("complete", "cancel", "promote", "lost_ack", "fail"))
async def test_local_committed_turn_capture_matches_canonical_and_never_claims_occupancy(
    tmp_path: Path, transition: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = InMemoryCacheBackend(max_bytes=2**24)
    async with _runtimes(
        tmp_path, cache=cache, source_authority=True, fail_next=(transition == "fail"),
    ) as (runtimes, entered, finish, block):
        owner, persistence = runtimes[0]
        reader, _ = runtimes[1]
        session = await owner.agents.get().create_session("session")
        warm = await session.start("previous")
        assert (await warm.wait()).result.status is ExecutionStatus.SUCCEEDED
        inactive = await owner.sessions.wait_inactive("session", principal=owner.default_principal, timeout_seconds=5)
        assert inactive.active_execution_id is None
        block()
        execution = await session.start("current")
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(AIError) as busy:
            await session.start("competing", idempotency_key="competing")
        assert busy.value.code is ErrorCode.SESSION_BUSY
        with (owner.sessions.capture_published_turn(
                  "session", execution.execution_id, principal=owner.default_principal,
              ) as captured,
              owner.sessions.capture_published_turn(
                  "different-session", execution.execution_id, principal=owner.default_principal,
              ) as wrong_session,
              reader.sessions.capture_published_turn(
                  "session", execution.execution_id, principal=reader.default_principal,
              ) as remote):
            assert await captured() is None and await remote() is None
            lost = False
            if transition == "lost_ack":
                from linktools.ai.persistence.contracts import ExecutionTerminalPublication

                group = persistence.execution.executions.state_store.storage_group
                mutate = group.mutate

                async def lose_terminal_ack(stores, callback):
                    nonlocal lost
                    value = await mutate(stores, callback)
                    if not lost and isinstance(value, ExecutionTerminalPublication):
                        lost = True
                        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
                    return value

                monkeypatch.setattr(group, "mutate", lose_terminal_ack)
            if transition == "cancel":
                await execution.cancel(idempotency_key="cancel")
            if transition == "promote":
                assert await owner.executions._service.acquire_dependency_hold(
                    execution.execution_id, tenant_id=owner.tenant_id, hold_id="hold",
                )
                await owner.executions._service.release_dependency_hold(
                    execution.execution_id, tenant_id=owner.tenant_id, hold_id="hold",
                )
            finish.set()
            result = (await execution.wait(timeout_seconds=5)).result
            async def forbidden_source_get(*args, **kwargs):
                raise AssertionError("local commit capture reached a durable lookup")

            with monkeypatch.context() as scoped:
                scoped.setattr(persistence.execution.executions, "get", forbidden_source_get)
                scoped.setattr(persistence.conversation.sessions, "get", forbidden_source_get)
                local = await captured()
            assert await wrong_session() is None
            # Both in-process runtimes borrow the same Memory Persistence:
            # peer sees the same after-image, not a claim on session occupancy.
            assert await remote() == local
            expected = await reader.history.session_detail(
                "session", principal=reader.default_principal, limit=1, max_age=0,
            )
            if transition == "lost_ack":
                assert lost and local is None
                assert expected.timeline.items[0].execution_id == execution.execution_id
                assert expected.timeline.items[0].status is result.status
                return
            assert local is not None
            assert expected.timeline.items == (local.turn,)
            assert expected.session.revision == local.session_revision
            assert local.turn_seq == 2
            assert local.turn.status == result.status
            if transition == "fail":
                assert result.status is ExecutionStatus.FAILED
                assert local.turn.items == () and local.turn.output is None

            class RevokedSessionRead:
                async def authorize(self, principal, action, resource):
                    from linktools.ai.core import AuthorizationAction
                    assert action is AuthorizationAction.SESSION_READ and resource.id == "session"
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)

            with monkeypatch.context() as scoped:
                scoped.setattr(owner.sessions._service, "_runtime_authorization", RevokedSessionRead())
                with pytest.raises(AIError) as denied:
                    await captured()
                assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED


            inactive = await owner.sessions.wait_inactive("session", principal=owner.default_principal, timeout_seconds=5)
            assert inactive.active_execution_id is None
            finish.clear()
            entered.clear()
            later = await session.start("later")
            await asyncio.wait_for(entered.wait(), 5)
            state = await reader.history.inspect_session("session", principal=reader.default_principal)
            assert state.active_execution_id == later.execution_id
            assert await captured() == local
            finish.set()
            await later.wait(timeout_seconds=5)
        assert await captured() is None
