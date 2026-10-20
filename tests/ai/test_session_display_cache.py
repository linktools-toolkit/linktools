#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Opt-in whole display snapshots retain fresh authority and bounded staleness."""

from linktools.ai.runtime.state import RuntimeState
import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from sqlalchemy import event

import linktools.ai.runtime._session as session_module
from linktools.ai.core import ExecutionStatus, HmacCursorSigner, Principal, TenantAuthorizationPolicy, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.capability import ExecutionCapability
from linktools.ai.runtime import RuntimeDomain, RuntimeSessions, RuntimeStorage, RuntimeStoragePlan
from linktools.ai.storage import StorageRoute
from linktools.ai.runtime._session import RuntimeSessionService
from linktools.ai.runtime.state._contracts import ConversationCursor
from linktools.ai.runtime.state._store import active_state_scope
from linktools.ai.storage import InMemoryCacheBackend, InMemoryStreamBackend

from .test_incremental_projection_boundaries import _storage
from .test_session_refresh import _admit
from .test_session_timeline import _commit_empty_turn, _commit_fields, _execution, _materialize_conversation, _session
from .test_runtime_content_cache_integration import WriteOnlyCache

pytestmark = pytest.mark.asyncio
_PRINCIPAL = Principal("owner", "tenant")


class _Cache(InMemoryCacheBackend):
    def __init__(self):
        super().__init__(max_bytes=1_000_000)
        self.reads = self.writes = 0
        self.keys = set()
        self.failure = False

    async def get(self, key):
        assert active_state_scope() is None
        self.reads += 1
        if self.failure:
            raise OSError("cache unavailable")
        return await super().get(key)

    async def put(self, key, content):
        assert active_state_scope() is None
        self.writes += 1
        self.keys.add(key)
        if self.failure:
            raise OSError("cache unavailable")
        await super().put(key, content)


def _service(state, cache, authorization=None):
    return RuntimeSessionService(
        state.conversation, state.execution.executions,
        authorization or TenantAuthorizationPolicy(), object(),
        HmacCursorSigner("display", b"display-test-key"), history_reader=object(),
        transcript_store=state.run_store.conversation_persistence, cache=cache,
    )


def _clock(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(session_module, "time", SimpleNamespace(time=lambda: clock.now))
    return clock


@pytest.fixture(autouse=True)
def _fixed_display_clock(monkeypatch):
    _clock(monkeypatch)


async def _prepare(state, *, body=False, namespace="display"):
    await state.initialize(
        namespace=namespace, tenant_id="tenant",
    )
    record = await state.conversation.sessions.create(replace(_session(), metadata={"nested": {"value": "old"}}))
    await _admit(state, status=ExecutionStatus.SUCCEEDED if body else ExecutionStatus.PENDING_START)
    if body:
        run_id, count = await _materialize_conversation(state, record.history_id)
        sessions = state.conversation.sessions
        execution = await state.execution.executions.get("active", tenant_id="tenant")

        async def commit(transaction):
            await sessions.commit_timeline_turn_in_transaction(
                transaction, "session", tenant_id="tenant", execution_id="active",
                start_message_index=0, end_message_index=count,
                **_commit_fields(execution),
            )
            await sessions.advance_continuation_in_transaction(
                transaction, "session", tenant_id="tenant", execution_id="active", expected=None,
                next_cursor=ConversationCursor(run_id, history_id=record.history_id, message_count=count),
                release_execution=True, history_quality="complete",
            )
        await sessions.state_store.mutate(commit)
    return record


async def _change_metadata(state, value):
    sessions = state.conversation.sessions
    record = await sessions.get("session", tenant_id="tenant")
    return await sessions.compare_and_swap(
        "session", tenant_id="tenant", expected_revision=record.revision,
        next_record=replace(record, revision=record.revision + 1, metadata={"value": value}),
    )


async def _change_execution_display(state, value, execution_id="active"):
    executions = state.execution.executions
    record = await executions.get(execution_id, tenant_id="tenant")
    return await executions.compare_and_swap(
        execution_id, tenant_id="tenant", expected_revision=record.revision,
        next_record=replace(record, revision=record.revision + 1, safe_error_details={"value": value}),
    )


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_display_cache_returns_whole_old_page_until_fixed_expiry(tmp_path, monkeypatch, backend):
    clock = _clock(monkeypatch)
    state, cache = _storage(backend, tmp_path), _Cache()
    state = RuntimeState(state)
    await _prepare(state)
    try:
        service = _service(state, cache)
        first = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        assert cache.writes == 1
        await _change_execution_display(state, "new")
        clock.now += 2.9
        warm = await service.refresh("session", principal=_PRINCIPAL, max_age=60)
        assert warm == first
        assert cache.writes == 1
        clock.now += 0.1
        fresh = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        assert fresh.timeline.items[0].safe_error_details == {"value": "new"}
        assert cache.writes == 2
        await _change_execution_display(state, "strong")
        reads = cache.reads
        strong = await service.refresh("session", principal=_PRINCIPAL)
        assert strong.timeline.items[0].safe_error_details == {"value": "strong"}
        assert cache.reads == reads
    finally:
        await state.close()


@pytest.mark.parametrize("body", (False, True))
async def test_display_cache_saves_body_sql_and_keeps_fresh_dependency_reads(tmp_path, body):
    state, cache = _storage("sqlite", tmp_path), _Cache()
    state = RuntimeState(state)
    await _prepare(state, body=body)
    engine = state.conversation.sessions.state_store.storage_group.context.engine
    statements = []

    def capture(_connection, _cursor, statement, *_args):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        service = _service(state, cache)
        cold = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        cold_count = len(statements)
        statements.clear()
        assert await service.refresh("session", principal=_PRINCIPAL, max_age=3) == cold
        assert len(statements) == (4 if body else 3)
        assert cold_count > len(statements)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await state.close()


async def test_display_cache_rechecks_policy_and_deletion_after_authorization(monkeypatch):
    state, cache = RuntimeStorage.in_memory(), _Cache()
    state = RuntimeState(state)
    await _prepare(state)
    calls = []
    mode = ["allow"]

    class Policy:
        async def authorize(self, principal, action, resource):
            assert active_state_scope() is None
            calls.append(resource)
            if mode[0] == "deny":
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            if mode[0] == "delete":
                async def remove(transaction):
                    records = await transaction.scan_records()
                    await transaction.delete_records(tuple(record.key_digest for record in records if record.kind == "session"))
                await state.conversation.sessions.state_store.mutate(remove)

    try:
        service = _service(state, cache, Policy())
        await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        mode[0] = "deny"
        with pytest.raises(AIError) as error:
            await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
        mode[0] = "delete"
        with pytest.raises(AIError) as error:
            await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert len(calls) == 3
    finally:
        await state.close()


async def test_display_cache_does_not_share_sources_or_mutable_results():
    cache = _Cache()
    states = [RuntimeStorage.in_memory(), RuntimeStorage.in_memory()]
    for index, state in enumerate(states):
        state = states[index] = RuntimeState(state)
        await _prepare(state, body=True, namespace=f"display-source-{index}")
    try:
        first = _service(states[0], cache)
        second = _service(states[1], cache)
        value = await first.refresh("session", principal=_PRINCIPAL, max_age=3)
        value.session.metadata["nested"]["value"] = "caller mutation"
        value.timeline.items[0].user_input["prompt"]["text"] = "caller mutation"
        reread = await first.refresh("session", principal=_PRINCIPAL, max_age=3)
        assert reread.session.metadata["nested"]["value"] == "old"
        assert reread.timeline.items[0].user_input["prompt"]["text"] == "active"
        await _change_metadata(states[1], "other source")
        assert (await second.refresh("session", principal=_PRINCIPAL, max_age=3)).session.metadata == {"value": "other source"}
        assert len(cache.keys) == 2
    finally:
        for state in states:
            await state.close()


@pytest.mark.parametrize("value", (-1, float("nan"), float("inf"), True, "3"))
async def test_display_cache_age_is_finite_non_negative(value):
    state = RuntimeStorage.in_memory()
    state = RuntimeState(state)
    await _prepare(state)
    try:
        with pytest.raises(ValueError, match="max_age"):
            await _service(state, _Cache()).refresh("session", principal=_PRINCIPAL, max_age=value)
    finally:
        await state.close()


@pytest.mark.parametrize("failure", ("io", "corrupt"))
async def test_display_cache_failures_return_origin(failure):
    state, cache = RuntimeStorage.in_memory(), _Cache()
    state = RuntimeState(state)
    await _prepare(state)
    try:
        service = _service(state, cache)
        await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        await _change_execution_display(state, "new")
        if failure == "io":
            cache.failure = True
        else:
            for key in cache.keys:
                await cache.put(key, b"not-json")
        assert (await service.refresh("session", principal=_PRINCIPAL, max_age=3)).timeline.items[0].safe_error_details == {"value": "new"}
    finally:
        await state.close()


@pytest.mark.parametrize("phase", ("get", "dependency", "body", "put"))
async def test_display_cache_slow_io_never_renews_capture_age(monkeypatch, phase):
    clock = _clock(monkeypatch)
    state, cache = RuntimeStorage.in_memory(), _Cache()
    state = RuntimeState(state)
    await _prepare(state, body=True)
    session = await state.conversation.sessions.get("session", tenant_id="tenant")
    await _admit(state, "second", ExecutionStatus.PENDING_START, expected=session.continuation)
    try:
        service = _service(state, cache)
        await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        await _change_execution_display(state, "new", "second")
        if phase in {"get", "dependency"}:
            owner = cache if phase == "get" else state.execution.executions
            name = "get" if phase == "get" else "get_many"
            read = getattr(owner, name)

            async def slow(*args, **kwargs):
                value = await read(*args, **kwargs)
                clock.now += 3
                return value
            monkeypatch.setattr(owner, name, slow)
            assert (await service.refresh("session", principal=_PRINCIPAL, max_age=3)).timeline.items[-1].safe_error_details == {"value": "new"}
        else:
            clock.now += 3
            if phase == "body":
                read = state.run_store.conversation_persistence.iter_conversation_message_range

                async def slow_body(**kwargs):
                    clock.now += 3
                    async for item in read(**kwargs):
                        yield item
                monkeypatch.setattr(state.run_store.conversation_persistence, "iter_conversation_message_range", slow_body)
            else:
                put = cache.put

                async def slow_put(*args):
                    clock.now += 3
                    await put(*args)
                monkeypatch.setattr(cache, "put", slow_put)
            await service.refresh("session", principal=_PRINCIPAL, max_age=3)
            await _change_execution_display(state, "latest", "second")
            assert (await service.refresh("session", principal=_PRINCIPAL, max_age=3)).timeline.items[-1].safe_error_details == {"value": "latest"}
    finally:
        await state.close()


async def test_display_cache_cancellation_is_request_owned(monkeypatch):
    state, cache = RuntimeStorage.in_memory(), _Cache()
    state = RuntimeState(state)
    await _prepare(state)
    entered, blocked, exited = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def blocked_get(key):
        try:
            entered.set()
            await blocked.wait()
        finally:
            exited.set()
    monkeypatch.setattr(cache, "get", blocked_get)
    try:
        task = asyncio.create_task(_service(state, cache).refresh("session", principal=_PRINCIPAL, max_age=3))
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert exited.is_set()
        assert cache.writes == 0
    finally:
        await state.close()


async def test_display_cache_outer_transaction_never_reads_or_publishes_cache():
    state, cache = RuntimeStorage.in_memory(), _Cache()
    state = RuntimeState(state)
    await _prepare(state)
    service = _service(state, cache)
    try:
        await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        counts = cache.reads, cache.writes
        sessions = state.conversation.sessions
        executions = state.execution.executions

        async def operation(_group):
            value = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
            assert value.session.session_id == "session"
            raise RuntimeError("rollback")
        with pytest.raises(RuntimeError, match="rollback"):
            await sessions.state_store.storage_group.mutate((sessions.state_store, executions.state_store), operation)
        assert (cache.reads, cache.writes) == counts
    finally:
        await state.close()


async def test_display_cache_facade_passes_max_age_to_custom_service():
    calls = []

    class Custom:
        async def refresh(self, session_id, *, principal, cursor, limit, max_age):
            calls.append((session_id, principal, cursor, limit, max_age))
            return "custom"
    facade = RuntimeSessions(Custom(), object(), ensure_open=lambda: None)
    assert await facade.refresh("session", principal=_PRINCIPAL, max_age=0) == "custom"
    assert await facade.refresh("session", principal=_PRINCIPAL, max_age=3) == "custom"
    assert calls == [("session", _PRINCIPAL, None, 100, 0), ("session", _PRINCIPAL, None, 100, 3)]


async def test_display_cache_split_routes_keep_strong_fallback(tmp_path):
    state = RuntimeStorage.from_plan(RuntimeStoragePlan(
        execution=StorageRoute.sqlite(tmp_path / "execution.db"),
        conversation=StorageRoute.sqlite(tmp_path / "conversation.db"),
    ))
    cache = _Cache()
    state = RuntimeState(state)
    await _prepare(state)
    try:
        service = _service(state, cache)
        await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        await _change_metadata(state, "new")
        assert (await service.refresh("session", principal=_PRINCIPAL, max_age=3)).session.metadata == {"value": "new"}
        assert cache.reads == cache.writes == 0
    finally:
        await state.close()


@pytest.mark.parametrize("corruption", ("session_status", "active_identity", "turn_status", "metadata", "item_content"))
async def test_display_cache_malformed_typed_json_falls_back(corruption):
    import json
    from linktools.ai.runtime.state._codec import decode_domain, encode_domain

    state, cache = RuntimeStorage.in_memory(), _Cache()
    state = RuntimeState(state)
    await _prepare(state, body=True)
    try:
        service = _service(state, cache)
        expected = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        key = next(iter(cache.keys))
        entry = decode_domain(json.loads(await cache.get(key))["payload"], dict[str, object])
        if corruption == "session_status":
            entry["detail"]["session"]["status"] = 123
        elif corruption == "active_identity":
            entry["detail"]["session"]["active_execution_id"] = "wrong-resource"
        elif corruption == "turn_status":
            entry["detail"]["timeline"]["items"][0]["status"] = {"unexpected": "shape"}
        elif corruption == "metadata":
            entry["detail"]["session"]["metadata"] = []
        else:
            entry["detail"]["timeline"]["items"][0]["items"][0]["content"] = expected.timeline.items[0].created_at
        payload = encode_domain(entry)
        await cache.put(key, json.dumps({"payload": payload, "digest": canonical_sha256(payload)}).encode())
        assert await service.refresh("session", principal=_PRINCIPAL, max_age=3) == expected
        assert cache.writes == 3  # initial capture, corruption, repaired capture
    finally:
        await state.close()


async def test_display_cache_execution_changes_without_session_revision_remain_whole(monkeypatch):
    clock = _clock(monkeypatch)
    state, cache = RuntimeStorage.in_memory(), _Cache()
    state = RuntimeState(state)
    await _prepare(state)
    try:
        service = _service(state, cache)
        first = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        record = await state.execution.executions.get("active", tenant_id="tenant")
        await state.execution.executions.compare_and_swap(
            "active", tenant_id="tenant", expected_revision=record.revision,
            next_record=replace(record, revision=record.revision + 1, status=ExecutionStatus.STARTED),
        )
        warm = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        assert warm == first
        strong = await service.refresh("session", principal=_PRINCIPAL)
        assert strong.session.revision == first.session.revision
        assert strong.timeline.items[0].status is ExecutionStatus.STARTED
        clock.now += 3
        assert await service.refresh("session", principal=_PRINCIPAL, max_age=3) == strong
    finally:
        await state.close()


@pytest.mark.parametrize("status,released", (
    (ExecutionStatus.SUCCEEDED, False),
    (ExecutionStatus.FAILED, False),
    (ExecutionStatus.CANCELLED, False),
    (ExecutionStatus.CANCELLED, True),
))
async def test_display_cache_terminal_active_refreshes_complete_page(status, released):
    state, cache = RuntimeStorage.in_memory(), _Cache()
    state = RuntimeState(state)
    session = await _prepare(state)
    try:
        service = _service(state, cache)
        first = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        assert first.session.active_execution_id == "active"
        executions = state.execution.executions
        record = await executions.get("active", tenant_id="tenant")
        await executions.compare_and_swap(
            "active", tenant_id="tenant", expected_revision=record.revision,
            next_record=replace(record, revision=record.revision + 1, status=status,
                result=_execution("active", status=status, prompt="active").result,
                safe_error_details={"finished": True}),
        )
        if status is ExecutionStatus.SUCCEEDED:
            run_id, count = await _materialize_conversation(state, session.history_id)
            sessions = state.conversation.sessions
            execution = await executions.get("active", tenant_id="tenant")

            async def commit(transaction):
                await sessions.commit_timeline_turn_in_transaction(
                    transaction, "session", tenant_id="tenant", execution_id="active",
                    start_message_index=0, end_message_index=count,
                    **_commit_fields(execution),
                )
                await sessions.advance_continuation_in_transaction(
                    transaction, "session", tenant_id="tenant", execution_id="active", expected=None,
                    next_cursor=ConversationCursor(run_id, history_id=session.history_id, message_count=count),
                )
            await sessions.state_store.mutate(commit)
        else:
            await _commit_empty_turn(state, await executions.get("active", tenant_id="tenant"))
        if released:
            await state.conversation.sessions.release_execution("session", tenant_id="tenant", execution_id="active")
        fresh = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        assert fresh.session.active_execution_id is None
        assert fresh.timeline.items[0].status is status
        assert fresh.timeline.items[0].safe_error_details == {"finished": True}
        assert fresh.timeline.items[0].conversation_committed
        if status is ExecutionStatus.SUCCEEDED:
            assert fresh.timeline.items[0].items
        else:
            assert fresh.timeline.items[0].items == ()
        assert (await state.conversation.sessions.get("session", tenant_id="tenant")).active_execution_id is None
        assert fresh == await service.refresh("session", principal=_PRINCIPAL)
        assert cache.writes == 2
        assert await service.refresh("session", principal=_PRINCIPAL, max_age=3) == fresh
        assert cache.writes == 2
    finally:
        await state.close()


@pytest.mark.parametrize("had_active,next_active", ((True, None), (True, "second"), (False, "second")))
async def test_display_cache_active_binding_change_refreshes_complete_page(had_active, next_active):
    state, cache = RuntimeStorage.in_memory(), _Cache()
    state = RuntimeState(state)
    await _prepare(state)
    try:
        if not had_active:
            await state.conversation.sessions.release_execution("session", tenant_id="tenant", execution_id="active")
        service = _service(state, cache)
        first = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        assert first.session.active_execution_id == ("active" if had_active else None)
        if had_active:
            await state.conversation.sessions.release_execution("session", tenant_id="tenant", execution_id="active")
        if next_active is not None:
            await _admit(state, next_active, ExecutionStatus.PENDING_START)
        fresh = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        assert fresh.session.active_execution_id == next_active
        assert fresh == await service.refresh("session", principal=_PRINCIPAL)
        assert cache.writes == 2
        assert await service.refresh("session", principal=_PRINCIPAL, max_age=3) == fresh
        assert cache.writes == 2
    finally:
        await state.close()


async def test_display_cache_idle_completion_observes_continuation_revision():
    state, cache = RuntimeStorage.in_memory(), _Cache()
    state = RuntimeState(state)
    session = await _prepare(state)
    sessions = state.conversation.sessions
    try:
        await sessions.release_execution("session", tenant_id="tenant", execution_id="active")
        service = _service(state, cache)
        first = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        assert first.session.active_execution_id is None
        await _admit(state, "second", ExecutionStatus.SUCCEEDED)
        run_id, count = await _materialize_conversation(state, session.history_id)
        execution = await state.execution.executions.get("second", tenant_id="tenant")

        async def commit(transaction):
            await sessions.commit_timeline_turn_in_transaction(
                transaction, "session", tenant_id="tenant", execution_id="second",
                start_message_index=0, end_message_index=count,
                **_commit_fields(execution),
            )
            return await sessions.advance_continuation_in_transaction(
                transaction, "session", tenant_id="tenant", execution_id="second", expected=None,
                next_cursor=ConversationCursor(run_id, history_id=session.history_id, message_count=count),
                release_execution=True,
            )
        committed = await sessions.state_store.mutate(commit)
        assert committed.active_execution_id is None
        assert committed.revision == first.session.revision + 1
        fresh = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        assert fresh.session.revision == committed.revision
        assert fresh.session.active_execution_id is None
        assert [turn.execution_id for turn in fresh.timeline.items] == ["active", "second"]
        assert fresh.timeline.items[1].conversation_committed
        assert fresh.timeline.items[1].items
        assert fresh == await service.refresh("session", principal=_PRINCIPAL)
        assert cache.writes == 2
    finally:
        await state.close()


@pytest.mark.parametrize("kind,code", (("execution", None), ("conversation_history", ErrorCode.SESSION_HISTORY_UNAVAILABLE)))
async def test_display_cache_rechecks_dependency_existence(kind, code):
    state, cache = RuntimeStorage.in_memory(), _Cache()
    state = RuntimeState(state)
    await _prepare(state, body=True)
    try:
        service = _service(state, cache)
        before = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        store = state.execution.executions.state_store if kind == "execution" else state.conversation.sessions.state_store

        async def remove(transaction):
            records = await transaction.scan_records()
            await transaction.delete_records(tuple(record.key_digest for record in records if record.kind == kind))
        await store.mutate(remove)
        if kind == "execution":
            assert await service.refresh("session", principal=_PRINCIPAL, max_age=3) == before
            assert await service.refresh("session", principal=_PRINCIPAL) == before
        else:
            with pytest.raises(AIError) as error:
                await service.refresh("session", principal=_PRINCIPAL, max_age=3)
            assert error.value.code is code
    finally:
        await state.close()


async def test_display_cache_independent_body_removal_is_bounded_by_max_age(monkeypatch):
    from linktools.ai.runtime import RuntimeDomain
    from linktools.ai.runtime.state._store import FactQuery

    clock = _clock(monkeypatch)
    state, cache = RuntimeStorage.in_memory(), _Cache()
    state = RuntimeState(state)
    session = await _prepare(state, body=True)
    try:
        service = _service(state, cache)
        first = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        archive = state.run_store.read_store(RuntimeDomain.CONVERSATION)
        stream = archive.transcript_repository.transcript_stream(session.history_id)
        facts = await archive.state_store.read(lambda tx: tx.list_facts(FactQuery(stream)))
        assert facts

        async def remove(transaction):
            for owner in {fact.owner_key_digest for fact in facts}:
                await transaction.delete_fact_streams(owner)
        await archive.state_store.mutate(remove)
        assert await service.refresh("session", principal=_PRINCIPAL, max_age=3) == first
        clock.now += 3
        with pytest.raises(AIError) as error:
            await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


async def test_display_cache_input_cursor_expiry_is_not_extended(monkeypatch):
    import time
    import linktools.ai.core._paging as paging_module
    from .test_session_refresh import _release

    state, cache = RuntimeStorage.in_memory(), _Cache()
    state = RuntimeState(state)
    await _prepare(state)
    try:
        await _release(state, "active")
        await _admit(state, "second", ExecutionStatus.PENDING_START)
        service = _service(state, cache)
        page = await service.timeline("session", principal=_PRINCIPAL, limit=1)
        signer = HmacCursorSigner("display", b"display-test-key")
        now = int(time.time())
        cursor = signer.encode(replace(signer.decode(page.next_cursor), expires_at=now + 1))
        await service.refresh("session", principal=_PRINCIPAL, limit=1, cursor=cursor, max_age=3)
        assert cache.writes == 1
        monkeypatch.setattr(paging_module, "time", SimpleNamespace(time=lambda: now + 2))
        with pytest.raises(AIError) as error:
            await service.refresh("session", principal=_PRINCIPAL, limit=1, cursor=cursor, max_age=3)
        assert error.value.code is ErrorCode.CURSOR_INVALID
    finally:
        await state.close()


async def test_display_cache_forks_and_ancestor_cursor_preserve_fresh_deletion():
    import json
    from .test_session_refresh import _release

    state, cache = RuntimeStorage.in_memory(), _Cache()
    state = RuntimeState(state)
    await _prepare(state)
    try:
        await _release(state, "active")
        sessions = state.conversation.sessions
        root = await sessions.get("session", tenant_id="tenant")
        await sessions.create(replace(_session(), session_id="child", timeline_parent_session_id="session", timeline_parent_turn_seq=1))
        service = _service(state, cache)
        first = await service.refresh("child", principal=_PRINCIPAL, max_age=3)
        assert first.timeline.items[0].execution_id == "active"
        assert cache.writes == 0
        # A valid signed cursor can explicitly name another source; keep it on
        # the original path even if the requested root itself is not forked.
        await _admit(state, "second", ExecutionStatus.PENDING_START)
        page = await service.timeline("session", principal=_PRINCIPAL, limit=1)
        signer = HmacCursorSigner("display", b"display-test-key")
        cursor = signer.encode(replace(signer.decode(page.next_cursor), position=json.dumps(["child", 1])))
        await service.refresh("session", principal=_PRINCIPAL, cursor=cursor, max_age=3)
        assert cache.writes == 0

        # Delete through the actual physical session key selected by its header.
        from linktools.ai.runtime.state._store import record_key_digest
        from linktools.ai.runtime import RuntimeDomain
        key = record_key_digest(sessions.namespace, sessions.tenant_id, RuntimeDomain.CONVERSATION.value, "session", root.session_id)
        await sessions.state_store.mutate(lambda tx: tx.delete_record(key))
        with pytest.raises(AIError) as error:
            await service.refresh("child", principal=_PRINCIPAL, max_age=3)
        assert error.value.code is ErrorCode.SESSION_HISTORY_UNAVAILABLE
    finally:
        await state.close()


async def test_display_cache_valid_json_byte_corruption_is_a_miss():
    import json
    from linktools.ai.runtime.state._codec import decode_domain, encode_domain

    state, cache = RuntimeStorage.in_memory(), _Cache()
    state = RuntimeState(state)
    await _prepare(state, body=True)
    try:
        service = _service(state, cache)
        expected = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        key = next(iter(cache.keys))
        envelope = json.loads(await cache.get(key))
        entry = decode_domain(envelope["payload"], dict[str, object])
        entry["detail"]["timeline"]["items"][0]["user_input"]["prompt"]["text"] = "different bytes"
        envelope["payload"] = encode_domain(entry)
        await cache.put(key, json.dumps(envelope).encode())
        assert await service.refresh("session", principal=_PRINCIPAL, max_age=3) == expected
        assert cache.writes == 3
    finally:
        await state.close()


async def test_display_cache_principal_and_page_shapes_do_not_share_entries():
    from .test_session_refresh import _release

    state, cache = RuntimeStorage.in_memory(), _Cache()
    state = RuntimeState(state)
    await _prepare(state)
    try:
        await _release(state, "active")
        await _admit(state, "second", ExecutionStatus.PENDING_START)

        class Allow:
            async def authorize(self, *args):
                return None
        service = _service(state, cache, Allow())
        await service.refresh("session", principal=_PRINCIPAL, max_age=3, limit=1)
        await _change_metadata(state, "new")
        other = await service.refresh("session", principal=Principal("other", "tenant"), max_age=3, limit=1)
        assert other.session.metadata == {"value": "new"}
        page = await service.refresh("session", principal=_PRINCIPAL, max_age=3, limit=2)
        assert page.session.metadata == {"value": "new"}
        assert len(page.timeline.items) == 2
        assert len(cache.keys) == 3
    finally:
        await state.close()


async def test_display_cache_same_id_recreation_does_not_reuse_old_page():
    from datetime import timedelta
    from linktools.ai.runtime import RuntimeDomain
    from linktools.ai.runtime.state._store import record_key_digest

    state, cache = RuntimeStorage.in_memory(), _Cache()
    state = RuntimeState(state)
    record = await _prepare(state)
    try:
        service = _service(state, cache)
        await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        sessions = state.conversation.sessions
        key = record_key_digest(sessions.namespace, sessions.tenant_id, RuntimeDomain.CONVERSATION.value, "session", "session")
        history = record.history_id
        keys = (key,) + tuple(record_key_digest(sessions.namespace, sessions.tenant_id, RuntimeDomain.CONVERSATION.value, kind, history)
                              for kind in ("conversation_history", "transcript_head"))
        from linktools.ai.runtime.state._store import sequence_key
        async def remove(transaction):
            await transaction.delete_records(keys)
            await transaction.delete_sequence(sequence_key(sessions.namespace, sessions.tenant_id, RuntimeDomain.CONVERSATION.value, "session_turn", ["session"]))
        await sessions.state_store.mutate(remove)
        await sessions.create(replace(record, created_at=record.created_at + timedelta(seconds=1), metadata={"rebuilt": True}))
        rebuilt = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        assert rebuilt.session.metadata == {"rebuilt": True}
    finally:
        await state.close()


async def test_display_cache_concurrent_misses_keep_original_capture_deadline(monkeypatch):
    clock = _clock(monkeypatch)
    state, cache = RuntimeStorage.in_memory(), _Cache()
    state = RuntimeState(state)
    await _prepare(state)
    entered, proceed = asyncio.Event(), asyncio.Event()
    put = cache.put
    calls = []

    async def blocked_put(key, content):
        calls.append(key)
        if len(calls) == 1:
            entered.set()
            await proceed.wait()
        await put(key, content)
    monkeypatch.setattr(cache, "put", blocked_put)
    try:
        service = _service(state, cache)
        first = asyncio.create_task(service.refresh("session", principal=_PRINCIPAL, max_age=3))
        await asyncio.wait_for(entered.wait(), 5)
        clock.now += 1
        await _change_execution_display(state, "second")
        await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        proceed.set()
        await first
        clock.now += 2
        assert (await service.refresh("session", principal=_PRINCIPAL, max_age=3)).timeline.items[0].safe_error_details == {"value": "second"}
    finally:
        proceed.set()
        await state.close()


async def test_storage_exposes_borrowed_cache_without_closing_it():
    cache = _Cache()
    state = RuntimeStorage.in_memory(cache=cache)
    assert state.plan.route(RuntimeDomain.CONVERSATION).cache is cache
    owner = RuntimeState(state)
    await owner.initialize(namespace="borrowed-cache", tenant_id="tenant")
    await owner.close()
    await cache.put("still-borrowed", b"content")
    assert await cache.get("still-borrowed") == b"content"


@pytest.mark.parametrize("retention", ("durable", "volatile", "transient"))
async def test_runtime_factory_borrows_display_cache_only_for_durable_domains(tmp_path, retention):
    from pydantic_ai.models.test import TestModel
    from linktools.ai.runtime import RuntimeFeatures
    from linktools.ai.runtime import Runtime, RuntimeContext
    from ._runtime_test_helpers import _Models

    cache = _Cache()
    if retention == "durable":
        storage = RuntimeStorage.sqlite(tmp_path / "factory.db", cache=cache)
    elif retention == "volatile":
        storage = RuntimeStorage.in_memory(cache=cache)
    else:
        storage = RuntimeStorage.from_plan(RuntimeStoragePlan(execution=StorageRoute.transient(cache=cache)))
    group = RuntimeFeatures[None]("display")
    group.capability(ExecutionCapability(retain_history=True), scope="runtime")
    group.agent("agent", model="default", allow_tools=())
    async with Runtime.open(
        "display", storage=storage, models=_Models(TestModel()), features=group,
        context=RuntimeContext(None, tenant_id="tenant"),
    ) as runtime:
        await runtime._state.conversation.sessions.create(_session())
        await _admit(runtime._state, status=ExecutionStatus.PENDING_START)
        first = await runtime.sessions.refresh("session", principal=_PRINCIPAL, max_age=3)
        await _change_execution_display(runtime._state, "changed")
        second = await runtime.sessions.refresh("session", principal=_PRINCIPAL, max_age=3)
        if retention == "durable":
            assert second == first
            assert len([key for key in cache.keys if key.startswith("session-display:")]) == 1
        else:
            assert second.timeline.items[0].safe_error_details == {"value": "changed"}
            assert not any(key.startswith("session-display:") for key in cache.keys)
    await cache.put("after-close", b"host-owned")
    assert await cache.get("after-close") == b"host-owned"
    with pytest.raises(AIError):
        await runtime.sessions.refresh("session", principal=_PRINCIPAL, max_age=3)


async def test_display_cache_custom_repository_keeps_original_read_path():
    from linktools.ai.runtime.state._conversation_repositories import SessionRepositoryImpl

    class CustomSessions(SessionRepositoryImpl):
        pass

    state, cache = RuntimeStorage.in_memory(), _Cache()
    state = RuntimeState(state)
    await _prepare(state)
    try:
        sessions = state.conversation.sessions
        custom = CustomSessions(sessions.state_store, namespace=sessions.namespace, tenant_id=sessions.tenant_id)
        service = RuntimeSessionService(
            replace(state.conversation, sessions=custom), state.execution.executions,
            TenantAuthorizationPolicy(), object(), HmacCursorSigner("display", b"display-test-key"),
            history_reader=object(), transcript_store=state.run_store.conversation_persistence, cache=cache,
        )
        await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        await _change_metadata(state, "changed")
        assert (await service.refresh("session", principal=_PRINCIPAL, max_age=3)).session.metadata == {"value": "changed"}
        assert cache.reads == cache.writes == 0
    finally:
        await state.close()


@pytest.mark.parametrize("dependency", ("foreign_session", "child_execution"))
async def test_display_cache_rejects_execution_dependencies_outside_authorized_top_level(dependency):
    import json
    from linktools.ai.core import ExecutionLineageKind
    from linktools.ai.runtime.state._codec import decode_domain, encode_domain
    from .test_session_timeline import _execution

    state, cache = RuntimeStorage.in_memory(), _Cache()
    state = RuntimeState(state)
    await _prepare(state)
    try:
        service = _service(state, cache)
        expected = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        key = next(iter(cache.keys))
        record = _execution("foreign-run", status=ExecutionStatus.PENDING_START, prompt="private prompt")
        if dependency == "foreign_session":
            record = replace(record, session_id="foreign", principal_id="other")
        else:
            record = replace(record, parent_execution_id="active", parent_invocation_id="invocation",
                             root_execution_id="active", lineage_kind=ExecutionLineageKind.SUBAGENT)
        await state.execution.executions.create(record)
        envelope = json.loads(await cache.get(key))
        entry = decode_domain(envelope["payload"], dict[str, object])
        # Preserve a true identity from the source but attach it to the wrong page.
        entry["executions"][record.execution_id] = (
            record.execution_id, record.session_id, record.parent_execution_id, record.created_at,
            record.principal_id, record.principal_kind, record.binding_digest,
        )
        turn = entry["detail"]["timeline"]["items"][0]
        turn["execution_id"] = record.execution_id
        turn["user_input"] = {"private": "prompt"}
        payload = encode_domain(entry)
        await cache.put(key, json.dumps({"payload": payload, "digest": canonical_sha256(payload)}).encode())
        assert await service.refresh("session", principal=_PRINCIPAL, max_age=3) == expected
        assert cache.writes == 3
    finally:
        await state.close()


@pytest.mark.parametrize("phase", ("get", "put"))
@pytest.mark.parametrize("cancel_read", (False, True))
async def test_runtime_close_drains_display_io_after_close_waiter_cancellation(tmp_path, monkeypatch, phase, cancel_read):
    from pydantic_ai.models.test import TestModel
    from linktools.ai.runtime import RuntimeFeatures
    from linktools.ai.runtime import Runtime, RuntimeContext
    from ._runtime_test_helpers import _Models

    cache = _Cache()
    storage = RuntimeStorage.sqlite(tmp_path / "close.db", cache=cache)
    group = RuntimeFeatures[None]("display")
    group.capability(ExecutionCapability(retain_history=True), scope="runtime")
    group.agent("agent", model="default", allow_tools=())
    entered, release, exited = asyncio.Event(), asyncio.Event(), asyncio.Event()
    method = getattr(cache, phase)

    async def blocked(*args):
        entered.set()
        try:
            await release.wait()
            return await method(*args)
        finally:
            exited.set()
    async with Runtime.open(
        "display", storage=storage, models=_Models(TestModel()), features=group,
        context=RuntimeContext(None, tenant_id="tenant"),
    ) as runtime:
        await runtime._state.conversation.sessions.create(_session())
        monkeypatch.setattr(cache, phase, blocked)
        read = asyncio.create_task(runtime.sessions.refresh("session", principal=_PRINCIPAL, max_age=3))
        await asyncio.wait_for(entered.wait(), 5)
        close = asyncio.create_task(runtime.close())
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(close), 0.02)
        assert not read.done()
        assert runtime._state.ready
        with pytest.raises(AIError):
            await runtime.sessions.refresh("session", principal=_PRINCIPAL, max_age=3)
        close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close
        again = asyncio.create_task(runtime.close())
        await asyncio.sleep(0)
        assert not again.done()
        if cancel_read:
            read.cancel()
            with pytest.raises(asyncio.CancelledError):
                await read
        else:
            release.set()
            assert (await read).session.session_id == "session"
        await asyncio.wait_for(again, 5)
        assert exited.is_set()
        assert not runtime._state.ready
        monkeypatch.setattr(cache, phase, method)
    await cache.put("after-close", b"host-owned")
    assert await cache.get("after-close") == b"host-owned"


async def test_session_display_drain_rejects_new_reads_and_preserves_original_error(monkeypatch):
    state, cache = RuntimeStorage.in_memory(), _Cache()
    state = RuntimeState(state)
    await _prepare(state)
    service = _service(state, cache)
    entered, release = asyncio.Event(), asyncio.Event()

    async def get_header(*args, **kwargs):
        entered.set()
        await release.wait()
        raise RuntimeError("source header failure")
    monkeypatch.setattr(state.conversation.sessions, "get_header", get_header)
    try:
        read = asyncio.create_task(service.refresh("session", principal=_PRINCIPAL, max_age=3))
        await asyncio.wait_for(entered.wait(), 5)
        drain = asyncio.create_task(service.drain_display_reads())
        await asyncio.sleep(0)
        assert not drain.done()
        with pytest.raises(AIError) as error:
            await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        assert error.value.code is ErrorCode.STORAGE_CLOSED
        release.set()
        with pytest.raises(RuntimeError, match="source header failure"):
            await read
        await asyncio.wait_for(drain, 5)
        await service.drain_display_reads()
    finally:
        release.set()
        await state.close()


@pytest.mark.parametrize("cache", (WriteOnlyCache(), InMemoryStreamBackend(max_bytes=1024)))
async def test_display_refresh_without_key_value_cache_keeps_fresh_source(cache: object) -> None:
    with pytest.raises(TypeError, match="cache must provide get, put and delete"):
        RuntimeStorage.in_memory(cache=cache)
    state = RuntimeStorage.in_memory()
    state = RuntimeState(state)
    await _prepare(state)
    try:
        service = _service(state, cache)
        await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        await _change_execution_display(state, "changed")
        fresh = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        assert fresh.timeline.items[0].safe_error_details == {"value": "changed"}
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
async def test_display_cache_is_shared_across_readers_without_renewing_capture(
    tmp_path, monkeypatch, backend,
):
    clock = _clock(monkeypatch)
    cache = _Cache()
    first_state = _storage(backend, tmp_path)
    first_state = RuntimeState(first_state)
    await _prepare(first_state, body=True)
    first = _service(first_state, cache)
    detail = await first.refresh("session", principal=_PRINCIPAL, max_age=3)
    await first_state.close()
    second_state = _storage(backend, tmp_path)
    second_state = RuntimeState(second_state)
    await second_state.initialize(namespace="display", tenant_id="tenant")
    try:
        second = _service(second_state, cache)
        writes = cache.writes
        captures = 0
        capture = session_module.capture_session_refresh

        async def count_capture(*args, **kwargs):
            nonlocal captures
            captures += 1
            return await capture(*args, **kwargs)

        monkeypatch.setattr(session_module, "capture_session_refresh", count_capture)
        clock.now += 2
        assert await second.refresh("session", principal=_PRINCIPAL, max_age=3) == detail
        assert captures == 0 and cache.writes == writes
        clock.now += 1
        assert await second.refresh("session", principal=_PRINCIPAL, max_age=3) == detail
        assert captures == 1 and cache.writes == writes + 1
    finally:
        await second_state.close()
        await first_state.close()


@pytest.mark.parametrize("warm_page", (False, True))
async def test_display_rebuild_does_not_stack_body_cache_expiry(tmp_path, monkeypatch, warm_page):
    from linktools.ai.runtime import RuntimeDomain
    from linktools.ai.runtime.state import _archive
    from linktools.ai.runtime.state._store import FactQuery
    from linktools.ai.storage import _write_cache

    clock = _clock(monkeypatch)
    monkeypatch.setattr(_write_cache, "time", lambda: clock.now)
    monkeypatch.setattr(_archive, "PAYLOAD_COMPRESS_MINIMUM", 0)
    cache = _Cache()
    state = RuntimeStorage.sqlite(tmp_path / "display.db", cache=cache)
    state = RuntimeState(state)
    session = await _prepare(state, body=True)
    try:
        service = _service(state, cache)
        archive = state.run_store.read_store(RuntimeDomain.CONVERSATION)
        transcript = archive.transcript_repository
        stream = transcript.transcript_stream(session.history_id)
        facts = await archive.state_store.read(lambda tx: tx.list_facts(FactQuery(stream)))
        contents = await transcript.content_archive()
        content_ids = []
        for fact in facts:
            content_ids.extend(await contents.content_ids(transcript.decode_chunk(fact).content))
        records = await contents.content_records(content_ids)
        references = [record.payload.ref for record in records]
        assert references and all(reference is not None for reference in references)
        if warm_page:
            detail = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        else:
            await transcript.load_messages(session.history_id)
        objects = state.object_store(RuntimeDomain.CONVERSATION)
        for reference in references:
            await objects.delete_object(reference.key, expected_digest=reference.digest)
        clock.now += 2
        if warm_page:
            assert await service.refresh("session", principal=_PRINCIPAL, max_age=3) == detail
        else:
            with pytest.raises(AIError) as raised:
                await service.refresh("session", principal=_PRINCIPAL, max_age=3)
            assert raised.value.code is ErrorCode.STORAGE_NOT_FOUND
        with pytest.raises(AIError) as raised:
            await service.refresh("session", principal=_PRINCIPAL, max_age=0)
        assert raised.value.code is ErrorCode.STORAGE_NOT_FOUND
        clock.now += 1
        with pytest.raises(AIError) as raised:
            await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        assert raised.value.code is ErrorCode.STORAGE_NOT_FOUND
    finally:
        await state.close()
