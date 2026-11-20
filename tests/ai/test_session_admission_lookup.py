#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recover accepted session requests without rebuilding mutable input context."""

import asyncio
import sqlite3
from contextlib import closing
from contextvars import ContextVar
from dataclasses import replace

import pytest

from linktools.ai.core import ExecutionStatus, IdempotencyStatus, Principal, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence.state import active_state_scope
from linktools.ai.storage import InMemoryCacheBackend, TurnScopeIdentity
from linktools.ai.storage._database import SqlStorageContext

from .test_pending_session_projection import _runtimes

pytestmark = pytest.mark.asyncio


async def test_lost_response_returns_frozen_input_without_rebuilding_context(tmp_path, monkeypatch):
    async with _runtimes(tmp_path, cache=None, shared=False) as (runtimes, *_):
        runtime, _ = runtimes[0]
        session = await runtime.agents.get().create_session("conversation")
        prompt = '{"question":"q","context_revision":1}'
        execution = await session.start(prompt, idempotency_key="request-1")
        await execution.wait()
        with pytest.raises(AIError) as conflict:
            await session.start('{"question":"q","context_revision":2}', idempotency_key="request-1")
        assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT

        async def no_write(*args, **kwargs):
            pytest.fail("admission lookup must not mutate durable state")

        with monkeypatch.context() as patch:
            patch.setattr(SqlStorageContext, "run_mutation", no_write)
            admission = await runtime.sessions.lookup_admission(
                "conversation", principal=runtime.default_principal, idempotency_key="request-1",
            )
            assert admission.execution_id == execution.execution_id
            assert admission.user_input == prompt
            assert await runtime.sessions.lookup_admission(
                "conversation", principal=runtime.default_principal, idempotency_key="absent",
            ) is None


@pytest.mark.parametrize("shared,source_authority", [(False, False), (True, False), (True, True)])
async def test_pending_admission_lookup_reads_local_and_shared_without_settling(tmp_path, monkeypatch, shared, source_authority):
    cache = InMemoryCacheBackend(max_bytes=2**24) if shared else None
    async with _runtimes(tmp_path, cache=cache, shared=shared, source_authority=source_authority) as (runtimes, entered, finish, block):
        owner, _ = runtimes[0]
        session = await owner.agents.get().create_session("conversation")
        block()
        execution = await session.start("accepted input", idempotency_key="pending")
        await asyncio.wait_for(entered.wait(), 5)
        async def no_settle(*args, **kwargs):
            pytest.fail("read-only lookup must not settle admission")
        with monkeypatch.context() as patch:
            patch.setattr(owner._execution_service._runtime_bridge.pending_turns, "settle", no_settle)
            for runtime, _ in runtimes:
                found = await runtime.sessions.lookup_admission(
                    "conversation", principal=runtime.default_principal, idempotency_key="pending",
                )
                assert found.execution_id == execution.execution_id
                assert found.user_input == "accepted input"
        finish.set()
        await execution.wait()


async def test_admission_lookup_checks_session_and_execution_authorization(tmp_path):
    async with _runtimes(tmp_path, cache=None, shared=False) as (runtimes, *_):
        runtime, _ = runtimes[0]
        first = await runtime.agents.get().create_session("first")
        await runtime.agents.get().create_session("second")
        execution = await first.start("private input", idempotency_key="same-key")
        await execution.wait()
        with pytest.raises(AIError) as cross_session:
            await runtime.sessions.lookup_admission("second", principal=runtime.default_principal, idempotency_key="same-key")
        assert cross_session.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        with pytest.raises(AIError) as foreign:
            await runtime.sessions.lookup_admission("first", principal=Principal("other", "other"), idempotency_key="same-key")
        assert foreign.value.code is ErrorCode.AUTHORIZATION_DENIED


@pytest.mark.parametrize("status", [IdempotencyStatus.RESERVED, IdempotencyStatus.START_UNKNOWN])
async def test_uncertain_admission_is_not_absence(tmp_path, monkeypatch, status):
    async with _runtimes(tmp_path, cache=None, shared=False) as (runtimes, *_):
        runtime, storage = runtimes[0]
        session = await runtime.agents.get().create_session("conversation")
        execution = await session.start("input", idempotency_key="request")
        await execution.wait()
        repo = storage.execution.idempotency
        original = await repo.get("session.resume", "request", tenant_id=runtime.tenant_id)
        async def uncertain(*args, **kwargs):
            return replace(original, status=status)
        with monkeypatch.context() as patch:
            patch.setattr(repo, "get", uncertain)
            with pytest.raises(AIError) as error:
                await runtime.sessions.lookup_admission("conversation", principal=runtime.default_principal, idempotency_key="request")
            assert error.value.code is ErrorCode.EXECUTION_START_UNKNOWN


async def test_lookup_survives_persistence_reopen(tmp_path):
    async with _runtimes(tmp_path, cache=None, shared=False) as (runtimes, *_):
        runtime, _ = runtimes[0]
        session = await runtime.agents.get().create_session("conversation")
        execution = await session.start("frozen input", idempotency_key="request")
        await execution.wait()
        execution_id = execution.execution_id
    async with _runtimes(tmp_path, cache=None, shared=False) as (runtimes, *_):
        runtime, _ = runtimes[0]
        admission = await runtime.sessions.lookup_admission(
            "conversation", principal=runtime.default_principal, idempotency_key="request",
        )
        assert admission.execution_id == execution_id
        assert admission.user_input == "frozen input"


@pytest.mark.parametrize("damage", ["execution", "input", "resource_kind"])
async def test_broken_admission_is_not_absence(tmp_path, monkeypatch, damage):
    from linktools.ai.core import ResourceKind
    async with _runtimes(tmp_path, cache=None, shared=False) as (runtimes, *_):
        runtime, storage = runtimes[0]
        session = await runtime.agents.get().create_session("conversation")
        execution = await session.start("input", idempotency_key="request")
        await execution.wait()
        service = runtime._execution_service
        original = await service.load_source_execution(execution.execution_id, tenant_id=runtime.tenant_id)
        with monkeypatch.context() as patch:
            if damage == "resource_kind":
                repo = storage.execution.idempotency
                identity = await repo.get("session.resume", "request", tenant_id=runtime.tenant_id)
                async def broken_identity(*args, **kwargs):
                    return replace(identity, resource_kind=ResourceKind.SESSION)
                patch.setattr(repo, "get", broken_identity)
            else:
                async def broken_execution(*args, **kwargs):
                    if damage == "execution":
                        return None
                    corrupt = replace(original)
                    object.__setattr__(corrupt, "stored_user_input", None)
                    return corrupt
                patch.setattr(service, "load_source_execution", broken_execution)
            with pytest.raises(AIError) as error:
                await runtime.sessions.lookup_admission("conversation", principal=runtime.default_principal, idempotency_key="request")
            assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


async def test_lookup_does_not_bypass_execution_read_policy(tmp_path, monkeypatch):
    from linktools.ai.core import ResourceKind
    async with _runtimes(tmp_path, cache=None, shared=False) as (runtimes, *_):
        runtime, _ = runtimes[0]
        session = await runtime.agents.get().create_session("conversation")
        execution = await session.start("private input", idempotency_key="request")
        await execution.wait()
        original = runtime._execution_service._authorization.authorize
        async def deny_execution(principal, action, resource):
            if resource.kind is ResourceKind.EXECUTION:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            return await original(principal, action, resource)
        with monkeypatch.context() as patch:
            patch.setattr(runtime._execution_service._authorization, "authorize", deny_execution)
            with pytest.raises(AIError) as error:
                await runtime.sessions.lookup_admission("conversation", principal=runtime.default_principal, idempotency_key="request")
            assert error.value.code is ErrorCode.AUTHORIZATION_DENIED


async def test_lookup_absence_race_does_not_change_admission_idempotency(tmp_path):
    async with _runtimes(tmp_path, cache=None, shared=False) as (runtimes, *_):
        runtime, _ = runtimes[0]
        session = await runtime.agents.get().create_session("conversation")
        assert await runtime.sessions.lookup_admission(
            "conversation", principal=runtime.default_principal, idempotency_key="request",
        ) is None
        winner = await session.start("original", idempotency_key="request")
        await winner.wait()
        with pytest.raises(AIError) as changed_request:
            await session.start("changed", idempotency_key="request")
        assert changed_request.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        admission = await runtime.sessions.lookup_admission(
            "conversation", principal=runtime.default_principal, idempotency_key="request",
        )
        assert admission.execution_id == winner.execution_id
        assert admission.user_input == "original"


async def test_uncertain_local_turn_does_not_settle_or_report_absence(tmp_path):
    async with _runtimes(tmp_path, cache=InMemoryCacheBackend(max_bytes=2**24), shared=False) as (runtimes, entered, finish, block):
        runtime, _ = runtimes[0]
        session = await runtime.agents.get().create_session("conversation")
        block()
        execution = await session.start("input", idempotency_key="request")
        await asyncio.wait_for(entered.wait(), 5)
        turn = runtime._execution_service._runtime_bridge.pending_turns.get(
            execution.execution_id, tenant_id=runtime.tenant_id,
        )
        turn.uncertain = True
        async def no_reconcile():
            pytest.fail("lookup must not reconcile")
        turn.reconcile = no_reconcile
        try:
            with pytest.raises(AIError) as error:
                await runtime.sessions.lookup_admission("conversation", principal=runtime.default_principal, idempotency_key="request")
            assert error.value.code is ErrorCode.EXECUTION_START_UNKNOWN
        finally:
            turn.uncertain = False
            turn.reconcile = None
            finish.set()
        await execution.wait()


@pytest.mark.parametrize("has_record", [False, True])
@pytest.mark.parametrize("status,code", [
    (IdempotencyStatus.FAILED, ErrorCode.EXECUTION_START_PERSISTENCE_FAILED),
    (IdempotencyStatus.CANCELLED, ErrorCode.EXECUTION_CANCELLED),
])
async def test_failed_start_retains_known_outcome_instead_of_unknown(tmp_path, monkeypatch, status, code, has_record):
    async with _runtimes(tmp_path, cache=None, shared=False) as (runtimes, *_):
        runtime, storage = runtimes[0]
        session = await runtime.agents.get().create_session("conversation")
        execution = await session.start("input", idempotency_key="request")
        await execution.wait()
        service = runtime._execution_service
        record = await service.load_source_execution(execution.execution_id, tenant_id=runtime.tenant_id)
        repo = storage.execution.idempotency
        identity = await repo.get("session.resume", "request", tenant_id=runtime.tenant_id)
        async def failed_identity(*args, **kwargs):
            return replace(identity, status=status, error_code=code.value)
        async def failed_execution(*args, **kwargs):
            return replace(
                record, started_at=None, status=ExecutionStatus(status.value),
                error_code=code.value, result=None, completed_at=None,
            ) if has_record else None
        with monkeypatch.context() as patch:
            patch.setattr(repo, "get", failed_identity)
            patch.setattr(service, "load_source_execution", failed_execution)
            with pytest.raises(AIError) as error:
                await runtime.sessions.lookup_admission("conversation", principal=runtime.default_principal, idempotency_key="request")
            assert error.value.code is code


async def test_terminal_admission_keeps_result_integrity_check(tmp_path, monkeypatch):
    async with _runtimes(tmp_path, cache=None, shared=False) as (runtimes, *_):
        runtime, storage = runtimes[0]
        session = await runtime.agents.get().create_session("conversation")
        execution = await session.start("input", idempotency_key="request")
        await execution.wait()
        repo = storage.execution.idempotency
        identity = await repo.get("session.resume", "request", tenant_id=runtime.tenant_id)
        async def damaged(*args, **kwargs):
            return replace(identity, result_digest="0" * 64)
        with monkeypatch.context() as patch:
            patch.setattr(repo, "get", damaged)
            with pytest.raises(AIError) as error:
                await runtime.sessions.lookup_admission("conversation", principal=runtime.default_principal, idempotency_key="request")
            assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("source_authority, pause_at", [
    (False, "owner"), (False, "cache"), (True, "cache"), (True, "receipt"),
])
async def test_receipt_wins_when_remote_turn_finishes_during_lookup(tmp_path, monkeypatch, source_authority, pause_at):
    cache = InMemoryCacheBackend(max_bytes=2**24)
    async with _runtimes(tmp_path, cache=cache, source_authority=source_authority) as (runtimes, entered, finish, block):
        if pause_at == "receipt":
            # Allow the owner to settle while the other runtime holds a read snapshot.
            with closing(sqlite3.connect(tmp_path / "state.db")) as connection:
                assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        owner, _ = runtimes[0]
        reader, state = runtimes[1]
        principal = reader.default_principal
        session = await owner.agents.get().create_session("conversation")
        block()
        execution = await session.start("frozen input", idempotency_key="request")
        await asyncio.wait_for(entered.wait(), 5)
        backend = owner._execution_service.runtime_backend()
        live_owner = backend._turn_owners[execution.execution_id]
        reading = ContextVar("test-reading-admission", default=False)
        reading_remote, continue_read = asyncio.Event(), asyncio.Event()
        target = state.execution.idempotency if pause_at == "receipt" else cache
        name = "read_turn_by_request" if pause_at == "owner" else "get"
        original = getattr(target, name)

        async def delayed(*args, **kwargs):
            if (not reading.get() or reading_remote.is_set()
                    or pause_at == "cache" and not str(args[0]).startswith("runtime-pending-execution:")):
                return await original(*args, **kwargs)
            if pause_at == "receipt":
                assert active_state_scope() is not None
                result = await original(*args, **kwargs)
                assert result is None
                reading_remote.set()
                await continue_read.wait()
                return result
            assert active_state_scope() is None
            reading_remote.set()
            await continue_read.wait()
            return await original(*args, **kwargs)

        async def lookup():
            token = reading.set(True)
            try:
                return await reader.sessions.lookup_admission(
                    "conversation", principal=principal, idempotency_key="request",
                )
            finally:
                reading.reset(token)

        monkeypatch.setattr(target, name, delayed)
        result = asyncio.create_task(lookup())
        try:
            await asyncio.wait_for(reading_remote.wait(), 5)
            finish.set()
            assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
            await asyncio.wait_for(live_owner.closed.wait(), 5)
            continue_read.set()
            admission = await asyncio.wait_for(result, 5)
            assert admission.execution_id == execution.execution_id
            assert admission.user_input == "frozen input"
        finally:
            continue_read.set()
            finish.set()
            await asyncio.gather(result, return_exceptions=True)


@pytest.mark.parametrize("source_authority", [False, True])
async def test_committed_replay_does_not_read_owner_authority(tmp_path, monkeypatch, source_authority):
    cache = InMemoryCacheBackend(max_bytes=2**24)
    async with _runtimes(tmp_path, cache=cache, source_authority=source_authority) as (runtimes, *_):
        owner, _ = runtimes[0]
        reader, state = runtimes[1]
        session = await owner.agents.get().create_session("conversation")
        execution = await session.start("frozen input", idempotency_key="request")
        await execution.wait(timeout_seconds=5)

        async def unavailable(*args, **kwargs):
            pytest.fail("a durable admission receipt does not depend on turn authority")

        monkeypatch.setattr(state.coordination_backend,
                            "_anchor" if source_authority else "read_turn_by_request", unavailable)
        admission = await reader.sessions.lookup_admission(
            "conversation", principal=reader.default_principal, idempotency_key="request",
        )
        assert admission.execution_id == execution.execution_id
        assert admission.user_input == "frozen input"


async def test_same_source_live_owner_without_pending_bytes_is_unknown(tmp_path):
    async with _runtimes(tmp_path, cache=None, shared=False, source_authority=True) as (runtimes, *_):
        runtime, state = runtimes[0]
        await runtime.agents.get().create_session("conversation")
        authority = state.coordination_backend
        owner = await authority.claim_turn(
            TurnScopeIdentity(state.namespace, state.tenant_id, "runtime-pending-turn"),
            session_key="session:conversation", request_key=canonical_sha256(["session.resume", "request"]),
            request_digest=canonical_sha256("request-digest"), execution_id="live",
            owner_id="f" * 32, lease_seconds=60,
        )
        assert owner is not None and owner.lease is not None
        try:
            with pytest.raises(AIError) as error:
                await runtime.sessions.lookup_admission(
                    "conversation", principal=runtime.default_principal, idempotency_key="request",
                )
            assert error.value.code is ErrorCode.EXECUTION_START_UNKNOWN
        finally:
            assert await authority.release_turn(owner.lease)


@pytest.mark.parametrize(("field", "invalid"), (
    ("owner_id", "f" * 33),
    ("request_digest", "d" * 65),
    ("session_key", "session:" + "界" * 43),
))
async def test_same_source_admission_rejects_oversized_keys_before_mutation(tmp_path, monkeypatch, field, invalid):
    async with _runtimes(tmp_path, cache=None, shared=False, source_authority=True) as (runtimes, *_):
        _, state = runtimes[0]
        authority = state.coordination_backend
        scope = TurnScopeIdentity(state.namespace, state.tenant_id, "runtime-pending-turn")
        identity = dict(session_key="session:conversation", request_key="a" * 64, request_digest="b" * 64,
                        execution_id="live", owner_id="f" * 32, lease_seconds=60)
        identity[field] = invalid
        async def unexpected(*args, **kwargs):
            pytest.fail("invalid typed input must not enter the mutation boundary")
        monkeypatch.setattr(authority, "_mutation", unexpected)
        with pytest.raises(AIError) as error:
            await authority.claim_turn(scope, **identity)
        assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID


async def test_same_source_lookup_authorizes_before_checking_receipt(tmp_path, monkeypatch):
    async with _runtimes(tmp_path, cache=None, source_authority=True) as (runtimes, *_):
        runtime, state = runtimes[0]
        await runtime.agents.get().create_session("conversation")

        async def corrupt(*args, **kwargs):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        monkeypatch.setattr(state.execution.idempotency, "get", corrupt)
        with pytest.raises(AIError) as error:
            await runtime.sessions.lookup_admission(
                "conversation", principal=Principal("other", "other"), idempotency_key="request",
            )
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED



async def test_builtin_session_inspection_reuses_its_authorized_row(tmp_path, monkeypatch):
    async with _runtimes(tmp_path, cache=None, shared=False) as (runtimes, *_):
        runtime, storage = runtimes[0]
        session = await runtime.agents.get().create_session("inspection")
        repository = storage.conversation.sessions
        get = repository.get
        reads = 0

        async def count(*args, **kwargs):
            nonlocal reads
            reads += 1
            return await get(*args, **kwargs)

        monkeypatch.setattr(repository, "get", count)
        view = await runtime.history.inspect_session(session.session_id, principal=runtime.default_principal)
        assert view.session_id == session.session_id
        assert reads == 1


async def test_remote_pending_refresh_does_not_materialize_an_absent_execution(tmp_path, monkeypatch):
    cache = InMemoryCacheBackend(max_bytes=2**24)
    async with _runtimes(tmp_path, cache=cache, source_authority=True) as (runtimes, entered, finish, block):
        owner, _ = runtimes[0]
        reader, storage = runtimes[1]
        session = await owner.agents.get().create_session("pending-refresh")
        block()
        execution = await session.start("pending input", idempotency_key="request")
        await asyncio.wait_for(entered.wait(), 5)
        repository = storage.execution.executions
        original = repository.get_timeline_many

        async def get_timeline_many(ids, **kwargs):
            assert execution.execution_id not in ids
            return await original(ids, **kwargs)

        try:
            with monkeypatch.context() as patch:
                patch.setattr(repository, "get_timeline_many", get_timeline_many)
                view = await reader.sessions.refresh("pending-refresh", principal=reader.default_principal)
            assert view.session.active_execution_id == execution.execution_id
            assert view.timeline.items[0].execution_id == execution.execution_id
        finally:
            finish.set()
            await execution.wait()
