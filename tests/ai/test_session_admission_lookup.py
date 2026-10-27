#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recover accepted session requests without rebuilding mutable input context."""

import asyncio
from dataclasses import replace

import pytest

from linktools.ai.core import ExecutionStatus, IdempotencyStatus, Principal, idempotency_key_digest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import InMemoryCacheBackend
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


@pytest.mark.parametrize("shared", [False, True])
async def test_pending_admission_lookup_reads_local_and_shared_without_settling(tmp_path, monkeypatch, shared):
    cache = InMemoryCacheBackend(max_bytes=2**24) if shared else None
    async with _runtimes(tmp_path, cache=cache, shared=shared) as (runtimes, entered, finish, block):
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
        original = await repo.get("session.resume", idempotency_key_digest("request"), tenant_id=runtime.tenant_id)
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
                identity = await repo.get("session.resume", idempotency_key_digest("request"), tenant_id=runtime.tenant_id)
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
        identity = await repo.get("session.resume", idempotency_key_digest("request"), tenant_id=runtime.tenant_id)
        async def failed_identity(*args, **kwargs):
            return replace(identity, status=status, error_code=code.value)
        async def failed_execution(*args, **kwargs):
            return replace(record, started_at=None, status=ExecutionStatus(status.value), error_code=code.value) if has_record else None
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
        identity = await repo.get("session.resume", idempotency_key_digest("request"), tenant_id=runtime.tenant_id)
        async def damaged(*args, **kwargs):
            return replace(identity, result_digest="0" * 64)
        with monkeypatch.context() as patch:
            patch.setattr(repo, "get", damaged)
            with pytest.raises(AIError) as error:
                await runtime.sessions.lookup_admission("conversation", principal=runtime.default_principal, idempotency_key="request")
            assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
