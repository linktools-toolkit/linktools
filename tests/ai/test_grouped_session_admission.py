#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native SQL turn admission retains one durable source and fenced ownership."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import event

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import ExecutionStatus, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.turns import SessionTurnKey
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


class _HostAuthorization(TenantAuthorizationPolicy):
    """Supply explicit session authorization to the canonical SQL runtime."""


@asynccontextmanager
async def _native_runtime(path, model):
    features = RuntimeFeatures("native-admission")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    features = model_features(_Models(_UsageFunctionModel(model)), features)
    async with (
        Persistence.open("native-admission", storage=PersistenceStorage.from_root(path)) as state,
        Runtime.open(persistence=state, features=features, authorization=_HostAuthorization()) as runtime,
    ):
        yield state, runtime


async def _native_start(runtime, *, prompt="question", request_id="request"):
    return await runtime.sessions.start(
        "session", prompt, principal=runtime.default_principal,
        agent=runtime.agents.get(), idempotency_key=request_id,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", [False, True], ids=("rolled-back", "committed"))
async def test_native_admission_reads_exact_receipt_after_unknown_ack(tmp_path, monkeypatch, committed):
    launched = 0

    async def model(_messages, _info):
        nonlocal launched
        launched += 1
        return ModelResponse(parts=[TextPart("answer")])

    async with _native_runtime(tmp_path, model) as (state, runtime):
        await runtime.agents.get().create_session("session")
        turns = state.session_turns
        original = turns.accept
        attempted = []
        unknown = AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        async def lose_ack(key, **kwargs):
            attempted.append(key)
            if committed:
                assert await original(key, **kwargs)
            raise unknown

        with monkeypatch.context() as patch:
            patch.setattr(turns, "accept", lose_ack)
            if committed:
                recovered_receipt = await _native_start(runtime)
            else:
                with pytest.raises(AIError) as rejected:
                    await _native_start(runtime)
                assert rejected.value is unknown

        [first_key] = attempted
        assert launched == 0
        session = await turns.get_session("session")
        row = await turns.find_request("session", "request")
        if committed:
            assert recovered_receipt.key == first_key
            assert row["turn_id"] == session["active_turn_id"] == first_key.turn_id.encode()
            accepted_row = await turns.get_turn(first_key)
            assert accepted_row["generation"] == 1 and accepted_row["state"] == "accepted"
            same = await _native_start(runtime)
            assert same.key == first_key and launched == 0
            assert not await runtime.sessions.recover_turn("session", first_key.turn_id, principal=runtime.default_principal)
            # Deterministically move only the fixture's lease to the past. The
            # public recovery call must acquire the next generation itself.
            async with turns._context.engine.begin() as connection:
                changed = await connection.execute(turns._turns.update().filter_by(
                    **turns._turn_values(first_key), generation=1, state="accepted",
                ).values(lease_until=0))
                assert changed.rowcount == 1
            assert await runtime.sessions.recover_turn("session", first_key.turn_id, principal=runtime.default_principal)
            assert (await same.wait_final(timeout_seconds=8)).status is ExecutionStatus.SUCCEEDED
            assert (await turns.get_turn(first_key))["generation"] == 2
        else:
            assert row is None and session["active_turn_id"] is None
            retry = await _native_start(runtime)
            assert retry.key != first_key
            assert (await retry.wait_final(timeout_seconds=8)).status is ExecutionStatus.SUCCEEDED
        assert launched == 1


@pytest.mark.asyncio
async def test_native_admission_caller_cancel_does_not_cancel_committed_turn(tmp_path, monkeypatch):
    committed, release = asyncio.Event(), asyncio.Event()
    launched = 0

    async def model(_messages, _info):
        nonlocal launched
        launched += 1
        return ModelResponse(parts=[TextPart("answer")])

    async with _native_runtime(tmp_path, model) as (state, runtime):
        await runtime.agents.get().create_session("session")
        turns = state.session_turns
        original = turns.accept
        keys = []

        async def delay_ack(key, **kwargs):
            accepted = await original(key, **kwargs)
            assert accepted
            keys.append(key)
            committed.set()
            await release.wait()
            return accepted

        monkeypatch.setattr(turns, "accept", delay_ack)
        sending = asyncio.create_task(_native_start(runtime))
        try:
            await asyncio.wait_for(committed.wait(), 8)
            sending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await sending
            duplicate = await _native_start(runtime)
            assert keys == [duplicate.key] and launched == 0
        finally:
            release.set()
            await asyncio.gather(sending, return_exceptions=True)

        result = await duplicate.wait_final(timeout_seconds=8)
        assert result.status is ExecutionStatus.SUCCEEDED and result.key == duplicate.key
        assert launched == 1
        assert (await turns.get_session("session"))["active_turn_id"] is None


@pytest.mark.asyncio
async def test_native_live_owner_conflict_never_consumes_another_request(tmp_path):
    entered, finish = asyncio.Event(), asyncio.Event()
    runs = 0

    async def model(_messages, _info):
        nonlocal runs
        runs += 1
        entered.set()
        await finish.wait()
        return ModelResponse(parts=[TextPart("answer")])

    async with (
        _native_runtime(tmp_path, model) as (_, owner),
        _native_runtime(tmp_path, model) as (state, contender),
    ):
        await owner.agents.get().create_session("session")
        first = await _native_start(owner)
        try:
            await asyncio.wait_for(entered.wait(), 8)
            duplicate = await _native_start(contender)
            assert duplicate.key == first.key
            with pytest.raises(AIError) as busy:
                await _native_start(contender, prompt="new", request_id="unused")
            assert busy.value.code is ErrorCode.SESSION_BUSY
            assert await state.session_turns.find_request("session", "unused") is None
            with pytest.raises(AIError) as conflict:
                await _native_start(contender, prompt="changed")
            assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
            assert runs == 1
        finally:
            finish.set()
        assert (await first.wait_final(timeout_seconds=8)).status is ExecutionStatus.SUCCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_turn", [False, True])
async def test_failed_native_insert_rolls_back_early_session_claim(tmp_path, prior_turn):
    model_calls = 0

    async def model(_messages, _info):
        nonlocal model_calls
        model_calls += 1
        return ModelResponse(parts=[TextPart("answer")])

    async with _native_runtime(tmp_path, model) as (state, runtime):
        await runtime.agents.get().create_session("session")
        if prior_turn:
            warm = await _native_start(runtime, request_id="warm")
            assert (await warm.wait_final(timeout_seconds=8)).status is ExecutionStatus.SUCCEEDED
        turns = state.session_turns
        before = await turns.get_session("session")
        calls_before = model_calls
        attempts = []
        blocked = AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        engine = turns._context.engine.sync_engine

        def reject_insert(_connection, _cursor, statement, *_args):
            if statement.lstrip().lower().startswith("insert into ai_session_turns"):
                attempts.append(statement)
                raise blocked

        event.listen(engine, "before_cursor_execute", reject_insert)
        try:
            with pytest.raises(AIError) as failure:
                await _native_start(runtime)
            assert failure.value is blocked
        finally:
            event.remove(engine, "before_cursor_execute", reject_insert)

        assert len(attempts) == 1 and model_calls == calls_before
        assert await turns.get_session("session") == before
        assert await turns.find_request("session", "request") is None
        accepted = await _native_start(runtime)
        assert (await accepted.wait_final(timeout_seconds=8)).status is ExecutionStatus.SUCCEEDED
        assert model_calls == calls_before + 1
        assert (await turns.get_session("session"))["last_turn_no"] == before["last_turn_no"] + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["live_replay", "terminal_replay", "digest_conflict", "foreign_request"])
async def test_native_admission_rechecks_late_winner_after_outer_read(tmp_path, monkeypatch, case):
    entered, finish, ready, resume = (asyncio.Event() for _ in range(4))
    calls = 0

    async def model(_messages, _info):
        nonlocal calls
        calls += 1
        entered.set()
        await finish.wait()
        return ModelResponse(parts=[TextPart("answer")])

    async with (
        _native_runtime(tmp_path, model) as (source, winner),
        _native_runtime(tmp_path, model) as (state, contender),
    ):
        await winner.agents.get().create_session("session")
        turns = state.session_turns
        original = turns.accept
        proposed = []

        async def pause(key, **kwargs):
            proposed.append(key)
            ready.set()
            await resume.wait()
            return await original(key, **kwargs)

        monkeypatch.setattr(turns, "accept", pause)
        sending = asyncio.create_task(_native_start(
            contender, prompt="different" if case == "digest_conflict" else "original",
            request_id="foreign" if case == "foreign_request" else "request",
        ))
        try:
            await asyncio.wait_for(ready.wait(), 8)
            first = await _native_start(winner, prompt="original")
            await asyncio.wait_for(entered.wait(), 8)
            if case == "terminal_replay":
                finish.set()
                assert (await first.wait_final(timeout_seconds=8)).status is ExecutionStatus.SUCCEEDED
            resume.set()
            if case in {"live_replay", "terminal_replay"}:
                assert (await sending).key == first.key
            else:
                with pytest.raises(AIError) as rejected:
                    await sending
                assert rejected.value.code is (
                    ErrorCode.IDEMPOTENCY_CONFLICT if case == "digest_conflict" else ErrorCode.SESSION_CONFLICT
                )
            [discarded] = proposed
            assert await source.session_turns.get_turn(discarded) is None
            assert await source.session_turns.find_request("session", "foreign") is None
        finally:
            resume.set()
            finish.set()
            await asyncio.gather(sending, return_exceptions=True)
        assert (await first.wait_final(timeout_seconds=8)).status is ExecutionStatus.SUCCEEDED
        assert calls == 1


@pytest.mark.asyncio
async def test_native_admission_never_launches_a_late_unowned_receipt(tmp_path, monkeypatch):
    ready, resume = asyncio.Event(), asyncio.Event()
    calls = 0

    async def model(_messages, _info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart("answer")])

    async with (
        _native_runtime(tmp_path, model) as (writer, owner),
        _native_runtime(tmp_path, model) as (state, contender),
    ):
        await owner.agents.get().create_session("session")
        turns = state.session_turns
        original = turns.accept
        proposed = []

        async def pause(key, **kwargs):
            proposed.append((key, kwargs))
            ready.set()
            await resume.wait()
            return await original(key, **kwargs)

        monkeypatch.setattr(turns, "accept", pause)
        sending = asyncio.create_task(_native_start(contender))
        try:
            await asyncio.wait_for(ready.wait(), 8)
            [(discarded, kwargs)] = proposed
            committed = SessionTurnKey("session", "existing-undispatched")
            assert await writer.session_turns.accept(committed, **kwargs)
            resume.set()
            replay = await sending
            assert replay.key == committed and calls == 0
            assert await turns.get_turn(discarded) is None
            row = await turns.get_turn(committed)
            assert row["state"] == "accepted" and row["generation"] == 1
        finally:
            resume.set()
            await asyncio.gather(sending, return_exceptions=True)


@pytest.mark.asyncio
async def test_native_admission_binds_exact_turn_with_plain_insert(tmp_path):
    entered, finish = asyncio.Event(), asyncio.Event()

    async def model(_messages, _info):
        entered.set()
        await finish.wait()
        return ModelResponse(parts=[TextPart("answer")])

    async with _native_runtime(tmp_path, model) as (state, runtime):
        await runtime.agents.get().create_session("session")
        turns = state.session_turns
        engine = turns._context.engine.sync_engine
        writes = []

        def observe(_connection, _cursor, statement, *_args):
            if statement.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
                writes.append(statement.lstrip().lower())

        event.listen(engine, "after_cursor_execute", observe)
        try:
            accepted = await _native_start(runtime)
            await asyncio.wait_for(entered.wait(), 8)
            row = await turns.get_turn(accepted.key)
            assert row["generation"] == 1
            assert (await turns.get_session("session"))["active_turn_id"] == accepted.key.turn_id.encode()
            [insert] = [sql for sql in writes if sql.startswith("insert into ai_session_turns")]
            assert " values " in insert and "select" not in insert
            assert not any("ai_execution_records" in sql for sql in writes)
            finish.set()
            assert (await accepted.wait_final(timeout_seconds=8)).status is ExecutionStatus.SUCCEEDED
            assert (await turns.get_session("session"))["active_turn_id"] is None
            assert (await turns.get_turn(accepted.key))["generation"] == 1
            assert not any("ai_execution_records" in sql for sql in writes)
        finally:
            finish.set()
            event.remove(engine, "after_cursor_execute", observe)
