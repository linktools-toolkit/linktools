#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Built-in source-coordinated synchronous Session turn admission."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus, IdempotencyStatus, TenantAuthorizationPolicy, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage, StateTurnResources
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


@asynccontextmanager
async def _runtime(path, model):
    features = RuntimeFeatures("grouped-admission")
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.sync()), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    features = model_features(_Models(_UsageFunctionModel(model)), features)
    storage = PersistenceStorage.sqlite(path, coordination=StateTurnResources())
    async with (
        Persistence.open("grouped-admission", storage=storage) as persistence,
        Runtime.open(persistence=persistence, features=features) as runtime,
    ):
        yield persistence, runtime


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
async def test_grouped_decline_rolls_back_lease_before_strict_retry(tmp_path, monkeypatch):
    async def model(messages, info):
        return ModelResponse(parts=[TextPart("answer")])

    async with _runtime(tmp_path / "state.db", model) as (persistence, runtime):
        await runtime.agents.get().create_session("session")
        backend = runtime._execution_service.runtime_backend()
        commands = backend._runtime_commands
        authority = persistence.coordination_backend
        original_reserve = commands._reserve_start_in_transaction
        attempts = claims = 0
        declined = []
        scope = backend._pending_turns.coordination._scope

        async def decline_once(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                _, reservation, *_ = args
                declined.append(reservation)
                assert kwargs["authority"] is not None
                return None
            return await original_reserve(*args, **kwargs)

        async def reject_fallback(*args, **kwargs):
            nonlocal claims
            claims += 1
            raise AssertionError("Strict grouped admission must not fall back to another claim path")

        monkeypatch.setattr(commands, "_reserve_start_in_transaction", decline_once)
        monkeypatch.setattr(authority, "claim_turn", reject_fallback)
        principal = runtime.default_principal
        with pytest.raises(AIError) as rejected:
            await runtime.sessions.start("session", "question", principal=principal, idempotency_key="request")
        assert rejected.value.code is ErrorCode.STORAGE_CONFLICT
        assert len(declined) == 1
        execution_id = declined[0].execution.execution_id
        assert await authority.read_turn(scope, execution_id) is None
        assert await authority.read_turn_by_session(scope, "session:session") is None
        request_key = canonical_sha256(["session.resume", "request"])
        assert await authority.read_turn_by_request(scope, request_key) is None
        assert await persistence.execution.executions.get(execution_id, tenant_id=principal.tenant_id) is None
        assert await persistence.execution.idempotency.get("session.resume", "request", tenant_id=principal.tenant_id) is None

        handle = await runtime.sessions.start("session", "question", principal=principal, idempotency_key="request")
        assert (await handle.wait(timeout_seconds=8)).result.status is ExecutionStatus.SUCCEEDED
        assert handle.execution_id != execution_id
        assert attempts == 2 and claims == 0


def _pause_grouped_command(runtime, monkeypatch):
    ready, resume = asyncio.Event(), asyncio.Event()
    commands = runtime._execution_service.runtime_backend()._runtime_commands
    original = commands.claim_and_reserve_start
    proposed = []

    async def pause(authority, scope, reservation, **kwargs):
        proposed.append(reservation)
        ready.set()
        await resume.wait()
        return await original(authority, scope, reservation, **kwargs)

    monkeypatch.setattr(commands, "claim_and_reserve_start", pause)
    return ready, resume, proposed


@pytest.mark.asyncio
@pytest.mark.parametrize("case", [
    "live_replay", "digest_conflict", "terminal_replay", "terminal_foreign",
    "terminal_mixed", "terminal_corrupt",
])
async def test_grouped_admission_rechecks_late_winner_after_outer_read(tmp_path, monkeypatch, case):
    model_entered, finish = asyncio.Event(), asyncio.Event()
    terminal_committed, release_terminal = asyncio.Event(), asyncio.Event()
    launched = 0

    async def model(messages, info):
        nonlocal launched
        launched += 1
        model_entered.set()
        await finish.wait()
        return ModelResponse(parts=[TextPart("answer")])

    path = tmp_path / "state.db"
    async with (_runtime(path, model) as (source, winner), _runtime(path, model) as (_, contender)):
        await winner.agents.get().create_session("session")
        principal = contender.default_principal
        ready, resume, proposed = _pause_grouped_command(contender, monkeypatch)
        if case.startswith("terminal"):
            backend = winner._execution_service.runtime_backend()
            old_cleanup = backend._discard_terminal_producer_tail

            async def held_cleanup(*args, **kwargs):
                terminal_committed.set()
                await release_terminal.wait()
                return await old_cleanup(*args, **kwargs)

            monkeypatch.setattr(backend, "_discard_terminal_producer_tail", held_cleanup)

        key = "foreign" if case == "terminal_foreign" else "request"
        prompt = "different" if case == "digest_conflict" else "original"
        task = asyncio.create_task(contender.sessions.start(
            "session", prompt, principal=principal, idempotency_key=key,
        ))
        try:
            # Contender completed its outer source read and has not entered
            # its grouped transaction. The winner now commits independently.
            await asyncio.wait_for(ready.wait(), 8)
            first = await winner.sessions.start(
                "session", "original", principal=winner.default_principal, idempotency_key="request",
            )
            await asyncio.wait_for(model_entered.wait(), 8)
            tenant = principal.tenant_id
            identities = source.execution.idempotency
            if case in {"terminal_mixed", "terminal_corrupt"}:
                stale_identity = await identities.get("session.resume", "request", tenant_id=tenant)
                assert stale_identity.status is IdempotencyStatus.STARTED

                async def read_stale_identity(_claim):
                    # Simulate READ COMMITTED's first read straddling terminal
                    # commit; the execution and later receipt reads stay real.
                    return stale_identity

                repositories = contender._execution_service.runtime_backend()._runtime_commands
                monkeypatch.setattr(repositories._execution, "get_start_idempotency", read_stale_identity)
            if case.startswith("terminal"):
                finish.set()
                await asyncio.wait_for(terminal_committed.wait(), 8)
                persisted = await source.execution.executions.get(first.execution_id, tenant_id=principal.tenant_id)
                assert persisted.status is ExecutionStatus.SUCCEEDED
                live_owner = await backend._pending_turns.coordination.read_session_owner(
                    "session", tenant_id=principal.tenant_id,
                )
                # Native terminal publication releases occupancy in the same
                # commit; producer cleanup held above is not the authority.
                assert live_owner is None
            if case == "terminal_corrupt":
                identity = await identities.get("session.resume", "request", tenant_id=tenant)
                assert identity.status is IdempotencyStatus.COMPLETED
                wrong_digest = "0" * 64
                assert identity.result_digest != wrong_digest
                await identities.compare_and_swap(
                    "session.resume", "request", tenant_id=tenant, expected_status=identity.status,
                    next_record=replace(identity, result_digest=wrong_digest),
                )

            resume.set()
            if case in {"live_replay", "terminal_replay", "terminal_mixed"}:
                replay = await task
                assert replay.execution_id == first.execution_id
            else:
                with pytest.raises(AIError) as rejected:
                    await task
                assert rejected.value.code is {
                    "digest_conflict": ErrorCode.IDEMPOTENCY_CONFLICT,
                    "terminal_foreign": ErrorCode.SESSION_BUSY,
                    "terminal_corrupt": ErrorCode.STORAGE_INTEGRITY_ERROR,
                }[case]
            assert len(proposed) == 1 and launched == 1
            assert await source.execution.executions.get(proposed[0].execution.execution_id, tenant_id=tenant) is None
            assert await source.execution.idempotency.get("session.resume", "foreign", tenant_id=tenant) is None
        finally:
            resume.set()
            finish.set()
            release_terminal.set()
            if not task.done():
                await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("start_unknown", [False, True])
async def test_grouped_admission_does_not_start_late_unowned_receipt(tmp_path, monkeypatch, start_unknown):
    launched = 0

    async def model(messages, info):
        nonlocal launched
        launched += 1
        return ModelResponse(parts=[TextPart("answer")])

    path = tmp_path / "state.db"
    async with (_runtime(path, model) as (source, writer), _runtime(path, model) as (_, contender)):
        await writer.agents.get().create_session("session")
        ready, resume, proposed = _pause_grouped_command(contender, monkeypatch)
        task = asyncio.create_task(contender.sessions.start(
            "session", "original", principal=contender.default_principal, idempotency_key="request",
        ))
        try:
            await asyncio.wait_for(ready.wait(), 8)
            reserved = await source.execution.executions.reserve_start(proposed[0])
            assert reserved.created and reserved.execution.status is ExecutionStatus.PENDING_START
            if start_unknown:
                from linktools.ai.execution import ExecutionStartUnknownCommit

                previous = reserved.execution
                identity = reserved.idempotency
                await source.execution.executions.mark_start_unknown(ExecutionStartUnknownCommit(
                    previous.execution_id, previous.revision, previous.event_seq,
                    identity.scope, identity.idempotency_key, identity.request_digest, previous.updated_at,
                ))
            resume.set()
            if start_unknown:
                with pytest.raises(AIError) as rejected:
                    await task
                assert rejected.value.code is ErrorCode.EXECUTION_START_UNKNOWN
                assert launched == 0
            else:
                with pytest.raises(AIError) as rejected:
                    await task
                assert rejected.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
                tenant = contender.default_principal.tenant_id
                assert await source.execution.executions.get(reserved.execution.execution_id, tenant_id=tenant) == reserved.execution
                assert await source.execution.idempotency.get("session.resume", "request", tenant_id=tenant) == reserved.idempotency
                assert launched == 0
            assert len(proposed) == 1
        finally:
            resume.set()
            if not task.done():
                await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_grouped_claim_binds_exact_execution_primary_key_with_plain_inserts(tmp_path):
    from sqlalchemy import event, select

    model_entered, finish = asyncio.Event(), asyncio.Event()

    async def model(messages, info):
        model_entered.set()
        await finish.wait()
        return ModelResponse(parts=[TextPart("answer")])

    async with _runtime(tmp_path / "state.db", model) as (persistence, runtime):
        await runtime.agents.get().create_session("session")
        store = persistence.execution.executions.state_store
        engine = store.storage_group.context.engine
        metadata = store.storage_group.metadata
        executions = metadata.tables["ai_execution_records"]
        sessions = metadata.tables["ai_state_records"]
        writes = []

        def observe(_connection, cursor, statement, *_args):
            if (statement.startswith(("INSERT INTO ai_execution_records", "DELETE FROM ai_execution_records"))
                    or statement.startswith("UPDATE ai_state_records SET strict_turn_")):
                writes.append((statement, cursor.rowcount))

        event.listen(engine.sync_engine, "after_cursor_execute", observe)
        try:
            handle = await runtime.sessions.start(
                "session", "question", principal=runtime.default_principal, idempotency_key="request",
            )
            await asyncio.wait_for(model_entered.wait(), 8)
            backend = runtime._execution_service.runtime_backend()
            lifetime = backend._turn_owners[handle.execution_id]
            async with engine.begin() as connection:
                row = (await connection.execute(select(
                    executions.c.id, executions.c.kind, executions.c.turn_token, executions.c.turn_receipt,
                    sessions.c.strict_turn_execution_id, sessions.c.strict_turn_fence,
                ).join(sessions, sessions.c.id == executions.c.turn_session_owner_id).where(
                    executions.c.store_key == store.store_key,
                    executions.c.key == persistence.execution.executions.execution_record_key(handle.execution_id),
                    sessions.c.strict_turn_fence == executions.c.turn_fence,
                ))).one()
                assert row.kind == "execution" and row.turn_token
                assert row.strict_turn_execution_id == handle.execution_id.encode("utf-8")
                assert row.turn_receipt["receipt_kind"] == "claim"
                execution_row_id = row.id
                admitted_fence = row.strict_turn_fence
            inserts = [entry for entry in writes if entry[0].startswith("INSERT INTO ai_execution_records")]
            assert len(inserts) == 2 and all(rowcount == 1 for _, rowcount in inserts)
            # One Execution with its receipt and one business idempotency row.
            (execution_insert,) = [entry for entry in inserts if "turn_receipt" in entry[0]]
            assert "turn_session_owner_id" in execution_insert[0]
            assert "strftime" in execution_insert[0]
            assert "VALUES" in execution_insert[0] and "SELECT" not in execution_insert[0]
            (slot,) = [entry for entry in writes if entry[0].startswith("UPDATE ai_state_records SET strict_turn_")]
            assert "strict_turn_execution_id" in slot[0] and "strict_turn_fence" in slot[0] and slot[1] == 1
            finish.set()
            assert (await handle.wait(timeout_seconds=8)).result.status is ExecutionStatus.SUCCEEDED
            await asyncio.wait_for(lifetime.closed.wait(), 8)
            async with engine.begin() as connection:
                retired = (await connection.execute(select(
                    executions.c.turn_token, executions.c.turn_receipt,
                    sessions.c.strict_turn_execution_id, sessions.c.strict_turn_fence,
                ).join(sessions, sessions.c.id == executions.c.turn_session_owner_id).where(
                    executions.c.id == execution_row_id,
                ))).one()
                assert retired.turn_token is None and retired.turn_receipt["receipt_kind"] == "release"
                assert retired.strict_turn_execution_id is None and retired.strict_turn_fence == admitted_fence
            assert not [entry for entry in writes if entry[0].startswith("DELETE FROM ai_execution_records")]
        finally:
            finish.set()
            event.remove(engine.sync_engine, "after_cursor_execute", observe)


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_turn", [False, True])
async def test_failed_grouped_reservation_rolls_back_early_slot_binding(tmp_path, monkeypatch, prior_turn):
    async def model(messages, info):
        return ModelResponse(parts=[TextPart("answer")])

    async with _runtime(tmp_path / "state.db", model) as (persistence, runtime):
        await runtime.agents.get().create_session("session")
        backend = runtime._execution_service.runtime_backend()
        start = runtime.sessions.start
        principal = runtime.default_principal
        if prior_turn:
            warm = await start("session", "warm", principal=principal)
            owner = backend._turn_owners.get(warm.execution_id)
            assert (await warm.wait(timeout_seconds=8)).result.status is ExecutionStatus.SUCCEEDED
            if owner is not None:
                await asyncio.wait_for(owner.closed.wait(), 8)
        store = persistence.execution.executions.state_store
        engine = store.storage_group.context.engine
        session_repository = persistence.conversation.sessions
        sessions = store.storage_group.metadata.tables["ai_state_records"]
        from sqlalchemy import select

        async def physical():
            async with engine.begin() as conn:
                return tuple((await conn.execute(select(
                    sessions.c.strict_turn_execution_id, sessions.c.strict_turn_fence, sessions.c.storage_revision,
                ).where(sessions.c.store_key == session_repository.state_store.store_key,
                        sessions.c.key == session_repository.session_record_key("session")))).all())

        before = await physical()
        commands = backend._runtime_commands
        original = commands._reserve_start_in_transaction

        async def fail_after_slot(*args, **kwargs):
            assert kwargs["authority"] is not None
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        monkeypatch.setattr(commands, "_reserve_start_in_transaction", fail_after_slot)
        with pytest.raises(AIError) as failure:
            await start("session", "question", principal=principal)
        assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await physical() == before
        monkeypatch.setattr(commands, "_reserve_start_in_transaction", original)
        handle = await start("session", "question", principal=principal)
        assert (await handle.wait(timeout_seconds=8)).result.status is ExecutionStatus.SUCCEEDED
