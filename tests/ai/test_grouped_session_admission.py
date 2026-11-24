#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Built-in source-coordinated synchronous Session turn admission."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus, IdempotencyStatus
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


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["owned", "not_committed", "retired"])
async def test_grouped_admission_confirms_both_receipts_after_unknown_ack(tmp_path, monkeypatch, outcome):
    launched = 0

    async def model(messages, info):
        nonlocal launched
        launched += 1
        return ModelResponse(parts=[TextPart("answer")])

    async with _runtime(tmp_path / "state.db", model) as (persistence, runtime):
        await runtime.agents.get().create_session("session")
        tenant = runtime.default_principal.tenant_id
        backend = runtime._execution_service.runtime_backend()
        source = persistence.execution.executions.state_store.storage_group
        original_mutate = source.mutate
        original_command = backend._runtime_commands.claim_and_reserve_start
        injecting = False
        attempted = 0
        unknown = AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        async def command(*args, **kwargs):
            nonlocal injecting
            injecting = True
            try:
                return await original_command(*args, **kwargs)
            finally:
                injecting = False

        async def mutate(stores, callback):
            nonlocal attempted, injecting
            if injecting:
                attempted += 1
                if outcome != "not_committed":
                    confirmed = await original_mutate(stores, callback)
                    assert confirmed[0].lease is not None and confirmed[1].created
                    if outcome == "retired":
                        injecting = False
                        assert await persistence.coordination_backend.release_turn(confirmed[0].lease)
                raise unknown
            return await original_mutate(stores, callback)

        monkeypatch.setattr(backend._runtime_commands, "claim_and_reserve_start", command)
        monkeypatch.setattr(source, "mutate", mutate)
        if outcome == "owned":
            handle = await runtime.sessions.start(
                "session", "question", principal=runtime.default_principal, idempotency_key="request",
            )
            assert (await handle.wait(timeout_seconds=8)).result.status is ExecutionStatus.SUCCEEDED
            receipt = await persistence.execution.idempotency.get("session.resume", "request", tenant_id=tenant)
            assert receipt.resource_id == handle.execution_id
            assert launched == 1
        else:
            with pytest.raises(AIError) as failure:
                await runtime.sessions.start(
                    "session", "question", principal=runtime.default_principal, idempotency_key="request",
                )
            assert failure.value is unknown
            receipt = await persistence.execution.idempotency.get("session.resume", "request", tenant_id=tenant)
            if outcome == "retired":
                assert receipt is not None and receipt.status is IdempotencyStatus.STARTED
            else:
                assert receipt is None
            assert launched == 0
        assert attempted == 1


@pytest.mark.asyncio
async def test_grouped_admission_caller_cancel_does_not_change_committed_turn(tmp_path, monkeypatch):
    entered, settle = asyncio.Event(), asyncio.Event()
    launched = 0

    async def model(messages, info):
        nonlocal launched
        launched += 1
        return ModelResponse(parts=[TextPart("answer")])

    async with _runtime(tmp_path / "state.db", model) as (persistence, runtime):
        await runtime.agents.get().create_session("session")
        tenant = runtime.default_principal.tenant_id
        backend = runtime._execution_service.runtime_backend()
        commands = backend._runtime_commands
        original = commands.claim_and_reserve_start

        async def delayed_ack(*args, **kwargs):
            result = await original(*args, **kwargs)
            assert result is not None and result[0].lease is not None and result[1].created
            entered.set()
            await settle.wait()
            return result

        monkeypatch.setattr(commands, "claim_and_reserve_start", delayed_ack)
        task = asyncio.create_task(runtime.sessions.start(
            "session", "question", principal=runtime.default_principal, idempotency_key="request",
        ))
        try:
            await asyncio.wait_for(entered.wait(), 8)
            task.cancel()
        finally:
            settle.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        monkeypatch.setattr(commands, "claim_and_reserve_start", original)
        replay = await runtime.sessions.start(
            "session", "question", principal=runtime.default_principal, idempotency_key="request",
        )
        owner = backend._turn_owners.get(replay.execution_id)
        assert (await replay.wait(timeout_seconds=8)).result.status is ExecutionStatus.SUCCEEDED
        if owner is not None:
            await asyncio.wait_for(owner.closed.wait(), 8)
        receipt = await persistence.execution.idempotency.get("session.resume", "request", tenant_id=tenant)
        assert receipt.status is IdempotencyStatus.COMPLETED and receipt.resource_id == replay.execution_id
        assert launched == 1
        assert (await runtime.sessions.inspect("session", principal=runtime.default_principal)).active_execution_id is None


@pytest.mark.asyncio
async def test_grouped_admission_live_owner_conflict_never_consumes_another_key(tmp_path):
    entered, finish = asyncio.Event(), asyncio.Event()
    runs = 0

    async def model(messages, info):
        nonlocal runs
        runs += 1
        entered.set()
        await finish.wait()
        return ModelResponse(parts=[TextPart("answer")])

    path = tmp_path / "state.db"
    async with (
        _runtime(path, model) as (_, owner),
        _runtime(path, model) as (source, contender),
    ):
        await owner.agents.get().create_session("session")
        await owner.agents.get().create_session("another")
        tenant = owner.default_principal.tenant_id
        first = await owner.sessions.start(
            "session", "original", principal=owner.default_principal, idempotency_key="first",
        )
        try:
            await asyncio.wait_for(entered.wait(), 8)
            duplicate = await contender.sessions.start(
                "session", "original", principal=contender.default_principal, idempotency_key="first",
            )
            assert duplicate.execution_id == first.execution_id
            with pytest.raises(AIError) as busy:
                await contender.sessions.start(
                    "session", "new", principal=contender.default_principal, idempotency_key="unused",
                )
            assert busy.value.code is ErrorCode.SESSION_BUSY
            assert await source.execution.idempotency.get("session.resume", "unused", tenant_id=tenant) is None
            with pytest.raises(AIError) as conflict:
                await contender.sessions.start(
                    "another", "original", principal=contender.default_principal, idempotency_key="first",
                )
            assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
            assert runs == 1
        finally:
            finish.set()
        assert (await first.wait(timeout_seconds=8)).result.status is ExecutionStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_grouped_decline_rolls_back_lease_before_ordinary_fallback(tmp_path, monkeypatch):
    async def model(messages, info):
        return ModelResponse(parts=[TextPart("answer")])

    async with _runtime(tmp_path / "state.db", model) as (persistence, runtime):
        await runtime.agents.get().create_session("session")
        backend = runtime._execution_service.runtime_backend()
        commands = backend._runtime_commands
        authority = persistence.coordination_backend
        original_reserve = commands._reserve_start_in_transaction
        original_claim = authority.claim_turn
        attempts = claims = 0

        async def decline_once(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return None
            return await original_reserve(*args, **kwargs)

        async def claim(scope, **kwargs):
            nonlocal claims
            claims += 1
            # The grouped attempt wrote a provisional claim before the declined
            # checkpoint; fallback is valid only after that claim rolled back.
            assert await authority.read_turn_by_session(scope, kwargs["session_key"]) is None
            assert await authority.read_turn_by_request(scope, kwargs["request_key"]) is None
            assert await authority.read_turn(scope, kwargs["execution_id"]) is None
            return await original_claim(scope, **kwargs)

        monkeypatch.setattr(commands, "_reserve_start_in_transaction", decline_once)
        monkeypatch.setattr(authority, "claim_turn", claim)
        handle = await runtime.sessions.start(
            "session", "question", principal=runtime.default_principal, idempotency_key="request",
        )
        assert (await handle.wait(timeout_seconds=8)).result.status is ExecutionStatus.SUCCEEDED
        assert attempts == 2 and claims == 1


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
                assert live_owner.execution_id == first.execution_id
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
async def test_grouped_admission_late_unprepared_receipt_retains_original_semantics(tmp_path, monkeypatch, start_unknown):
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
                handle = await task
                assert handle.execution_id == reserved.execution.execution_id
                assert (await handle.wait(timeout_seconds=8)).result.status is ExecutionStatus.SUCCEEDED
                assert launched == 1
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
        executions, admissions = metadata.tables["ai_execution_records"], metadata.tables["ai_turn_admissions"]
        writes = []

        def observe(_connection, cursor, statement, *_args):
            if statement.startswith(("INSERT INTO ai_execution_records", "INSERT INTO ai_turn_admissions",
                                     "DELETE FROM ai_execution_records", "INSERT INTO ai_turn_slots",
                                     "UPDATE ai_turn_slots")):
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
                row = (await connection.execute(select(executions.c.id, executions.c.kind, executions.c.turn_token).select_from(
                    admissions.join(executions, admissions.c.control_id == executions.c.id),
                ).where(admissions.c.store_key == store.store_key,
                        admissions.c.execution_id == handle.execution_id.encode("utf-8")))).one()
                assert row.kind == "execution" and row.turn_token
                execution_row_id = row.id
            execution_insert, receipt_insert = [entry for entry in writes if entry[0].startswith("INSERT INTO ai_execution_records")]
            assert execution_insert[1] == receipt_insert[1] == 1
            assert "strftime" in execution_insert[0]
            (slot,) = [entry[0] for entry in writes if entry[0].startswith(
                ("INSERT INTO ai_turn_slots", "UPDATE ai_turn_slots"),
            )]
            assert "active_execution_id" in slot and "lease_fence" in slot
            (admission,) = [entry for entry in writes if entry[0].startswith("INSERT INTO ai_turn_admissions")]
            assert "VALUES" in admission[0] and "SELECT" not in admission[0] and admission[1] == 1
            finish.set()
            assert (await handle.wait(timeout_seconds=8)).result.status is ExecutionStatus.SUCCEEDED
            await asyncio.wait_for(lifetime.closed.wait(), 8)
            async with engine.begin() as connection:
                retired = (await connection.execute(select(admissions.c.control_id, admissions.c.receipt_kind).where(
                    admissions.c.store_key == store.store_key,
                    admissions.c.execution_id == handle.execution_id.encode("utf-8"),
                ))).one()
                assert retired.control_id is None and retired.receipt_kind == "release"
                (token,) = (await connection.execute(select(executions.c.turn_token).where(
                    executions.c.id == execution_row_id,
                ))).one()
                assert token is None
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
        slots = store.storage_group.metadata.tables["ai_turn_slots"]
        from sqlalchemy import select

        async def physical():
            async with engine.begin() as conn:
                return tuple((await conn.execute(select(
                    slots.c.session_kind, slots.c.session_key, slots.c.active_execution_id,
                    slots.c.lease_fence, slots.c.revision,
                ).where(slots.c.store_key == store.store_key))).all())

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
