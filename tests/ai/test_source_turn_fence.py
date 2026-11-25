#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A stale owner cannot publish despite unchanged durable producer control."""

import asyncio
from contextlib import asynccontextmanager, nullcontext
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import update

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence.commands import StateCommands

from .test_grouped_session_admission import _runtime as _base_runtime


@asynccontextmanager
async def _runtime(path, model):
    async with _base_runtime(path, model) as (state, runtime):
        assert state.coordination_backend._coowned
        yield state, runtime



@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("attempt", "attempt_expired", "attempt_with_successor", "plain_attempt", "terminal"))
async def test_takeover_revokes_write_before_execution_producer_cas_moves(tmp_path, monkeypatch, boundary) -> None:
    captured, resume, rejected = asyncio.Event(), asyncio.Event(), asyncio.Event()
    evidence = {}

    async def model(messages, info):
        return ModelResponse(parts=[TextPart("answer")])

    if boundary == "plain_attempt":
        from linktools.ai.runtime._local import LocalExecutionBackend

        async def cannot_prepare(*args, **kwargs):
            return False

        monkeypatch.setattr(LocalExecutionBackend, "_can_prepare_initial_attempt", cannot_prepare)
        method = "commit_agent_attempt_checkpoint"
    else:
        method = ("commit_prepared_agent_attempt_checkpoint" if boundary.startswith("attempt") else "commit_terminal_checkpoint")
    original = getattr(StateCommands, method)

    async def pause(command, *args, **kwargs):
        fence = kwargs.get("turn_fence")
        if fence is None or captured.is_set():
            return await original(command, *args, **kwargs)
        evidence["fence"] = fence
        captured.set()
        await resume.wait()
        try:
            result = await original(command, *args, **kwargs)
        except BaseException as error:
            evidence["error"] = error
            if boundary == "attempt_with_successor":
                evidence["successor_execution"] = await original(
                    command, *args, **{**kwargs, "turn_fence": evidence["successor_fence"]},
                )
            raise
        else:
            evidence["result"] = result
            return result
        finally:
            rejected.set()

    monkeypatch.setattr(StateCommands, method, pause)
    path = tmp_path / "state.db"
    close_expected = nullcontext() if boundary == "attempt_with_successor" else pytest.raises(AIError)
    with close_expected as closed:
        async with (_runtime(path, model) as (source, owner), _runtime(path, model) as (peer, _)):
            await owner.agents.get().create_session("session")
            first = asyncio.create_task(owner.sessions.start(
                "session", "question", principal=owner.default_principal, idempotency_key="original",
            ))
            try:
                await asyncio.wait_for(captured.wait(), 7)
                fence = evidence["fence"]
                backend = owner._execution_service.runtime_backend()
                lifetime = backend._turn_owners[fence.authority.execution_id]
                tenant = owner.default_principal.tenant_id
                before = await source.execution.executions.get_observed(fence.authority.execution_id, tenant_id=tenant)
                assert before is not None
                peer_store = peer.execution.executions.state_store
                slot = peer_store.storage_group.metadata.tables["ai_execution_records"]
                async with peer_store.storage_group.context.engine.begin() as conn:
                    expired = await conn.execute(update(slot).where(
                        slot.c.store_key == peer_store.store_key,
                        slot.c.key == fence.authority.key,
                    ).values(turn_expiry=datetime(2000, 1, 1, tzinfo=timezone.utc)))
                    assert expired.rowcount == 1
                if boundary != "attempt_expired":
                    session_store = peer.conversation.sessions.state_store
                    async def regain(_group):
                        tx = peer_store._turn_transaction()
                        _, proof, _ = await tx._read_native_execution_receipt(
                            fence.authority.scope, route="execution", identity=fence.authority.execution_id,
                        )
                        return await tx._regrant_native_execution_receipt(
                            fence.authority.scope, session_store=session_store, session_key=fence.authority.slot,
                            request_key=proof.request_key, request_digest=proof.request_digest,
                            execution_id=fence.authority.execution_id, owner_id="b" * 32, duration=90, token="c" * 32,
                        )
                    successor = await peer_store.storage_group.mutate((peer_store, session_store), regain)
                    assert successor is not None and successor.lease is not None
                    assert successor.lease.fence > fence.authority.fence
                    evidence["successor_fence"] = replace(fence, authority=replace(fence.authority, token=successor.lease.token, fence=successor.lease.fence))
                assert await peer.execution.executions.get_observed(fence.authority.execution_id, tenant_id=tenant) == before
                resume.set()
                await asyncio.wait_for(rejected.wait(), 7)
                assert "result" not in evidence
                assert isinstance(evidence["error"], AIError)
                assert evidence["error"].code is ErrorCode.STORAGE_OWNER_MISMATCH
                await asyncio.wait_for(lifetime.closed.wait(), 7)
            finally:
                resume.set()
                await asyncio.gather(first, return_exceptions=True)

    if boundary == "attempt_with_successor":
        after = evidence["successor_execution"]
        assert after.value.producer_generation > before.value.producer_generation
    else:
        assert closed.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        assert closed.value.safe_details["phase"] == "source_turn_revoked"


@pytest.mark.asyncio
async def test_revoked_turn_cannot_admit_new_effect_but_existing_effect_can_settle(tmp_path) -> None:
    from sqlalchemy import event
    from linktools.ai.core import ExecutionStatus, ToolOperationStatus
    from linktools.ai.execution import ExecutionRecoveryDetail
    from linktools.ai.persistence import Persistence, PersistenceStorage
    from linktools.ai.persistence.coordination import PendingTurnCoordination
    from linktools.ai.persistence.turns import StateTurnCoordination
    from linktools.ai.storage import StoredPayload, TurnScopeIdentity

    from .test_history_projection_conformance import _record
    from .test_tool_execution_boundary import _commands, _request

    async with (
        Persistence.open("tool-boundary", tenant_id="tenant", storage=PersistenceStorage.sqlite(tmp_path / "tool.db")) as state,
        Persistence.open("tool-boundary", tenant_id="tenant", storage=PersistenceStorage.sqlite(tmp_path / "split.db")) as split,
    ):
        repository = state.execution.executions
        authority = StateTurnCoordination(repository.state_store, namespace="tool-boundary", tenant_id="tenant", _coowned=True)
        scope = TurnScopeIdentity("tool-boundary", "tenant", "runtime-pending-turn")

        coordinator = PendingTurnCoordination.for_state(None, authority, namespace=scope.namespace, tenant_id=scope.tenant_id)
        execution = replace(_record(ExecutionStatus.STARTED, 1), producer_generation=1, producer_claim_id="worker",
                            recovery=ExecutionRecoveryDetail(agent_run_id="run"))
        await repository.create_execution(execution)

        async def own(owner_id: str):
            return await authority.claim_turn(
                scope, session_key="request:" + "a" * 64, request_key="a" * 64, request_digest="b" * 64,
                execution_id="execution", owner_id=owner_id, lease_seconds=60,
            )

        first = await own("1" * 32)
        fence = coordinator.durable_write_fence(first, repository.state_store)
        assert fence is not None
        with pytest.raises(AIError) as wrong_source:
            coordinator.durable_write_fence(first, split.execution.executions.state_store)
        assert wrong_source.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
        unsupported = PendingTurnCoordination.select(authority, namespace=scope.namespace, tenant_id=scope.tenant_id)
        assert unsupported.durable_write_fence(first, repository.state_store) is None

        observed = await repository.get_observed("execution", tenant_id="tenant")
        commands = _commands(state)
        statements = []

        def capture(_connection, _cursor, statement, *_args):
            statements.append(statement)

        engine = repository.state_store.storage_group.context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", capture)
        try:
            accepted = await commands.commit_tool_admission_boundary(
                _request(), execution=observed, known_new=True, turn_fence=fence,
            )
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        (authority_read,) = [statement for statement in statements if statement.lstrip().startswith("SELECT")]
        assert "ai_execution_records" in authority_read
        (control_write,) = [statement for statement in statements if statement.lstrip().startswith("UPDATE")]
        assert "ai_execution_records" in control_write
        assert await authority.release_turn(first.lease)
        second = await own("2" * 32)
        assert second.lease.fence > first.lease.fence
        assert await repository.get_observed("execution", tenant_id="tenant") == accepted.execution

        with pytest.raises(AIError) as stale:
            await commands.commit_tool_admission_boundary(
                replace(_request(), tool_operation_id="next", tool_call_id="next"),
                execution=accepted.execution, known_new=True, turn_fence=fence,
            )
        assert stale.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
        assert await state.checkpoint.tools.get_operation("next", tenant_id="tenant") is None
        settled = await commands.commit_tool_terminal_boundary(
            "operation", tenant_id="tenant", owner="worker", fence=1, execution=accepted.execution,
            observed=accepted.operation, result_payload=StoredPayload.inline_text("known"),
        )
        assert settled.operation.value.status is ToolOperationStatus.COMPLETED
        assert await repository.get_observed("execution", tenant_id="tenant") == accepted.execution
        assert await authority.release_turn(second.lease)


@pytest.mark.asyncio
async def test_piggyback_cas_rechecks_expiry_after_real_slot_lock_wait(tmp_path, monkeypatch) -> None:
    from datetime import timedelta
    from sqlalchemy import event, literal
    from linktools.ai.core import ExecutionStatus
    from linktools.ai.execution import ExecutionRecoveryDetail
    from linktools.ai.persistence import Persistence, PersistenceStorage
    from linktools.ai.persistence.turns import StateTurnCoordination
    from linktools.ai.storage import TurnScopeIdentity

    from .test_history_projection_conformance import _record
    from .test_tool_execution_boundary import _commands, _request

    scope = TurnScopeIdentity("tool-boundary", "tenant", "runtime-pending-turn")
    async with Persistence.open("tool-boundary", tenant_id="tenant",
                                storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as state:
        repository = state.execution.executions
        store = repository.state_store
        engine = store.storage_group.context.engine
        authority = StateTurnCoordination(store, namespace=scope.namespace, tenant_id=scope.tenant_id, _coowned=True)
        from linktools.ai.persistence.coordination import PendingTurnCoordination
        coordinator = PendingTurnCoordination.for_state(None, authority, namespace=scope.namespace, tenant_id=scope.tenant_id)
        execution = replace(_record(ExecutionStatus.STARTED, 1), producer_generation=1, producer_claim_id="worker",
                            recovery=ExecutionRecoveryDetail(agent_run_id="run"))
        await repository.create_execution(execution)
        first = await authority.claim_turn(
            scope, session_key="request:" + "a" * 64, request_key="a" * 64, request_digest="b" * 64,
            execution_id="execution", owner_id="c" * 32, lease_seconds=60,
        )
        fence = coordinator.durable_write_fence(first, store)
        observed = await repository.get_observed("execution", tenant_id="tenant")
        sampled = datetime.now(timezone.utc)
        samples = []
        guard_waiting = asyncio.Event()
        control = store.storage_group.metadata.tables["ai_execution_records"]

        def clock():
            samples.append(sampled)
            return literal(sampled)

        def before_guard(_conn, _cursor, statement, *_args):
            if statement.startswith("UPDATE ai_execution_records SET storage_revision="):
                guard_waiting.set()

        monkeypatch.setattr(store.storage_group.context.dialect, "read_snapshot_time_expression", clock)
        async with engine.begin() as other:
            await other.execute(update(control).values(turn_fence=control.c.turn_fence))
            event.listen(engine.sync_engine, "before_cursor_execute", before_guard)
            attempted = asyncio.create_task(_commands(state).commit_tool_admission_boundary(
                _request(), execution=observed, known_new=True, turn_fence=fence,
            ))
            try:
                await asyncio.wait_for(guard_waiting.wait(), 5)
                assert not attempted.done() and not samples
                sampled += timedelta(minutes=2)
            finally:
                event.remove(engine.sync_engine, "before_cursor_execute", before_guard)
        with pytest.raises(AIError) as expired:
            await asyncio.wait_for(attempted, 5)
        assert expired.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
        assert samples and set(samples) == {sampled}
        assert await repository.get_observed("execution", tenant_id="tenant") == observed
        assert await state.checkpoint.tools.get_operation("operation", tenant_id="tenant") is None


@pytest.mark.asyncio
async def test_attempt_retry_keeps_fence_after_optimistic_conflict(tmp_path, monkeypatch) -> None:
    from linktools.ai.persistence._execution_repositories import ExecutionRepositoryImpl

    first = asyncio.Event()
    rejected = asyncio.Event()
    model_calls = 0
    outcomes = {}
    original_attempt = StateCommands._commit_agent_attempt_checkpoint
    original_claim = ExecutionRepositoryImpl.claim_prepared_agent_run_in_transaction

    async def model(messages, info):
        nonlocal model_calls
        model_calls += 1
        return ModelResponse(parts=[TextPart("answer")])

    with pytest.raises(AIError) as closed:
        async with _runtime(tmp_path / "retry.db", model) as (state, runtime):
            await runtime.agents.get().create_session("session")
            store = state.execution.executions.state_store
            engine = store.storage_group.context.engine
            table = store.storage_group.metadata.tables["ai_execution_records"]

            async def force_conflict(repository, *args, **kwargs):
                if not first.is_set():
                    first.set()
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                return await original_claim(repository, *args, **kwargs)

            async def attempt(command, claim, **kwargs):
                fence = kwargs.get("turn_fence")
                if not first.is_set():
                    assert fence is not None and kwargs["observed"] is not None
                    outcomes["initial"] = fence
                    return await original_attempt(command, claim, **kwargs)
                outcomes["retry"] = fence
                async with engine.begin() as conn:
                    changed = await conn.execute(update(table).where(
                        outcomes["initial"].authority.owner_matches(table, store.store_key),
                    ).values(turn_expiry=datetime(2000, 1, 1, tzinfo=timezone.utc)))
                    assert changed.rowcount == 1
                try:
                    return await original_attempt(command, claim, **kwargs)
                except AIError as error:
                    outcomes["error"] = error.code
                    raise
                finally:
                    rejected.set()

            monkeypatch.setattr(ExecutionRepositoryImpl, "claim_prepared_agent_run_in_transaction", force_conflict)
            monkeypatch.setattr(StateCommands, "_commit_agent_attempt_checkpoint", attempt)
            handle = await runtime.sessions.start("session", "question", principal=runtime.default_principal)
            backend = runtime._execution_service.runtime_backend()
            owner = backend._turn_owners[handle.execution_id]
            await asyncio.wait_for(rejected.wait(), 8)
            await asyncio.wait_for(owner.closed.wait(), 8)
            assert outcomes["retry"] is outcomes["initial"]
            assert outcomes["error"] is ErrorCode.STORAGE_OWNER_MISMATCH
            assert model_calls == 0
    assert closed.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    assert closed.value.safe_details["phase"] == "source_turn_revoked"
