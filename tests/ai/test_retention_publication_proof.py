#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""A durable retention closure proves publication across process boundaries."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from linktools.ai.core import ExecutionStatus, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import Persistence, PersistencePlan, PersistenceStorage
from linktools.ai.persistence.state import active_state_scope
from linktools.ai.persistence.contracts import ExecutionTerminalPublication
from linktools.ai.storage import StorageRoute

from ._runtime_test_helpers import model_features
from .test_conversation_output_retention import _NAMESPACE, _features, _models
from .test_incremental_projection_boundaries import _storage
from .test_retention_read_io import _controller, _execution

pytestmark = pytest.mark.asyncio


class _HostAuthorization(TenantAuthorizationPolicy):
    """Authorize this fixture’s native SQL Session operations."""


@pytest.mark.parametrize("cached", ("absent", "pending"))
async def test_stale_unadmitted_cache_cannot_prove_publication(tmp_path, monkeypatch, cached):
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="retention-io", tenant_id="tenant")
    try:
        repository = state.execution.executions
        terminal = replace(
            _execution(), session_id="session", conversation=True,
            status=ExecutionStatus.SUCCEEDED, retention_closed=False,
        )
        await repository.create_execution(terminal)
        original_get = repository.get
        stale = None if cached == "absent" else replace(
            terminal, status=ExecutionStatus.PENDING_START, started_at=None,
        )

        source_reads = []

        async def cached_get(*args, **kwargs):
            if active_state_scope() is None:
                return stale
            source_reads.append(1)
            return await original_get(*args, **kwargs)

        original_source = repository.get_retention_in_transaction

        async def source_get(*args, **kwargs):
            assert active_state_scope() is not None
            source_reads.append(1)
            return await original_source(*args, **kwargs)

        monkeypatch.setattr(repository, "get", cached_get)
        monkeypatch.setattr(repository, "get_retention_in_transaction", source_get)
        calls = []
        controller = _controller(repository, calls)

        async def missing_receipt(transaction, **kwargs):
            del transaction
            # An authoritative terminal read must be required even if the old
            # cache says admission never happened. This source is admitted.
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)

        async def no_session(*args, **kwargs):
            return None

        async def release_session(*args, **kwargs):
            calls.append("released")

        controller._conversation = SimpleNamespace(sessions=SimpleNamespace(
            state_store=repository.state_store,
            pending_turn_commit_in_transaction=missing_receipt,
            get=no_session,
            release_execution=release_session,
        ))
        error = None
        try:
            await controller.release_execution_handoff("execution", tenant_id="tenant")
        except AIError as caught:
            error = caught
        async def read(transaction):
            del transaction
            return await original_get("execution", tenant_id="tenant")
        current = await repository.state_store.read(read)
        assert source_reads, "publication requires an authoritative terminal snapshot"
        assert not current.retention_closed
        assert calls == []
        assert error is not None and error.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    finally:
        await state.close()


@pytest.mark.parametrize("split", (False, True))
async def test_closed_public_results_reuse_proof_across_runtime(tmp_path, monkeypatch, split):
    plan = PersistencePlan(
        execution=StorageRoute.sqlite(tmp_path / "execution.db"),
        conversation=StorageRoute.sqlite(tmp_path / ("conversation.db" if split else "execution.db")),
    )
    storage = PersistenceStorage.from_plan(plan)
    authorization = _HostAuthorization()
    async with (
        Persistence.open(_NAMESPACE, storage=storage) as state,
        Runtime.open(persistence=state, features=model_features(_models(), _features()),
                     authorization=authorization) as runtime,
    ):
        result = (await runtime.agents.get().run("retained input", timeout_seconds=5)).result
        principal = runtime.default_principal
        execution_id = result.execution_id
        await runtime.sessions.create_turn_session("s", principal=principal)
        turn = await runtime.sessions.start_turn("s", "canonical input", agent="default",
                                            principal=principal, idempotency_key="turn")
        assert (await turn.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED

    async with (
        Persistence.open(_NAMESPACE, storage=storage) as state,
        Runtime.open(persistence=state, features=model_features(_models(), _features()),
                     authorization=authorization) as runtime,
    ):
        record = await state.execution.executions.get(execution_id, tenant_id=principal.tenant_id)
        assert record.retention_closed and record.session_id is None
        turns = state.session_turns
        turn_reads = []
        original = turns.get_history_turn

        async def count_turn_read(*args, **kwargs):
            turn_reads.append(1)
            return await original(*args, **kwargs)

        monkeypatch.setattr(turns, "get_history_turn", count_turn_read)
        assert (await runtime.executions.result(execution_id, principal=principal)).output == result.output
        assert await runtime._execution_service.result_payload_size(execution_id, principal=principal) > 0
        assert turn_reads == []

        # Corrupt only the real canonical Session owner. A retained independent
        # Execution neither uses nor vouches for that transcript.
        row = await turns.get_turn(turn.key)
        assert row is not None and row["result_json"]["transcript"]
        async with turns._context.engine.begin() as connection:
            mutation = turns._turns.update().filter_by(**turns._turn_values(turn.key)).values(
                result_json={**row["result_json"], "transcript": {}},
            )
            assert (await connection.execute(mutation)).rowcount == 1
        assert (await runtime.executions.result(execution_id, principal=principal)).output == result.output
        assert turn_reads == []

    async with (
        Persistence.open(_NAMESPACE, storage=storage, tenant_id="default", read_only=True) as state,
        History.open(persistence=state, authorization=authorization) as history,
    ):
        assert (await history.result(execution_id, principal=principal)).output == result.output
        with pytest.raises(AIError) as raised:
            await history.turn_result("s", turn.key.turn_id, principal=principal)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("committed", (False, True))
async def test_unknown_close_preserves_only_durable_publication_proof(tmp_path, monkeypatch, committed):
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="retention-io", tenant_id="tenant")
    try:
        repository = state.execution.executions
        await repository.create_execution(replace(_execution(), status=ExecutionStatus.SUCCEEDED))
        calls = []
        controller = _controller(repository, calls)
        original = repository.state_store.mutate

        async def unknown(callback):
            outer = active_state_scope() is None
            if not outer:
                return await original(callback)
            if committed:
                await original(callback)
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        with monkeypatch.context() as patch:
            patch.setattr(repository.state_store, "mutate", unknown)
            with pytest.raises(AIError) as raised:
                await controller.release_execution_handoff("execution", tenant_id="tenant")
            assert raised.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        assert calls == []
        async def source(transaction):
            del transaction
            return await repository.get("execution", tenant_id="tenant")
        durable = await repository.state_store.read(source)
        assert durable.retention_closed is committed
        assert await controller.release_execution_handoff("execution", tenant_id="tenant")
        assert calls == [("objects", "execution")]
    finally:
        await state.close()


@pytest.mark.parametrize("confirmed", (False, True))
async def test_dependency_hold_added_after_source_confirmation_prevents_close(tmp_path, monkeypatch, confirmed):
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="retention-io", tenant_id="tenant")
    try:
        repository = state.execution.executions
        terminal = replace(_execution(), status=ExecutionStatus.SUCCEEDED)
        await repository.create_execution(terminal)
        original = repository.close_retention_in_transaction

        async def acquire_before_close(transaction, execution_id, *, tenant_id, observed=None):
            assert await repository.acquire_dependency_hold(
                execution_id, tenant_id=tenant_id, hold_id="late-dependent",
            )
            return await original(transaction, execution_id, tenant_id=tenant_id, observed=observed)

        monkeypatch.setattr(repository, "close_retention_in_transaction", acquire_before_close)
        calls = []
        controller = _controller(repository, calls)
        if confirmed:
            controller.confirm_terminal_publication(terminal)
        assert not await controller.release_execution_handoff("execution", tenant_id="tenant")
        assert calls == []
        async def source(transaction):
            del transaction
            return await repository.get("execution", tenant_id="tenant")
        durable = await repository.state_store.read(source)
        assert not durable.retention_closed
        assert durable.dependency_hold_ids == ("late-dependent",)
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_same_runtime_confirmed_terminal_reuses_publication_receipt(tmp_path, monkeypatch, backend):
    storage = _storage(backend, tmp_path)
    async with Persistence.open(_NAMESPACE, storage=storage) as state:
        retention = state.retention
        confirmed = []
        original_confirm = retention.confirm_terminal_publication

        def confirm(execution):
            original_confirm(execution)
            confirmed.append(execution)

        monkeypatch.setattr(retention, "confirm_terminal_publication", confirm)
        async with Runtime.open(persistence=state, features=model_features(_models(), _features())) as runtime:
            receipt_reads = []
            if backend == "memory":
                owner = state.conversation.sessions
                read_source = "pending_turn_commit_in_transaction"
                session = await runtime.agents.get().create_session("receipt-session")
                run = session.run
            else:
                assert state.conversation.sessions is None
                owner = state.execution.executions
                read_source = "get_retention_in_transaction"
                run = runtime.agents.get().run
            original_read = getattr(owner, read_source)

            async def read_receipt(*args, **kwargs):
                receipt_reads.append(1)
                return await original_read(*args, **kwargs)

            monkeypatch.setattr(owner, read_source, read_receipt)
            result = (await run("confirmed publication", timeout_seconds=5)).result
            assert result.status is ExecutionStatus.SUCCEEDED
            assert len(confirmed) == 1
            receipt = confirmed[0]
            assert isinstance(receipt, ExecutionTerminalPublication)
            terminal = receipt.execution
            assert receipt.observed.value == terminal
            assert receipt.session_released is (backend == "memory")
            assert (terminal.session_id is not None) is (backend == "memory")
            assert terminal.execution_id == result.execution_id
            assert receipt_reads == []
            assert not retention.terminal_publication_confirmed(terminal)
            current = await state.execution.executions.get(result.execution_id, tenant_id=runtime.tenant_id)
            assert current.retention_closed


async def test_publication_receipt_matches_exact_source_revision_and_clears_on_close(tmp_path):
    async with Persistence.open("retention-io", storage=_storage("memory", tmp_path), tenant_id="tenant") as state:
        terminal = replace(_execution(), status=ExecutionStatus.SUCCEEDED)
        retention = state.retention
        assert not retention.terminal_publication_confirmed(terminal)
        retention.confirm_terminal_publication(terminal)
        assert retention.terminal_publication_confirmed(terminal)
        assert not retention.terminal_publication_confirmed(replace(terminal, revision=terminal.revision + 1))
        assert not retention.terminal_publication_confirmed(replace(terminal, execution_id="other"))
        with pytest.raises(ValueError):
            retention.confirm_terminal_publication(replace(terminal, status=ExecutionStatus.STARTED))
    assert not retention.terminal_publication_confirmed(terminal)


async def test_publication_receipt_does_not_assume_session_released(tmp_path):
    async with Persistence.open("retention-io", storage=_storage("sqlite", tmp_path), tenant_id="tenant") as state:
        repository = state.execution.executions
        terminal = replace(_execution(), status=ExecutionStatus.SUCCEEDED,
                           session_id="session", conversation=True)
        await repository.create_execution(terminal)
        calls = []
        controller = _controller(repository, calls)
        controller.confirm_terminal_publication(terminal)
        session_reads = []
        original_release = controller._conversation.sessions.release_execution

        async def read_session(session_id, *, tenant_id):
            session_reads.append((session_id, tenant_id))
            return SimpleNamespace(active_execution_id=terminal.execution_id)

        controller._conversation.sessions = SimpleNamespace(
            state_store=repository.state_store, get=read_session,
            release_execution=original_release,
        )
        assert await controller.release_execution_handoff(terminal.execution_id, tenant_id="tenant")
        assert session_reads == [("session", "tenant")]
        assert calls == [("session", "session"), ("objects", terminal.execution_id)]
