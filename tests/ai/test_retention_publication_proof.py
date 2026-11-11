#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""A durable retention closure proves publication across process boundaries."""
from ._runtime_test_helpers import model_features

from dataclasses import replace
from types import SimpleNamespace

import pytest

from linktools.ai.persistence import Persistence
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage, PersistencePlan
from linktools.ai.persistence.state import active_state_scope
from linktools.ai.persistence.contracts import ExecutionTerminalPublication
from linktools.ai.storage import StorageRoute

from .test_conversation_output_retention import _NAMESPACE, _features, _models
from .test_incremental_projection_boundaries import _storage
from .test_retention_read_io import _controller, _execution

pytestmark = pytest.mark.asyncio


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

        original_source = repository.get_in_transaction

        async def source_get(*args, **kwargs):
            source_reads.append(1)
            return await original_source(*args, **kwargs)

        monkeypatch.setattr(repository, "get", cached_get)
        monkeypatch.setattr(repository, "get_in_transaction", source_get)
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
    async with Persistence.open(_NAMESPACE, storage=PersistenceStorage.from_plan(plan)) as persistence_106, Runtime.attach(persistence=persistence_106, features=model_features(_models(), _features())) as runtime:
        session = await runtime.agents.get().create_session("s")
        result = (await session.run("retained input", timeout_seconds=5)).result
        principal = runtime.default_principal
        execution_id = result.execution_id

    storage = PersistenceStorage.from_plan(plan)
    async with Persistence.open(_NAMESPACE, storage=storage) as persistence_115, Runtime.attach(persistence=persistence_115, features=model_features(_models(), _features())) as runtime:
        record = await runtime._state.execution.executions.get(execution_id, tenant_id=principal.tenant_id)
        assert record.retention_closed
        receipt_reads = []
        sessions = runtime._state.conversation.sessions
        original = sessions.pending_turn_commit_in_transaction

        async def count_receipt(*args, **kwargs):
            receipt_reads.append(1)
            return await original(*args, **kwargs)

        monkeypatch.setattr(sessions, "pending_turn_commit_in_transaction", count_receipt)
        assert (await runtime.executions.result(execution_id, principal=principal)).output == result.output
        assert await runtime._execution_service.result_payload_size(execution_id, principal=principal) > 0
        assert receipt_reads == []

        # Timeline still owns current integrity validation; a closed Execution
        # does not waive the independent Conversation reader's responsibility.
        async def corrupt(transaction):
            stored = await transaction.get_record(sessions._timeline_commit_key("s", 1))
            assert stored is not None
            assert await transaction.replace_record(
                replace(stored, data=dict(stored.data, execution_id="missing-execution"),
                        storage_revision=stored.storage_revision + 1),
                expected_storage_revision=stored.storage_revision,
            )
        await sessions.state_store.mutate(corrupt)
        assert (await runtime.executions.result(execution_id, principal=principal)).output == result.output

    async with Persistence.open(_NAMESPACE, storage=PersistenceStorage.from_plan(plan), tenant_id="default", read_only=True) as history_persistence_145, History.open(persistence=history_persistence_145) as history:
        assert (await history.result(execution_id, principal=principal)).output == result.output
        with pytest.raises(AIError) as raised:
            await history.session_timeline("s", principal=principal)
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
        async with Runtime.attach(persistence=state, features=model_features(_models(), _features())) as runtime:
            session = await runtime.agents.get().create_session("receipt-session")
            sessions = state.conversation.sessions
            original_read = sessions.pending_turn_commit_in_transaction
            receipt_reads = []

            async def read_receipt(*args, **kwargs):
                receipt_reads.append(kwargs["execution_id"])
                return await original_read(*args, **kwargs)

            monkeypatch.setattr(sessions, "pending_turn_commit_in_transaction", read_receipt)
            result = (await session.run("confirmed publication", timeout_seconds=5)).result
            assert result.status is ExecutionStatus.SUCCEEDED
            assert len(confirmed) == 1
            receipt = confirmed[0]
            terminal = receipt.execution if isinstance(receipt, ExecutionTerminalPublication) else receipt
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
