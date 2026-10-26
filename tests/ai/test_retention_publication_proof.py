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
        await repository.create_with_history_head(terminal)
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

        async def release_staging(**kwargs):
            calls.append("staging")

        controller._run_store = SimpleNamespace(release_staging_many=release_staging)
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
    async with Persistence.open(_NAMESPACE, storage=PersistenceStorage.from_plan(plan)) as persistence_106, Runtime.open(persistence=persistence_106, features=model_features(_models(), _features())) as runtime:
        session = await runtime.agents.get().create_session("s")
        result = (await session.run("retained input", timeout_seconds=5)).result
        principal = runtime.default_principal
        execution_id = result.execution_id

    storage = PersistenceStorage.from_plan(plan)
    async with Persistence.open(_NAMESPACE, storage=storage) as persistence_115, Runtime.open(persistence=persistence_115, features=model_features(_models(), _features())) as runtime:
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
                replace(stored, data=dict(stored.data, status="invalid"),
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
        await repository.create_with_history_head(replace(_execution(), status=ExecutionStatus.SUCCEEDED))
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
        assert calls == [("staging", "execution")]
    finally:
        await state.close()


async def test_dependency_hold_added_after_source_confirmation_prevents_close(tmp_path, monkeypatch):
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="retention-io", tenant_id="tenant")
    try:
        repository = state.execution.executions
        await repository.create_with_history_head(replace(_execution(), status=ExecutionStatus.SUCCEEDED))
        original = repository.close_retention

        async def acquire_before_close(execution_id, *, tenant_id):
            assert await repository.acquire_dependency_hold(
                execution_id, tenant_id=tenant_id, hold_id="late-dependent",
            )
            return await original(execution_id, tenant_id=tenant_id)

        monkeypatch.setattr(repository, "close_retention", acquire_before_close)
        calls = []
        assert not await _controller(repository, calls).release_execution_handoff("execution", tenant_id="tenant")
        assert calls == []
        async def source(transaction):
            del transaction
            return await repository.get("execution", tenant_id="tenant")
        durable = await repository.state_store.read(source)
        assert not durable.retention_closed
        assert durable.dependency_hold_ids == ("late-dependent",)
    finally:
        await state.close()
