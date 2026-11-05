#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Confirmed publication receipts preserve live holds and public read boundaries."""

import asyncio
from dataclasses import replace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.core import ExecutionStatus, StopReason, UsageMetrics
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ResultRecord
from linktools.ai.persistence import Persistence, PersistencePlan, PersistenceStorage
from linktools.ai.persistence.contracts import ExecutionTerminalPublication
from linktools.ai.runtime import Runtime
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.storage import StorageRoute

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_conversation_output_retention import _NAMESPACE, _features, _models
from .test_incremental_projection_boundaries import _storage
from .test_retention_read_io import _controller, _execution

pytestmark = pytest.mark.asyncio


async def _publication(repository, execution):
    observed = await repository.get_observed(execution.execution_id, tenant_id="tenant")
    assert observed is not None and observed.value == execution
    return ExecutionTerminalPublication(
        execution, ResultRecord(None, StopReason.END_TURN, UsageMetrics(), execution.updated_at),
        observed=observed,
    )


async def test_sessionless_split_handoff_does_not_retain_precompletion_observation(tmp_path, monkeypatch):
    storage = PersistenceStorage.from_plan(PersistencePlan(
        execution=StorageRoute.sqlite(tmp_path / "execution.db"),
        conversation=StorageRoute.sqlite(tmp_path / "conversation.db"),
    ))
    async with Persistence.open(_NAMESPACE, storage=storage) as state:
        publications = []
        handoffs = []
        original_confirm = state.retention.confirm_terminal_publication
        original_complete = LocalExecutionBackend._complete_handoff

        async def complete(backend, checkpoint):
            handoffs.append(checkpoint)
            return await original_complete(backend, checkpoint)

        def confirm(publication):
            publications.append(publication)
            if isinstance(publication, ExecutionTerminalPublication):
                assert publication.observed.value == publication.execution
            original_confirm(publication)

        monkeypatch.setattr(state.retention, "confirm_terminal_publication", confirm)
        monkeypatch.setattr(LocalExecutionBackend, "_complete_handoff", complete)
        async with Runtime.attach(persistence=state, features=model_features(_models(), _features())) as runtime:
            handle = await runtime.agents.get().start("standalone split conversation")
            result = (await handle.wait(timeout_seconds=5)).result
            assert result.status is ExecutionStatus.SUCCEEDED
            assert publications
            current = await state.execution.executions.get(result.execution_id, tenant_id=runtime.tenant_id)
            assert current.conversation and not current.durable_trace and not current.restart_checkpoints
            assert handoffs
            assert current.retention_closed


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_receipt_with_hold_allows_later_handoff_after_hold_release(tmp_path, backend):
    async with Persistence.open("retention-io", storage=_storage(backend, tmp_path), tenant_id="tenant") as state:
        repository = state.execution.executions
        terminal = replace(_execution(), status=ExecutionStatus.SUCCEEDED, dependency_hold_ids=("dependent",))
        await repository.create_execution(terminal)
        calls = []
        controller = _controller(repository, calls)
        publication = await _publication(repository, terminal)
        controller.confirm_terminal_publication(publication)

        assert not await controller.release_execution_handoff(terminal.execution_id, tenant_id="tenant")
        assert calls == []
        assert controller.terminal_publication(terminal.execution_id, tenant_id="tenant") is publication
        await repository.release_dependency_hold(terminal.execution_id, tenant_id="tenant", hold_id="dependent")

        assert await controller.release_execution_handoff(terminal.execution_id, tenant_id="tenant")
        assert calls == [("objects", terminal.execution_id)]
        current = await repository.get(terminal.execution_id, tenant_id="tenant")
        assert current.retention_closed and current.dependency_hold_ids == ()
        assert controller.terminal_publication(terminal.execution_id, tenant_id="tenant") is None


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_receipt_without_hold_rechecks_after_late_hold_cas_conflict(tmp_path, monkeypatch, backend):
    async with Persistence.open("retention-io", storage=_storage(backend, tmp_path), tenant_id="tenant") as state:
        repository = state.execution.executions
        terminal = replace(_execution(), status=ExecutionStatus.SUCCEEDED)
        await repository.create_execution(terminal)
        calls = []
        controller = _controller(repository, calls)
        publication = await _publication(repository, terminal)
        controller.confirm_terminal_publication(publication)
        assert await repository.acquire_dependency_hold(terminal.execution_id, tenant_id="tenant", hold_id="late")
        original_close = repository.close_retention_in_transaction
        observations = []

        async def close(transaction, execution_id, *, tenant_id, observed=None):
            observations.append(observed)
            return await original_close(transaction, execution_id, tenant_id=tenant_id, observed=observed)

        monkeypatch.setattr(repository, "close_retention_in_transaction", close)
        assert not await controller.release_execution_handoff(terminal.execution_id, tenant_id="tenant")
        assert observations == [publication.observed, None]
        assert calls == []
        current = await repository.get(terminal.execution_id, tenant_id="tenant")
        assert not current.retention_closed and current.dependency_hold_ids == ("late",)
        await repository.release_dependency_hold(terminal.execution_id, tenant_id="tenant", hold_id="late")
        assert await controller.release_execution_handoff(terminal.execution_id, tenant_id="tenant")
        assert calls == [("objects", terminal.execution_id)]


@pytest.mark.parametrize("blocker", ("authorization", "worker_failure"))
async def test_public_wait_receipt_preserves_authorization_and_worker_failure(tmp_path, monkeypatch, blocker):
    entered, release, waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def respond(messages, info):
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("finished")])

    async with Persistence.open(_NAMESPACE, storage=_storage("sqlite", tmp_path)) as state:
        async with Runtime.attach(persistence=state, features=model_features(_Models(_UsageFunctionModel(respond)), _features())) as runtime:
            session = await runtime.agents.get().create_session("receipt-wait")
            execution = await session.start("question")
            await asyncio.wait_for(entered.wait(), 5)
            service = runtime._execution_service
            backend = service._backend
            original_wait = backend.wait_terminal
            original_authorize = service.authorize_loaded_execution
            original_failure = backend.worker_failure
            receipt_reads = []
            authorizations = []
            original_receipt = service._retained_publication
            failure = AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

            async def wait_terminal(*args, **kwargs):
                waiting.set()
                await original_wait(*args, **kwargs)

            def receipt(*args, **kwargs):
                value = original_receipt(*args, **kwargs)
                if value is not None:
                    receipt_reads.append(value)
                return value

            async def authorize(value, *args, **kwargs):
                authorizations.append(value)
                if blocker == "authorization":
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                await original_authorize(value, *args, **kwargs)

            def worker_failure(execution_id, *, tenant_id):
                if blocker == "worker_failure" and state.retention.terminal_publication(execution_id, tenant_id=tenant_id) is not None:
                    return failure
                return original_failure(execution_id, tenant_id=tenant_id)

            monkeypatch.setattr(backend, "wait_terminal", wait_terminal)
            monkeypatch.setattr(service, "_retained_publication", receipt)
            monkeypatch.setattr(service, "authorize_loaded_execution", authorize)
            monkeypatch.setattr(backend, "worker_failure", worker_failure)
            task = asyncio.create_task(service.wait(execution.execution_id, principal=runtime.default_principal, timeout_seconds=5))
            try:
                await asyncio.wait_for(waiting.wait(), 5)
                release.set()
                with pytest.raises(AIError) as raised:
                    await task
                expected = ErrorCode.AUTHORIZATION_DENIED if blocker == "authorization" else ErrorCode.STORAGE_INTEGRITY_ERROR
                assert raised.value.code is expected
                assert receipt_reads and authorizations
                assert all(value.execution.execution_id == execution.execution_id for value in receipt_reads)
                assert all(value.status is ExecutionStatus.SUCCEEDED for value in authorizations)
                current = await state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
                assert not current.retention_closed
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
