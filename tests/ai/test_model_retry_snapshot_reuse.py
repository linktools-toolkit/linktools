#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A transport retry reuses its recovery point but still fences provider dispatch."""

import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

import pytest
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, DynamicCapability
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai_harness.step_persistence import StepPersistence
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.core import ExecutionStatus
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import Persistence, PersistenceStorage, SqlStepStore
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.persistence.runs import OfficialStepReader

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


@pytest.mark.asyncio
@pytest.mark.parametrize("dynamic", (False, True))
async def test_wrapper_retry_reuses_snapshot_and_keeps_dispatch_fence(tmp_path: Path, dynamic: bool) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'runtime.db'}")
    await provision_database(engine)
    native = SqlStepStore(engine, namespace="native-boundary", tenant_id="default")
    snapshots = []
    execution_updates = []
    provider_updates = []
    provider_snapshots = []
    from sqlalchemy import event
    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def sql_probe(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("UPDATE ai_state_records"):
            execution_updates.append(statement)
        if statement.startswith("INSERT INTO ai_step_snapshots"):
            snapshots.append((statement, parameters))
    order = []
    factory_calls = []

    class OutsideModel(WrapperModel):
        def __init__(self, model, ctx):
            super().__init__(model)
            self.ctx = ctx

        async def prepare(self):
            await asyncio.sleep(0)
            order.append("wrapper")
            self.ctx.messages[-1].metadata = {"final_wrapper": True}

        async def request(self, messages, model_settings, model_request_parameters):
            await self.prepare()
            try:
                return await self.wrapped.request(messages, model_settings, model_request_parameters)
            except RuntimeError:
                return await self.wrapped.request(messages, model_settings, model_request_parameters)

        @asynccontextmanager
        async def request_stream(self, messages, model_settings, model_request_parameters, run_context=None):
            await self.prepare()
            try:
                async with self.wrapped.request_stream(messages, model_settings, model_request_parameters, run_context) as response:
                    yield response
            except RuntimeError:
                async with self.wrapped.request_stream(messages, model_settings, model_request_parameters, run_context) as response:
                    yield response

    class LateHooks(AbstractCapability):
        id = "late-model-hooks"

        def get_ordering(self):
            return CapabilityOrdering(position="innermost")

        async def prepare_model_request(self, ctx, request_context):
            self.ctx = ctx
            order.append("prepare")
            return request_context

        def get_wrapper_model(self, model):
            return OutsideModel(model, self.ctx)

    async def factory(ctx):
        factory_calls.append(ctx.run_id)
        return LateHooks()

    async def provider(messages, info):
        order.append("provider")
        provider_updates.append(len(execution_updates))
        runs = await native.list_runs()
        assert len(runs) == 1
        snapshot = await OfficialStepReader(native).latest_snapshot(run_id=runs[0].run_id)
        provider_snapshots.append(snapshot)
        assert messages[-1].metadata == {"final_wrapper": True}
        assert snapshot.messages[-1].metadata == {"final_wrapper": True}
        if order.count("provider") == 1:
            raise RuntimeError("retry transport")
        return ModelResponse(parts=[TextPart("done")])

    features = RuntimeFeatures("native-boundary")
    features.capability(StepPersistence(store=native, capture_frontier=True), scope="runtime")
    features.capability(DynamicCapability(factory, id="dynamic-model-hooks") if dynamic else LateHooks(), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    try:
        async with Persistence.open("native-boundary", storage=PersistenceStorage.sql(engine)) as state:
            async with Runtime.attach(persistence=state,
                features=model_features(_Models(_UsageFunctionModel(provider)), features)) as runtime:
                execution = await runtime.agents.get().start("work", idempotency_key="work")
                result = await execution.wait(timeout_seconds=5)
                assert result.result.status is ExecutionStatus.SUCCEEDED, result.result
        assert order == ["prepare", "wrapper", "provider", "provider"]
        assert provider_updates[1] == provider_updates[0] + 1
        assert provider_snapshots[0] == provider_snapshots[1]
        assert len(snapshots) == 3
        assert len(factory_calls) == int(dynamic)
        # A fresh reader reconstructs the same retry budget and completed output.
        from linktools.ai.persistence import NativeCheckpoint
        runs = await native.list_runs()
        cold = SqlStepStore(engine, namespace="native-boundary", tenant_id="default")
        try:
            restored = NativeCheckpoint.decode(await cold.latest_snapshot(run_id=runs[0].run_id))
            assert restored.output_retries_used == 0
            assert restored.messages[-1].parts[0].content == "done"
        finally:
            await cold.close()
    finally:
        await native.close()
        await engine.dispose()

@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_retry_reuse_keeps_complete_budget_and_discards_failed_publication(failure):
    from types import SimpleNamespace
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai_harness.step_persistence import InMemoryStepStore
    from linktools.ai.capability.steps import AtomicStepPersistence
    from linktools.ai.runtime._step_store import RuntimeStepStore

    writes = []
    notifications = []
    retry_budget = 0
    fail = False
    provider_calls = []

    async def boundary(*, run=None, events=(), snapshot=None):
        writes.append(snapshot)
        if fail:
            raise failure()

    async def emit(event):
        notifications.append(event)

    store = RuntimeStepStore(InMemoryStepStore(), InMemoryStepStore(),
                             commit_boundary=boundary, output_retries_used=lambda: retry_budget)
    capability = AtomicStepPersistence(store=store, run_id="run", capture_frontier=True)
    ctx = SimpleNamespace(run_id="run", run_step=1, conversation_id="conversation",
                          messages=[ModelRequest(parts=[UserPromptPart("same")])], emit=emit)
    capability._atomic_model_context = ctx
    def provider(messages, info):
        provider_calls.append(1)
        return ModelResponse(parts=[TextPart("done")])
    wrapper = capability.get_wrapper_model(FunctionModel(provider))
    await wrapper._publish()
    await wrapper._publish()
    assert writes[0] is not None and writes[1] is None
    assert len(notifications) == 2
    retry_budget = 1
    await wrapper._publish()
    assert writes[-1] is not None
    fail = True
    with pytest.raises(failure):
        await wrapper.request(ctx.messages, None, None)
    assert not provider_calls
    assert writes[-1] is None
    assert len(notifications) == 3
    fail = False
    await wrapper._publish()
    assert writes[-1] is not None
    assert len(notifications) == 4
    # A new logical model request has no reusable prior publication.
    await capability.get_wrapper_model(wrapper.wrapped)._publish()
    assert writes[-1] is not None

@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["messages", "step", "state", "retry"])
async def test_model_retry_changed_recovery_point_is_written(change, monkeypatch):
    from types import SimpleNamespace
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai_harness.step_persistence import InMemoryStepStore
    from linktools.ai.capability.steps import AtomicStepPersistence
    from linktools.ai.runtime._step_store import RuntimeStepStore
    from dataclasses import replace

    writes = []
    budget = 0
    async def boundary(**values):
        writes.append(values["snapshot"])
    async def emit(event):
        pass
    store = RuntimeStepStore(InMemoryStepStore(), InMemoryStepStore(),
                             commit_boundary=boundary, output_retries_used=lambda: budget)
    capability = AtomicStepPersistence(store=store, run_id="run", capture_frontier=True)
    ctx = SimpleNamespace(run_id="run", run_step=1, conversation_id="conversation",
                          messages=[ModelRequest(parts=[UserPromptPart("before")])], emit=emit)
    capability._atomic_model_context = ctx
    wrapper = capability.get_wrapper_model(FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("done")])))
    await wrapper._publish()
    if change == "messages":
        ctx.messages[0].parts[0].content = "after"
    elif change == "step":
        ctx.run_step += 1
    elif change == "retry":
        budget += 1
    else:
        original = store.capture_snapshot
        monkeypatch.setattr(store, "capture_snapshot", lambda snapshot: replace(original(snapshot), state="interrupted"))
    await wrapper._publish()
    assert len(writes) == 2 and all(snapshot is not None for snapshot in writes)

@pytest.mark.asyncio
async def test_model_retry_without_bound_runtime_writer(tmp_path):
    from types import SimpleNamespace
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai_harness.step_persistence import InMemoryStepStore
    from linktools.ai.capability.steps import AtomicStepPersistence
    from linktools.ai.runtime._step_store import RuntimeStepStore
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'standalone.db'}")
    native = SqlStepStore(engine, namespace="retry", tenant_id="default")
    notifications = []
    async def emit(event):
        notifications.append(event)
    try:
        await native.initialize(local=True)
        store = RuntimeStepStore(native, InMemoryStepStore())
        capability = AtomicStepPersistence(store=store, run_id="run", capture_frontier=True)
        ctx = SimpleNamespace(run_id="run", run_step=1, conversation_id="conversation",
                              messages=[ModelRequest(parts=[UserPromptPart("same")])], emit=emit)
        capability._atomic_model_context = ctx
        wrapper = capability.get_wrapper_model(FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("done")])))
        await wrapper._publish()
        await wrapper._publish()
        assert len(await native.list_snapshots(run_id="run")) == 1
        assert len(notifications) == 2
    finally:
        await native.close()
        await engine.dispose()

@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["cancel", "ack_loss"])
async def test_empty_retry_boundary_keeps_database_fence_and_resolves_lost_ack(tmp_path, monkeypatch, fault):
    from dataclasses import replace
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import AsyncSessionTransaction
    from linktools.ai.errors import AIError, ErrorCode

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'fenced-retry.db'}")
    await provision_database(engine)
    native = SqlStepStore(engine, namespace="fenced-retry", tenant_id="default")
    first_call = asyncio.Event()
    resume = asyncio.Event()
    calls = []
    snapshots = []
    fence_rowcounts = []
    lost = []
    marker = "retry_boundary_ack_loss"
    original_exit = AsyncSessionTransaction.__aexit__

    def record_sql(connection, cursor, statement, parameters, context, many):
        if statement.startswith("INSERT INTO ai_step_snapshots"):
            snapshots.append(parameters)
        if len(calls) == 1 and resume.is_set() and statement.startswith("UPDATE ai_state_records"):
            fence_rowcounts.append(cursor.rowcount)
            if fault == "ack_loss" and not lost:
                connection.info[marker] = True

    async def lose_ack(transaction, error_type, error, traceback):
        selected = False
        if error_type is None and not transaction.nested and not lost:
            connection = await transaction.session.connection()
            selected = connection.info.pop(marker, False)
        result = await original_exit(transaction, error_type, error, traceback)
        if selected:
            assert len(snapshots) == 1
            lost.append(True)
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN, "retry fence committed; acknowledgement lost")
        return result

    event.listen(engine.sync_engine, "after_cursor_execute", record_sql)
    monkeypatch.setattr(AsyncSessionTransaction, "__aexit__", lose_ack)

    class RetryModel(WrapperModel):
        async def request(self, messages, model_settings, model_request_parameters):
            try:
                return await self.wrapped.request(messages, model_settings, model_request_parameters)
            except RuntimeError:
                return await self.wrapped.request(messages, model_settings, model_request_parameters)

        @asynccontextmanager
        async def request_stream(self, messages, model_settings, model_request_parameters, run_context=None):
            try:
                async with self.wrapped.request_stream(messages, model_settings, model_request_parameters, run_context) as response:
                    yield response
            except RuntimeError:
                async with self.wrapped.request_stream(messages, model_settings, model_request_parameters, run_context) as response:
                    yield response

    class RetryHooks(AbstractCapability):
        id = "retry-model"
        def get_wrapper_model(self, model):
            return RetryModel(model)

    async def provider(messages, info):
        calls.append(True)
        if len(calls) == 1:
            first_call.set()
            await resume.wait()
            raise RuntimeError("transport failed")
        return ModelResponse(parts=[TextPart("done")])

    features = RuntimeFeatures("fenced-retry")
    features.capability(StepPersistence(store=native, capture_frontier=True), scope="runtime")
    features.capability(RetryHooks(), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    try:
        async with Persistence.open("fenced-retry", storage=PersistenceStorage.sql(engine)) as state:
            async with Runtime.attach(persistence=state,
                features=model_features(_Models(_UsageFunctionModel(provider)), features)) as runtime:
                execution = await runtime.agents.get().start("work", idempotency_key="work")
                await asyncio.wait_for(first_call.wait(), timeout=5)
                assert len(snapshots) == 1
                if fault == "cancel":
                    repository = state.execution.executions
                    current = await repository.get(execution.execution_id, tenant_id="default")
                    await repository.compare_and_swap(execution.execution_id, tenant_id="default",
                        expected_revision=current.revision,
                        next_record=replace(current, revision=current.revision + 1, status=ExecutionStatus.CANCELLING))
                resume.set()
                result = await execution.wait(timeout_seconds=5)
                if fault == "cancel":
                    assert result.result.status is ExecutionStatus.CANCELLED
                    assert len(calls) == 1
                    assert 0 in fence_rowcounts
                else:
                    assert result.result.status is ExecutionStatus.SUCCEEDED
                    assert len(calls) == 2 and lost == [True]
                    assert len(snapshots) == 3
                    assert fence_rowcounts[0] == 1
    finally:
        resume.set()
        await native.close()
        await engine.dispose()
