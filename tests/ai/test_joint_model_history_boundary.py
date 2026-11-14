#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Same-group request history and native dispatch retain one producer fence."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai_harness.step_persistence import StepPersistence
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.capability import TraceCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage, SqlStepStore
from linktools.ai.persistence.runs import RuntimeAgentRunStore
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_step_message_cache import _SqlTraffic


@pytest.mark.asyncio
async def test_joint_request_preserves_sync_history_with_less_sql(tmp_path: Path, monkeypatch) -> None:
    measurements = []
    join = RuntimeAgentRunStore.can_join_step_projection
    for enabled in (False, True):
        monkeypatch.setattr(RuntimeAgentRunStore, "can_join_step_projection", join if enabled else lambda self, run_id: False)
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / str(enabled)}")
        await provision_database(engine)
        native = SqlStepStore(engine, namespace="joint", tenant_id="default")
        traffic = _SqlTraffic()
        observed = []
        transactions = {"begins": 0, "commits": 0, "rollbacks": 0, "write_transactions": 0}
        writes = {}

        def begin(connection):
            transactions["begins"] += 1
            writes[connection] = False

        def statement(connection, cursor, statement, parameters, context, many):
            if statement.split()[0].upper() in {"INSERT", "UPDATE", "DELETE", "REPLACE"}:
                writes[connection] = True

        def commit(connection):
            transactions["commits"] += 1
            transactions["write_transactions"] += int(writes.pop(connection, False))

        def rollback(connection):
            transactions["rollbacks"] += 1
            writes.pop(connection, None)

        listeners = (("before_cursor_execute", traffic.capture), ("before_cursor_execute", statement),
                     ("begin", begin), ("commit", commit), ("rollback", rollback))

        async def provider(messages, info):
            source, = await native.list_runs()
            assert any(value.kind == "model_request_started" for value in await native.list_events(run_id=source.run_id))
            archive = state.run_store.read_store(PersistenceDomain.TRACE)
            interactions = await archive.list_model_interactions(agent_run_id=source.run_id)
            assert len(interactions) == 1
            observed.append(interactions[0].model_request_seq)
            return ModelResponse(parts=[TextPart("done")])

        features = RuntimeFeatures("joint")
        features.capability(StepPersistence(store=native, capture_frontier=True), scope="runtime")
        features.capability(TraceCapability(commit=CommitPolicy.sync()), scope="runtime")
        features.agent("default", model="default", allow_tools=())
        try:
            async with Persistence.open("joint", storage=PersistenceStorage.sql(engine)) as state:
                async with Runtime.attach(persistence=state,
                    features=model_features(_Models(_UsageFunctionModel(provider)), features)) as runtime:
                    for name, listener in listeners:
                        event.listen(engine.sync_engine, name, listener)
                    execution = await runtime.agents.get().start("work", idempotency_key="work")
                    result = await execution.wait(timeout_seconds=5)
                    assert result.result.status is ExecutionStatus.SUCCEEDED, result.result
                    for name, listener in listeners:
                        event.remove(engine.sync_engine, name, listener)
            assert observed == [1]
            measurements.append({"statements": dict(traffic.statements), "transactions": transactions,
                                 "bound_bytes": traffic.bound_bytes})
        finally:
            await native.close()
            await engine.dispose()
    print("joint request SQL:", measurements)
    assert measurements[1]["statements"]["UPDATE"] < measurements[0]["statements"]["UPDATE"]
    assert measurements[1]["transactions"]["commits"] < measurements[0]["transactions"]["commits"]
    assert measurements[1]["transactions"]["write_transactions"] < measurements[0]["transactions"]["write_transactions"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("prepare", "trace_rollback", "lost_ack", "cancel_prepare", "cancel_commit"))
async def test_joint_boundary_settles_preparation_rollback_and_lost_ack(tmp_path: Path, monkeypatch, fault: str) -> None:
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.persistence._step_archive import StateStepArchive
    from linktools.ai.persistence.state import active_state_scope

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / fault}")
    await provision_database(engine)
    native = SqlStepStore(engine, namespace="joint-fault", tenant_id="default")
    injected = False
    native_written = False
    ready, release = asyncio.Event(), asyncio.Event()
    caller = None
    provider_calls = []
    prepare = native.prepare_step
    write = native.write_prepared_step
    sync = StateStepArchive.sync_prepared_projection

    async def prepare_step(**kwargs):
        nonlocal injected
        if fault in {"prepare", "cancel_prepare"} and not injected and any(value.kind == "model_request_started" for value in kwargs.get("events", ())):
            injected = True
            if fault == "cancel_prepare":
                ready.set()
                await release.wait()
            else:
                raise RuntimeError("native preparation failed")
        return await prepare(**kwargs)

    async def write_step(transaction, prepared, **kwargs):
        nonlocal native_written
        receipt = await write(transaction, prepared, **kwargs)
        if any(value.kind == "model_request_started" for value in prepared.events):
            native_written = True
        return receipt

    async def sync_projection(self, *args, **kwargs):
        nonlocal injected
        if fault == "trace_rollback" and native_written and not injected:
            injected = True
            raise RuntimeError("trace publication failed")
        return await sync(self, *args, **kwargs)

    from linktools.ai.capability.steps import AtomicStepPersistence
    publish_model = AtomicStepPersistence.publish_model_boundary

    async def observe_caller(cls, model):
        nonlocal caller
        caller = asyncio.current_task()
        return await publish_model(model)

    monkeypatch.setattr(AtomicStepPersistence, "publish_model_boundary", classmethod(observe_caller))
    monkeypatch.setattr(native, "prepare_step", prepare_step)
    monkeypatch.setattr(native, "write_prepared_step", write_step)
    monkeypatch.setattr(StateStepArchive, "sync_prepared_projection", sync_projection)

    async def provider(messages, info):
        provider_calls.append(True)
        return ModelResponse(parts=[TextPart("done")])

    features = RuntimeFeatures("joint-fault")
    features.capability(StepPersistence(store=native, capture_frontier=True), scope="runtime")
    features.capability(TraceCapability(commit=CommitPolicy.sync()), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    try:
        async with Persistence.open("joint-fault", storage=PersistenceStorage.sql(engine)) as state:
            group = state.run_store._execution_repository.state_store.storage_group
            mutate = group.mutate

            async def lose_ack(stores, callback):
                nonlocal injected
                outer = active_state_scope() is None
                value = await mutate(stores, callback)
                if fault in {"lost_ack", "cancel_commit"} and native_written and outer and not injected:
                    injected = True
                    if fault == "cancel_commit":
                        ready.set()
                        await release.wait()
                    else:
                        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
                return value

            monkeypatch.setattr(group, "mutate", lose_ack)
            async with Runtime.attach(persistence=state,
                features=model_features(_Models(_UsageFunctionModel(provider)), features)) as runtime:
                execution = await runtime.agents.get().start("work", idempotency_key="work")
                if fault.startswith("cancel"):
                    await asyncio.wait_for(ready.wait(), 5)
                    cancellation = asyncio.create_task(execution.cancel())
                    assert caller is not None
                    caller.cancel()
                    await asyncio.sleep(0)
                    release.set()
                    await asyncio.wait_for(cancellation, 5)
                result = await execution.wait(timeout_seconds=5)
                assert injected
                expected = (ExecutionStatus.SUCCEEDED if fault == "lost_ack" else
                            ExecutionStatus.CANCELLED if fault.startswith("cancel") else ExecutionStatus.FAILED)
                assert result.result.status is expected
                assert provider_calls == ([True] if fault == "lost_ack" else [])
                assert not state.run_store._durability_flights
                source, = await native.list_runs()
                archive = state.run_store.read_store(PersistenceDomain.TRACE)
                interactions = await archive.list_model_interactions(agent_run_id=source.run_id)
                assert len(interactions) == 1
                assert interactions[0].status == expected.value.upper()
    finally:
        await native.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_unwrapped_dispatch_does_not_grant_retry_a_fence_credit() -> None:
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.wrapper import WrapperModel
    from linktools.ai.capability.steps import AtomicStepPersistence, _BoundaryModel

    fences = []
    calls = []
    pending = True

    async def publish():
        nonlocal pending
        fences.append(True)
        pending = False

    def provider(messages, info):
        calls.append(True)
        return ModelResponse(parts=[TextPart("done")])

    model = FunctionModel(provider)
    boundary = _BoundaryModel(model, publish, lambda: pending)
    dispatched = await AtomicStepPersistence.publish_model_boundary(boundary)
    assert dispatched is model
    await dispatched.request([], None, ModelRequestParameters())
    await boundary.request([], None, ModelRequestParameters())
    assert len(fences) == len(calls) == 2
    assert await AtomicStepPersistence.publish_model_boundary(boundary) is None
    assert await AtomicStepPersistence.publish_model_boundary(WrapperModel(boundary)) is None
    assert len(fences) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", (False, True))
async def test_custom_early_exit_keeps_prior_sync_trace(tmp_path: Path, monkeypatch, failure: bool) -> None:
    from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
    from pydantic_ai.models.wrapper import WrapperModel
    from types import SimpleNamespace
    from linktools.ai.persistence._sql import SqlStateStore

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'custom'}")
    await provision_database(engine)
    native = SqlStepStore(engine, namespace="custom-joint", tenant_id="default")
    visible = []
    calls = []

    active_store = None
    gate = RuntimeAgentRunStore.can_join_step_projection
    def remember_store(store, run_id):
        nonlocal active_store
        active_store = store
        return gate(store, run_id)
    monkeypatch.setattr(RuntimeAgentRunStore, "can_join_step_projection", remember_store)

    class EarlyModel(WrapperModel):
        async def request(self, messages, settings, parameters):
            runs = await state.run_store.read_store(PersistenceDomain.TRACE).list_agent_runs()
            source = runs[0]
            rows = await state.run_store.read_store(PersistenceDomain.TRACE).list_model_interactions(agent_run_id=source.agent_run_id)
            visible.append(len(rows))
            assert active_store is not None
            original = active_store._execution_repository
            assert gate(active_store, source.agent_run_id)
            split = SqlStateStore(engine, context=original.state_store.context)
            try:
                active_store._execution_repository = SimpleNamespace(state_store=split)
                assert not gate(active_store, source.agent_run_id)
            finally:
                active_store._execution_repository = original
            if failure:
                raise RuntimeError("custom early failure")
            return ModelResponse(parts=[TextPart("short circuit")])

        @asynccontextmanager
        async def request_stream(self, messages, settings, parameters, run_context=None):
            response = await self.request(messages, settings, parameters)
            async def respond(messages, info):
                return response
            async with _UsageFunctionModel(respond).request_stream(messages, settings, parameters, run_context) as stream:
                yield stream

    class EarlyExit(AbstractCapability):
        id = "early-exit"

        def get_ordering(self):
            return CapabilityOrdering(position="innermost")

        def get_wrapper_model(self, model):
            return EarlyModel(model)

    async def provider(messages, info):
        calls.append(True)
        return ModelResponse(parts=[TextPart("provider")])

    features = RuntimeFeatures("custom-joint")
    features.capability(StepPersistence(store=native, capture_frontier=True), scope="runtime")
    features.capability(TraceCapability(commit=CommitPolicy.sync()), scope="runtime")
    features.capability(EarlyExit(), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    try:
        async with Persistence.open("custom-joint", storage=PersistenceStorage.sql(engine)) as state:
            async with Runtime.attach(persistence=state,
                features=model_features(_Models(_UsageFunctionModel(provider)), features)) as runtime:
                execution = await runtime.agents.get().start("work", idempotency_key="work")
                result = await execution.wait(timeout_seconds=5)
                assert result.result.status is (ExecutionStatus.FAILED if failure else ExecutionStatus.SUCCEEDED), result.result
                assert visible == [1]
                assert calls == []
    finally:
        await native.close()
        await engine.dispose()
