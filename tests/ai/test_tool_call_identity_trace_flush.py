#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Call identity needs no history reads; explicit Trace publication is visible."""

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from linktools.ai.capability import AgentContext, TraceCapability
from linktools.ai.core import CommitPolicy
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import InMemoryCacheBackend

from ._runtime_test_helpers import model_features
from .test_explicit_runtime_features import _FeatureFunctionModel, _FeatureModels


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("mode", ["batch", "terminal", "sync", "none"])
async def test_tool_identity_and_explicit_trace_frontier(tmp_path, monkeypatch, backend, mode):
    requests = 0
    identities = []
    visibility = []
    statements = []
    identity_sql = []
    flush_sql = []
    buffer = InMemoryCacheBackend(max_bytes=4 * 1024 * 1024)
    policy = {
        "batch": CommitPolicy.batch(buffer=buffer, batch_size=128, flush_interval=60),
        "terminal": CommitPolicy.terminal(buffer=buffer),
        "sync": CommitPolicy.sync(),
    }

    async def inspect_call(ctx: AgentContext[None]) -> str:
        identity = ctx.tool_call
        assert identity is not None
        identities.append(identity)
        assert identity.execution_id == ctx.execution_id
        assert identity.root_execution_id == identity.execution_id
        assert identity.parent_execution_id is None
        assert identity.agent_id == "default" and identity.agent_run_seq == 1
        assert identity.agent_run_id
        assert identity.tool_call_id == "call-1" and identity.tool_name == "inspect_call"
        assert identity.model_request_seq == (None if mode == "none" else 1)
        if mode != "none":
            trace = ctx.trace
            assert trace is not None
            archive = runtime._run_store.read_store(PersistenceDomain.TRACE)
            before = await archive.list_events(agent_run_id=identity.agent_run_id)
            start = len(statements)
            await trace.flush()
            flush_sql.append(len(statements) - start)
            after = await archive.list_events(agent_run_id=identity.agent_run_id)
            visibility.append((len(before), len(after)))
            assert any(event.tool_call_id == identity.tool_call_id for event in after)
            if mode in {"batch", "terminal"}:
                assert not before
            await trace.flush()
            assert await archive.list_events(agent_run_id=identity.agent_run_id) == after
            assert trace.commit.mode == mode
        return "done"

    async def model(messages, info):
        nonlocal requests
        requests += 1
        if requests == 1:
            return ModelResponse(parts=[ToolCallPart("inspect_call", {}, tool_call_id="call-1")])
        return ModelResponse(parts=[TextPart("finished")])

    features = RuntimeFeatures("tool-call-identity")
    if mode != "none":
        features.capability(TraceCapability(commit=policy[mode]))
    features.tool(inspect_call, effect_policy="none")
    features.agent("default", model="default", allow_tools=("inspect_call",))
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "state.db")
    async with (
        Persistence.open("tool-call-identity", storage=storage) as persistence,
        Runtime.attach(persistence=persistence, features=model_features(
            _FeatureModels(_FeatureFunctionModel(model)), features,
        ), metrics=None) as runtime,
    ):
        from linktools.ai.runtime import _features as feature_module
        original_context = feature_module._tool_context

        def measured_context(ctx):
            before = len(statements)
            value = original_context(ctx)
            identity_sql.append(len(statements) - before)
            return value

        monkeypatch.setattr(feature_module, "_tool_context", measured_context)
        if backend == "sqlite":
            from sqlalchemy import event
            engine = persistence.execution.executions.state_store.storage_group.context.engine
            event.listen(engine.sync_engine, "before_cursor_execute",
                         lambda *args: statements.append(args[2]))
        execution = await runtime.agents.get().start("call inspect_call")
        assert (await execution.wait(timeout_seconds=10)).result.output == {"text": "finished"}
        assert len(identities) == 1
        assert identity_sql == [0]
        print(f"TRACE_VISIBILITY backend={backend} mode={mode} before_after={visibility} "
              f"identity_sql={identity_sql} flush_sql={flush_sql}")


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_waiter", [False, True])
async def test_public_trace_flush_settles_source_and_preserves_later_batch(tmp_path, monkeypatch, cancel_waiter):
    import asyncio
    from functools import partial
    from pydantic_ai.capabilities import AbstractCapability
    from .test_observation_coalescing import _setup, _observe, _add_run

    state = await _setup(tmp_path, "sqlite")
    store = state.run_store
    owner = store.trace_persistence
    owner.configure("run", buffer=owner._caches["run"], batch_size=128, flush_interval=60)
    await _add_run(state, "other-run", "other-execution")
    owner.configure("other-run", buffer=owner._caches["other-run"], batch_size=128, flush_interval=60)
    original = owner._publish
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        await release.wait()
        await original(*args, **kwargs)

    trace = TraceCapability().bind(AbstractCapability(), flush=partial(
        store.flush_execution_projection, "run", execution_id="execution",
        producer_generation=1, require_durable=True,
    ))
    archive = store.read_store(PersistenceDomain.TRACE)
    try:
        await _observe(store, 1)
        await _observe(store, 1, run_id="other-run", execution_id="other-execution")
        monkeypatch.setattr(owner, "_publish", blocked)
        task = asyncio.create_task(trace.flush())
        await asyncio.wait_for(entered.wait(), 5)
        assert not await archive.list_events(agent_run_id="run")
        if cancel_waiter:
            task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        if cancel_waiter:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
        assert len(await archive.list_events(agent_run_id="run")) == 1
        assert not await archive.list_events(agent_run_id="other-run")
        monkeypatch.setattr(owner, "_publish", original)
        await _observe(store, 2)
        assert len(await archive.list_events(agent_run_id="run")) == 1
        await trace.flush()
        assert len(await archive.list_events(agent_run_id="run")) == 2
    finally:
        release.set()
        monkeypatch.setattr(owner, "_publish", original)
        await state.close()


@pytest.mark.asyncio
async def test_public_trace_flush_rejects_transient_source(tmp_path):
    from functools import partial
    from pydantic_ai.capabilities import AbstractCapability
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.persistence import PersistencePlan
    from linktools.ai.persistence.runs import AgentRunRecord
    from linktools.ai.storage import StorageRoute

    storage = PersistenceStorage(PersistencePlan(trace=StorageRoute.transient()))
    async with Persistence.open("transient-flush", storage=storage) as state:
        store = state.run_store
        await store.register_agent_run(AgentRunRecord("run"))
        trace = TraceCapability().bind(AbstractCapability(), flush=partial(
            store.flush_execution_projection, "run", execution_id="execution", require_durable=True,
        ))
        with pytest.raises(AIError) as error:
            await trace.flush()
        assert error.value.code is ErrorCode.CAPABILITY_REQUIRED_MISSING
        # Existing local-only internal boundaries retain their supported behavior.
        await store.flush_execution_projection("run", execution_id="execution")


@pytest.mark.asyncio
async def test_public_trace_flush_does_not_report_source_failure_as_visibility(tmp_path, monkeypatch):
    from functools import partial
    from pydantic_ai.capabilities import AbstractCapability
    from linktools.ai.errors import AIError, ErrorCode
    from .test_observation_coalescing import _setup, _observe

    state = await _setup(tmp_path, "sqlite")
    store = state.run_store
    owner = store.trace_persistence
    group = state.execution.executions.state_store.storage_group
    owner.configure("run", buffer=owner._caches["run"], batch_size=128, flush_interval=60)
    error = AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

    async def fail(*args, **kwargs):
        raise error

    trace = TraceCapability().bind(AbstractCapability(), flush=partial(
        store.flush_execution_projection, "run", execution_id="execution",
        producer_generation=1, require_durable=True,
    ))
    try:
        await _observe(store, 1)
        monkeypatch.setattr(owner, "_publish", fail)
        with pytest.raises(AIError) as caught:
            await trace.flush()
        assert caught.value is error
        assert not await store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id="run")
        assert owner._buffers["run"].pending
        with pytest.raises(AIError):
            await state.close()
    finally:
        # Do not pretend an unresolved source publication was safely replayed.
        assert all(task.done() for task in owner._background_tasks)
        await group.close()
        await group.context.close()
