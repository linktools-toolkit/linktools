#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Atomic hook boundaries preserve the original native recovery transcript."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.step_persistence import (
    InMemoryStepStore,
    StepPersistence,
    continue_run,
)
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.capability.steps import AtomicStepPersistence
from linktools.ai.persistence import SqlStepStore
from linktools.ai.runtime._step_store import RuntimeStepStore


class _ObservedStore(SqlStepStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.boundaries = []
        self.fail = False

    async def commit_step(self, *, run=None, events=(), snapshot=None):
        if self.fail:
            raise RuntimeError("rollback")
        await super().commit_step(run=run, events=events, snapshot=snapshot)
        self.boundaries.append((run, tuple(events), snapshot))


def _context(messages, emit):
    return SimpleNamespace(
        run_id="run",
        run_step=1,
        conversation_id="conversation",
        messages=messages,
        emit=emit,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("frontier", [False, True])
async def test_atomic_hooks_match_native_records_and_original_tool_results(
    tmp_path: Path, frontier: bool
) -> None:
    stores = []
    engines = []
    try:
        for label in ("native", "atomic"):
            engine = create_async_engine(
                f"sqlite+aiosqlite:///{tmp_path / (label + '.db')}"
            )
            engines.append(engine)
            store = _ObservedStore(
                engine,
                namespace="steps",
                tenant_id="one",
                tool_effect_store=InMemoryStepStore(),
            )
            stores.append(store)
            await store.initialize(local=True)
            capability = (
                StepPersistence if label == "native" else AtomicStepPersistence
            )(
                store=store if label == "native" else RuntimeStepStore(store, InMemoryStepStore()),
                run_id="run",
                capture_frontier=frontier,
            )

            def model(messages, info):
                if any(
                    isinstance(part, ToolReturnPart)
                    for message in messages
                    for part in message.parts
                ):
                    return ModelResponse(parts=[TextPart("done")])
                return ModelResponse(
                    parts=[ToolCallPart("inspect", {}, tool_call_id="call")]
                )

            agent = Agent(FunctionModel(model), capabilities=[capability])

            @agent.tool_plain
            def inspect() -> dict[str, object]:
                return {"original": [False, None, {"result": 42}]}

            assert (
                await agent.run("work", conversation_id="conversation")
            ).output == "done"
        native, atomic = stores

        def normalize(value):
            if isinstance(value, dict):
                return {
                    k: normalize(v)
                    for k, v in value.items()
                    if k
                    not in {
                        "timestamp",
                        "started_at",
                        "ended_at",
                        "registration_id",
                        "idempotency_key",
                        "run_id",
                        "provider_response_id",
                    }
                }
            if isinstance(value, (list, tuple)):
                return [normalize(v) for v in value]
            return value

        assert normalize(
            [asdict(event) for event in await native.list_events(run_id="run") if not event.kind.startswith("tool_call_")]
        ) == normalize(
            [asdict(event) for event in await atomic.list_events(run_id="run")]
        )
        assert normalize(asdict(await native.latest_snapshot(run_id="run"))) == normalize(
            asdict(await atomic.latest_snapshot(run_id="run"))
        )
        assert normalize(
            [asdict(m) for m in await continue_run(native, run_id="run")]
        ) == normalize([asdict(m) for m in await continue_run(atomic, run_id="run")])
        assert atomic.boundaries[0][0] is not None
        assert atomic.boundaries[0][1][0].kind == "run_started"
        model_groups = [
            group
            for group in atomic.boundaries
            if any(event.kind == "model_request_started" for event in group[1])
        ]
        assert len(model_groups) == 2
        assert (model_groups[0][2] is not None) == frontier
        assert model_groups[1][2] is not None
        # The terminal node snapshot is committed with the completed event.
        assert atomic.boundaries[-1][1][0].kind == "run_completed"
        assert atomic.boundaries[-1][2] is not None
    finally:
        for store in stores:
            await store.close()
        for engine in engines:
            await engine.dispose()


@pytest.mark.asyncio
async def test_notification_observes_committed_pair_and_failure_does_not_retract_it(
    tmp_path: Path,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'notifications.db'}")
    store = _ObservedStore(engine, namespace="steps", tenant_id="one")
    await store.initialize(local=True)
    confirmed = []
    invalidated = []
    working = SimpleNamespace(
        confirm_step_snapshot=confirmed.append, invalidate_step_run=invalidated.append
    )
    wrapped = RuntimeStepStore(store, InMemoryStepStore(), working_set=working)
    capability = AtomicStepPersistence(
        store=wrapped, run_id="run", capture_frontier=True
    )
    messages = [ModelRequest(parts=[UserPromptPart("accepted")])]
    notifications = []

    async def emit(event):
        assert await store.latest_snapshot(run_id="run") is not None
        assert (await store.list_events(run_id="run"))[
            -1
        ].kind == "model_request_started"
        assert confirmed
        notifications.append(event)
        raise RuntimeError("notification failed")

    try:
        with pytest.raises(RuntimeError, match="notification failed"):
            await capability.before_model_request(_context(messages, emit), None)
            await capability._atomic_publish_model(_context(messages, emit))
        assert len(confirmed) == len(notifications) == 1
        assert not invalidated
        assert await store.latest_snapshot(run_id="run") is not None
        store.fail = True
        with pytest.raises(RuntimeError, match="rollback"):
            await capability.before_model_request(_context(messages, emit), None)
            await capability._atomic_publish_model(_context(messages, emit))
        assert len(confirmed) == len(notifications) == 1
        assert invalidated == ["run"]
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_error_snapshot_and_failure_event_commit_before_original_error_propagates(
    tmp_path: Path,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'error.db'}")
    store = _ObservedStore(engine, namespace="steps", tenant_id="one")
    await store.initialize(local=True)
    capability = AtomicStepPersistence(store=RuntimeStepStore(store, InMemoryStepStore()), run_id="run", capture_frontier=True)
    notifications = []

    async def emit(event):
        notifications.append(event)

    ctx = _context([ModelRequest(parts=[UserPromptPart("accepted")])], emit)
    try:
        await capability.before_model_request(ctx, None)
        await capability._atomic_publish_model(ctx)
        error = RuntimeError("model failed")
        with pytest.raises(RuntimeError) as raised:
            await capability.on_run_error(ctx, error=error)
        assert raised.value is error
        assert store.boundaries[-1][1][0].kind == "run_failed"
        assert store.boundaries[-1][2] is not None
        assert len(notifications) == 2
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_inherited_run_scopes_preserve_nested_lineage_and_per_run_state(
    tmp_path: Path,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'nested.db'}")
    store = _ObservedStore(
        engine,
        namespace="steps",
        tenant_id="one",
        tool_effect_store=InMemoryStepStore(),
    )
    await store.initialize(local=True)
    child_capability = AtomicStepPersistence(
        store=RuntimeStepStore(store, InMemoryStepStore()), agent_name="child", capture_frontier=True
    )
    child = Agent(
        FunctionModel(
            lambda messages, info: ModelResponse(parts=[TextPart("child result")])
        ),
        capabilities=[child_capability],
    )

    def parent_model(messages, info):
        if any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            return ModelResponse(parts=[TextPart("parent result")])
        return ModelResponse(parts=[ToolCallPart("delegate", {}, tool_call_id="call")])

    parent = Agent(
        FunctionModel(parent_model),
        capabilities=[
            AtomicStepPersistence(
                store=RuntimeStepStore(store, InMemoryStepStore()), agent_name="parent", capture_frontier=True
            )
        ],
    )

    @parent.tool_plain
    async def delegate() -> str:
        return (await child.run("nested")).output

    try:
        await parent.run("first")
        await parent.run("second")
        runs = await store.list_runs()
        parents = [run for run in runs if run.agent_name == "parent"]
        children = [run for run in runs if run.agent_name == "child"]
        assert len(parents) == len(children) == 2
        assert len({run.run_id for run in runs}) == 4
        assert {run.parent_run_id for run in children} == {
            run.run_id for run in parents
        }
        for run in runs:
            events = await store.list_events(run_id=run.run_id)
            assert events[0].idempotency_key == "0:0:run_started:"
            assert events[-1].kind == "run_completed"
            assert await continue_run(store, run_id=run.run_id)
        assert child_capability.compaction_transcript_handle() is None
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("capability_type", [StepPersistence, AtomicStepPersistence])
async def test_model_failure_rescues_live_rebound_tool_history(
    tmp_path: Path, capability_type
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'live.db'}")
    store = _ObservedStore(
        engine,
        namespace="steps",
        tenant_id="one",
        tool_effect_store=InMemoryStepStore(),
    )
    await store.initialize(local=True)
    failure = RuntimeError("second model request failed")

    def model(messages, info):
        if any(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        ):
            raise failure
        return ModelResponse(parts=[ToolCallPart("inspect", {}, tool_call_id="call")])

    agent = Agent(
        FunctionModel(model),
        capabilities=[
            capability_type(store=store if capability_type is StepPersistence else RuntimeStepStore(store, InMemoryStepStore()), run_id="run", capture_frontier=True)
        ],
    )

    @agent.tool_plain
    def inspect() -> str:
        return "exact original result"

    try:
        with pytest.raises(RuntimeError) as raised:
            await agent.run("accepted prompt")
        assert raised.value is failure
        messages = await continue_run(store, run_id="run")
        results = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert len(results) == 1
        assert results[0].content == "exact original result"
        assert (await store.list_events(run_id="run"))[-1].kind == "run_failed"
        if capability_type is AtomicStepPersistence:
            assert store.boundaries[-1][2].messages == messages
    finally:
        await store.close()
        await engine.dispose()


def test_durable_operation_registration_has_no_inherited_name_collisions() -> None:
    from pydantic_ai.durable_exec._capability_operation import (
        collect_capability_operations,
    )

    operations = collect_capability_operations(AtomicStepPersistence())
    assert "commit_boundary" in operations


@pytest.mark.asyncio
async def test_same_registration_retry_keeps_one_run_and_one_started_event(
    tmp_path: Path, monkeypatch
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'registration.db'}")
    store = _ObservedStore(engine, namespace="steps", tenant_id="one")
    await store.initialize(local=True)

    async def registration(self):
        return "journaled-registration"

    monkeypatch.setattr(AtomicStepPersistence, "_atomic_registration_id", registration)

    async def emit(event):
        pass

    try:
        for _ in range(2):
            capability = AtomicStepPersistence(store=RuntimeStepStore(store, InMemoryStepStore()), run_id="run")
            ctx = _context([], emit)
            await capability.before_run(ctx)
            assert not store.boundaries or len(store.boundaries) == 1
            await capability.before_model_request(ctx, None)
            await capability._atomic_publish_model(ctx)
        assert len(await store.list_runs()) == 1
        assert len(await store.list_events(run_id="run")) == 2

        async def different_registration(self):
            return "different-registration"

        monkeypatch.setattr(
            AtomicStepPersistence, "_atomic_registration_id", different_registration
        )
        with pytest.raises(ValueError, match="single-shot"):
            capability = AtomicStepPersistence(store=RuntimeStepStore(store, InMemoryStepStore()), run_id="run")
            ctx = _context([], emit)
            await capability.before_run(ctx)
            await capability.before_model_request(ctx, None)
            await capability._atomic_publish_model(ctx)
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_runtime_adaptation_preserves_all_native_capability_configuration(
    tmp_path: Path,
) -> None:
    from linktools.ai.core import PromptLimits
    from linktools.ai.persistence.runs import StagingAgentRunStore
    from linktools.ai.runtime._capabilities import (
        compose_platform_capabilities,
        runtime_feature_leaves,
    )
    from linktools.ai.runtime._compaction import CompactionPolicy

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'binding.db'}")
    store = _ObservedStore(engine, namespace="steps", tenant_id="one")
    original = StepPersistence(
        id="custom-id",
        description="custom description",
        defer_loading=True,
        store=store,
        capture_frontier=True,
        metadata={"custom": "metadata"},
    )
    try:
        values = await compose_platform_capabilities(
            agent_id="agent",
            agent_run_id="run",
            agent_run_seq=1,
            history_id=None,
            memory_scope=None,
            run_store=StagingAgentRunStore(),
            memory_store=None,
            tool_policy=(),
            compaction_policy=CompactionPolicy(),
            limits=PromptLimits(),
            planning=False,
            context_target_tokens=None,
            parent_agent_run_id="parent",
            plan_store_resolver=None,
            runtime_capabilities=[original],
        )
        adapted = [
            leaf
            for value in values
            for leaf in runtime_feature_leaves(value)
            if isinstance(leaf, AtomicStepPersistence)
        ][0]
        assert adapted.id == original.id
        assert adapted.description == original.description
        assert adapted.defer_loading is True
        assert adapted.capture_frontier is True
        assert adapted.store is store
        assert adapted.metadata["custom"] == "metadata"
        assert adapted.run_id == "run" and adapted.parent_run_id == "parent"
        assert adapted.agent_name == "agent"
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", (False, True))
@pytest.mark.parametrize("boundary_first", (False, True))
async def test_model_leaf_commits_final_context_after_hooks_and_model_wrappers(
    tmp_path: Path, streaming: bool, boundary_first: bool,
) -> None:
    import asyncio
    from contextlib import asynccontextmanager
    from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
    from pydantic_ai.models.wrapper import WrapperModel

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'leaf.db'}")
    store = _ObservedStore(engine, namespace="leaf", tenant_id="one")
    await store.initialize(local=True)
    order = []

    async def boundary(*, run=None, events=(), snapshot=None):
        if any(event.kind == "model_request_started" for event in events):
            order.append("commit")
            assert snapshot.messages[-1].metadata == {"prepare": True, "wrapper": True}
        await store.commit_step(run=run, events=events, snapshot=snapshot)

    class OutsideModel(WrapperModel):
        def __init__(self, model, ctx):
            super().__init__(model)
            self.ctx = ctx

        async def ready(self):
            await asyncio.sleep(0)
            order.append("wrapper")
            self.ctx.messages[-1].metadata["wrapper"] = True

        async def request(self, messages, settings, parameters):
            await self.ready()
            return await self.wrapped.request(messages, settings, parameters)

        @asynccontextmanager
        async def request_stream(self, messages, settings, parameters, run_context=None):
            await self.ready()
            async with self.wrapped.request_stream(messages, settings, parameters, run_context) as response:
                yield response

    class LateHooks(AbstractCapability):
        def get_ordering(self):
            return CapabilityOrdering(position="innermost")

        async def before_model_request(self, ctx, request_context):
            assert not store.boundaries
            await asyncio.sleep(0)
            order.append("before")
            ctx.messages[-1].parts.append(UserPromptPart("persistent hook edit"))
            return request_context

        async def prepare_model_request(self, ctx, request_context):
            await asyncio.sleep(0)
            order.append("prepare")
            self.ctx = ctx
            ctx.messages[-1].metadata = {"prepare": True}
            return request_context

        def get_wrapper_model(self, model):
            return OutsideModel(model, self.ctx)

    async def check_provider():
        order.append("provider")
        saved = await store.latest_snapshot(run_id="run")
        assert saved.messages[-1].parts[-1].content == "persistent hook edit"
        assert saved.messages[-1].metadata == {"prepare": True, "wrapper": True}

    async def model(messages, info):
        await check_provider()
        return ModelResponse(parts=[TextPart("done")])

    async def stream(messages, info):
        await check_provider()
        yield "done"

    capability = AtomicStepPersistence(
        store=RuntimeStepStore(store, InMemoryStepStore(), commit_boundary=boundary),
        run_id="run", capture_frontier=True,
    )
    hooks = LateHooks()
    agent = Agent(FunctionModel(model, stream_function=stream),
        capabilities=[capability, hooks] if boundary_first else [hooks, capability])
    try:
        if streaming:
            async with agent.run_stream("work") as result:
                assert await result.get_output() == "done"
        else:
            assert (await agent.run("work")).output == "done"
        assert order == ["before", "prepare", "wrapper", "commit", "provider"]
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_standalone_atomic_capability_requires_runtime_effect_owner(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'unbound.db'}")
    store = _ObservedStore(engine, namespace="steps", tenant_id="one")
    await store.initialize(local=True)
    agent = Agent(FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("unreachable")])),
                  capabilities=[AtomicStepPersistence(store=store)])
    try:
        with pytest.raises(TypeError, match="Runtime-bound"):
            await agent.run("work")
        assert not store.boundaries
    finally:
        await store.close()
        await engine.dispose()
