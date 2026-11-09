#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime run-level and dynamic model wrappers precede native publication."""

import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

import pytest
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, DynamicCapability
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai_harness.step_persistence import StepPersistence
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.capability import AgentContext
from linktools.ai.core import ExecutionStatus
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import Persistence, PersistenceStorage, SqlStepStore
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.persistence.runs import OfficialStepReader

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


@pytest.mark.asyncio
@pytest.mark.parametrize("dynamic", (False, True))
async def test_runtime_resolves_all_model_wrappers_before_placing_publication_leaf(tmp_path: Path, dynamic: bool) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'runtime.db'}")
    await provision_database(engine)
    native = SqlStepStore(engine, namespace="native-boundary", tenant_id="default")
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
            return await self.wrapped.request(messages, model_settings, model_request_parameters)

        @asynccontextmanager
        async def request_stream(self, messages, model_settings, model_request_parameters, run_context=None):
            await self.prepare()
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
        runs = await native.list_runs()
        assert len(runs) == 1
        snapshot = await OfficialStepReader(native).latest_snapshot(run_id=runs[0].run_id)
        assert messages[-1].metadata == {"final_wrapper": True}
        assert snapshot.messages[-1].metadata == {"final_wrapper": True}
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
        assert order == ["prepare", "wrapper", "provider"]
        assert len(factory_calls) == int(dynamic)
    finally:
        await native.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("override_hooks", (None, False, True))
async def test_custom_native_capability_preserves_hooks_without_provisional_effects(
    tmp_path: Path, override_hooks: bool | None,
) -> None:
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart
    from linktools.ai.core import ToolOperationStatus

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'custom.db'}")
    await provision_database(engine)
    native = SqlStepStore(engine, namespace="custom-native", tenant_id="default")
    hooks, effects = [], []
    source = None

    class CustomStep(StepPersistence):
        pass

    class HookedStep(StepPersistence):
        async def before_tool_execute(self, ctx, *, call, tool_def, args):
            hooks.append("before")
            args = await super().before_tool_execute(ctx, call=call, tool_def=tool_def, args=args)
            assert await source.checkpoint.tools.list_by_execution(ctx.deps.execution_id, tenant_id="default") == ()
            return args

        async def after_tool_execute(self, ctx, *, call, tool_def, args, result):
            hooks.append("after")
            result = await super().after_tool_execute(ctx, call=call, tool_def=tool_def, args=args, result=result)
            operations = await source.checkpoint.tools.list_by_execution(ctx.deps.execution_id, tenant_id="default")
            assert len(operations) == 1 and operations[0].status is ToolOperationStatus.COMPLETED
            return result

    async def perform(ctx: AgentContext[None]) -> str:
        effects.append("effect")
        return "confirmed"

    async def provider(messages, info):
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("perform", {}, tool_call_id="call")])

    features = RuntimeFeatures("custom-native")
    capability_type = StepPersistence if override_hooks is None else HookedStep if override_hooks else CustomStep
    features.capability(capability_type(store=native, capture_frontier=True), scope="runtime")
    features.tool(perform, effect_policy="non_replay_safe")
    features.agent("default", model="default", allow_tools=("perform",))
    try:
        async with Persistence.open("custom-native", storage=PersistenceStorage.sql(engine)) as state:
            source = state
            async with Runtime.attach(persistence=state,
                features=model_features(_Models(_UsageFunctionModel(provider)), features)) as runtime:
                execution = await runtime.agents.get().start("work", idempotency_key="work")
                result = await execution.wait(timeout_seconds=5)
                assert result.result.status is ExecutionStatus.SUCCEEDED, result.result
                operations = await source.checkpoint.tools.list_by_execution(execution.execution_id, tenant_id="default")
                assert len(operations) == 1 and operations[0].status is ToolOperationStatus.COMPLETED
            runs = await native.list_runs()
            events = await native.list_events(run_id=runs[0].run_id)
            tool_events = [event for event in events if event.kind.startswith("tool_call")]
            assert [event.kind for event in tool_events] == ["tool_call_started", "tool_call_completed"]
            assert all(event.metadata == runs[0].metadata and event.agent_name == runs[0].agent_name
                       and event.parent_run_id == runs[0].parent_run_id for event in tool_events)
        assert effects == ["effect"]
        assert hooks == (["before", "after"] if override_hooks else [])
    finally:
        await native.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_native_once", (False, True))
async def test_unknown_tool_fact_and_native_failure_event_share_storage(
    tmp_path: Path, fail_native_once: bool,
) -> None:
    from pydantic_ai.messages import ToolCallPart
    from linktools.ai.core import ToolOperationStatus
    from linktools.ai.errors import AIError, ErrorCode

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'unknown.db'}")
    await provision_database(engine)

    class NativeStore(SqlStepStore):
        interrupted = False

        async def _write_events(self, session, events):
            if (fail_native_once and not self.interrupted
                    and any(event.error == ErrorCode.TOOL_EFFECT_UNKNOWN.value for event in events)):
                self.interrupted = True
                raise OSError("native event storage failed")
            await super()._write_events(session, events)

        async def append_event(self, event):
            raise AssertionError("same-group unknown publication must join the effect transaction")

    native = NativeStore(engine, namespace="unknown-native", tenant_id="default")
    effects = []

    async def perform(ctx: AgentContext[None]) -> str:
        effects.append("effect")
        raise RuntimeError("private provider error must not become event content")

    async def provider(messages, info):
        return ModelResponse(parts=[ToolCallPart("perform", {}, tool_call_id="call")])

    features = RuntimeFeatures("unknown-native")
    features.capability(StepPersistence(store=native, capture_frontier=True), scope="runtime")
    features.tool(perform, effect_policy="non_replay_safe")
    features.agent("default", model="default", allow_tools=("perform",))
    try:
        async with Persistence.open("unknown-native", storage=PersistenceStorage.sql(engine)) as state:
            async with Runtime.attach(persistence=state,
                features=model_features(_Models(_UsageFunctionModel(provider)), features)) as runtime:
                execution = await runtime.agents.get().start("work", idempotency_key="work")
                with pytest.raises(AIError) as failure:
                    await execution.wait(timeout_seconds=5)
                assert failure.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
                operations = await state.checkpoint.tools.list_by_execution(execution.execution_id, tenant_id="default")
                assert len(operations) == 1 and operations[0].status is ToolOperationStatus.EFFECT_UNKNOWN
                assert operations[0].effect_summary is None
            runs = await native.list_runs()
            events = [event for event in await native.list_events(run_id=runs[0].run_id) if event.kind.startswith("tool_call")]
            assert [event.kind for event in events] == ["tool_call_started", "tool_call_failed"]
            assert events[-1].error == ErrorCode.TOOL_EFFECT_UNKNOWN.value
        assert effects == ["effect"]
        assert native.interrupted is fail_native_once
    finally:
        await native.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_checkpoint_control_route_rejects_an_independent_group(tmp_path: Path) -> None:
    from dataclasses import replace
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.storage import StorageRoute

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'control.db'}")
    await provision_database(engine)
    storage = PersistenceStorage(replace(PersistenceStorage.sql(engine).plan,
        checkpoint=StorageRoute.sqlite(tmp_path / "separate-control.db")))
    try:
        with pytest.raises(AIError) as invalid:
            async with Persistence.open("separate-checkpoint-control", storage=storage):
                pytest.fail("checkpoint controls cannot leave the execution transaction group")
        assert invalid.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
    finally:
        await engine.dispose()
