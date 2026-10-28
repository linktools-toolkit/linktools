#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Runtime callbacks borrow the same per-run instances as native SDK hooks."""

import asyncio
import hashlib
from dataclasses import dataclass

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, CombinedCapability, WrapperCapability
from pydantic_ai.messages import ToolReturn, UserPromptPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel

from linktools.ai.asset import AssetKey, AssetVersionRef
from linktools.ai.capability import AgentContext, ReferencedInputsCapability, ToolResultsCapability
from linktools.ai.core import ExecutionStatus, service_principal
from linktools.ai.model import ReferencedContent
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.capability import RunCallbacks, bind_run_callbacks
from linktools.ai.spec import CapabilityRef
from linktools.ai.storage import StorageEntryRevision

_BODY = b"Recoverable run-local text"
_SOURCE = CapabilityRef("capability", "source", 1)
_REF = ReferencedContent(
    AssetVersionRef(AssetKey("file", "document"), "configured", StorageEntryRevision(1),
                    hashlib.sha256(_BODY).hexdigest(), len(_BODY)),
    media_type="text/plain", source=_SOURCE, representation="text",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ("none", "replay_safe"))
@pytest.mark.parametrize("wrapped", ("direct", "wrapper", "combined"))
async def test_two_concurrent_runs_use_their_initialized_instance_through_close(policy: str, wrapped: str) -> None:
    events = []
    arrived = 0
    both = asyncio.Event()

    class Project(ToolResultsCapability):
        id = "project"

        def __init__(self, run=None):
            self.run, self.ready = run, False

        async def for_run(self, ctx):
            assert self.run is None
            bound = type(self)(ctx.deps.execution_id)
            events.append(("project.bind", bound.run, id(bound)))
            return bound

        async def before_run(self, ctx):
            assert self.run == ctx.deps.execution_id and not self.ready
            self.ready = True
            events.append(("project.open", self.run, id(self)))

        async def wrap_run(self, ctx, *, handler):
            try:
                return await handler()
            finally:
                assert self.ready
                events.append(("project.close", self.run, id(self)))
                self.ready = False

        async def process_tool_result(self, context, tool, arguments, result):
            nonlocal arrived
            assert self.ready and self.run == context.execution_id
            events.append(("project.process", self.run, id(self)))
            arrived += 1
            if arrived == 2:
                both.set()
            await asyncio.wait_for(both.wait(), 5)
            return ToolReturn(return_value="summary", content=[_REF])

        async def after_tool_execute(self, ctx, *, result, **kwargs):
            assert self.ready
            events.append(("project.after", self.run, id(self)))
            return result

    class Source(ReferencedInputsCapability):
        id = "source"

        def __init__(self, run=None):
            self.run, self.ready = run, False
            async def load(context, content):
                assert self.ready and self.run == context.execution_id and content == _REF
                events.append(("source.load", self.run, id(self)))
                return _BODY
            super().__init__(load)

        async def for_run(self, ctx):
            assert self.run is None
            bound = type(self)(ctx.deps.execution_id)
            events.append(("source.bind", bound.run, id(bound)))
            return bound

        async def before_run(self, ctx):
            self.ready = True
            events.append(("source.open", self.run, id(self)))

        async def wrap_run(self, ctx, *, handler):
            try:
                return await handler()
            finally:
                assert self.ready
                events.append(("source.close", self.run, id(self)))
                self.ready = False

    class Binding:
        provider, model_identity = "test", "test:run-lifecycle"
        contract = {"input_mime_types": []}
        def materialize(self):
            requests = 0
            async def stream(messages, info):
                nonlocal requests
                requests += 1
                if requests == 1:
                    yield {0: DeltaToolCall(name="inspect", json_args="{}", tool_call_id="same-call")}
                else:
                    assert requests == 2
                    assert any(_BODY.decode() in part.content for message in messages for part in message.parts
                               if isinstance(part, UserPromptPart) and not isinstance(part.content, str))
                    yield "done"
            return FunctionModel(stream_function=stream)

    async def inspect(context: AgentContext[None]) -> str:
        return "original"

    features = RuntimeFeatures("run-lifecycle")
    features.model("default", Binding())
    features.tool(inspect, effect_policy=policy)
    project, source = Project(), Source()
    def container(capability):
        if wrapped == "combined":
            return CombinedCapability([WrapperCapability(capability)])
        return WrapperCapability(capability) if wrapped == "wrapper" else capability
    features.capability(container(project), id="project", scope="agent")
    features.capability(container(source), id="source", scope="agent")
    features.agent("reader", allow_tools=("inspect",), allow_skills=(), allow_subagents=())
    async with Persistence.open('run-lifecycle', storage=PersistenceStorage.in_memory()) as persistence_142, Runtime.attach(persistence=persistence_142, features=features) as runtime:
        executions = await asyncio.gather(*(runtime.agents.get("reader").start("Inspect.") for _ in range(2)))
        outcomes = await asyncio.gather(*(execution.wait(timeout_seconds=10) for execution in executions))
        assert all(outcome.result.status is ExecutionStatus.SUCCEEDED for outcome in outcomes)
    runs = {execution.execution_id for execution in executions}
    for run in runs:
        for kind, phases in (("project", ("bind", "open", "process", "after", "close")),
                             ("source", ("bind", "open", "load", "close"))):
            observed = [(phase, instance) for phase, event_run, instance in events if event_run == run and phase.startswith(kind)]
            assert [phase for phase, _ in observed] == [f"{kind}.{phase}" for phase in phases]
            assert len({instance for _, instance in observed}) == 1
    assert len({instance for phase, _, instance in events if phase.endswith("bind")}) == 4
    assert not project.ready and not source.ready


@pytest.mark.asyncio
@pytest.mark.parametrize("deferred", (False, True))
async def test_native_ordering_type_requirements_and_duplicate_merge_remain_transparent(deferred: bool) -> None:
    async def execute(adapt: bool):
        events = []
        @dataclass
        class Project(ToolResultsCapability):
            label: str
            id = "same-project"
            defer_loading = deferred
            async def for_run(self, ctx):
                events.append(("bind", self.label))
                return type(self)(self.label + "-bound")
            async def before_run(self, ctx):
                events.append(("open", self.label))
        class Needs(AbstractCapability):
            def get_ordering(self):
                return CapabilityOrdering(requires=(Project,))
        capabilities = [Project("first"), Project("last"), Needs()]
        if adapt:
            callbacks = RunCallbacks()
            capabilities = [bind_run_callbacks(capability, _SOURCE, callbacks) for capability in capabilities]
        await Agent(TestModel(), capabilities=capabilities).run("Inspect.")
        return events
    assert await execute(True) == await execute(False)


@pytest.mark.asyncio
async def test_parallel_call_scopes_are_isolated_and_cancelled_scopes_are_removed() -> None:
    callbacks = RunCallbacks()
    first, second = ToolResultsCapability(), ToolResultsCapability()
    entered = asyncio.Event()
    release = asyncio.Event()
    async def call(call_id, capability):
        with callbacks.tool_scope(call_id, capability):
            entered.set()
            await release.wait()
    task = asyncio.create_task(call("first", first))
    await entered.wait()
    with callbacks.tool_scope("second", second):
        assert callbacks.tool_results("first") == (first,)
        assert callbacks.tool_results("second") == (second,)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert callbacks.tool_results("first") == ()
        assert callbacks.tool_results("second") == (second,)
    assert callbacks.tool_results("second") == ()


@pytest.mark.asyncio
async def test_native_replacement_and_removal_refresh_the_bound_source() -> None:
    from ._runtime_test_helpers import tool_run_context
    async def load(context, content):
        return _BODY
    original, replacement = ReferencedInputsCapability(load), ReferencedInputsCapability(load)
    original.id = replacement.id = "source"
    callbacks, context = RunCallbacks(), tool_run_context()
    bound = await bind_run_callbacks(original, _SOURCE, callbacks).for_run(context)
    context.capabilities = {"source": original}
    callbacks.context = context
    assert callbacks.referenced_inputs() == {_SOURCE: original}
    changed = bound.visit_and_replace(lambda value: replacement if value is original else value)
    context.capabilities = {"source": replacement}
    assert callbacks.referenced_inputs() == {_SOURCE: replacement}
    assert changed.visit_and_replace(lambda value: None) is None
    assert callbacks.referenced_inputs() == {}


@pytest.mark.asyncio
async def test_deferred_source_replacement_preserves_native_identity_and_releases_binding() -> None:
    from ._runtime_test_helpers import tool_run_context

    async def load(context, content):
        return _BODY

    original, replacement = ReferencedInputsCapability(load), ReferencedInputsCapability(load)
    original.id = replacement.id = "source"
    original.defer_loading = replacement.defer_loading = True
    callbacks, context = RunCallbacks(), tool_run_context()
    bound = await bind_run_callbacks(original, _SOURCE, callbacks).for_run(context)
    visited = []
    bound.apply(visited.append)
    assert len(visited) == 1 and visited[0] is original

    changed = bound.visit_and_replace(lambda value: replacement if value is original else value)
    visited.clear()
    changed.apply(visited.append)
    assert len(visited) == 1 and visited[0] is replacement
    assert changed.visit_and_replace(lambda value: None) is None
    await bind_run_callbacks(original, _SOURCE, callbacks).for_run(context)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ("leaf", "wrapper", "combined"))
async def test_deferred_callbacks_follow_native_loading_and_returned_run_shape(shape: str) -> None:
    calls = []
    def container(capability):
        if shape == "combined":
            return CombinedCapability([capability])
        return WrapperCapability(capability) if shape == "wrapper" else capability

    class Project(ToolResultsCapability):
        id, defer_loading = "project", True
        def __init__(self, bound=False):
            self.bound = bound
        async def for_run(self, ctx):
            assert not self.bound
            calls.append("project.bind")
            return container(type(self)(True))
        async def process_tool_result(self, context, tool, arguments, result):
            assert self.bound and tool.name == "inspect"
            calls.append("project.process")
            return ToolReturn(return_value="summary", content=[_REF])

    class Source(ReferencedInputsCapability):
        id, defer_loading = "source", True
        def __init__(self, bound=False):
            self.bound = bound
            async def load(context, content):
                assert self.bound
                calls.append("source.load")
                return _BODY
            super().__init__(load)
        async def for_run(self, ctx):
            assert not self.bound
            calls.append("source.bind")
            return container(type(self)(True))

    requests = 0
    async def stream(messages, info):
        nonlocal requests
        requests += 1
        if requests == 1:
            yield {0: DeltaToolCall(name="inspect", json_args="{}", tool_call_id="before")}
        elif requests == 2:
            assert calls == ["project.bind", "source.bind"]
            yield {0: DeltaToolCall(name="load_capability", json_args='{"id":"project"}', tool_call_id="load-project"),
                   1: DeltaToolCall(name="load_capability", json_args='{"id":"source"}', tool_call_id="load-source")}
        elif requests == 3:
            yield {0: DeltaToolCall(name="inspect", json_args="{}", tool_call_id="after")}
        else:
            assert requests == 4 and calls[-2:] == ["project.process", "source.load"]
            yield "done"
    class Binding:
        provider, model_identity = "test", "test:deferred-lifecycle"
        contract = {"input_mime_types": []}
        def materialize(self):
            return FunctionModel(stream_function=stream)
    async def inspect() -> str:
        return "original"
    from pydantic_ai.toolsets import FunctionToolset
    from linktools.ai.runtime._harness import SummaryProviderModel
    from linktools.ai.capability import RunCallbackScope
    from linktools.ai.runtime._tool_boundary import BoundaryToolset, ManagedToolDescriptor
    from ._runtime_test_helpers import tool_run_context, tool_with_metadata
    callbacks, context = RunCallbacks(), tool_run_context()
    context.deps = AgentContext(None, service_principal("tenant", "owner"), "deferred", None, "run", {})
    descriptor = ManagedToolDescriptor(effect_owner="none", effect_policy="none", tool_class="business")
    boundary = BoundaryToolset(
        (FunctionToolset([tool_with_metadata(inspect, descriptor)]),),
        {"inspect": descriptor}, id="deferred-test", result_callbacks=callbacks,
    )
    binding = Binding()
    model = SummaryProviderModel(binding.materialize(), context=context.deps, binding=binding,
                                 referenced_inputs=callbacks.referenced_inputs)
    capabilities = [RunCallbackScope(callbacks),
                    bind_run_callbacks(Project(), CapabilityRef("capability", "project", 1), callbacks),
                    bind_run_callbacks(Source(), _SOURCE, callbacks)]
    agent = Agent(model, toolsets=[boundary], capabilities=capabilities)
    async with agent.run_stream("Inspect.", deps=context.deps) as result:
        assert await result.get_output() == "done"
    assert callbacks.context is None and callbacks.tool_results("after") == ()
    assert calls == ["project.bind", "source.bind", "project.process", "source.load"]


@pytest.mark.asyncio
async def test_boundary_retains_current_callbacks_across_the_original_tool_await() -> None:
    from pydantic_ai.toolsets import FunctionToolset
    from linktools.ai.runtime._tool_boundary import BoundaryToolset, ManagedToolDescriptor
    from ._runtime_test_helpers import tool_run_context, tool_with_metadata
    entered, release = asyncio.Event(), asyncio.Event()
    class Project(ToolResultsCapability):
        async def process_tool_result(self, context, tool, arguments, result):
            assert result == "original"
            return "projected"
    async def inspect() -> str:
        entered.set()
        await release.wait()
        return "original"
    callbacks, context = RunCallbacks(), tool_run_context()
    descriptor = ManagedToolDescriptor(effect_owner="none", effect_policy="none", tool_class="business")
    boundary = BoundaryToolset((FunctionToolset([tool_with_metadata(inspect, descriptor)]),),
                               {"inspect": descriptor}, id="await", result_callbacks=callbacks)
    tools = await boundary.get_tools(context)
    with callbacks.tool_scope(context.tool_call_id, Project()):
        task = asyncio.create_task(boundary.call_tool("inspect", {}, context, tools["inspect"]))
        await entered.wait()
    assert callbacks.tool_results(context.tool_call_id) == ()
    release.set()
    assert await task == "projected"
