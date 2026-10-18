#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native feature binding preserves wrappers, type ordering and run isolation."""

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, CombinedCapability, WrapperCapability
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import RunContext

from linktools.ai.core import ExecutionStatus
from linktools.ai.capability import CheckpointCapability, ConversationCapability, TraceCapability
from linktools.ai.runtime import Runtime, RuntimeStorage

from .test_explicit_runtime_features import _FeatureModels, _features


@dataclass
class _ObservedWrapper(WrapperCapability):
    label: str = "wrapper"
    events: list[tuple[str, str, str]] = field(default_factory=list)

    async def before_run(self, ctx: RunContext) -> None:
        self.events.append((ctx.deps.execution_id, self.label, "before"))
        await super().before_run(ctx)

    async def after_run(self, ctx: RunContext, *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
        result = await super().after_run(ctx, result=result)
        self.events.append((ctx.deps.execution_id, self.label, "after"))
        return result


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", (False, True))
async def test_wrapped_features_keep_native_order_and_isolate_concurrent_runs(nested: bool) -> None:
    events: list[tuple[str, str, str]] = []
    trace = TraceCapability()
    inner = _ObservedWrapper(trace, label="inner", events=events)
    tree = (_ObservedWrapper(CombinedCapability([inner, CheckpointCapability(), ConversationCapability()]),
                             id="wrapped.features", label="outer", events=events) if nested else inner)
    features = _features(tree)
    state = RuntimeStorage.in_memory()
    async with Runtime.open("wrapped-features", models=_FeatureModels(), storage=state,
                            features=features) as runtime:
        executions = await asyncio.gather(*(runtime.agents.get().start(f"input {index}") for index in range(3)))
        results = await asyncio.gather(*(execution.wait(timeout_seconds=5) for execution in executions))
        assert all(result.result.status is ExecutionStatus.SUCCEEDED for result in results)
        for index, execution in enumerate(executions):
            expected = ([("outer", "before"), ("inner", "before"), ("inner", "after"), ("outer", "after")]
                        if nested else [("inner", "before"), ("inner", "after")])
            assert [(label, phase) for owner, label, phase in events if owner == execution.execution_id] == expected
            interactions = await execution.model_interactions()
            assert len(interactions.items) == 1
            assert interactions.items[0].execution_id == execution.execution_id
            transcript = await execution.transcript(include_content=True)
            assert [item.text for item in transcript.items] == [f"input {index}", "done"]
    assert inner.wrapped is trace
    assert trace._service is None


@dataclass(kw_only=True, eq=False)
class _ObservedTrace(TraceCapability):
    events: list[tuple[str, str]] = field(default_factory=list)
    id: str | None = "runtime.trace"

    def __post_init__(self) -> None:
        super().__init__()

    async def before_run(self, ctx: RunContext) -> None:
        self.events.append((ctx.deps.execution_id, "trace"))
        await super().before_run(ctx)


@dataclass(kw_only=True, eq=False)
class _RequiresFeatureTypes(AbstractCapability):
    events: list[tuple[str, str]] = field(default_factory=list)
    id: str | None = "test.requires-features"

    def get_ordering(self) -> CapabilityOrdering:
        features = (ConversationCapability, CheckpointCapability, TraceCapability)
        return CapabilityOrdering(requires=features, wraps=features)

    async def before_run(self, ctx: RunContext) -> None:
        self.events.append((ctx.deps.execution_id, "requires"))


@pytest.mark.asyncio
async def test_bound_features_preserve_public_types_and_subclass_hooks_for_native_ordering() -> None:
    events: list[tuple[str, str]] = []
    trace = _ObservedTrace(events=events)
    checkpoint = CheckpointCapability()
    conversation = ConversationCapability()
    features = _features(trace, checkpoint, conversation, _RequiresFeatureTypes(events=events))
    async with Runtime.open("type-ordering", models=_FeatureModels(), storage=RuntimeStorage.in_memory(),
                            features=features) as runtime:
        executions = await asyncio.gather(*(runtime.agents.get().start(str(index)) for index in range(2)))
        results = await asyncio.gather(*(execution.wait(timeout_seconds=5) for execution in executions))
        assert all(result.result.status is ExecutionStatus.SUCCEEDED for result in results)
        for execution in executions:
            assert [label for owner, label in events if owner == execution.execution_id] == ["requires", "trace"]
            assert len((await execution.model_interactions()).items) == 1
    assert trace._service is checkpoint._service is conversation._service is None
