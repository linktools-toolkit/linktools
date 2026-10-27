#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Execution owns captured history policy without consuming native SDK capability trees."""
from ._runtime_test_helpers import model_features

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, CombinedCapability, WrapperCapability
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from linktools.ai.capability import ExecutionCapability, SessionCapability
from linktools.ai.core import ExecutionStatus, PromptLimits
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime._capabilities import compose_platform_capabilities, runtime_feature_leaves
from linktools.ai.runtime._compaction import CompactionPolicy
from linktools.ai.persistence.runs import RuntimeAgentRunStore, StagingAgentRunStore

from .test_explicit_runtime_features import _FeatureModels, _features


@dataclass
class _Service(AbstractCapability):
    events: list[str] = field(default_factory=list)

    async def after_run(self, ctx: RunContext, *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
        self.events.append('after')
        return result

    async def on_run_error(self, ctx: RunContext, *, error: BaseException) -> AgentRunResult[Any]:
        self.events.append('error')
        raise error


@pytest.mark.asyncio
@pytest.mark.parametrize('retained', [False, True])
async def test_execution_callbacks_use_bound_promise_without_mutating_declaration(retained: bool) -> None:
    declaration = ExecutionCapability(retain_history=not retained)
    service = _Service()
    bound = declaration.bind(service, retain_history=retained)
    result = await Agent(TestModel(custom_output_text='done'), capabilities=[bound]).run('task')
    assert result.output == 'done'
    assert service.events == (['after'] if retained else [])
    for error in (ValueError('failed'), asyncio.CancelledError()):
        with pytest.raises(type(error)) as raised:
            await bound.on_run_error(RunContext(deps=None, model=TestModel(), usage=RunUsage()), error=error)
        assert raised.value is error
    assert service.events == (['after', 'error', 'error'] if retained else [])
    assert declaration.retain_history is not retained
    assert declaration._service is None


@pytest.mark.asyncio
@pytest.mark.parametrize('retained', [False, True])
@pytest.mark.parametrize('combined', [False, True])
async def test_wrapped_execution_owner_is_bound_to_captured_policy_once(retained: bool, combined: bool) -> None:
    staging = StagingAgentRunStore()
    store = RuntimeAgentRunStore(staging, conversation_archive=StagingAgentRunStore(),
                                execution_archive=None, recovery_archive=None)
    store.bind_execution_producer('run', execution_id='execution', producer_generation=1,
                                  conversation=retained, restart_checkpoints=False, durable_trace=False)
    declaration = ExecutionCapability(retain_history=not retained)
    tree = WrapperCapability(CombinedCapability([declaration]) if combined else declaration, id='wrapped-execution')
    values = await compose_platform_capabilities(
        agent_id='agent', agent_run_id='run', execution_id='execution', agent_run_seq=1,
        history_id=None, memory_scope=None, run_store=store, memory_store=None,
        tool_policy=(), compaction_policy=CompactionPolicy(), limits=PromptLimits(),
        planning=False, context_target_tokens=None, parent_agent_run_id=None,
        plan_store_resolver=None, runtime_capabilities=[tree],
    )
    owners = [leaf for value in values for leaf in runtime_feature_leaves(value)
              if isinstance(leaf, ExecutionCapability)]
    assert len(owners) == 1 and owners[0].retain_history is retained
    assert owners[0] is not declaration
    assert declaration.retain_history is not retained and declaration._service is None
    assert any(isinstance(value, WrapperCapability) and value.id == 'wrapped-execution' for value in values)
    assert (tree.wrapped.capabilities[0] if combined else tree.wrapped) is declaration


@dataclass
class _RequiresExecution(AbstractCapability):
    observed: list[bool] = field(default_factory=list)
    id: str = 'test.requires-execution-owner'

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(requires=(ExecutionCapability,), wraps=(ExecutionCapability,))

    async def before_run(self, ctx: RunContext) -> None:
        owners = [value for value in ctx.capabilities.values() if isinstance(value, ExecutionCapability)]
        assert len(owners) == 1
        self.observed.append(owners[0].retain_history)


@pytest.mark.asyncio
async def test_default_execution_owner_exists_without_implying_history() -> None:
    observer = _RequiresExecution()
    async with Persistence.open('default-execution-owner', storage=PersistenceStorage.in_memory()) as persistence_103, Runtime.attach(persistence=persistence_103, features=model_features(_FeatureModels(), _features(observer))) as runtime:
        execution = await runtime.agents.get().start('task')
        assert (await execution.wait()).result.status is ExecutionStatus.SUCCEEDED
    assert observer.observed == [False]


@pytest.mark.asyncio
async def test_retained_execution_forks_with_sessions_disabled() -> None:
    observer = _RequiresExecution()
    declaration = ExecutionCapability(retain_history=True)
    features = _features(WrapperCapability(CombinedCapability([declaration]), id='execution-wrapper'),
                         SessionCapability(enabled=False), observer)
    async with Persistence.open('sessionless-execution-owner', storage=PersistenceStorage.in_memory()) as persistence_116, Runtime.attach(persistence=persistence_116, features=model_features(_FeatureModels(), features)) as runtime:
        source = await runtime.agents.get().start('task')
        assert (await source.wait()).result.status is ExecutionStatus.SUCCEEDED
        fork = await source.fork('follow up')
        assert (await fork.wait()).result.status is ExecutionStatus.SUCCEEDED
    assert observer.observed == [True, True]
    assert declaration._service is None and declaration.retain_history is True


def test_native_wrapper_type_requirements_match_registered_sdk_tree() -> None:
    from pydantic_ai.exceptions import UserError

    class Owner(AbstractCapability):
        pass

    class Consumer(AbstractCapability):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(requires=(Owner,))

    with pytest.raises(UserError, match='requires'):
        CombinedCapability([WrapperCapability(Owner(), id='wrapped-owner'), Consumer()])
    CombinedCapability([
        WrapperCapability(CombinedCapability([Owner()]), id='wrapped-owner'), Consumer(),
    ])
