#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Execution owns captured history policy without consuming native SDK capability trees."""
from ._runtime_test_helpers import model_features

import asyncio

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, CombinedCapability, WrapperCapability
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext

from linktools.ai.capability import SessionCapability
from linktools.ai.core import ExecutionStatus, PromptLimits
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime._capabilities import compose_platform_capabilities
from linktools.ai.runtime._compaction import CompactionPolicy
from linktools.ai.persistence.runs import RuntimeAgentRunStore, StagingAgentRunStore

from .test_explicit_runtime_features import _FeatureModels, _features


@pytest.mark.asyncio
@pytest.mark.parametrize("retained", [False, True])
async def test_admitted_promise_controls_conversation_service(retained: bool) -> None:
    from linktools.ai.runtime._conversation_service import ConversationService

    store = RuntimeAgentRunStore(
        StagingAgentRunStore(), conversation_archive=StagingAgentRunStore(), execution_archive=None,
    )
    store.bind_execution_producer("run", execution_id="execution", producer_generation=1,
                                  conversation=retained, restart_checkpoints=False, durable_trace=False)
    values = await compose_platform_capabilities(
        agent_id="agent", agent_run_id="run", execution_id="execution", agent_run_seq=1,
        history_id=None, memory_scope=None, run_store=store, memory_store=None,
        tool_policy=(), compaction_policy=CompactionPolicy(), limits=PromptLimits(), planning=False,
        context_target_tokens=None, parent_agent_run_id=None, plan_store_resolver=None,
        retain_history=not retained,
    )
    assert any(isinstance(value, ConversationService) for value in values) is retained


@pytest.mark.asyncio
async def test_retained_execution_forks_with_sessions_disabled() -> None:
    features = _features(SessionCapability(enabled=False), retain_history=True)
    async with Persistence.open('sessionless-execution-owner', storage=PersistenceStorage.in_memory()) as persistence_116, Runtime.open(persistence=persistence_116, features=model_features(_FeatureModels(), features)) as runtime:
        source = await runtime.agents.get().start('task')
        assert (await source.wait()).result.status is ExecutionStatus.SUCCEEDED
        fork = await source.fork('follow up')
        assert (await fork.wait()).result.status is ExecutionStatus.SUCCEEDED


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
