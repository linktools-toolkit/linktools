#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Native fallback dispatch retains Runtime candidate and observation boundaries."""

import pytest
from pydantic_ai.models.decision import DecisionHandOff

from linktools.ai.capability import ExecutionCapability, TraceCapability
from linktools.ai.core import ExecutionStatus
from linktools.ai.model import FallbackBinding
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.persistence import PersistenceStorage
from tests.ai.test_decision_model_binding import _Binding, _Decision, _OfflineDecisionModel


class _HandoffModel(_OfflineDecisionModel):
    async def decide(self, request, model_settings):
        self.requests.append(request)
        raise DecisionHandOff(self.model_name, "delegate", 0.5, "synthetic handoff")


class _AlwaysThinkingModel(_OfflineDecisionModel):
    @property
    def profile(self):
        return {**super().profile, "thinking_always_enabled": True}


def _features(first, second):
    features = RuntimeFeatures("fallback-runtime")
    features.agent("default", allow_tools=(), allow_subagents=())
    features.capability(TraceCapability(), scope="runtime")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    loads = []

    def load():
        loads.append(True)
        return FallbackBinding(_Binding(first), _Binding(second), fallback_on=DecisionHandOff)

    features.model("default", load)
    return features, loads


@pytest.mark.asyncio
async def test_typed_handoff_reaches_one_candidate_and_reopen_preserves_usage(tmp_path):
    first, second = _HandoffModel(), _OfflineDecisionModel()
    features, loads = _features(first, second)
    storage = PersistenceStorage.sqlite(tmp_path / "fallback.sqlite")
    async with Persistence.open('fallback-runtime', storage=storage) as persistence_47, Runtime.attach(persistence=persistence_47, features=features) as runtime:
        execution = await runtime.agents.get().start("classify", output=_Decision)
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert result.usage.model_requests == 1
        assert result.usage.input_tokens == 12
        assert len(first.requests) == len(second.requests) == 1
        assert len(loads) == 1
        principal = runtime.default_principal
        execution_id = execution.execution_id
    async with Persistence.open('fallback-runtime', storage=storage) as persistence_57, Runtime.attach(persistence=persistence_57, features=features) as runtime:
        result = await runtime.executions.result(execution_id, principal=principal)
        assert result.status is ExecutionStatus.SUCCEEDED
        assert result.usage.input_tokens == 12
        assert len(loads) == 1


@pytest.mark.asyncio
async def test_unused_incompatible_candidate_is_not_checked():
    first, second = _OfflineDecisionModel(), _AlwaysThinkingModel()
    features, _ = _features(first, second)
    async with Persistence.open('fallback-runtime', storage=PersistenceStorage.in_memory()) as persistence_68, Runtime.attach(persistence=persistence_68, features=features) as runtime:
        execution = await runtime.agents.get().start("classify", output=_Decision, thinking=False)
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert len(first.requests) == 1
        assert not second.requests


@pytest.mark.asyncio
async def test_selected_always_on_candidate_preserves_native_false_semantics():
    first, second = _HandoffModel(), _AlwaysThinkingModel()
    features, _ = _features(first, second)
    async with Persistence.open('fallback-runtime', storage=PersistenceStorage.in_memory()) as persistence_80, Runtime.attach(persistence=persistence_80, features=features) as runtime:
        execution = await runtime.agents.get().start("classify", output=_Decision, thinking=False)
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert len(first.requests) == len(second.requests) == 1
