#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Actual serving models retain prepared request capture after SDK reselection."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai import Agent, RetryModelRequest
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelRequest, ModelResponse, TextPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.tools import RunContext

from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.capability import AgentContext, PRELOAD_SKILL_READS_METADATA_KEY, SkillCapability, SkillDefinition, SkillSourceRegistry
from linktools.ai.core import ExecutionStatus, JsonValue
from linktools.ai.model.messages import decode_model_messages
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._model_interaction import StagedContextInline
from linktools.ai.persistence._step_contracts import AgentRunRecord
from linktools.ai.persistence.runs import StagingAgentRunStore
from linktools.ai.runtime import Runtime
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder
from linktools.ai.runtime._metric_capability import ModelObservationCapability
from linktools.ai.spec import SkillSpec
from linktools.ai.storage import StorageOverlay

from ._runtime_test_helpers import _UsageFunctionModel
from .test_preload_skill_inputs import _declaration, _features


class _ReselectModel(AbstractCapability[AgentContext[object]]):
    def __init__(self, model: _UsageFunctionModel) -> None:
        self.model = model

    async def prepare_model_request(
        self, ctx: RunContext[AgentContext[object]], request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        if request_context.attempt == 1:
            raise RetryModelRequest(self.model)
        return request_context


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", (False, True))
@pytest.mark.parametrize("skill_first", (False, True))
async def test_reselected_provider_capture_includes_final_skill_receipt(
    streaming: bool, skill_first: bool,
) -> None:
    provider_requests: list[JsonValue] = []

    async def provider(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert "PINNED_RULE" in (info.instructions or "")
        assert isinstance(messages[-1], ModelRequest)
        assert PRELOAD_SKILL_READS_METADATA_KEY in (messages[-1].metadata or {})
        provider_requests.append(ModelMessagesTypeAdapter.dump_python(messages, mode="json"))
        return ModelResponse(parts=[TextPart("done")])

    async def unselected(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError("the model replaced during preparation must not be called")

    store = StagingAgentRunStore()
    recorder = AgentRunRecorder(store, execution_id="execution", agent_run_id="run")
    await recorder.register_agent_run(AgentRunRecord("run"))
    observation = ModelObservationCapability(
        None, source_namespace="reselection", tenant_id="default", execution_id="execution",
        session_id=None, agent_run_id="run", agent_id="agent", interaction_recorder=recorder,
    )
    skill = SkillCapability((), SkillSourceRegistry(),
                            preload_skills=(SkillDefinition(SkillSpec("guide", "PINNED_RULE")),))
    capabilities = [skill, observation] if skill_first else [observation, skill]
    agent = Agent(_UsageFunctionModel(unselected),
                  capabilities=[*capabilities, _ReselectModel(_UsageFunctionModel(provider))])
    if streaming:
        async with agent.run_stream("go", deps=SimpleNamespace(correlation={})) as run:
            assert await run.get_output() == "done"
    else:
        assert (await agent.run("go", deps=SimpleNamespace(correlation={}))).output == "done"
    interaction, = await store.list_model_interactions(agent_run_id="run")
    captured: list[ModelMessage] = []
    for item in interaction.request_context.items:
        assert isinstance(item, StagedContextInline)
        captured.extend(decode_model_messages(store.staged_payload("run", item.payload_digest)))
    assert ModelMessagesTypeAdapter.dump_python(captured, mode="json") == provider_requests[0]
    envelope = json.loads(store.staged_payload("run", interaction.request_envelope_digest))
    assert envelope["streaming"] is streaming
    assert "PINNED_RULE" in str(envelope["parameters"]["instruction_parts"])


@pytest.mark.asyncio
async def test_reselected_model_history_retains_preload_receipt(tmp_path: Path) -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    await store.put(AssetKey("skill", "guide/SKILL.md"), _declaration("guide", "PINNED_RULE"))

    async def provider(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert "PINNED_RULE" in (info.instructions or "")
        assert isinstance(messages[-1], ModelRequest)
        assert PRELOAD_SKILL_READS_METADATA_KEY in (messages[-1].metadata or {})
        return ModelResponse(parts=[TextPart("done")])

    features = _features(store, provider, allowed=())
    features.capability(_ReselectModel(_UsageFunctionModel(provider)), id="select-model", scope="agent")
    try:
        async with Persistence.open("preload-reselection", storage=PersistenceStorage.sqlite(tmp_path / "runtime.db")) as persistence:
            async with Runtime.attach(persistence=persistence, features=features) as runtime:
                execution = await runtime.agents.get().start("go")
                result = (await execution.wait(timeout_seconds=15)).result
                assert result.status is ExecutionStatus.SUCCEEDED, result
                interaction, = await runtime.history.read_model_interactions(
                    execution.execution_id, principal=runtime.default_principal,
                    identities=((1, 1),), include_content=True,
                )
                assert interaction.request["preload_skill_reads"]
                assert "PINNED_RULE" in str(interaction.request["instructions"])
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_provider_error_reselection_observes_the_successful_model() -> None:
    from pydantic_ai.usage import RequestUsage

    from .test_model_metric_projection_boundary import _Recorder

    attempts: list[str] = []

    async def first(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        attempts.append("first")
        raise ValueError("first provider failed")

    async def second(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        attempts.append("second")
        return ModelResponse(parts=[TextPart("recovered")], usage=RequestUsage(input_tokens=3, output_tokens=5))

    selected = _UsageFunctionModel(second, model_name="successful-provider")

    class RecoverRequest(AbstractCapability[AgentContext[object]]):
        async def on_model_request_error(
            self, ctx: RunContext[AgentContext[object]], *,
            request_context: ModelRequestContext, error: Exception,
        ) -> ModelResponse:
            if request_context.attempt == 1:
                raise RetryModelRequest(selected)
            raise error

    metric = _Recorder("accept", [])
    observation = ModelObservationCapability(
        metric, source_namespace="reselection", tenant_id="default", execution_id="execution",
        session_id=None, agent_run_id="run", agent_id="agent",
    )
    result = await Agent(_UsageFunctionModel(first, model_name="failed-provider"),
                         capabilities=[observation, RecoverRequest()]).run("go", deps=SimpleNamespace(correlation={}))
    assert result.output == "recovered"
    assert attempts == ["first", "second"]
    assert result.usage.input_tokens == 3 and result.usage.output_tokens == 5
    recorded, = metric.observations
    assert recorded.status == "SUCCEEDED"
    assert recorded.dimensions["selected_model_name"] == selected.model_name
    assert {item.name: item.value for item in recorded.measurements if item.name.endswith("tokens")} == {
        "input_tokens": 3, "output_tokens": 5, "total_tokens": 8,
    }
