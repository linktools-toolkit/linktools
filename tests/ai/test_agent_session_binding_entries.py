#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Captured output contracts survive every Agent and Session admission facade."""

from dataclasses import replace

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo

from linktools.ai.agent import bind_output
from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import ExecutionStatus
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


class _CapturedAnswer(BaseModel):
    captured_count: int


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ("agent", "session", "create-session", "fork-session"))
@pytest.mark.parametrize("mode", ("start", "plan"))
async def test_captured_binding_is_used_by_agent_and_session_entries(entry: str, mode: str) -> None:
    requests: list[str] = []
    output_properties: list[set[str]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        requests.append(str(messages))
        if not info.output_tools:
            output_properties.append(set())
            return ModelResponse(parts=[TextPart("live text output")])
        tool = info.output_tools[0]
        output_properties.append(set(tool.parameters_json_schema["properties"]))
        return ModelResponse(parts=[ToolCallPart(tool.name, {"captured_count": 7})])

    features = RuntimeFeatures("binding-entries")
    features.agent("default", model="default", system_prompt="CURRENT_DEFINITION",
                   allow_tools=(), allow_skills=(), allow_subagents=())
    features.capability(ExecutionCapability(retain_history=True))
    features = model_features(_Models(_UsageFunctionModel(respond)), features)
    async with Persistence.open("binding-entries", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence, features=features) as runtime:
            current = runtime.agents.get()
            captured_base = current.derive(system_prompt="CAPTURED_DEFINITION").execution_port.binding_contract
            output = bind_output(_CapturedAnswer)
            captured = replace(captured_base, output_mode=output.mode, output_schema=output.schema_definition)
            assert captured.binding_digest != current.execution_port.binding_contract.binding_digest
            # Explicit per-admission output is not an AgentSpec default. Rebinding only
            # the compiled Agent loses this contract even if its prompt survives.
            assert captured.agent_spec.output_schema is None
            restored = runtime.agents.restore(captured)
            if entry == "agent":
                handle = restored
            elif entry == "session":
                handle = restored.session("target")
            elif entry == "create-session":
                handle = await restored.create_session("target", idempotency_key="create-target")
            else:
                source = await restored.create_session("source", idempotency_key="create-source")
                handle = await source.fork("target", idempotency_key="fork-target")

            if mode == "start":
                execution = await handle.start("answer", idempotency_key="captured-start")
                result = (await execution.wait(timeout_seconds=10)).result
            else:
                result = (await handle.plan("answer", idempotency_key="captured-plan", timeout_seconds=10)).result
            assert result.status is ExecutionStatus.SUCCEEDED
            assert result.output == {"captured_count": 7}
            assert output_properties == [{"captured_count"}]
            assert "CAPTURED_DEFINITION" in requests[0]
            assert "CURRENT_DEFINITION" not in requests[0]
            record = await persistence.execution.executions.get(result.execution_id, tenant_id=runtime.tenant_id)
            assert record is not None
            assert record.binding_digest == captured.binding_digest
            assert record.binding == captured
            assert record.session_id == (None if entry == "agent" else "target")
