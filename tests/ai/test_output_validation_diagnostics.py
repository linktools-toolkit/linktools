#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Schema corrections retain useful paths without retaining rejected values."""

import json
import traceback
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

import pytest
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.usage import RunUsage, UsageLimits

from linktools.ai.agent import OutputBinding, bind_output, restore_output
from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import ExecutionStatus, JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime._agent_executor import _execution_error


_SECRET = "sentinel" + "-private-rejected-value"
_ENUM_SECRET = "sentinel" + "-private-schema-literal"


def _binding(properties: dict[str, JsonValue], **constraints: JsonValue) -> OutputBinding:
    return OutputBinding.create("structured", {"type": "object", "properties": properties, **constraints})


def _map(error: Exception) -> AIError:
    return _execution_error(error, usage_limits=UsageLimits(), run_usage=RunUsage())


@pytest.mark.parametrize(
    ("binding", "payload", "path", "rule"),
    [
        (_binding({"count": {"type": "integer"}}, required=["count"]), {"other": _SECRET}, '$["count"]', "required"),
        (_binding({"count": {"type": "integer"}}), {"count": _SECRET}, '$["count"]', "type"),
        (_binding({"choice": {"enum": [_ENUM_SECRET]}}), {"choice": _SECRET}, '$["choice"]', "enum"),
        (_binding({"choice": {"const": _ENUM_SECRET}}), {"choice": _SECRET}, '$["choice"]', "const"),
        (_binding({}, additionalProperties=False), {_SECRET: _SECRET}, "$", "additionalProperties"),
        (_binding({"labels": {"type": "object", "additionalProperties": {"type": "integer"}}}), {"labels": {_SECRET: _SECRET}}, '$["labels"]["<key>"]', "type"),
        (_binding({"labels": {"type": "object", "patternProperties": {"^sentinel": {"type": "integer"}}}}), {"labels": {_SECRET: _SECRET}}, '$["labels"]["<key>"]', "type"),
        (_binding({"records": {"type": "array", "items": {"type": "object", "properties": {"count": {"type": "integer"}}}}}), {"records": [{"count": _SECRET}]}, '$["records"][0]["count"]', "type"),
        (_binding({"count": {"type": "integer"}}), [_SECRET], "$", "type"),
    ],
)
def test_schema_diagnostic_is_actionable_and_does_not_echo_values(
    binding: OutputBinding, payload: JsonValue, path: str, rule: str
) -> None:
    with pytest.raises(ValidationError) as failed:
        TypeAdapter(binding.runtime_output_type).validate_python(payload)
    mapped = _map(failed.value)
    assert mapped.code is ErrorCode.OUTPUT_VALIDATION_FAILED
    assert mapped.retryable is False
    assert mapped.safe_details["output_validation"]["path"] == path
    assert mapped.safe_details["output_validation"]["rule"] == rule
    retry = RetryPromptPart.from_error(failed.value, tool_name="final_result")
    rendered = "\n".join((
        str(failed.value), failed.value.json(), retry.model_response(), str(mapped),
        str(mapped.safe_details), str(mapped.diagnostics), "".join(traceback.format_exception(failed.value)),
    ))
    assert rule in retry.model_response()
    assert _SECRET not in rendered
    assert _ENUM_SECRET not in rendered
    assert all(detail.get("input") is None for detail in failed.value.errors())


def test_final_payload_uses_the_same_safe_schema_diagnostic() -> None:
    binding = _binding({"count": {"type": "integer"}})
    with pytest.raises(AIError) as failed:
        binding.validate_payload({"count": _SECRET})
    with pytest.raises(ValidationError) as sdk_failed:
        TypeAdapter(binding.runtime_output_type).validate_python({"count": _SECRET})
    assert failed.value.safe_details == _map(sdk_failed.value).safe_details
    assert _SECRET not in "".join(traceback.format_exception(failed.value))
    assert failed.value.code is ErrorCode.OUTPUT_VALIDATION_FAILED


def test_schema_diagnostic_bounds_long_paths_and_retains_alias_contract() -> None:
    class Item(BaseModel):
        count: int = Field(alias="externalCount")

    class Output(BaseModel):
        items: list[Item]

    binding = bind_output(Output)
    restored = restore_output(binding.mode, binding.schema_definition)
    assert restored == binding
    with pytest.raises(ValidationError) as failed:
        TypeAdapter(restored.runtime_output_type).validate_python({"items": [{"externalCount": _SECRET}]})
    detail = _map(failed.value).safe_details["output_validation"]
    assert detail == {
        "path": '$["items"][0]["externalCount"]', "rule": "type",
        "expected_type": "integer", "actual_type": "string",
    }
    assert TypeAdapter(restored.runtime_output_type).validate_python({"items": [{"externalCount": 3}]}) == {"items": [{"externalCount": 3}]}

    schema: dict[str, JsonValue] = {"type": "integer"}
    payload: JsonValue = _SECRET
    for index in range(20):
        name = f"field{index}" + "x" * 100
        schema = {"type": "object", "properties": {name: schema}}
        payload = {name: payload}
    with pytest.raises(ValidationError) as deep_failed:
        TypeAdapter(OutputBinding.create("structured", schema).runtime_output_type).validate_python(payload)
    deep_detail = _map(deep_failed.value).safe_details["output_validation"]
    assert len(deep_detail["path"]) <= 512
    assert len(str(deep_failed.value)) < 1024
    assert _SECRET not in str(deep_failed.value)


@pytest.mark.asyncio
async def test_model_corrects_output_from_safe_retry_feedback() -> None:
    retry_feedback: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        retries = [part for message in messages for part in message.parts if isinstance(part, RetryPromptPart)]
        if retries:
            retry_feedback.append(retries[-1].model_response())
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"count": 3 if retries else _SECRET})])

    binding = _binding({"count": {"type": "integer"}})
    result = await Agent(FunctionModel(respond), output_type=binding.runtime_output_type, retries=1).run("count")
    assert result.output == {"count": 3}
    assert len(retry_feedback) == 1
    assert "count" in retry_feedback[0] and "expected integer" in retry_feedback[0]
    assert _SECRET not in retry_feedback[0]


@pytest.mark.asyncio
async def test_exhausted_schema_retries_preserve_safe_diagnostics_and_traceback() -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"count": _SECRET})])

    binding = _binding({"count": {"type": "integer"}})
    with pytest.raises(UnexpectedModelBehavior) as failed:
        await Agent(FunctionModel(respond), output_type=binding.runtime_output_type, retries=1).run("count")
    mapped = _map(failed.value)
    assert mapped.code is ErrorCode.MODEL_RESPONSE_INVALID
    assert mapped.safe_details["output_validation"]["path"] == '$["count"]'
    assert mapped.diagnostics is not None
    assert mapped.diagnostics.exception_type == "ValidationError"
    assert "expected integer" in mapped.diagnostics.exception_message
    assert _SECRET not in "".join(traceback.format_exception(failed.value))


class _InvalidOutputModels:
    route_id = "default"
    provider = "test"
    model_identity = "test:invalid-output"
    vision = False
    contract: dict[str, JsonValue] = {"provider": "test", "model": "invalid-output"}

    def capture(self) -> "_InvalidOutputModels":
        return self

    def resolve(self, route_id: str) -> "_InvalidOutputModels":
        assert route_id == self.route_id
        return self

    def restore(self, payload: Mapping[str, JsonValue], *, route_id: str | None = None) -> "_InvalidOutputModels":
        assert dict(payload) == self.contract
        assert route_id in (None, self.route_id)
        return self

    def materialize(self) -> FunctionModel:
        async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[dict[int, DeltaToolCall]]:
            yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args=json.dumps({"count": _SECRET}))}
        return FunctionModel(stream_function=respond)


@pytest.mark.asyncio
async def test_failed_runtime_persists_safe_diagnostic_and_logs_no_output_values(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    class Output(BaseModel):
        count: int

    models = _InvalidOutputModels()
    group = CapabilityGroup("output-diagnostics")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    async with Runtime.open("output-diagnostics", models=models, capabilities=(group,), storage=RuntimeStorage.filesystem(tmp_path)) as runtime:
        result = (await runtime.agents.get().run("count", output=Output, timeout_seconds=10)).result
        assert result.status is ExecutionStatus.FAILED
        assert result.error_code == ErrorCode.MODEL_RESPONSE_INVALID.value
        assert result.safe_error_details["output_validation"]["path"] == '$["count"]'
        assert result.error_diagnostics is not None
        assert "expected integer" in result.error_diagnostics.exception_message
        assert _SECRET not in str(result.error_diagnostics)
        execution_id = result.execution_id
    assert "expected integer" in caplog.text
    assert _SECRET not in caplog.text
    async with Runtime.open("output-diagnostics", models=models, capabilities=(group,), storage=RuntimeStorage.filesystem(tmp_path)) as runtime:
        restored = await runtime.history.result(execution_id, principal=runtime.default_principal)
        assert restored.safe_error_details == result.safe_error_details
        assert restored.error_diagnostics == result.error_diagnostics
