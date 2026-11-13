#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Actual accepted Runtime outputs reuse only exact final Native response parts."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo

from linktools.ai.agent import AssistantTextOutput
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model.messages import encode_model_messages
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._execution_content import restore_native_text_payload
from linktools.ai.persistence._message_root import NativeMessageRoot, NativeMessageSlice, NativeTextReference
from linktools.ai.persistence.codec import decode_envelope, execution_payload_slots, native_text_reference
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.steps import native_store_key
from linktools.ai.storage import StoredPayload

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features, TraceCapability
from .test_agent_default_output import _Count


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ("assistant", "structured", "coerced"))
async def test_actual_runtime_output_selects_exact_native_projection(tmp_path: Path, case: str) -> None:
    def respond(messages: list, info: AgentInfo) -> ModelResponse:
        if not info.output_tools:
            return ModelResponse(parts=[TextPart("plain")])
        value = {"count": "3" if case == "coerced" else 3}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, value)])

    features = RuntimeFeatures("native-output")
    features.agent("default", output_schema=_Count.model_json_schema(), allow_tools=(), allow_skills=(), allow_subagents=())
    features.capability(TraceCapability(), scope="runtime")
    config = PersistenceStorage.sqlite(tmp_path / "outputs.db")
    expected = {"text": "plain"} if case == "assistant" else {"count": 3}
    async with Persistence.open("native-output", tenant_id="tenant", storage=config) as state, Runtime.attach(
        persistence=state, features=model_features(_Models(_UsageFunctionModel(respond)), features),
    ) as runtime:
        result = (await runtime.agents.get().run("review", output=AssistantTextOutput if case == "assistant" else _Count,
                                                timeout_seconds=10)).result
        if case == "coerced":
            assert result.status is ExecutionStatus.FAILED
            assert result.error_code == ErrorCode.MODEL_RESPONSE_INVALID.value
            assert result.output is None
        else:
            assert result.status is ExecutionStatus.SUCCEEDED
            assert result.output == expected
        records = await state.execution.executions.state_store.read(lambda tx: tx.scan_records())
        stored, = (record for record in records if record.kind == "execution")
        value = decode_envelope(stored.data).value["payload"]
        identity = value["fields"]["execution_id"]
        refs = [ref for _, payload in execution_payload_slots(value) if (ref := native_text_reference(payload)) is not None]
        if case == "coerced":
            assert refs == []
        else:
            assert len(refs) == 1
            assert refs[0].projection == ("assistant-text-output" if case == "assistant" else "tool-call-output")
    async with Persistence.open("native-output", tenant_id="tenant", storage=config, read_only=True) as cold:
        restored = await cold.execution.executions.get_result_record(identity, tenant_id="tenant", fresh=True)
        assert restored.result.output == (None if case == "coerced" else StoredPayload.inline_json(expected))
        if case == "coerced":
            assert restored.error_code == ErrorCode.MODEL_RESPONSE_INVALID.value


def test_output_projection_rejects_wrong_part_kind_and_tool_identity() -> None:
    root = NativeMessageRoot("a" * 32, native_store_key("native-output", "tenant"),
                             (NativeMessageSlice("b" * 64, 1, 0, 1),))
    reference = NativeTextReference(root, "tool-call-output", 0, "final_result", "call")
    raw = encode_model_messages((ModelResponse(parts=[ToolCallPart("final_result", {"count": 3}, tool_call_id="call")]),))
    assert restore_native_text_payload(reference, raw) == StoredPayload.inline_json({"count": 3})
    assert NativeTextReference.from_json(reference.to_json()) == reference
    for invalid in (replace(reference, tool_call_id="foreign"), replace(reference, part_index=1), NativeTextReference(root)):
        with pytest.raises(AIError):
            restore_native_text_payload(invalid, raw)
    with pytest.raises(ValueError):
        NativeTextReference.from_json({"$native_text": root.to_json(), "projection": "unknown"})


@pytest.mark.asyncio
async def test_terminal_output_projection_preserves_json_value_types(tmp_path: Path) -> None:
    from .test_native_text_result_runtime import _prepare, _publish

    config = PersistenceStorage.sqlite(tmp_path / "typed-output.db")
    async with Persistence.open("native-result", tenant_id="tenant", storage=config) as state:
        commands, commit, current, checkpoint = await _prepare(state, "placeholder", "placeholder")
        accepted = StoredPayload.inline_json({"count": 3})
        commit = replace(commit, result=replace(commit.result, output=accepted))
        checkpoint = replace(checkpoint, messages=[ModelResponse(parts=[
            ToolCallPart("final_result", {"count": "3"}, tool_call_id="output-call")])])
        await _publish(commands, commit, current, checkpoint)
        repository = state.execution.executions
        raw = await repository.state_store.read_record(repository.execution_record_key("execution"))
        assert not any(native_text_reference(payload) is not None
                       for _, payload in execution_payload_slots(decode_envelope(raw.data).value["payload"]))
    async with Persistence.open("native-result", tenant_id="tenant", storage=config, read_only=True) as cold:
        assert (await cold.execution.executions.get_result_record(
            "execution", tenant_id="tenant", fresh=True)).result.output == accepted
