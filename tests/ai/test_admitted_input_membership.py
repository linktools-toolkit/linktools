#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Semantic output membership uses original admitted references, not schema enums."""

from collections.abc import Sequence
import hashlib
import json
from typing import Any

import pytest
from pydantic_ai import ModelRetry
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.tools import RunContext

from linktools.ai.asset import AssetKey, AssetVersionRef
from linktools.ai.capability import AgentContext, ReferencedInputResult, ReferencedInputsCapability
from linktools.ai.core import ExecutionStatus, PromptLimits
from linktools.ai.model import ReferencedContent
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.spec import CapabilityRef
from linktools.ai.storage import StorageEntryRevision

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


_SCHEMA = {
    "type": "object",
    "properties": {"sources": {"type": "array", "items": {"type": "string"}}},
    "required": ["sources"],
    "additionalProperties": False,
}
_BODY = b"synthetic material"
_SOURCE = CapabilityRef("capability", "admitted-materials", 1)


class _AdmittedMembership(AbstractCapability[AgentContext[None]]):
    id = "admitted-source-membership"

    async def after_output_validate(
        self, ctx: RunContext[AgentContext[None]], *, output_context: Any, output: Any,
    ) -> Any:
        del output_context
        assert ctx.deps.original_input is not None
        allowed = {item["input_identifier"] for item in ctx.deps.original_input["attachments"]}
        selected = output["sources"]
        if len(set(selected)) != len(selected) or not set(selected).issubset(allowed):
            raise ModelRetry("Use distinct source identities from the admitted input.")
        return output


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", (("source-0",), ("foreign",), ("source-0", "source-0")))
async def test_semantic_membership_uses_original_admitted_file_set(selected: tuple[str, ...]) -> None:
    count = 1000 if selected == ("foreign",) else 3
    original = [ReferencedContent(AssetVersionRef(
        AssetKey("document", f"source-{index}"), "admitted", StorageEntryRevision(1),
        hashlib.sha256(_BODY).hexdigest(), len(_BODY),
    ), media_type="text/plain", representation="text", source=_SOURCE, identifier=f"source-{index}")
        for index in range(count)]
    attempts: list[list[str]] = []
    effects: list[str] = []
    loaded: list[str] = []
    schemas: list[str] = []

    async def authorized_load_many(
        context: AgentContext[None], contents: Sequence[ReferencedContent],
    ) -> list[ReferencedInputResult]:
        assert context.original_input is not None
        admitted = {item["input_identifier"]: item for item in context.original_input["attachments"]}
        results = []
        for content in contents:
            expected = admitted[content.identifier]
            assert (expected["digest"], expected["size"], expected["media_type"]) == (
                content.ref.digest, content.ref.size, content.media_type,
            )
            loaded.append(content.identifier)
            results.append(ReferencedInputResult(content, _BODY))
        return results

    def record_effect(context: AgentContext[None]) -> str:
        effects.append(context.execution_id)
        return "recorded"

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        schema = json.dumps(info.output_tools[0].parameters_json_schema, sort_keys=True)
        schemas.append(schema)
        assert "source-999" not in schema and len(schema) < 1024
        assert "source-999" not in str(messages)
        if not effects:
            original[:] = []
            return ModelResponse(parts=[ToolCallPart("record_effect", {}, tool_call_id="effect")])
        if attempts:
            feedback = [part for message in messages for part in message.parts
                        if isinstance(part, ToolReturnPart) and part.outcome == "retried"]
            assert any("Use distinct source identities" in part.model_response_str() for part in feedback)
        output = ["source-0"] if attempts else list(selected)
        attempts.append(output)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"sources": output})])

    features = RuntimeFeatures("admitted-membership", limits=PromptLimits(max_binary_input_parts=1000))
    features.capability(_AdmittedMembership(), revision=1, scope="agent")
    features.capability(ReferencedInputsCapability(authorized_load_many=authorized_load_many),
                        id=_SOURCE.id, revision=1, scope="agent")
    features.tool(record_effect, effect_policy="replay_safe")
    features.agent("default", allow_tools=("record_effect",), allow_skills=(), allow_subagents=(),
                   output_schema=_SCHEMA, output_retries=1)
    model_features(_Models(_UsageFunctionModel(model)), features)
    async with Persistence.open("admitted-membership", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence, features=features) as runtime:
            execution = await runtime.agents.get().start(["Review the admitted materials.", *original])
            result = (await execution.wait(timeout_seconds=30)).result
            assert result.status is ExecutionStatus.SUCCEEDED, result
            assert result.output == {"sources": ["source-0"]}
    assert len(effects) == 1
    assert len(attempts) == (1 if selected == ("source-0",) else 2)
    assert len(set(schemas)) == 1
    assert set(loaded) == {f"source-{index}" for index in range(count)}
