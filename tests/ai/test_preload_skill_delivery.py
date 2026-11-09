#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preload delivery follows the final SDK request, not earlier hook inputs."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai import Agent, RetryModelRequest
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart

from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.capability import (
    PRELOAD_SKILL_READS_METADATA_KEY, SkillCapability, SkillDefinition,
    SkillReadReceipt, SkillSourceRegistry,
)
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime
from linktools.ai.spec import SkillSpec
from linktools.ai.storage import StorageOverlay

from ._runtime_test_helpers import _UsageFunctionModel
from .test_preload_skill_inputs import _declaration, _features


class _ChangePreload(AbstractCapability):
    def __init__(self, phase, *, replacement=None, unset=False):
        self.phase = phase
        self.replacement = replacement
        self.unset = unset
        self.attempts = []

    def change(self, request_context):
        parameters = request_context.model_request_parameters
        parts = None if self.unset else [
            part for part in parameters.instruction_parts
            if not (part.name or "").startswith("preload-skill-")
        ]
        return replace(request_context, model_request_parameters=replace(parameters, instruction_parts=parts))

    async def before_model_request(self, ctx, request_context):
        return self.change(request_context) if self.phase == "before" else request_context

    async def prepare_model_request(self, ctx, request_context):
        self.attempts.append(request_context.attempt)
        if self.phase == "retry" and request_context.attempt == 1:
            raise RetryModelRequest(self.replacement)
        return self.change(request_context) if self.phase in {"prepare", "retry"} else request_context


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("before", "prepare", "retry"))
async def test_preload_removed_by_request_hooks_fails_without_delivery_receipt(tmp_path: Path, phase: str):
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    await store.put(AssetKey("skill", "guide/SKILL.md"), _declaration("guide", "PINNED_RULE"))
    delivered = []

    async def provider(messages, info):
        delivered.append(info.instructions)
        return ModelResponse(parts=[TextPart("done")])

    hook = _ChangePreload(phase, replacement=_UsageFunctionModel(provider))
    features = _features(store, provider, allowed=())
    features.capability(hook, id="change-preload", revision=1, scope="agent")
    try:
        async with Persistence.open("preload-delivery", storage=PersistenceStorage.sqlite(tmp_path / "runtime.db")) as persistence:
            async with Runtime.attach(persistence=persistence, features=features) as runtime:
                execution = await runtime.agents.get().start("go")
                result = (await execution.wait(timeout_seconds=15)).result
                assert result.status is ExecutionStatus.FAILED, result
                assert result.error_code == ErrorCode.BINDING_CONFLICT.value
                assert delivered == []
                if phase == "retry":
                    assert hook.attempts == [1, 2]
                interactions = await runtime.history.read_model_interactions(execution.execution_id,
                    principal=runtime.default_principal, identities=((1, 1),), include_content=True)
                interaction, = interactions
                assert interaction.status == "FAILED"
                assert interaction.request is None or "preload_skill_reads" not in interaction.request
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("unset", (False, True))
@pytest.mark.parametrize("resumed", (False, True))
async def test_preload_delivery_accepts_rendered_instructions_and_native_continuation(unset: bool, resumed: bool):
    definition = SkillDefinition(SkillSpec("guide", "PINNED_RULE\n"))
    skill = SkillCapability((), SkillSourceRegistry(), preload_skills=(definition,))
    received = []

    async def provider(messages, info):
        assert "PINNED_RULE" in info.instructions
        current = messages[-2] if isinstance(messages[-1], ModelResponse) else messages[-1]
        assert isinstance(current, ModelRequest)
        receipt, = current.metadata[PRELOAD_SKILL_READS_METADATA_KEY]
        assert SkillReadReceipt.from_payload(receipt).skill.id == "guide"
        received.append(info.instructions)
        if not resumed and len(received) == 1:
            return ModelResponse(parts=[TextPart("partial")], state="suspended", provider_response_id="continuation")
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(_UsageFunctionModel(provider), capabilities=[skill, _ChangePreload("prepare", unset=unset)]
                  if unset else [skill])
    if resumed:
        history = [ModelRequest.user_text_prompt("go", instructions="PINNED_RULE"),
                   ModelResponse(parts=[TextPart("partial")], state="suspended", provider_response_id="continuation")]
        result = await agent.run(message_history=history)
    else:
        result = await agent.run("go")
    assert result.output == "partialdone"
    assert len(received) == (1 if resumed else 2)


@pytest.mark.asyncio
async def test_standalone_preload_wrapper_validates_nonstream_request_before_provider():
    definition = SkillDefinition(SkillSpec("guide", "PINNED_RULE"))
    skill = SkillCapability((), SkillSourceRegistry(), preload_skills=(definition,))
    delivered = []

    async def provider(messages, info):
        delivered.append(info.instructions)
        return ModelResponse(parts=[TextPart("done")])

    from pydantic_ai.models import ModelRequestParameters
    model = skill.get_wrapper_model(_UsageFunctionModel(provider))
    message = ModelRequest.user_text_prompt("go", instructions="PINNED_RULE")
    with pytest.raises(AIError) as removed:
        await model.request([message], None, ModelRequestParameters(instruction_parts=[]))
    assert removed.value.code is ErrorCode.BINDING_CONFLICT
    assert message.metadata is None and delivered == []
    await model.request([message], None, ModelRequestParameters(instruction_parts=None))
    assert delivered == ["PINNED_RULE"]
    assert PRELOAD_SKILL_READS_METADATA_KEY in message.metadata
