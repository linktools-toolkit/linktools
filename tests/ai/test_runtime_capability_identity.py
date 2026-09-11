#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for opaque capability semantic identity."""

from dataclasses import dataclass, fields

import pytest
from linktools.ai.agent import OutputBinding
from linktools.ai.capability import CapabilityContribution, CapabilityGroup, AgentContext
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.spec import AgentSpec
from pydantic_ai.capabilities import AbstractCapability, PrepareTools, SelectModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset


@dataclass
class _Capability(AbstractCapability[AgentContext[None]]):
    id: str = "test-capability"

    @classmethod
    def get_serialization_name(cls) -> "str | None":
        return None


@dataclass
class _OtherCapability(AbstractCapability[AgentContext[None]]):
    id: str = "test-capability"


@dataclass
class _ModelCapability(AbstractCapability[AgentContext[None]]):
    id: str = "model-capability"

    def get_model(self) -> str:
        return "openai:gpt-test"


@dataclass
class _ToolsetCapability(AbstractCapability[AgentContext[None]]):
    id: str = "toolset-capability"

    def get_toolset(self) -> FunctionToolset[AgentContext[None]]:
        return FunctionToolset()


@dataclass
class _DynamicCapability(AbstractCapability[AgentContext[None]]):
    id: str = "dynamic-capability"

    async def for_run(
        self, ctx: RunContext[AgentContext[None]]
    ) -> AbstractCapability[AgentContext[None]]:
        del ctx
        return self


@dataclass
class _ToolWrapperCapability(AbstractCapability[AgentContext[None]]):
    id: str = "tool-wrapper-capability"

    def get_wrapper_toolset(
        self, toolset: AbstractToolset[AgentContext[None]]
    ) -> AbstractToolset[AgentContext[None]]:
        return toolset


@dataclass
class _PrepareToolsCapability(AbstractCapability[AgentContext[None]]):
    id: str = "prepare-tools-capability"

    async def prepare_tools(self, ctx, tool_defs):  # type: ignore[no-untyped-def]
        del ctx
        return tool_defs


@dataclass
class _ToolExecuteCapability(AbstractCapability[AgentContext[None]]):
    id: str = "tool-execute-capability"

    async def before_tool_execute(self, ctx, *, call, tool_def, args):  # type: ignore[no-untyped-def]
        del ctx, call, tool_def
        return args


@dataclass
class _ModelRequestCapability(AbstractCapability[AgentContext[None]]):
    id: str = "model-request-capability"

    async def before_model_request(self, ctx, request_context):  # type: ignore[no-untyped-def]
        del ctx
        return request_context


@dataclass
class _DeferredResolverCapability(AbstractCapability[AgentContext[None]]):
    id: str = "deferred-resolver-capability"

    async def handle_deferred_tool_calls(self, ctx, *, requests):  # type: ignore[no-untyped-def]
        del ctx, requests
        return None


@dataclass
class _OutputTransformCapability(AbstractCapability[AgentContext[None]]):
    id: str = "output-transform-capability"

    async def after_output_validate(self, ctx, *, output_context, output):  # type: ignore[no-untyped-def]
        del ctx, output_context
        return output


@pytest.mark.asyncio
async def test_capability_revision_is_fingerprint_input_only() -> None:
    first = CapabilityGroup[None]("first")
    first.capability(_Capability(), revision=1, semantic_config={})
    second = CapabilityGroup[None]("second")
    second.capability(_Capability(), revision=2, semantic_config={})

    first_candidate = (await first.freeze())[0]
    second_candidate = (await second.freeze())[0]

    assert tuple(item.name for item in fields(CapabilityContribution)) == (
        "kind",
        "id",
        "fingerprint",
        "value",
    )
    assert first_candidate.kind == "capability"
    assert first_candidate.id == "test-capability"
    assert not hasattr(first_candidate, "semantic_revision")
    assert first_candidate.semantic_contract["revision"] == 1
    assert second_candidate.semantic_contract["revision"] == 2
    assert first_candidate.fingerprint != second_candidate.fingerprint
    assert "restore_locator" not in first_candidate.semantic_contract


@pytest.mark.asyncio
async def test_identical_capability_semantics_have_stable_fingerprint() -> None:
    left = CapabilityGroup[None]("left")
    left.capability(_Capability(), revision=7, semantic_config={"mode": "strict"})
    right = CapabilityGroup[None]("right")
    right.capability(_Capability(), revision=7, semantic_config={"mode": "strict"})

    left_candidate = (await left.freeze())[0]
    right_candidate = (await right.freeze())[0]

    assert left_candidate.fingerprint == right_candidate.fingerprint
    assert left_candidate.semantic_contract == right_candidate.semantic_contract
    assert left_candidate.semantic_contract["config"] == {"mode": "strict"}


@pytest.mark.asyncio
async def test_public_semantic_config_changes_capability_fingerprint() -> None:
    strict = CapabilityGroup[None]("strict")
    strict.capability(_Capability(), revision=1, semantic_config={"mode": "strict"})
    relaxed = CapabilityGroup[None]("relaxed")
    relaxed.capability(_Capability(), revision=1, semantic_config={"mode": "relaxed"})

    strict_candidate = (await strict.freeze())[0]
    relaxed_candidate = (await relaxed.freeze())[0]

    assert strict_candidate.fingerprint != relaxed_candidate.fingerprint


def test_opaque_contribution_factory_rejects_canonical_declarations() -> None:
    with pytest.raises(AIError) as error:
        CapabilityContribution.from_opaque(
            "agent",  # type: ignore[arg-type]
            "agent",
            AgentSpec("agent"),  # type: ignore[arg-type]
        )
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.parametrize("revision", [0, -1, True])
def test_invalid_capability_revision_is_rejected(revision: object) -> None:
    group = CapabilityGroup[None]("group")
    with pytest.raises(AIError) as error:
        group.capability(_Capability(), revision=revision)  # type: ignore[arg-type]
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
async def test_deferred_generic_capability_keeps_native_semantics() -> None:
    capability = _Capability()
    capability.defer_loading = True
    capability.description = "load on demand"
    group = CapabilityGroup[None]("group")
    group.capability(capability)

    candidate = (await group.freeze())[0]
    assert candidate.value is capability
    assert candidate.value.defer_loading is True
    assert candidate.semantic_contract["config"] == {}


@pytest.mark.parametrize(
    "capability_id",
    [
        "linktools.custom",
        "linktools.workspace",
    ],
)
def test_custom_capability_cannot_claim_runtime_or_mcp_namespace(capability_id: str) -> None:
    group = CapabilityGroup[None]("group")
    with pytest.raises(AIError) as error:
        group.capability(_Capability(capability_id), semantic_config={})
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
async def test_duplicate_capability_identity_is_rejected_when_group_freezes() -> None:
    group = CapabilityGroup[None]("group")
    group.capability(_Capability(), revision=1, semantic_config={})
    group.capability(_Capability(), revision=1, semantic_config={})

    with pytest.raises(AIError) as error:
        await group.freeze()

    assert error.value.code is ErrorCode.CAPABILITY_CONFLICT


@pytest.mark.asyncio
async def test_capability_semantic_config_defaults_to_empty() -> None:
    group = CapabilityGroup[None]("group")
    group.capability(_Capability())

    candidate = (await group.freeze())[0]
    assert candidate.semantic_contract["config"] == {}


@pytest.mark.asyncio
async def test_capability_implementation_identity_is_not_fingerprint_input() -> None:
    first = CapabilityGroup[None]("first")
    first.capability(_Capability(), revision=1, semantic_config={})
    second = CapabilityGroup[None]("second")
    second.capability(_OtherCapability(), revision=1, semantic_config={})

    first_candidate = (await first.freeze())[0]
    second_candidate = (await second.freeze())[0]

    assert first_candidate.semantic_contract == second_candidate.semantic_contract
    assert first_candidate.fingerprint == second_candidate.fingerprint


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "capability",
    (
        _ModelCapability(),
        _ToolsetCapability(),
        _ToolWrapperCapability(),
        _PrepareToolsCapability(),
        _ToolExecuteCapability(),
        _ModelRequestCapability(),
        _DeferredResolverCapability(),
        _DynamicCapability(),
    ),
)
async def test_external_capability_keeps_native_pydantic_extension_surface(
    capability: AbstractCapability[AgentContext[None]],
) -> None:
    group = CapabilityGroup[None]("group")
    group.capability(capability)

    candidate = (await group.freeze())[0]
    assert candidate.value is capability
    assert candidate.semantic_contract["revision"] == 1


@pytest.mark.asyncio
async def test_external_capability_keeps_output_transformation_hooks() -> None:
    group = CapabilityGroup[None]("group")
    group.capability(
        _OutputTransformCapability(),
        semantic_config={"mode": "identity"},
    )

    candidate = (await group.freeze())[0]
    assert candidate.id == "output-transform-capability"


def test_output_binding_revalidates_final_payload() -> None:
    binding = OutputBinding.create(
        "structured",
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )

    binding.validate_payload({"answer": "ok"})
    with pytest.raises(AIError) as error:
        binding.validate_payload({"unexpected": True})
    assert error.value.code is ErrorCode.OUTPUT_VALIDATION_FAILED


@pytest.mark.asyncio
async def test_anonymous_native_capabilities_require_explicit_semantic_ids() -> None:
    select_model = SelectModel(lambda ctx: ctx.model)
    prepare_tools = PrepareTools(lambda _ctx, tool_defs: tool_defs)
    group = CapabilityGroup[None]("group")

    group.capability(select_model, semantic_id="select-model")
    group.capability(prepare_tools, semantic_id="prepare-tools")
    candidates = await group.freeze()

    assert select_model.id is None
    assert prepare_tools.id is None
    assert {candidate.id for candidate in candidates} == {
        "prepare-tools",
        "select-model",
    }
