#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool identity is explicit id plus revision."""

import pytest
from pydantic_ai import Tool

from linktools.ai.capability import CapabilityContribution, tool_metadata
from linktools.ai.errors import AIError, ErrorCode


async def _probe() -> str:
    return "ok"


@pytest.mark.parametrize(
    "metadata",
    (
        tool_metadata(effect_policy="none"),
        tool_metadata(tool_class="business"),
    ),
)
def test_tool_contribution_rejects_incomplete_runtime_contract(
    metadata: dict[str, object],
) -> None:
    tool = Tool(_probe, takes_ctx=False, name="probe", metadata=metadata)
    with pytest.raises(AIError) as raised:
        CapabilityContribution.from_opaque("tool", "probe", tool)
    assert raised.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


def test_tool_revision_defines_named_identity() -> None:
    def sample(value: str) -> str:
        return value

    baseline = Tool(
        sample,
        name="sample",
        metadata=tool_metadata(
            effect_policy="none", plan_safe=True, tool_class="business"
        ),
    )
    changed = Tool(
        sample,
        name="sample",
        timeout=2.0,
        max_retries=3,
        metadata=tool_metadata(
            effect_policy="replay_safe", plan_safe=True, tool_class="business"
        ),
    )

    first = CapabilityContribution.from_opaque(
        "tool", "sample", baseline, revision=3
    )
    same_revision = CapabilityContribution.from_opaque(
        "tool", "sample", changed, revision=3
    )
    next_revision = CapabilityContribution.from_opaque(
        "tool", "sample", changed, revision=4
    )

    assert first.revision == same_revision.revision == 3
    assert first.contract != same_revision.contract
    assert next_revision.revision == 4


def test_upstream_metadata_is_contract_data_not_identity() -> None:
    def sample(value: str) -> str:
        return value

    metadata = tool_metadata(
        effect_policy="none", plan_safe=True, tool_class="business"
    )
    first = CapabilityContribution.from_opaque(
        "tool",
        "sample",
        Tool(sample, name="sample", metadata={**metadata, "upstream.trace": "a"}),
        revision=2,
    )
    second = CapabilityContribution.from_opaque(
        "tool",
        "sample",
        Tool(sample, name="sample", metadata={**metadata, "upstream.trace": "b"}),
        revision=2,
    )

    assert first.revision == second.revision == 2
    assert first.contract != second.contract
