#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool semantic identity uses explicit revision rather than field projection."""

import pytest
from pydantic_ai import Tool

from linktools.ai.capability import CapabilityContribution, tool_semantic_metadata
from linktools.ai.errors import AIError, ErrorCode


async def _probe() -> str:
    return "ok"


@pytest.mark.parametrize(
    "metadata",
    (
        tool_semantic_metadata(effect="none"),
        tool_semantic_metadata(tool_class="business"),
    ),
)
def test_tool_contribution_rejects_incomplete_runtime_semantics(
    metadata: dict[str, object],
) -> None:
    tool = Tool(_probe, takes_ctx=False, name="probe", metadata=metadata)
    with pytest.raises(AIError) as raised:
        CapabilityContribution.from_opaque("tool", "probe", tool)
    assert raised.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


def test_tool_fingerprint_uses_explicit_revision() -> None:
    def sample(value: str) -> str:
        return value

    baseline = Tool(
        sample,
        name="sample",
        metadata=tool_semantic_metadata(
            effect="none",
            plan_safe=True,
            tool_class="business",
        ),
    )
    changed = Tool(
        sample,
        name="sample",
        timeout=2.0,
        max_retries=3,
        metadata=tool_semantic_metadata(
            effect="replay_safe",
            plan_safe=True,
            tool_class="business",
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

    assert first.semantic_contract != same_revision.semantic_contract
    assert first.fingerprint == same_revision.fingerprint
    assert first.fingerprint != next_revision.fingerprint
    assert first.semantic_contract["revision"] == 3
    assert next_revision.semantic_contract["revision"] == 4


def test_tool_fingerprint_ignores_upstream_metadata_at_same_revision() -> None:
    def sample(value: str) -> str:
        return value

    semantic = tool_semantic_metadata(
        effect="none",
        plan_safe=True,
        tool_class="business",
    )
    first = Tool(sample, name="sample", metadata={**semantic, "upstream.trace": "a"})
    second = Tool(sample, name="sample", metadata={**semantic, "upstream.trace": "b"})

    assert (
        CapabilityContribution.from_opaque(
            "tool", "sample", first, revision=2
        ).fingerprint
        == CapabilityContribution.from_opaque(
            "tool", "sample", second, revision=2
        ).fingerprint
    )
