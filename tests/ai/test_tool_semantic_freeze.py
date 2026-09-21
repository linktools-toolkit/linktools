#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool semantic completeness at the frozen contribution boundary."""

import pytest
from pydantic_ai import Tool

from linktools.ai.capability import (
    CapabilityContribution,
    tool_semantic_metadata,
)
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


def test_tool_fingerprint_ignores_unrelated_upstream_metadata() -> None:

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
        CapabilityContribution.from_opaque("tool", "sample", first).fingerprint
        == CapabilityContribution.from_opaque("tool", "sample", second).fingerprint
    )


def test_tool_fingerprint_changes_with_linktools_execution_semantics() -> None:

    def sample(value: str) -> str:
        return value

    first = Tool(
        sample,
        name="sample",
        metadata=tool_semantic_metadata(
            effect="none",
            plan_safe=True,
            tool_class="business",
        ),
    )
    second = Tool(
        sample,
        name="sample",
        metadata=tool_semantic_metadata(
            effect="replay_safe",
            plan_safe=True,
            tool_class="business",
        ),
    )

    assert (
        CapabilityContribution.from_opaque("tool", "sample", first).fingerprint
        != CapabilityContribution.from_opaque("tool", "sample", second).fingerprint
    )

@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("max_retries", 2),
        ("sequential", True),
        ("requires_approval", True),
        ("timeout", 1.0),
        ("defer_loading", True),
        ("include_return_schema", False),
    ),
)
def test_tool_fingerprint_tracks_nondefault_execution_configuration(
    option: str,
    value: object,
) -> None:
    def sample(value: str) -> str:
        return value

    metadata = tool_semantic_metadata(
        effect="none",
        plan_safe=True,
        tool_class="business",
    )
    baseline = Tool(sample, name="sample", metadata=metadata)
    configured = Tool(
        sample,
        name="sample",
        metadata=metadata,
        **{option: value},
    )

    assert (
        CapabilityContribution.from_opaque("tool", "sample", baseline).fingerprint
        != CapabilityContribution.from_opaque("tool", "sample", configured).fingerprint
    )


def test_tool_timeout_identity_normalizes_integer_and_float_values() -> None:
    def sample(value: str) -> str:
        return value

    metadata = tool_semantic_metadata(
        effect="none",
        plan_safe=True,
        tool_class="business",
    )
    integer = Tool(sample, name="sample", metadata=metadata, timeout=1)
    floating = Tool(sample, name="sample", metadata=metadata, timeout=1.0)

    assert (
        CapabilityContribution.from_opaque("tool", "sample", integer).fingerprint
        == CapabilityContribution.from_opaque("tool", "sample", floating).fingerprint
    )

