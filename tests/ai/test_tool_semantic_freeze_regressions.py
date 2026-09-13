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
