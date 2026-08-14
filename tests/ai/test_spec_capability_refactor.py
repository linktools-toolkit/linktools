#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability selection and immutable skill catalog contracts."""

import pytest
from linktools.ai.capability._skill import (
    SkillCatalogSnapshot,
    SkillDescriptor,
    bind_skill_capability,
)
from linktools.ai.agent import select_platform_tool_names
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.spec import AgentCapabilityRef, SkillSpec


def test_platform_tool_selection_is_allowlist_scoped() -> None:
    assert select_platform_tool_names(allow_tools=("write_plan",), memory_scope="memory") == ("write_plan",)
    assert select_platform_tool_names(allow_tools=("*",), memory_scope=None) == ("write_plan",)


def test_skill_catalog_snapshot_is_sorted_and_immutable() -> None:
    first = SkillSpec("z", 1, "z skill")
    second = SkillSpec("a", 1, "a skill")
    catalog = SkillCatalogSnapshot(
        (SkillDescriptor("z", 1, "z"), SkillDescriptor("a", 1, "a")),
        (first, second),
    )
    assert tuple(item.id for item in catalog.descriptors) == ("a", "z")
    assert tuple(item.id for item in catalog.specifications) == ("a", "z")


def test_required_missing_skill_is_rejected() -> None:
    with pytest.raises(AIError) as error:
        bind_skill_capability((AgentCapabilityRef("skill", "missing", 1, True),), (None,))
    assert error.value.code is ErrorCode.CAPABILITY_REQUIRED_MISSING
