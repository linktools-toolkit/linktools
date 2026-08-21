#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability selection and immutable skill catalog contracts."""

import pytest

from linktools.ai.agent import select_platform_tool_names
from linktools.ai.asset import AssetRef
from linktools.ai.capability._skill import (
    SkillCatalogSnapshot,
    SkillDescriptor,
    bind_skill_capability,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.spec import SkillSpec


def test_platform_tool_selection_keeps_planning_outside_allow_tools() -> None:
    assert (
        select_platform_tool_names(
            allow_tools=("write_plan",),
            memory_scope="memory",
        )
        == ()
    )
    assert select_platform_tool_names(
        allow_tools=(),
        memory_scope=None,
        planning=True,
    ) == ("write_plan",)
    assert select_platform_tool_names(
        allow_tools=(),
        memory_scope=None,
        subagent_available=True,
    ) == ("delegate_task",)


def test_skill_catalog_snapshot_is_sorted_and_immutable() -> None:
    first = SkillSpec("z", 1, "z skill")
    second = SkillSpec("a", 1, "a skill")
    catalog = SkillCatalogSnapshot(
        (SkillDescriptor("z", 1, "z"), SkillDescriptor("a", 1, "a")),
        (first, second),
    )
    assert tuple(item.id for item in catalog.descriptors) == ("a", "z")
    assert tuple(item.id for item in catalog.specifications) == ("a", "z")


def test_skill_binding_requires_one_spec_per_discovered_asset() -> None:
    with pytest.raises(AIError) as error:
        bind_skill_capability((AssetRef("skill", "missing"),), ())
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
