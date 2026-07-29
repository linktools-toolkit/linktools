#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai.agent.skill: SkillProvider (prompt catalog + list_skills/read_skill).
Skills are a read-only discovery capability; execution is left to
the caller."""

from .models import SkillContent, SkillSummary
from .spec import SkillSpec, SkillSpecProvider
from .prompt import render_skill_catalog
from .provider import SkillProvider
from .toolset import build_skill_toolset

__all__ = [
    "SkillSummary",
    "SkillSpec",
    "SkillSpecProvider",
    "SkillContent",
    "render_skill_catalog",
    "build_skill_toolset",
    "SkillProvider",
]
