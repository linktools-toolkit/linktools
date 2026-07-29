#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SkillProvider: the AgentFeatureProvider for ``skill:*`` / ``skill:<id>``.

- skill:*   -> inject the catalog prompt (lightweight summaries only) + expose
               list_skills/read_skill authorized for every skill.
- skill:<id>-> expose list_skills/read_skill authorized for that one skill only;
               no full content is injected into the prompt.

Extension skills surface their extension_id in summaries; deeper extension-asset
access is a separate ``extension-asset`` feature, not auto-enabled here."""

from dataclasses import dataclass
from typing import Any, ClassVar

from ..assembly.models import AgentContribution, AgentFeatureRef
from ..assembly.provider import AgentFeatureContext
from .spec import SkillSpecProvider
from ...governance.policy.rule import RiskLevel, SideEffectKind
from ..tool.models import (
    ToolCategory,
    ToolDescriptor,
    ToolSource,
    declared_tool_definitions,
)
from .prompt import render_skill_catalog
from .toolset import build_skill_toolset, summary_from_spec


@dataclass
class SkillProvider:
    """AgentFeatureProvider for skills. ``skill_provider`` is any SkillSpecProvider
    (default SkillSpecIndex or a business backend)."""

    skill_provider: SkillSpecProvider
    # When set, read_skill activates the skill in the current task context so a
    # later call_subagent(instruction_path=...) can resolve under it.
    active_skill_lookup: Any = None
    kind: str = "skill"
    supported_kinds: "ClassVar[tuple[str, ...]]" = ("skill",)

    async def resolve(
        self,
        ref: AgentFeatureRef,
        context: AgentFeatureContext,
    ) -> AgentContribution:
        emit = None
        if ref.name == "*":
            return await self._resolve_wildcard(ref, context, emit)
        return self._resolve_single(ref, emit)

    async def _resolve_wildcard(self, ref, context, emit=None) -> AgentContribution:
        ids = await self.skill_provider.list_ids()
        summaries = []
        for sid in ids:
            try:
                spec = await self.skill_provider.get(sid)
            except (KeyError, LookupError):
                continue
            summaries.append(summary_from_spec(sid, spec))
        toolset = build_skill_toolset(
            self.skill_provider,
            authorized=set(ids),
            emit=emit,
            active_skill_lookup=self.active_skill_lookup,
        )
        sections = {}
        if summaries:
            sections["skills"] = render_skill_catalog(summaries)
        tools = _skill_tools(toolset, ref)
        return AgentContribution(
            prompt_sections=sections,
            tools=tools,
        )

    def _resolve_single(self, ref, emit=None) -> AgentContribution:
        # Single-skill ref also respects expose_discovery_tools.
        if not emit:
            pass  # emit check is handled by caller's exposure policy
        toolset = build_skill_toolset(
            self.skill_provider,
            authorized={ref.name},
            emit=emit,
            active_skill_lookup=self.active_skill_lookup,
        )
        return AgentContribution(tools=_skill_tools(toolset, ref))


def _skill_tools(toolset, ref: AgentFeatureRef):
    """Both skill tools are read-only discovery."""
    kw = dict(
        source=ToolSource.SKILL,
        feature=ref,
        category=ToolCategory.DISCOVERY,
        risk=RiskLevel.LOW,
        side_effect=SideEffectKind.READ_ONLY,
    )
    descriptors = (
        ToolDescriptor(name="list_skills", **kw),
        ToolDescriptor(name="read_skill", **kw),
    )
    return declared_tool_definitions(toolset, descriptors)
