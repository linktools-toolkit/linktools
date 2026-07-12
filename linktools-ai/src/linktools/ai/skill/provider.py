#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SkillProvider: the CapabilityProvider for ``skill:*`` / ``skill:<id>``.

- skill:*   -> inject the catalog prompt (lightweight summaries only) + expose
               list_skills/read_skill authorized for every skill.
- skill:<id>-> expose list_skills/read_skill authorized for that one skill only;
               no full content is injected into the prompt.

Package skills surface their package_id in summaries; deeper package-resource
access is a separate ``package-resource`` capability, not auto-enabled here."""

from dataclasses import dataclass
from typing import ClassVar

from ..capability.models import CapabilityBundle
from ..capability.provider import CapabilityContext, make_event_emitter
from ..capability.models import CapabilityRef
from ..providers.skill import SkillSpecProvider
from ..tool.models import ToolDescriptor
from ..tool.models import ToolContribution, declared_tool_definitions
from .prompt import render_skill_catalog
from .toolset import _summary_from_spec, build_skill_toolset


@dataclass
class SkillProvider:
    """CapabilityProvider for skills. ``skill_provider`` is any SkillSpecProvider
    (default SkillRegistry or a business backend)."""

    skill_provider: SkillSpecProvider
    kind: str = "skill"
    supported_kinds: "ClassVar[tuple[str, ...]]" = ("skill",)

    async def resolve(
        self,
        ref: CapabilityRef,
        context: CapabilityContext,
    ) -> CapabilityBundle:
        emit = make_event_emitter(context)
        if ref.name == "*":
            return await self._resolve_wildcard(context, emit)
        return self._resolve_single(ref.name, emit)

    async def _resolve_wildcard(self, context, emit=None) -> CapabilityBundle:
        ids = await self.skill_provider.list_ids()
        # When discovery tools are disabled, only inject the prompt catalog (if
        # enabled); list_skills/read_skill are NOT exposed.
        if not context.exposure_policy.expose_discovery_tools:
            summaries = []
            for sid in ids:
                try:
                    spec = await self.skill_provider.get(sid)
                except (KeyError, LookupError):
                    continue
                summaries.append(_summary_from_spec(sid, spec))
            sections: "dict[str, str]" = {}
            if context.exposure_policy.expose_prompt_catalog and summaries:
                sections["skills"] = render_skill_catalog(summaries)
            return CapabilityBundle(prompt_sections=sections)
        # Discovery tools enabled: expose list_skills/read_skill.
        summaries = []
        for sid in ids:
            try:
                spec = await self.skill_provider.get(sid)
            except (KeyError, LookupError):
                continue
            summaries.append(_summary_from_spec(sid, spec))
        toolset = build_skill_toolset(
            self.skill_provider, authorized=set(ids), emit=emit
        )
        sections = {}
        if context.exposure_policy.expose_prompt_catalog and summaries:
            sections["skills"] = render_skill_catalog(summaries)
        contribution = _skill_contribution(toolset)
        return CapabilityBundle(
            prompt_sections=sections, tool_contributions=(contribution,)
        )

    def _resolve_single(self, skill_id, emit=None) -> CapabilityBundle:
        # Single-skill ref also respects expose_discovery_tools.
        if not emit:
            pass  # emit check is handled by caller's exposure policy
        toolset = build_skill_toolset(
            self.skill_provider, authorized={skill_id}, emit=emit
        )
        contribution = _skill_contribution(toolset)
        return CapabilityBundle(tool_contributions=(contribution,))


def _skill_contribution(toolset) -> ToolContribution:
    """Both skill tools are read-only discovery."""
    kw = dict(
        source="skill",
        capability_kind="skill",
        category="discovery",
        risk="low",
        mutating=False,
    )
    descriptors = (
        ToolDescriptor(name="list_skills", **kw),
        ToolDescriptor(name="read_skill", **kw),
    )
    return ToolContribution(tools=declared_tool_definitions(toolset, descriptors))
