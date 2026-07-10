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

from ..capability.bundle import CapabilityBundle
from ..capability.provider import CapabilityContext, make_event_emitter
from ..capability.ref import CapabilityRef
from ..providers.skill import SkillSpecProvider
from .prompt import render_skill_catalog
from .toolset import _summary_from_spec, build_skill_toolset


@dataclass
class SkillProvider:
    """CapabilityProvider for skills. ``skill_provider`` is any SkillSpecProvider
    (default SkillRegistry or a business backend)."""

    skill_provider: SkillSpecProvider
    kind: str = "skill"

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
        summaries = []
        for sid in ids:
            try:
                spec = await self.skill_provider.get(sid)
            except (KeyError, LookupError):
                continue
            summaries.append(_summary_from_spec(sid, spec))
        toolset = build_skill_toolset(self.skill_provider, authorized=set(ids), emit=emit)
        sections: "dict[str, str]" = {}
        if context.exposure_policy.expose_prompt_catalog and summaries:
            sections["skills"] = render_skill_catalog(summaries)
        return CapabilityBundle(prompt_sections=sections, toolsets=(toolset,))

    def _resolve_single(self, skill_id, emit=None) -> CapabilityBundle:
        toolset = build_skill_toolset(self.skill_provider, authorized={skill_id}, emit=emit)
        return CapabilityBundle(toolsets=(toolset,))
