#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill prompt catalog. Only id/name/description/tags/extension_id
are injected -- never the full SKILL.md body, which the model must fetch via
read_skill only when needed."""

from typing import Iterable

from .models import SkillSummary

_HEADER = (
    "## Available Skills\n\n"
    "Inspect available skills with list_skills(). "
    "Use read_skill(skill_id) to load full instructions only when needed; "
    "do not assume a skill's full content before reading it. "
    "Some skills may be extensions -- if extension-asset tools are enabled, "
    "inspect their references/assets with list_extension_content() and "
    "read_extension_content().\n\nAvailable skill summaries:"
)


def render_skill_catalog(summaries: "Iterable[SkillSummary]") -> str:
    lines = [_HEADER]
    for s in summaries:
        tags = f" [{', '.join(s.tags)}]" if s.tags else ""
        ext = f" (extension: {s.extension_id})" if s.extension_id else ""
        desc = f": {s.description}" if s.description else ""
        lines.append(f"- {s.id}{tags}{ext}{desc}")
    return "\n".join(lines)
