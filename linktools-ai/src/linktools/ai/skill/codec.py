#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SkillSpecCodec: the CatalogCodec[SkillSpec] for the skill domain.

Owns the skill-specific parsing (moved here from registry/skill.py): a
``{name}.md`` item is markdown with YAML frontmatter. The codec splits the raw
text, strictly validates the frontmatter, and builds a SkillSpec. Parse failures
propagate the domain's existing errors (RegistryParseError / InvalidSpecError).
"""

from __future__ import annotations

from typing import Any

from ..catalog import CatalogCodec
from ..catalog.parsing import (
    StrictConfigReader,
    parse_markdown_text,
    resolved_name,
)
from .models import SkillSpec


def parse_skill_spec(skill_id: str, payload: "dict[str, Any]", body: str) -> SkillSpec:
    """Build a SkillSpec from a parsed frontmatter dict + markdown body.

    - name falls back to skill_id when the frontmatter omits it.
    - description defaults to "".
    - instructions is the stripped markdown body.
    - metadata is copied through from the optional `metadata` mapping.
    """
    allowed = {"name", "description", "metadata"}
    reader = StrictConfigReader(payload, allowed=allowed, context=f"skill {skill_id}")
    name = resolved_name(reader, skill_id)
    description = reader.optional_str("description") or ""
    instructions = body.strip()
    metadata = reader.mapping("metadata") or {}
    return SkillSpec(
        id=skill_id,
        name=name,
        description=description,
        instructions=instructions,
        metadata=metadata,
    )


class SkillSpecCodec:
    """CatalogCodec[SkillSpec]: decode one ``{id}.md`` item's raw text into a
    SkillSpec. Strict (rejects unknown frontmatter fields); propagates the
    domain's existing rich errors."""

    def decode(self, item_id: str, raw: str) -> SkillSpec:
        source = f"{item_id}.md"
        payload, body = parse_markdown_text(raw, source=source)
        return parse_skill_spec(item_id, payload, body)


__all__: "list[str]" = ["SkillSpecCodec", "parse_skill_spec"]
