#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SkillSpec  + SkillRegistry: loads skill declarations from
{name}.md (markdown + YAML frontmatter) via SpecLoader, revision-cached. The
frontmatter holds name/description/metadata; the markdown body becomes the
SkillSpec.instructions (mirrors the AgentRegistry markdown pattern)."""

from dataclasses import dataclass, field
import asyncio
from typing import Any, Mapping

from ..errors import RegistryNotFoundError
from .parser import SpecLoader, StrictConfigReader, parse_markdown_text, resolved_name


@dataclass(frozen=True, slots=True)
class SkillSpec:
    id: str
    name: str
    description: str = ""
    instructions: str = ""
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


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


class SkillRegistry:
    """Loads SkillSpecs from `{name}.md` files via a SpecLoader, revision-cached.

    Mirrors AgentRegistry: the loader exposes a revision() monotonic clock;
    whenever it changes the per-(id, revision) cache and the id listing are
    dropped so the next get() re-reads and re-parses the markdown.
    """

    def __init__(self, loader: SpecLoader, *, suffix: str = ".md") -> None:
        self._loader = loader
        self._suffix = suffix
        self._cache: "dict[tuple[str, int], SkillSpec]" = {}
        self._cached_revision: "int | None" = None
        self._ids: "tuple[str, ...] | None" = None
        self._refresh_lock = asyncio.Lock()

    async def _ensure_fresh(self) -> None:
        async with self._refresh_lock:
            revision = await self._loader.revision()
            if revision != self._cached_revision:
                self._cache.clear()
                self._ids = None
                self._cached_revision = revision

    async def list_ids(self) -> "tuple[str, ...]":
        await self._ensure_fresh()
        if self._ids is None:
            self._ids = await self._loader.list_ids(self._suffix)
        return self._ids

    async def get(self, skill_id: str) -> SkillSpec:
        await self._ensure_fresh()
        revision = self._cached_revision if self._cached_revision is not None else 0
        cache_key = (skill_id, revision)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            text = await self._loader.read(f"{skill_id}{self._suffix}")
        except RegistryNotFoundError:
            raise
        payload, body = parse_markdown_text(text, source=f"{skill_id}{self._suffix}")
        spec = parse_skill_spec(skill_id, payload, body)
        self._cache[cache_key] = spec
        return spec
