#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill toolset: list_skills / read_skill with an authorization boundary
(spec §14.4/§14.7). An agent may only read skills it declared; unauthorized
reads raise SkillNotFoundError so existence is not leaked."""

from typing import Any, Awaitable, Callable, Iterable

from pydantic_ai.toolsets import FunctionToolset

from ..errors import SkillNotFoundError
from ..events.payloads import SkillListed, SkillRead
from ..providers.skill import SkillSpecProvider
from .models import SkillContent, SkillSummary

# Optional async emitter (event_store-backed) for skill operation events.
SkillEmitter = "Callable[[Any], Awaitable[None]]"


def _summary_from_spec(skill_id: str, spec, *, authorized: bool = True) -> SkillSummary:
    meta = dict(getattr(spec, "metadata", {}) or {})
    return SkillSummary(
        id=skill_id,
        name=getattr(spec, "name", skill_id),
        description=getattr(spec, "description", None) or None,
        tags=list(meta.get("tags", []) or []),
        package_id=meta.get("package_id"),
        metadata=meta,
    )


def build_skill_toolset(
    skill_provider: SkillSpecProvider,
    *,
    authorized: "Iterable[str]",
    emit: "SkillEmitter | None" = None,
) -> FunctionToolset:
    """Level-1 skill discovery tools scoped to ``authorized`` skill ids."""
    toolset: FunctionToolset = FunctionToolset()
    allowed = set(authorized)

    async def list_skills(query: "str | None" = None) -> "dict[str, Any]":
        """List skills available to this agent (optionally filtered by query)."""
        ids = await skill_provider.list_ids()
        out: "list[dict[str, Any]]" = []
        for sid in ids:
            if sid not in allowed:
                continue
            try:
                spec = await skill_provider.get(sid)
            except (KeyError, LookupError):
                continue
            summary = _summary_from_spec(sid, spec)
            if query:
                haystack = f"{summary.name} {summary.description or ''}".lower()
                if query.lower() not in haystack:
                    continue
            out.append(summary.model_dump())
        if emit is not None:
            await emit(SkillListed(query=query, count=len(out)))
        return {"skills": out}

    async def read_skill(skill_id: str) -> "dict[str, Any]":
        """Read one skill's full content. Only declared skills are readable."""
        allowed_read = skill_id in allowed
        if emit is not None:
            await emit(SkillRead(skill_id=skill_id, allowed=allowed_read))
        if not allowed_read:
            # Do not leak whether the skill exists.
            raise SkillNotFoundError(f"skill not available: {skill_id}")
        spec = await skill_provider.get(skill_id)
        meta = dict(getattr(spec, "metadata", {}) or {})
        content = SkillContent(
            id=skill_id,
            name=getattr(spec, "name", skill_id),
            description=getattr(spec, "description", None) or None,
            content=getattr(spec, "instructions", ""),
            package_id=meta.get("package_id"),
            metadata=meta,
        )
        return content.model_dump()

    toolset.add_function(list_skills)
    toolset.add_function(read_skill)
    return toolset
