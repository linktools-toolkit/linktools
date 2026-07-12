#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentRegistry: resolves AgentSpec from {name}.md (markdown + YAML frontmatter)
via SpecLoader, revision-cached. The frontmatter holds name/model/tools/middleware;
the markdown body becomes the PromptSpec.instructions."""

from typing import Any

from ..agent.spec import AgentSpec, MiddlewareRef, PromptSpec
from ..errors import InvalidSpecError, RegistryNotFoundError
from .parser import (
    StrictConfigReader,
    SpecLoader,
    parse_markdown_text,
    parse_model_policy,
    parse_tool_refs,
)


def _parse_middleware_refs(items: Any) -> "tuple[MiddlewareRef, ...]":
    """Build a tuple[MiddlewareRef] from a list of names or {name, config} dicts."""
    if items is None:
        return ()
    if not isinstance(items, (list, tuple)):
        raise InvalidSpecError("middleware must be a list")
    refs: list[MiddlewareRef] = []
    for item in items:
        if isinstance(item, str):
            refs.append(MiddlewareRef(name=item))
        elif isinstance(item, dict) and "name" in item:
            if not isinstance(item["name"], str) or not item["name"].strip():
                raise InvalidSpecError("middleware name must be a non-empty string")
            config = item.get("config")
            if config is None:
                config = {}
            elif not isinstance(config, dict):
                raise InvalidSpecError("middleware config must be a mapping")
            refs.append(
                MiddlewareRef(
                    name=item["name"],
                    config=config,
                )
            )
        else:
            raise InvalidSpecError(f"invalid middleware ref: {item!r}")
    return tuple(refs)


def parse_agent_spec(agent_id: str, payload: "dict[str, Any]", body: str) -> AgentSpec:
    """Build an AgentSpec from a parsed frontmatter dict + markdown body."""
    allowed = {"name", "model", "tools", "sections", "middleware", "metadata"}
    reader = StrictConfigReader(payload, allowed=allowed, context=f"agent {agent_id}")
    name = reader.optional_str("name") or agent_id
    model_payload = payload.get("model")
    if not isinstance(model_payload, dict):
        raise InvalidSpecError(f"agent {agent_id}: 'model' must be a mapping")
    model = parse_model_policy(model_payload)
    sections = reader.string_mapping("sections") or {}
    instructions = PromptSpec(
        instructions=body.strip(),
        sections=sections,
    )
    return AgentSpec(
        id=agent_id,
        name=name,
        model=model,
        instructions=instructions,
        tools=parse_tool_refs(payload.get("tools")),
        middleware=_parse_middleware_refs(payload.get("middleware")),
        output_schema=None,
        metadata=reader.mapping("metadata") or {},
    )


class AgentRegistry:
    """Loads AgentSpecs from `{name}.md` files via a SpecLoader, revision-cached.

    Mirrors ToolRegistry: the loader exposes a revision() monotonic clock; whenever
    it changes the per-(id, revision) cache and the id listing are dropped so the
    next get() re-reads and re-parses the markdown.
    """

    def __init__(self, loader: SpecLoader, *, suffix: str = ".md") -> None:
        self._loader = loader
        self._suffix = suffix
        self._cache: "dict[tuple[str, int], AgentSpec]" = {}
        self._cached_revision: "int | None" = None
        self._ids: "tuple[str, ...] | None" = None

    async def _ensure_fresh(self) -> None:
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

    async def get(self, agent_id: str) -> AgentSpec:
        await self._ensure_fresh()
        revision = self._cached_revision if self._cached_revision is not None else 0
        cache_key = (agent_id, revision)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            text = await self._loader.read(f"{agent_id}{self._suffix}")
        except RegistryNotFoundError:
            raise
        payload, body = parse_markdown_text(text, source=f"{agent_id}{self._suffix}")
        spec = parse_agent_spec(agent_id, payload, body)
        self._cache[cache_key] = spec
        return spec
