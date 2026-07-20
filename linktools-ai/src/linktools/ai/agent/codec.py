#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentSpecCodec: the CatalogCodec[AgentSpec] for the agent domain.

Owns the agent-specific parsing (moved here from the old registry/agent.py):
a ``{name}.md`` item is markdown with a YAML frontmatter. The codec splits the
raw text, strictly validates the frontmatter, and builds an AgentSpec. Unknown
frontmatter fields are rejected; parse failures surface the domain's existing
rich errors (RegistryParseError for malformed frontmatter, InvalidSpecError
for a bad/missing spec field) -- the same errors the inlined registry used to
raise, so callers and tests are unchanged by the Catalog migration.

The shared markdown / strict-config primitives live in catalog/parsing (moved
out of registry); this module imports them one-way (no cycle).
"""

from __future__ import annotations

from typing import Any

from collections.abc import Mapping

from ..catalog import CatalogCodec
from ..catalog.parsing import (
    StrictConfigReader,
    parse_markdown_text,
    parse_model_policy,
    resolved_name,
)
from ..tool.codec import parse_tool_refs
from ..errors import InvalidSpecError
from .spec import AgentSpec, MiddlewareRef, PromptSpec


def parse_middleware_refs(items: Any) -> "tuple[MiddlewareRef, ...]":
    """Build a tuple[MiddlewareRef] from a list of names or {name, config}
    mappings. Unknown fields are rejected and names are stripped."""
    if items is None:
        return ()
    if not isinstance(items, (list, tuple)):
        raise InvalidSpecError("middleware must be a list")
    refs: "list[MiddlewareRef]" = []
    for index, item in enumerate(items):
        if isinstance(item, str):
            name = item.strip()
            if not name:
                raise InvalidSpecError(f"middleware[{index}]: name must not be blank")
            refs.append(MiddlewareRef(name=name))
            continue
        if not isinstance(item, Mapping):
            raise InvalidSpecError(
                f"middleware[{index}]: invalid middleware ref: {item!r}"
            )
        item_reader = StrictConfigReader(
            item,
            allowed={"name", "config"},
            context=f"middleware[{index}]",
        )
        name = item_reader.required_str("name").strip()
        if not name:
            raise InvalidSpecError(f"middleware[{index}]: name must not be blank")
        config = item_reader.mapping("config") or {}
        refs.append(MiddlewareRef(name=name, config=config))
    return tuple(refs)


def parse_agent_spec(agent_id: str, payload: "dict[str, Any]", body: str) -> AgentSpec:
    """Build an AgentSpec from a parsed frontmatter dict + markdown body."""
    allowed = {"name", "model", "tools", "sections", "middleware", "metadata"}
    reader = StrictConfigReader(payload, allowed=allowed, context=f"agent {agent_id}")
    name = resolved_name(reader, agent_id)
    model_payload = payload.get("model")
    if not isinstance(model_payload, dict):
        raise InvalidSpecError(f"agent {agent_id}: 'model' must be a mapping")
    model = parse_model_policy(model_payload)
    sections = reader.string_mapping("sections") or {}
    instructions = PromptSpec(
        instructions=body.strip(),
        sections=sections,
    )
    # tools/middleware are parsed by value, so distinguish a missing key (unset
    # / empty-list) from an explicit null here -- null is rejected.
    if "tools" in payload and payload["tools"] is None:
        raise InvalidSpecError(f"agent {agent_id}: 'tools' must not be null")
    if "middleware" in payload and payload["middleware"] is None:
        raise InvalidSpecError(f"agent {agent_id}: 'middleware' must not be null")
    return AgentSpec(
        id=agent_id,
        name=name,
        model=model,
        instructions=instructions,
        tools=parse_tool_refs(payload.get("tools")),
        middleware=parse_middleware_refs(payload.get("middleware")),
        output_schema=None,
        metadata=reader.mapping("metadata") or {},
    )


class AgentSpecCodec:
    """CatalogCodec[AgentSpec]: decode one ``{id}.md`` item's raw text into an
    AgentSpec. Strict (rejects unknown frontmatter fields). Propagates the
    domain's existing rich errors (RegistryParseError / InvalidSpecError)
    carrying the item id + field path."""

    def decode(self, item_id: str, raw: str) -> AgentSpec:
        source = f"{item_id}.md"
        payload, body = parse_markdown_text(raw, source=source)
        return parse_agent_spec(item_id, payload, body)


__all__: "list[str]" = ["AgentSpecCodec", "parse_agent_spec", "parse_middleware_refs"]
