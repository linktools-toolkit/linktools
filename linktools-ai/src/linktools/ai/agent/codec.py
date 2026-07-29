#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent spec codecs: document parsing and JSON round-trip.

Two codecs share this module:

* :class:`AgentSpecDocumentCodec` is the ``SpecCodec[AgentSpec]`` that decodes a
  ``{name}.md`` item (markdown + YAML frontmatter) into an AgentSpec for the
  spec index. It strictly validates the frontmatter and propagates the domain's
  rich errors (SpecParseError / InvalidSpecError).
* :class:`AgentSpecCodec` is the JSON round-trip codec: ``encode``
  produces the canonical JsonValue persisted in ``RunDefinition.spec`` and
  ``decode`` rebuilds a semantically equal AgentSpec, including the structured
  output type via :class:`OutputTypeRegistry`.

The shared markdown and strict-config primitives live in ``spec.parsing``;
this module imports them one-way.
"""

from __future__ import annotations

from typing import Any

from collections.abc import Mapping

from ..spec import SpecCodec
from ..spec.parsing import (
    StrictConfigReader,
    parse_markdown_text,
    resolved_name,
)
from decimal import Decimal

from ..json import JsonValue
from ..model.codec import parse_model_policy
from ..model.policy import ModelPolicy
from .assembly.models import AgentFeatureRef
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


def _parse_feature_refs(items: Any) -> tuple[AgentFeatureRef, ...]:
    if items is None:
        return ()
    if not isinstance(items, (list, tuple)):
        raise InvalidSpecError("features must be a list")
    refs: list[AgentFeatureRef] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise InvalidSpecError(f"features[{index}]: invalid feature ref")
        reader = StrictConfigReader(
            item,
            allowed={"kind", "name", "config"},
            context=f"features[{index}]",
        )
        refs.append(
            AgentFeatureRef(
                kind=reader.required_str("kind").strip(),
                name=reader.required_str("name").strip(),
                config=reader.mapping("config") or {},
            )
        )
    return tuple(refs)


def parse_agent_spec(agent_id: str, payload: "dict[str, Any]", body: str) -> AgentSpec:
    """Build an AgentSpec from a parsed frontmatter dict + markdown body."""
    allowed = {"name", "model", "features", "sections", "middleware", "metadata"}
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
    if "features" in payload and payload["features"] is None:
        raise InvalidSpecError(f"agent {agent_id}: 'features' must not be null")
    if "middleware" in payload and payload["middleware"] is None:
        raise InvalidSpecError(f"agent {agent_id}: 'middleware' must not be null")
    return AgentSpec(
        id=agent_id,
        name=name,
        model=model,
        instructions=instructions,
        features=_parse_feature_refs(payload.get("features")),
        middleware=parse_middleware_refs(payload.get("middleware")),
        output_schema=None,
        metadata=reader.mapping("metadata") or {},
    )


class AgentSpecDocumentCodec:
    """SpecCodec[AgentSpec]: decode one ``{id}.md`` item's raw text into an
    AgentSpec. Strict (rejects unknown frontmatter fields). Propagates the
    domain's existing rich errors (SpecParseError / InvalidSpecError)
    carrying the item id + field path."""

    def decode(self, item_id: str, raw: str) -> AgentSpec:
        source = f"{item_id}.md"
        payload, body = parse_markdown_text(raw, source=source)
        return parse_agent_spec(item_id, payload, body)


class OutputTypeRegistry:
    """Stable ref <-> structured output type mapping so a RunDefinition can
    reference an output type by string (persisted in the spec JSON) and resume
    can resolve the exact class without re-fetching it from the caller or
    synthesizing an empty model."""

    def __init__(self) -> None:
        self._by_ref: "dict[str, type]" = {}
        self._by_type: "dict[type, str]" = {}

    def register(self, ref: str, output_type: type) -> None:
        existing_ref = self._by_type.get(output_type)
        existing_type = self._by_ref.get(ref)
        if existing_ref == ref and existing_type is output_type:
            return
        if existing_ref is not None and existing_ref != ref:
            raise InvalidSpecError(f"output type already registered as {existing_ref!r}")
        if existing_type is not None and existing_type is not output_type:
            raise InvalidSpecError(f"output ref {ref!r} already registered to a different type")
        self._by_ref[ref] = output_type
        self._by_type[output_type] = ref

    def resolve(self, ref: str) -> type:
        try:
            return self._by_ref[ref]
        except KeyError as exc:
            raise InvalidSpecError(f"unknown output type ref {ref!r}") from exc

    def ref_for(self, output_type: type) -> "str | None":
        return self._by_type.get(output_type)


class AgentSpecCodec:
    """JSON round-trip codec for AgentSpec. ``encode`` produces the
    canonical JsonValue stored in ``RunDefinition.spec``; ``decode`` rebuilds a
    semantically equal AgentSpec from it. Every field round-trips -- model,
    instructions + sections, tools, middleware, metadata, and the structured
    output type (by registry ref). Output types are never stored as Python
    reprs and never restored via a dynamic model factory."""

    def __init__(self, output_types: "OutputTypeRegistry | None" = None) -> None:
        self._output_types = output_types

    def encode(self, spec: AgentSpec) -> JsonValue:
        output_ref = None
        if spec.output_schema is not None:
            if self._output_types is None:
                raise InvalidSpecError("cannot encode a structured output without an OutputTypeRegistry")
            output_ref = self._output_types.ref_for(spec.output_schema)
            if output_ref is None:
                raise InvalidSpecError("structured output type is not registered")
        return {
            "schema": "agent-spec.v1",
            "id": spec.id,
            "name": spec.name,
            "model": {
                "primary": spec.model.primary,
                "fallbacks": list(spec.model.fallbacks),
                "request_retries": spec.model.request_retries,
                "timeout_seconds": spec.model.timeout_seconds,
                "max_tokens": spec.model.max_tokens,
                "budget": (
                    format(spec.model.budget, "f")
                    if spec.model.budget is not None
                    else None
                ),
            },
            "instructions": {
                "instructions": spec.instructions.instructions,
                "sections": dict(spec.instructions.sections),
            },
            "features": [
                {"kind": item.kind, "name": item.name, "config": dict(item.config)}
                for item in spec.features
            ],
            "middleware": [{"name": item.name, "config": dict(item.config)} for item in spec.middleware],
            "output_ref": output_ref,
            "metadata": dict(spec.metadata),
        }

    def decode(self, value: JsonValue) -> AgentSpec:
        data = dict(value)
        raw_features = data.get("features") or ()
        instructions = data["instructions"]
        output_schema = None
        output_ref = data.get("output_ref")
        if output_ref is not None:
            if self._output_types is None:
                raise InvalidSpecError("cannot decode a structured output ref without an OutputTypeRegistry")
            output_schema = self._output_types.resolve(output_ref)
        return AgentSpec(
            id=data["id"],
            name=data["name"],
            model=ModelPolicy(
                primary=data["model"]["primary"],
                fallbacks=tuple(data["model"].get("fallbacks") or ()),
                request_retries=data["model"].get("request_retries"),
                timeout_seconds=data["model"].get("timeout_seconds"),
                max_tokens=data["model"].get("max_tokens"),
                budget=(
                    Decimal(data["model"]["budget"])
                    if data["model"].get("budget") is not None
                    else None
                ),
            ),
            instructions=PromptSpec(
                instructions=instructions["instructions"],
                sections=instructions.get("sections") or {},
            ),
            features=tuple(
                AgentFeatureRef(
                    kind=item["kind"],
                    name=item["name"],
                    config=item.get("config") or {},
                )
                for item in raw_features
            ),
            middleware=tuple(
                MiddlewareRef(name=item["name"], config=item.get("config") or {})
                for item in data.get("middleware") or ()
            ),
            output_schema=output_schema,
            metadata=data.get("metadata") or {},
        )


def parse_agent_spec_markdown(content: str, *, agent_id: str) -> AgentSpec:
    """Decode one AgentSpec from Markdown without touching the filesystem."""
    return AgentSpecDocumentCodec().decode(agent_id, content)


__all__: "list[str]" = [
    "AgentSpecCodec",
    "AgentSpecDocumentCodec",
    "OutputTypeRegistry",
    "parse_agent_spec",
    "parse_agent_spec_markdown",
    "parse_middleware_refs",
]
