#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent definition lookup for Runtime services."""

from collections.abc import Mapping
from types import MappingProxyType

from ..core import validate_agent_id
from ..errors import AIError, ErrorCode
from ._definition import AgentDefinition


class AgentDefinitionCatalog:
    """Keep immutable named roots plus Runtime-local effective definitions by digest."""

    def __init__(self, roots: Mapping[str, AgentDefinition]) -> None:
        by_digest: dict[str, AgentDefinition] = {}
        for definition in roots.values():
            existing = by_digest.get(definition.digest)
            if existing is not None and not _same_definition(existing, definition):
                raise AIError(ErrorCode.BINDING_CONFLICT)
            by_digest[definition.digest] = definition
        self._roots = MappingProxyType(dict(roots))
        self._definitions = by_digest

    @property
    def root_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._roots))

    def root_definition(self, agent_id: str) -> AgentDefinition:
        validate_agent_id(agent_id)
        try:
            return self._roots[agent_id]
        except KeyError as error:
            raise AIError(
                ErrorCode.AGENT_DEFINITION_UNAVAILABLE,
                safe_details={"agent_id": agent_id},
            ) from error

    def subagent_definition(self, agent_id: str) -> AgentDefinition:
        return self.root_definition(agent_id)

    def register(self, definition: AgentDefinition) -> AgentDefinition:
        """Register one effective definition without changing named Agent lookup."""
        existing = self._definitions.get(definition.digest)
        if existing is not None:
            if not _same_definition(existing, definition):
                raise AIError(ErrorCode.BINDING_CONFLICT)
            return existing
        self._definitions[definition.digest] = definition
        return definition

    def definition(self, digest: str) -> AgentDefinition:
        try:
            return self._definitions[digest]
        except KeyError as error:
            raise AIError(
                ErrorCode.AGENT_DEFINITION_UNAVAILABLE,
                safe_details={"binding_digest": digest},
            ) from error


def _same_definition(left: AgentDefinition, right: AgentDefinition) -> bool:
    if left.digest != right.digest:
        return False
    try:
        left_capabilities = tuple(
            (capability.provider, capability.id, capability.fingerprint)
            for capability in left.effective_capabilities
        )
        right_capabilities = tuple(
            (capability.provider, capability.id, capability.fingerprint)
            for capability in right.effective_capabilities
        )
        return (
            left.spec == right.spec
            and left.model.fingerprint == right.model.fingerprint
            and left.output_binding.descriptor == right.output_binding.descriptor
            and left_capabilities == right_capabilities
            and left.binding_snapshot == right.binding_snapshot
            and left.trusted_tool_classes == right.trusted_tool_classes
            and left.trusted_mcp_selectors == right.trusted_mcp_selectors
        )
    except (AttributeError, TypeError) as error:
        raise AIError(
            ErrorCode.INTERNAL_ERROR,
            safe_details={"phase": "agent_catalog_compare"},
        ) from error


__all__ = ["AgentDefinitionCatalog"]
