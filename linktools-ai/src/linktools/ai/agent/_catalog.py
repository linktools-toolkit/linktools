#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable Agent definition lookup for Runtime services."""

from collections.abc import Mapping
from types import MappingProxyType

from ..core import validate_agent_id
from ..errors import AIError, ErrorCode
from ._definition import AgentDefinition


class AgentDefinitionCatalog:
    """Expose only frozen root, subagent, and digest lookups after composition."""

    def __init__(
        self,
        roots: Mapping[str, AgentDefinition],
        subagents: Mapping[str, AgentDefinition],
    ) -> None:
        values = (*roots.values(), *subagents.values())
        by_digest: dict[str, AgentDefinition] = {}
        for definition in values:
            existing = by_digest.get(definition.digest)
            if existing is not None and existing != definition:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            by_digest[definition.digest] = definition
        self._roots = MappingProxyType(dict(roots))
        self._subagents = MappingProxyType(dict(subagents))
        self._definitions = MappingProxyType(by_digest)

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
        validate_agent_id(agent_id)
        try:
            return self._subagents[agent_id]
        except KeyError as error:
            raise AIError(
                ErrorCode.AGENT_DEFINITION_UNAVAILABLE,
                safe_details={"agent_id": agent_id},
            ) from error

    def definition(self, digest: str) -> AgentDefinition:
        try:
            return self._definitions[digest]
        except KeyError as error:
            raise AIError(
                ErrorCode.AGENT_DEFINITION_UNAVAILABLE,
                safe_details={"binding_digest": digest},
            ) from error


__all__ = ["AgentDefinitionCatalog"]
