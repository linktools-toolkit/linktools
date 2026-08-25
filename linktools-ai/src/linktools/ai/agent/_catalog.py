#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent definition and exact binding lookup for Runtime services."""

from collections.abc import Mapping
from types import MappingProxyType

from ..core import validate_agent_id
from ..errors import AIError, ErrorCode
from ._binding import AgentBinding
from ._definition import AgentDefinition


class AgentCatalog:
    """Keep immutable named roots plus Runtime-local definitions and bindings."""

    def __init__(self, roots: Mapping[str, AgentDefinition]) -> None:
        by_digest: dict[str, AgentDefinition] = {}
        for agent_id, definition in roots.items():
            validate_agent_id(agent_id)
            if definition.spec.id != agent_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            existing = by_digest.get(definition.digest)
            if existing is not None and not _same_definition(existing, definition):
                raise AIError(ErrorCode.BINDING_CONFLICT)
            by_digest[definition.digest] = definition
        self._roots = MappingProxyType(dict(roots))
        self._definitions = by_digest
        self._bindings: dict[str, AgentBinding] = {}

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

    def register_definition(self, definition: AgentDefinition) -> AgentDefinition:
        existing = self._definitions.get(definition.digest)
        if existing is not None:
            if not _same_definition(existing, definition):
                raise AIError(ErrorCode.BINDING_CONFLICT)
            return existing
        self._definitions[definition.digest] = definition
        return definition

    def definition(self, agent_digest: str) -> AgentDefinition:
        try:
            return self._definitions[agent_digest]
        except KeyError as error:
            raise AIError(
                ErrorCode.AGENT_DEFINITION_UNAVAILABLE,
                safe_details={"agent_digest": agent_digest},
            ) from error

    def register_binding(self, binding: AgentBinding) -> AgentBinding:
        definition = self.register_definition(binding.definition)
        if definition is not binding.definition and not _same_definition(
            definition, binding.definition
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        existing = self._bindings.get(binding.digest)
        if existing is not None:
            if not _same_binding(existing, binding):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return existing
        self._bindings[binding.digest] = binding
        return binding

    def binding(self, binding_digest: str) -> AgentBinding:
        try:
            return self._bindings[binding_digest]
        except KeyError as error:
            raise AIError(
                ErrorCode.AGENT_DEFINITION_UNAVAILABLE,
                safe_details={"binding_digest": binding_digest},
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
            and left_capabilities == right_capabilities
            and left.trusted_tool_classes == right.trusted_tool_classes
            and left.trusted_mcp_selectors == right.trusted_mcp_selectors
        )
    except (AttributeError, TypeError) as error:
        raise AIError(
            ErrorCode.INTERNAL_ERROR,
            safe_details={"phase": "agent_catalog_compare"},
        ) from error


def _same_binding(left: AgentBinding, right: AgentBinding) -> bool:
    return (
        left.digest == right.digest
        and _same_definition(left.definition, right.definition)
        and left.output_binding.descriptor == right.output_binding.descriptor
        and left.snapshot == right.snapshot
    )


__all__ = ["AgentCatalog"]
