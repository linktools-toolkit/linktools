#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable Agent definitions plus exact binding lookup."""

from collections.abc import Mapping
from types import MappingProxyType

from ..core import validate_agent_id
from ..errors import AIError, ErrorCode
from ..spec import AgentSpecCodec, agent_spec_identity_payload
from ._binding import AgentBinding
from ._definition import AgentDefinition


class AgentCatalog:
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
            if _same_runtime_definition(existing, definition):
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
        if definition.digest != binding.definition.digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        existing = self._bindings.get(binding.digest)
        if existing is not None and _same_runtime_definition(
            existing.definition,
            binding.definition,
        ):
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
    return (
        left.digest == right.digest
        and agent_spec_identity_payload(AgentSpecCodec().to_payload(left.spec))
        == agent_spec_identity_payload(AgentSpecCodec().to_payload(right.spec))
        and left.model.fingerprint == right.model.fingerprint
        and tuple((item.kind, item.id, item.fingerprint) for item in left.selected_tools) == tuple((item.kind, item.id, item.fingerprint) for item in right.selected_tools)
        and tuple((item.kind, item.id, item.fingerprint) for item in left.selected_skills) == tuple((item.kind, item.id, item.fingerprint) for item in right.selected_skills)
        and tuple((item.kind, item.id, item.fingerprint) for item in left.selected_mcp) == tuple((item.kind, item.id, item.fingerprint) for item in right.selected_mcp)
        and tuple((item.kind, item.id, item.fingerprint) for item in left.selected_capabilities) == tuple((item.kind, item.id, item.fingerprint) for item in right.selected_capabilities)
        and left.selected_subagents == right.selected_subagents
        and left.ordinary_tool_policy == right.ordinary_tool_policy
        and left.mcp_selector_policy == right.mcp_selector_policy
    )


def _same_runtime_definition(
    left: AgentDefinition,
    right: AgentDefinition,
) -> bool:
    if not _same_definition(left, right) or left.model is not right.model:
        return False
    for left_values, right_values in (
        (left.selected_tools, right.selected_tools),
        (left.selected_skills, right.selected_skills),
        (left.selected_mcp, right.selected_mcp),
        (left.selected_capabilities, right.selected_capabilities),
    ):
        if len(left_values) != len(right_values) or any(
            left_value.value is not right_value.value
            for left_value, right_value in zip(
                left_values,
                right_values,
                strict=True,
            )
        ):
            return False
    return True


__all__ = ["AgentCatalog"]
