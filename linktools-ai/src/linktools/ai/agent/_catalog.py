#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable compiled Agent semantics plus exact binding lookup."""

from collections.abc import Mapping
from types import MappingProxyType

from ..core import validate_agent_id
from ..errors import AIError, ErrorCode
from ..spec import AgentSpecCodec
from ._binding import AgentBinding
from ._compiled import CompiledAgent


class AgentCatalog:
    def __init__(self, roots: Mapping[str, CompiledAgent]) -> None:
        for agent_id, compiled_agent in roots.items():
            validate_agent_id(agent_id)
            if compiled_agent.spec.id != agent_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._roots = MappingProxyType(dict(roots))
        self._bindings: dict[str, AgentBinding] = {}

    @property
    def root_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._roots))

    def root_agent(self, agent_id: str) -> CompiledAgent:
        validate_agent_id(agent_id)
        try:
            return self._roots[agent_id]
        except KeyError as error:
            raise AIError(
                ErrorCode.AGENT_BINDING_UNAVAILABLE,
                safe_details={"agent_id": agent_id},
            ) from error

    def register_binding(self, binding: AgentBinding) -> AgentBinding:
        existing = self._bindings.get(binding.binding_digest)
        if existing is not None and _same_runtime_compiled_agent(
            existing.compiled_agent,
            binding.compiled_agent,
        ):
            return existing
        self._bindings[binding.binding_digest] = binding
        return binding

    def binding(self, binding_digest: str) -> AgentBinding:
        try:
            return self._bindings[binding_digest]
        except KeyError as error:
            raise AIError(
                ErrorCode.AGENT_BINDING_UNAVAILABLE,
                safe_details={"binding_digest": binding_digest},
            ) from error


def _same_compiled_agent(left: CompiledAgent, right: CompiledAgent) -> bool:
    return (
        AgentSpecCodec().to_payload(left.spec)
        == AgentSpecCodec().to_payload(right.spec)
        and dict(left.model.contract) == dict(right.model.contract)
        and tuple((item.kind, item.id, item.revision, item.contract) for item in left.selected_tools) == tuple((item.kind, item.id, item.revision, item.contract) for item in right.selected_tools)
        and tuple((item.kind, item.id, item.revision, item.contract) for item in left.selected_skills) == tuple((item.kind, item.id, item.revision, item.contract) for item in right.selected_skills)
        and tuple((item.kind, item.id, item.revision, item.contract) for item in left.selected_mcp) == tuple((item.kind, item.id, item.revision, item.contract) for item in right.selected_mcp)
        and tuple((item.kind, item.id, item.revision, item.contract) for item in left.selected_runtime_capabilities) == tuple((item.kind, item.id, item.revision, item.contract) for item in right.selected_runtime_capabilities)
        and left.selected_subagents == right.selected_subagents
        and left.ordinary_tool_policy == right.ordinary_tool_policy
        and left.mcp_selector_policy == right.mcp_selector_policy
    )


def _same_runtime_compiled_agent(
    left: CompiledAgent,
    right: CompiledAgent,
) -> bool:
    if not _same_compiled_agent(left, right) or left.model is not right.model:
        return False
    for left_values, right_values in (
        (left.selected_tools, right.selected_tools),
        (left.selected_skills, right.selected_skills),
        (left.selected_mcp, right.selected_mcp),
        (left.selected_runtime_capabilities, right.selected_runtime_capabilities),
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
