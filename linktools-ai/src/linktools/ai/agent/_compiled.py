#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable output-independent Agent semantics."""

from dataclasses import dataclass

from ..capability import CapabilityContribution, SkillDefinition
from ..errors import AIError, ErrorCode
from ..model import ModelBinding
from ..spec import AgentSpec, MCPServerSpec


@dataclass(frozen=True, slots=True)
class CompiledAgent:
    spec: AgentSpec
    model: ModelBinding
    selected_tools: "tuple[CapabilityContribution[object], ...]"
    selected_skills: "tuple[CapabilityContribution[object], ...]"
    selected_mcp: "tuple[CapabilityContribution[object], ...]"
    selected_runtime_capabilities: "tuple[CapabilityContribution[object], ...]"
    selected_subagents: "tuple[str, ...]"
    tool_policy: "tuple[str, ...]"
    mcp_policy: "tuple[str, ...]"

    def __post_init__(self) -> None:
        if not isinstance(self.spec, AgentSpec):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        groups = (
            ("tool", self.selected_tools),
            ("skill", self.selected_skills),
            ("mcp", self.selected_mcp),
            ("runtime_capability", self.selected_runtime_capabilities),
        )
        identities: set[tuple[str, str]] = set()
        for expected_kind, values in groups:
            previous: str | None = None
            for value in values:
                if value.kind != expected_kind:
                    raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
                if expected_kind != "runtime_capability" and previous is not None and value.id < previous:
                    raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
                identity = (value.kind, value.id)
                if identity in identities:
                    raise AIError(ErrorCode.CAPABILITY_CONFLICT)
                identities.add(identity)
                previous = value.id
        if tuple(sorted(set(self.selected_subagents))) != self.selected_subagents:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)

    @property
    def skill_definitions(self) -> "tuple[SkillDefinition, ...]":
        return tuple(value.value for value in self.selected_skills)

    @property
    def mcp_servers(self) -> "tuple[MCPServerSpec, ...]":
        return tuple(value.value for value in self.selected_mcp)

    @property
    def static_tool_names(self) -> "tuple[str, ...]":
        return tuple(value.id for value in self.selected_tools)


__all__ = ["CompiledAgent"]
