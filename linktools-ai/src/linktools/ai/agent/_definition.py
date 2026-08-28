#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable output-independent Agent semantics."""

from dataclasses import dataclass

from ..capability import CapabilityContribution, SkillDefinition
from ..errors import AIError, ErrorCode
from ..model import ModelBinding
from ..spec import AgentSpec, MCPServerSpec


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True, slots=True)
class AgentDefinition:
    digest: str
    spec: AgentSpec
    model: ModelBinding
    selected_tools: "tuple[CapabilityContribution[object], ...]"
    selected_skills: "tuple[CapabilityContribution[object], ...]"
    selected_mcp: "tuple[CapabilityContribution[object], ...]"
    selected_capabilities: "tuple[CapabilityContribution[object], ...]"
    selected_subagents: "tuple[str, ...]"
    ordinary_tool_policy: "tuple[str, ...]"
    mcp_selector_policy: "tuple[str, ...]"

    def __post_init__(self) -> None:
        if not _is_digest(self.digest) or not isinstance(self.spec, AgentSpec):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        groups = (
            ("tool", self.selected_tools),
            ("skill", self.selected_skills),
            ("mcp", self.selected_mcp),
            ("capability", self.selected_capabilities),
        )
        identities: set[tuple[str, str]] = set()
        for expected_kind, values in groups:
            previous: str | None = None
            for value in values:
                if value.kind != expected_kind or (previous is not None and value.id < previous):
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
        return tuple(
            value.value
            for value in self.selected_skills
            if isinstance(value.value, SkillDefinition)
        )

    @property
    def mcp_servers(self) -> "tuple[MCPServerSpec, ...]":
        return tuple(value.value for value in self.selected_mcp if isinstance(value.value, MCPServerSpec))

    @property
    def static_tool_names(self) -> "tuple[str, ...]":
        return tuple(value.id for value in self.selected_tools)


__all__ = ["AgentDefinition"]
