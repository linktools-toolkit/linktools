#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Agent runtime assembly for resolved capabilities and per-run execution context."""

from __future__ import annotations

from dataclasses import dataclass

from .registry import AgentSpec, MCPServerSpec, SkillSpec, SubagentSpec
from .session import Session


@dataclass(frozen=True, slots=True)
class CapabilityBundle:
    builtin_tools: list[str]
    skills: list[SkillSpec]
    subagents: list[SubagentSpec]
    mcp_servers: list[MCPServerSpec]
    missing_mcp_sources: list[str]


@dataclass(slots=True)
class AgentExecutionContext:
    session: Session
    capabilities: CapabilityBundle
    kernel: "AgentKernel | None" = None


class AgentKernel:
    """Resolve an agent spec into the concrete capabilities used at execution time."""

    def __init__(self, environ) -> None:
        self.environ = environ

    def build_context(
        self,
        spec: AgentSpec,
        session: Session,
        *,
        builtin_tool_names: frozenset[str],
    ) -> AgentExecutionContext:
        return AgentExecutionContext(
            session=session,
            capabilities=self.resolve_capabilities(spec, builtin_tool_names=builtin_tool_names),
            kernel=self,
        )

    def resolve_capabilities(
        self,
        spec: AgentSpec,
        *,
        builtin_tool_names: frozenset[str],
    ) -> CapabilityBundle:
        builtin_tools = self._resolve_builtin_tools(spec, builtin_tool_names)
        skills = self._resolve_skills(spec)
        subagents = self._resolve_subagents(spec)
        mcp_servers, missing_mcp_sources = self._resolve_mcp_servers(spec, builtin_tool_names)
        return CapabilityBundle(
            builtin_tools=builtin_tools,
            skills=skills,
            subagents=subagents,
            mcp_servers=mcp_servers,
            missing_mcp_sources=missing_mcp_sources,
        )

    def _resolve_builtin_tools(
        self,
        spec: AgentSpec,
        builtin_tool_names: frozenset[str],
    ) -> list[str]:
        if spec.allowed_tools is None:
            return list(builtin_tool_names)
        return [tool for tool in spec.allowed_tools if tool in builtin_tool_names]

    def _resolve_skills(self, spec: AgentSpec) -> list[SkillSpec]:
        registry = self.environ.get_skill_registry()
        if spec.allowed_skills is not None:
            return [registry.get(skill_id) for skill_id in spec.allowed_skills if skill_id in registry]
        return list(registry.all())

    def _resolve_subagents(self, spec: AgentSpec) -> list[SubagentSpec]:
        if not spec.allowed_subagents:
            return []
        registry = self.environ.get_subagent_registry()
        return [registry.get(subagent_id) for subagent_id in spec.allowed_subagents if subagent_id in registry]

    def _resolve_mcp_servers(
        self,
        spec: AgentSpec,
        builtin_tool_names: frozenset[str],
    ) -> tuple[list[MCPServerSpec], list[str]]:
        if spec.allowed_tools is None:
            return [], []
        registry = self.environ.get_mcp_registry()
        specs: list[MCPServerSpec] = []
        missing: list[str] = []
        seen: set[str] = set()
        for source in spec.allowed_tools:
            if source in builtin_tool_names:
                continue
            resolved = registry.get(source) if source in registry else registry.resolve_by_capability(source)
            if resolved is None:
                missing.append(source)
                continue
            if resolved.name in seen:
                continue
            seen.add(resolved.name)
            specs.append(resolved)
        return specs, missing
