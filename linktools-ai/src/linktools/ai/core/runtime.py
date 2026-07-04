#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Agent runtime assembly for resolved capabilities and per-run execution context."""

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from linktools.runtime import EventHandlerMixin

from .registry import AgentSpec
from ..mcp.registry import MCPServerSpec
from ..skill.registry import SkillSpec
from ..subagent.registry import SubagentSpec
from ..session.types import Session
from ..session.local import InMemoryRunStatusStore

if TYPE_CHECKING:
    from ..session.protocols import RunStatus, RunStatusStore
    from ..skill.registry import SkillRegistry
    from ..subagent.registry import SubagentRegistry
    from ..mcp.registry import MCPRegistry


@dataclass(frozen=True, slots=True)
class CapabilityBundle:
    builtin_tools: "list[str]"
    skills: "list[SkillSpec]"
    subagents: "list[SubagentSpec]"
    mcp_servers: "list[MCPServerSpec]"
    missing_mcp_sources: "list[str]"


@dataclass(slots=True)
class AgentExecutionContext:
    session: Session
    capabilities: CapabilityBundle
    kernel: "AgentKernel"
    context: "dict[str, Any]" = field(default_factory=dict)


class AgentKernel(EventHandlerMixin):
    """Resolve an agent spec into the concrete capabilities used at execution time."""

    def __init__(
        self,
        skill_registry: "SkillRegistry",
        subagent_registry: "SubagentRegistry",
        mcp_registry: "MCPRegistry",
        run_status_store: "RunStatusStore | None" = None,
    ) -> None:
        self._skill_registry = skill_registry
        self._subagent_registry = subagent_registry
        self._mcp_registry = mcp_registry
        self.run_status_store: "RunStatusStore" = run_status_store or InMemoryRunStatusStore()
        self._background_tasks: "dict[str, asyncio.Task]" = {}

    def build_context(
        self,
        spec: AgentSpec,
        session: Session,
        *,
        builtin_tool_names: "frozenset[str]",
        context: "dict[str, Any] | None" = None,
    ) -> AgentExecutionContext:
        return AgentExecutionContext(
            session=session,
            capabilities=self.resolve_capabilities(spec, builtin_tool_names=builtin_tool_names),
            kernel=self,
            context=context or {},
        )

    def resolve_capabilities(
        self,
        spec: AgentSpec,
        *,
        builtin_tool_names: "frozenset[str]",
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
        builtin_tool_names: "frozenset[str]",
    ) -> "list[str]":
        if spec.allowed_tools is None:
            return list(builtin_tool_names)
        return [tool for tool in spec.allowed_tools if tool in builtin_tool_names]

    def _resolve_skills(self, spec: AgentSpec) -> "list[SkillSpec]":
        registry = self._skill_registry
        if spec.allowed_skills is not None:
            return [registry.get(skill_id) for skill_id in spec.allowed_skills if skill_id in registry]
        return list(registry.all())

    def _resolve_subagents(self, spec: AgentSpec) -> "list[SubagentSpec]":
        if not spec.allowed_subagents:
            return []
        registry = self._subagent_registry
        return [registry.get(subagent_id) for subagent_id in spec.allowed_subagents if subagent_id in registry]

    def _resolve_mcp_servers(
        self,
        spec: AgentSpec,
        builtin_tool_names: "frozenset[str]",
    ) -> "tuple[list[MCPServerSpec], list[str]]":
        if spec.allowed_tools is None:
            return [], []
        registry = self._mcp_registry
        specs: "list[MCPServerSpec]" = []
        missing: "list[str]" = []
        seen: "set[str]" = set()
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

    async def start_background(
        self,
        spec: SubagentSpec,
        session: Session,
        input_data: Any,
        *,
        call_id: "str | None" = None,
    ) -> str:
        """Start a subagent run in the background, returning immediately with a run_id.
        Call `check_background(run_id)` later to poll for its status."""
        # Deferred import: agent.py imports AgentKernel from this module at load
        # time, so importing SubAgent here at module level would be circular.
        from ..agent import SubAgent

        run_id = call_id or str(uuid.uuid4())
        await self.run_status_store.start(run_id)

        async def _run() -> None:
            from ..session.protocols import RunStatus

            try:
                child_context = self.build_context(
                    spec,
                    session,
                    builtin_tool_names=SubAgent._BUILTIN_TOOL_NAMES,
                )
                agent = SubAgent(
                    spec,
                    session,
                    execution_context=child_context,
                )
                result = await agent.generate(input_data, call_id=run_id)
                await self.run_status_store.update(run_id, RunStatus(state="done", result=result))
            except Exception as exc:
                from ..session.protocols import RunStatus as _RunStatus

                await self.run_status_store.update(run_id, _RunStatus(state="failed", error=str(exc)))

        task = asyncio.create_task(_run())
        self._background_tasks[run_id] = task
        return run_id

    async def check_background(self, run_id: str) -> "RunStatus":
        return await self.run_status_store.get(run_id)
