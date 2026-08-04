#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Composition helpers for the CLI runtime client."""

import asyncio
from dataclasses import dataclass, replace
from uuid import uuid4

from linktools.core import environ

from ..agent.index import AgentSpecIndex
from ..agent.assembly.models import AgentFeatureRef
from ..agent.sandbox import LocalSandbox
from ..agent.spec import AgentSpec, PromptSpec
from ..spec.parsing import SpecLoader
from ..agent.mcp.index import MCPServerSpecIndex
from ..model.policy import ModelPolicy
from ..model.resolver import ModelResolver
from ..agent.skill.private import (
    ActiveSkillContext,
    get_active_skill,
    reset_active_skill,
    set_active_skill,
)
from ..agent.skill.provider import SkillProvider
from ..agent.subagent import (
    AgentBackedSubagentProvider,
    SubagentProvider,
    SubagentResult,
)
from ..agent.subagent.skill_resolver import (
    SkillSubagentProvider,
    UnifiedSubagentResolver,
)
from ..agent.tool.exposure import ToolExposurePolicy
from ..agent.tool.persistence import LocalToolStateBackend
from ..agent.tool.policy import ResolvedToolPolicy
from ..execution.commands import ParentLeaseGuard
from ..execution.domain import RunStatus
from ..execution.live_events import ExecutionEventHub
from ..governance.identity import trusted_local_principal
from ..runtime import (
    LocalDirectoryStorage,
    RuntimeDependencies,
    RuntimeRequirements,
    build_runtime,
)
from ..agent.skill.index import SkillSpecIndex
from .skill_index import DirectorySkillIndex

logger = environ.get_logger("ai.cli.runtime")

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Mapping

    from ..execution.live_events import RunLiveEventSink
    from ..execution.context import RunContext
    from ..agent.tool.models import ToolDescriptor
    from ..runtime import Runtime, RuntimeStorage
    from ..governance.identity import PrincipalContext
    from ..agent.extension.scope import ExtensionScope
    from .project import CliProject


_DEFAULT_CLI_FEATURES = (
    AgentFeatureRef(kind="builtin", name="*"),
    AgentFeatureRef(kind="skill", name="*"),
    AgentFeatureRef(kind="subagent", name="*"),
)


class _DefaultFeatureAgentProvider:
    """Apply the CLI's full feature surface to agents without declarations."""

    def __init__(self, source: AgentSpecIndex) -> None:
        self._source = source

    async def list_ids(self) -> "tuple[str, ...]":
        return await self._source.list_ids()

    async def get(self, agent_id: str) -> AgentSpec:
        return _with_default_features(await self._source.get(agent_id))


def _with_default_features(spec: AgentSpec) -> AgentSpec:
    if spec.features:
        return spec
    return replace(spec, features=_DEFAULT_CLI_FEATURES)


@dataclass(slots=True)
class _AllowAllCliToolPolicy:
    require_approval: bool = False

    async def resolve(
        self,
        descriptor: "ToolDescriptor",
        context: "RunContext",
    ) -> ResolvedToolPolicy:
        return ResolvedToolPolicy(
            enabled=True,
            require_approval=self.require_approval and descriptor.mutating,
        )


@dataclass(slots=True)
class _LocalSubagentExecutor:
    runtime: "Runtime"
    principal: "PrincipalContext"

    async def execute(
        self,
        *,
        agent_spec: AgentSpec,
        task: str,
        context: "dict[str, Any] | None",
        parent: "Any",
        scope: "ExtensionScope | None",
        timeout_seconds: "float | None",
    ) -> SubagentResult:
        parent_record = await self.runtime.get_execution_record(
            parent.run_id, principal=self.principal
        )
        if parent_record is None or parent_record.lease.owner is None:
            raise RuntimeError("parent execution lease is unavailable")
        child_run_id = uuid4().hex
        skill_token = set_active_skill(None)
        try:
            child = self.runtime.run_child(
                agent_spec,
                task,
                principal=self.principal,
                session_id=parent.session_id,
                execution_id=child_run_id,
                root_execution_id=parent.root_execution_id,
                parent_execution_id=parent.run_id,
                parent_guard=ParentLeaseGuard(
                    run_id=parent.run_id,
                    owner=parent_record.lease.owner,
                    fence=parent_record.lease.fence,
                ),
                metadata=context,
            )
            result = (
                await asyncio.wait_for(child, timeout=timeout_seconds)
                if timeout_seconds is not None
                else await child
            )
        finally:
            reset_active_skill(skill_token)
        status = (
            "succeeded"
            if result.status is RunStatus.COMPLETED
            else "cancelled"
            if result.status is RunStatus.CANCELLED
            else "failed"
        )
        error = None
        if result.error is not None:
            error = {
                "type": result.error.error_type,
                "message": result.error.message,
            }
        usage = result.usage
        scope_value = None
        if scope is not None:
            scope_value = {
                "extension_id": scope.extension_id,
                "extension_kind": scope.extension_kind,
            }
        return SubagentResult(
            agent_id=agent_spec.id,
            scope=scope_value,
            session_id=parent.session_id,
            run_id=result.run_id,
            status=status,
            output=result.output,
            error=error,
            token_usage={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
                "total_cost": str(usage.total_cost)
                if usage.total_cost is not None
                else None,
            },
        )


@dataclass(frozen=True, slots=True)
class CliRuntimeBundle:
    project: "CliProject"
    runtime: "Runtime"
    storage: "RuntimeStorage"
    agents: "AgentSpecIndex | _DefaultFeatureAgentProvider"
    skills: SkillSpecIndex
    mcp: MCPServerSpecIndex
    skill_index: DirectorySkillIndex
    live_events: "RunLiveEventSink | None" = None


_BUILTIN_DEFAULT = AgentSpec(
    id="default",
    name="default",
    model=ModelPolicy(primary="standard", request_retries=1, timeout_seconds=120),
    instructions=PromptSpec(instructions="You are a concise local assistant."),
)


def build_cli_runtime(
    *,
    project: "CliProject",
    model_resolver: "ModelResolver | None",
    live_events: "RunLiveEventSink | None" = None,
    require_tool_approval: bool = False,
) -> CliRuntimeBundle:
    """Build the CLI bundle with the v4 runtime storage composition.

    ``live_events`` (default Noop) lets a caller capture the engine's
    streaming events; ``LocalRuntimeClient`` passes a queue-backed sink so
    ``run_stream`` can stream model text/tools live instead of replaying
    the trace after the run finishes."""
    source_agents = AgentSpecIndex.from_specloader(
        SpecLoader.from_filesystem(project.agents_root)
    )
    skills = SkillSpecIndex.from_specloader(
        SpecLoader.from_filesystem(project.skills_root), suffix=""
    )
    mcp = MCPServerSpecIndex.from_specloader(
        SpecLoader.from_filesystem(project.mcp_root)
    )
    agents = _DefaultFeatureAgentProvider(source_agents)
    skill_index = DirectorySkillIndex(project.skills_root)

    async def active_skill_lookup(skill_id: str) -> "ActiveSkillContext | None":
        info = await skill_index.get(skill_id)
        if info is None:
            return None
        return ActiveSkillContext(
            skill_id=info.id,
            skill_root=info.root,
            revision=info.revision,
        )

    skill_provider = SkillProvider(
        skill_provider=skill_index,
        active_skill_lookup=active_skill_lookup,
    )
    skill_subagents = SkillSubagentProvider(
        skills=skill_index,
        default_timeout_seconds=project.subagent_timeout_seconds,
    )
    subagent_provider = SubagentProvider(
        subagent_provider=AgentBackedSubagentProvider(agents),
        skill_resolver=UnifiedSubagentResolver(
            project_agents=agents,
            skill_agents=skill_subagents,
        ),
        active_skill_provider=get_active_skill,
    )
    storage = LocalDirectoryStorage(
        project.state_root,
        tools=LocalToolStateBackend(),
    )
    sandbox = LocalSandbox(runtime_dir=project.root, base_dirs=[project.root])
    dependencies = RuntimeDependencies(
        model_resolver=model_resolver or ModelResolver(),
        skill_provider=skill_provider,
        subagent_provider=subagent_provider,
        sandbox=sandbox,
        tool_policy=_AllowAllCliToolPolicy(require_approval=require_tool_approval),
        live_events=live_events,
    )
    runtime = build_runtime(
        storage=storage,
        dependencies=dependencies,
        requirements=RuntimeRequirements(
            tools=True,
            tool_exposure=ToolExposurePolicy(expose_execution_tools=True),
        ),
        event_hub=live_events if isinstance(live_events, ExecutionEventHub) else None,
    )
    subagent_provider.executor = _LocalSubagentExecutor(
        runtime=runtime,
        principal=trusted_local_principal(),
    )
    if environ.debug:
        logger.debug(
            "CLI runtime ready: tools=all skills=all subagents=all project=%s",
            project.root,
        )
    return CliRuntimeBundle(
        project=project,
        runtime=runtime,
        storage=storage,
        agents=agents,
        skills=skills,
        mcp=mcp,
        skill_index=skill_index,
        live_events=live_events,
    )


async def load_agent_spec(
    bundle: CliRuntimeBundle, agent_id: "str | None"
) -> AgentSpec:
    target = agent_id or bundle.project.default_agent
    try:
        return _with_default_features(await bundle.agents.get(target))
    except Exception:
        if target == bundle.project.default_agent:
            return _with_default_features(_BUILTIN_DEFAULT)
        raise


__all__ = ["CliRuntimeBundle", "build_cli_runtime", "load_agent_spec"]
