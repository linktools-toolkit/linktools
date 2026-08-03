#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Composition helpers for the v4 CLI client."""

from dataclasses import dataclass
from ..agent.index import AgentSpecIndex
from ..agent.spec import AgentSpec, PromptSpec
from ..spec.parsing import SpecLoader
from ..agent.mcp.index import MCPServerSpecIndex
from ..model.policy import ModelPolicy
from ..model.resolver import ModelResolver
from ..runtime import LocalDirectoryStorage, RuntimeDependencies, build_runtime
from ..agent.skill.index import SkillSpecIndex
from .skill_index import DirectorySkillIndex

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..execution.live_events import RunLiveEventSink
    from ..runtime import Runtime, RuntimeStorage
    from .project import CliProject


@dataclass(frozen=True, slots=True)
class CliRuntimeBundle:
    project: "CliProject"
    runtime: "Runtime"
    storage: "RuntimeStorage"
    agents: AgentSpecIndex
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
) -> CliRuntimeBundle:
    """Build the CLI bundle with the v4 runtime storage composition.

    ``live_events`` (default Noop) lets a caller capture the engine's
    streaming events; ``LocalRuntimeClient`` passes a queue-backed sink so
    ``run_stream`` can stream model text/tools live instead of replaying
    the trace after the run finishes."""
    agents = AgentSpecIndex.from_specloader(
        SpecLoader.from_filesystem(project.agents_root)
    )
    skills = SkillSpecIndex.from_specloader(
        SpecLoader.from_filesystem(project.skills_root), suffix=""
    )
    mcp = MCPServerSpecIndex.from_specloader(
        SpecLoader.from_filesystem(project.mcp_root)
    )
    storage = LocalDirectoryStorage(project.state_root)
    dependencies = RuntimeDependencies(
        model_resolver=model_resolver or ModelResolver(),
        live_events=live_events,
    )
    runtime = build_runtime(storage=storage, dependencies=dependencies)
    return CliRuntimeBundle(
        project=project,
        runtime=runtime,
        storage=storage,
        agents=agents,
        skills=skills,
        mcp=mcp,
        skill_index=DirectorySkillIndex(project.skills_root),
        live_events=live_events,
    )


async def load_agent_spec(
    bundle: CliRuntimeBundle, agent_id: "str | None"
) -> AgentSpec:
    target = agent_id or bundle.project.default_agent
    try:
        return await bundle.agents.get(target)
    except Exception:
        if target == bundle.project.default_agent:
            return _BUILTIN_DEFAULT
        raise


__all__ = ["CliRuntimeBundle", "build_cli_runtime", "load_agent_spec"]
