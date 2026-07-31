"""Composition helpers for the v4 CLI client."""

from dataclasses import dataclass

from ..agent.index import AgentSpecIndex
from ..agent.spec import AgentSpec, PromptSpec
from ..spec.parsing import SpecLoader
from ..execution.persistence.local import LocalExecutionBackend
from ..execution.store import ExecutionStore
from ..agent.mcp.index import MCPServerSpecIndex
from ..model.policy import ModelPolicy
from ..runtime import Runtime, RuntimeStorage, build_runtime
from ..agent.skill.index import SkillSpecIndex

from .project import CliProject
from .skill_index import DirectorySkillIndex


@dataclass(frozen=True, slots=True)
class CliRuntimeBundle:
    project: CliProject
    runtime: Runtime
    storage: RuntimeStorage
    agents: AgentSpecIndex
    skills: SkillSpecIndex
    mcp: MCPServerSpecIndex
    skill_index: DirectorySkillIndex


_BUILTIN_DEFAULT = AgentSpec(
    id="default",
    name="default",
    model=ModelPolicy(primary="standard", request_retries=1, timeout_seconds=120),
    instructions=PromptSpec(instructions="You are a concise local assistant."),
)


def build_cli_runtime(*, project: CliProject, model_resolver) -> CliRuntimeBundle:
    """Build the CLI bundle with the v4 runtime storage composition."""
    agents = AgentSpecIndex.from_specloader(SpecLoader.from_filesystem(project.agents_root))
    skills = SkillSpecIndex.from_specloader(SpecLoader.from_filesystem(project.skills_root), suffix="")
    mcp = MCPServerSpecIndex.from_specloader(SpecLoader.from_filesystem(project.mcp_root))
    storage = RuntimeStorage(execution=ExecutionStore(LocalExecutionBackend(project.state_root / "execution")))
    runtime = build_runtime(storage=storage, model_resolver=model_resolver)
    return CliRuntimeBundle(
        project=project,
        runtime=runtime,
        storage=storage,
        agents=agents,
        skills=skills,
        mcp=mcp,
        skill_index=DirectorySkillIndex(project.skills_root),
    )


async def load_agent_spec(bundle: CliRuntimeBundle, agent_id: str | None) -> AgentSpec:
    target = agent_id or bundle.project.default_agent
    try:
        return await bundle.agents.get(target)
    except Exception:
        if target == bundle.project.default_agent:
            return _BUILTIN_DEFAULT
        raise


__all__ = ["CliRuntimeBundle", "build_cli_runtime", "load_agent_spec"]
