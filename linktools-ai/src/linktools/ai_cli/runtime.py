#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Runtime assembly for a discovered project.

Wires ``.linktools/{agents,skills,mcp}`` into ``Runtime.build`` via a real
``ProviderBundle`` (AgentRegistry / SkillRegistry / MCPRegistry built from the
project directories), a project-scoped ``FileStorage`` (state isolation), an
``MCPConnectionManager`` (so ``aclose`` can release connections), and
``CapabilityRuntimeOptions``. Also builds the directory skill index +
skill-private subagent resolver for the CLI's own use (inspect/list/doctor and
the live ``call_subagent(instruction_path=...)`` routing).

Project agents are exposed as subagents through the runtime's existing
``call_subagent`` (the ``name`` branch); the skill-private ``instruction_path``
branch is resolved by :class:`UnifiedSubagentResolver`, composed here and
available to the CLI."""

from dataclasses import dataclass
from pathlib import Path

from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.capability.exposure import CapabilityToolExposurePolicy
from linktools.ai.capability.models import CapabilityRuntimeOptions
from linktools.ai.execution.local import LocalExecutionBackend
from linktools.ai.mcp.client import MCPConnectionManager
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.providers.bundle import ProviderBundle
from linktools.ai.registry.agent import AgentRegistry
from linktools.ai.registry.mcp import MCPRegistry
from linktools.ai.registry.parser import SpecLoader
from linktools.ai.registry.skill import SkillRegistry
from linktools.ai.runtime import Runtime
from linktools.ai.skill.private import ActiveSkillContext, get_active_skill
from linktools.ai.storage.facade import FileStorage
from linktools.ai.subagent.skill_resolver import (
    SkillSubagentProvider,
    UnifiedSubagentResolver,
)

from .project import CliProject
from .skill_index import DirectorySkillIndex


async def _activate_skill(skill_index: DirectorySkillIndex, skill_id: str):
    """Build an ActiveSkillContext from the on-disk skill directory, so a
    read_skill activates exactly the skill (root + revision) the model read."""
    info = await skill_index.get(skill_id)
    if info is None:
        return None
    return ActiveSkillContext(
        skill_id=info.id, skill_root=info.root, revision=info.revision
    )


def skill_spec_loader(skills_root: Path) -> SpecLoader:
    """A SpecLoader that reads self-contained ``skills/<id>/SKILL.md``
    directories instead of flat ``<id>.md`` files.

    ``SkillRegistry`` calls ``read(f"{id}{suffix}")`` and ``list_ids(suffix)``;
    with an empty suffix this becomes ``read(id)`` (mapped to ``<id>/SKILL.md``)
    and ``list_ids`` returning the directory names. The revision covers every
    skill's ``SKILL.md`` + ``agents/*.md`` tree so a change anywhere refreshes
    the registry cache."""

    def _ids() -> "tuple[str, ...]":
        if not skills_root.is_dir():
            return ()
        return tuple(
            sorted(
                p.name
                for p in skills_root.iterdir()
                if p.is_dir() and (p / "SKILL.md").is_file()
            )
        )

    async def read(name: str) -> str:
        # The registry may pass "<id>" or "<id>/" -- normalize to the stem.
        stem = name.rstrip("/")
        path = skills_root / stem / "SKILL.md"
        if not path.is_file():
            from linktools.ai.errors import RegistryNotFoundError

            raise RegistryNotFoundError(f"skill not found: {stem}")
        return path.read_text(encoding="utf-8")

    async def list_ids(suffix: str) -> "tuple[str, ...]":
        return _ids()

    async def revision() -> int:
        from hashlib import sha256

        state: "list[tuple[str, int, int]]" = []
        for skill_id in _ids():
            root = skills_root / skill_id
            for sub in ("SKILL.md",):
                p = root / sub
                if p.is_file():
                    st = p.stat()
                    state.append((f"{skill_id}/{sub}", st.st_mtime_ns, st.st_size))
            agents = root / "agents"
            if agents.is_dir():
                for p in sorted(agents.iterdir()):
                    if p.is_file() and p.suffix == ".md":
                        st = p.stat()
                        state.append(
                            (f"{skill_id}/agents/{p.name}", st.st_mtime_ns, st.st_size)
                        )
        return int.from_bytes(
            sha256(repr(tuple(sorted(state))).encode("utf-8")).digest()[:8], "big"
        )

    return SpecLoader(read=read, list_ids=list_ids, revision=revision)


@dataclass(frozen=True, slots=True)
class CliRuntimeBundle:
    project: CliProject
    runtime: Runtime
    storage: FileStorage
    agents: AgentRegistry
    skills: SkillRegistry
    mcp: MCPRegistry
    skill_index: DirectorySkillIndex
    subagents: UnifiedSubagentResolver


_BUILTIN_DEFAULT = AgentSpec(
    id="default",
    name="default",
    model=ModelPolicy(primary="standard", max_retries=1, timeout_seconds=120),
    instructions=PromptSpec(
        instructions=(
            "You are a general-purpose local assistant running in a terminal. "
            "You can read/write files and run shell commands in the current "
            "working directory via your file and terminal tools. Be direct and concise."
        )
    ),
)


async def load_agent_spec(bundle: CliRuntimeBundle, agent_id: "str | None"):
    """Resolve an AgentSpec by id (default: the project's default agent).

    Falls back to a built-in default agent when no agents are found on disk,
    so ``lt ai run``/``lt ai tui`` work without ``lt ai init``."""
    from linktools.cli import CommandError

    target = agent_id or bundle.project.default_agent
    try:
        return await bundle.agents.get(target)
    except Exception:
        if target == bundle.project.default_agent:
            return _BUILTIN_DEFAULT
        raise CommandError(f"agent not found: {target}")


def build_cli_runtime(*, project: CliProject, model_router) -> CliRuntimeBundle:
    """Assemble a ``Runtime`` from a project's ``.linktools/``."""
    agents = AgentRegistry(SpecLoader.from_filesystem(project.agents_root))
    skills = SkillRegistry(skill_spec_loader(project.skills_root), suffix="")
    mcp = MCPRegistry(SpecLoader.from_filesystem(project.mcp_root))

    skill_index = DirectorySkillIndex(project.skills_root)
    skill_subagents = SkillSubagentProvider(
        skills=skill_index,
        default_timeout_seconds=project.subagent_timeout_seconds,
    )
    subagents = UnifiedSubagentResolver(
        project_agents=agents,
        skill_agents=skill_subagents,
    )

    providers = ProviderBundle(
        agents=agents,
        skills=skills,
        mcp_servers=mcp,
        # AgentRegistry structurally satisfies SubagentSpecProvider (async
        # get/list_ids), so project agents are callable via call_subagent(name).
        subagents=agents,
        # Skill-private subagent wiring: read_skill activates the skill
        # (active_skill_lookup); call_subagent(instruction_path=...) resolves it
        # through the UnifiedSubagentResolver against the active skill.
        skill_resolver=subagents,
        active_skill_provider=get_active_skill,
        active_skill_lookup=lambda sid: _activate_skill(skill_index, sid),
        child_model_policy=ModelPolicy(primary="standard"),
        # parent_delegated_tools is left None here on purpose: the live
        # SubagentProvider.resolve derives it per-resolution from the parent
        # agent's own declared tools (via context.agent_id), so the permission
        # intersection IS enforced at runtime without a static value here.
        parent_delegated_tools=None,
    )

    storage = FileStorage(root=project.state_root)
    runtime = Runtime.build(
        storage=storage,
        model_router=model_router,
        execution=LocalExecutionBackend(runtime_dir=project.root),
        providers=providers,
        options=CapabilityRuntimeOptions(
            tool_exposure=CapabilityToolExposurePolicy(
                expose_prompt_catalog=True,
                expose_discovery_tools=True,
                expose_execution_tools=True,
                max_tools_total=64,
                max_tools_per_capability=16,
            ),
            allow_mcp_wildcard=project.allow_mcp_wildcard,
        ),
        allow_mcp_wildcard=project.allow_mcp_wildcard,
        mcp_connection_manager=MCPConnectionManager(),
    )
    return CliRuntimeBundle(
        project=project,
        runtime=runtime,
        storage=storage,
        agents=agents,
        skills=skills,
        mcp=mcp,
        skill_index=skill_index,
        subagents=subagents,
    )
