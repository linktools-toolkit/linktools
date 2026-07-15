#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ProviderBundle: the declaration/configuration bundle handed to
Runtime.build. Holds the optional spec providers for each capability domain.
Distinct from Storage (state) and CapabilityRuntimeOptions (policy)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .agent import AgentSpecProvider
from .mcp import MCPServerSpecProvider
from .package import PackageResourceProvider, PackageSpecProvider
from .skill import SkillSpecProvider
from .subagent import SubagentSpecProvider
from .swarm import SwarmSpecProvider
from .tool_policy import ToolPolicyMetadataSource

if TYPE_CHECKING:
    from ..capability.provider import CapabilityProvider
    from ..package.resolver import EntrypointResolver


@dataclass(frozen=True)
class ProviderPrefixes:
    """Resource-path prefixes for building default registries from a shared
    ResourceStore. An empty/None prefix means that domain is not
    constructed."""

    agents: str = "specs/agents"
    skills: str = "specs/skills"
    mcp: str = "specs/mcp"
    tools: str = "specs/tools"
    packages: "str | None" = (
        None  # packages need a filesystem root (see from_resources)
    )


@dataclass(frozen=True)
class ProviderBundle:
    agents: "AgentSpecProvider | None" = None
    skills: "SkillSpecProvider | None" = None
    mcp_servers: "MCPServerSpecProvider | None" = None
    tool_policies: "ToolPolicyMetadataSource | None" = None
    swarms: "SwarmSpecProvider | None" = None
    subagents: "SubagentSpecProvider | None" = None
    packages: "PackageSpecProvider | None" = None
    package_resources: "PackageResourceProvider | None" = None
    entrypoints: "EntrypointResolver | None" = None
    capabilities: "tuple[CapabilityProvider, ...]" = ()
    # Skill-private subagent support (optional; None preserves legacy behavior).
    # skill_resolver: a UnifiedSubagentResolver for call_subagent(instruction_path).
    # active_skill_provider: returns the ActiveSkillContext for the current task.
    # active_skill_lookup: async(skill_id)->ActiveSkillContext set by read_skill.
    # child_model_policy / parent_delegated_tools: build the child AgentSpec with
    # the permission intersection.
    skill_resolver: Any = None
    active_skill_provider: Any = None
    active_skill_lookup: Any = None
    child_model_policy: Any = None
    parent_delegated_tools: "set[str] | None" = None

    def is_empty(self) -> bool:
        return (
            not any(
                v is not None
                for v in (
                    self.agents,
                    self.skills,
                    self.mcp_servers,
                    self.tool_policies,
                    self.swarms,
                    self.subagents,
                    self.packages,
                    self.package_resources,
                    self.entrypoints,
                )
            )
            and not self.capabilities
        )

    @classmethod
    def from_resources(
        cls,
        resource_store: Any,
        *,
        prefixes: "ProviderPrefixes | None" = None,
        packages_base: "Any | None" = None,
    ) -> "ProviderBundle":
        """Build a ProviderBundle of default registries from a shared
        ResourceStore. Each Spec-backed registry (agents/skills/
        mcp/tools) is constructed via SpecLoader.from_resources under its prefix.
        ``packages_base`` (a filesystem root Path) optionally builds a
        PackageRegistry + DirectoryEntrypointResolver; packages are filesystem
        trees, not single Spec files, so they take a root rather than a prefix.
        """
        from ..registry.agent import AgentRegistry
        from ..registry.mcp import MCPRegistry
        from ..registry.parser import SpecLoader
        from ..registry.skill import SkillRegistry
        from ..registry.tool import ToolRegistry

        prefixes = prefixes or ProviderPrefixes()
        kwargs: "dict[str, Any]" = {}
        if prefixes.agents:
            kwargs["agents"] = AgentRegistry(
                SpecLoader.from_resources(resource_store, prefix=prefixes.agents)
            )
        if prefixes.skills:
            kwargs["skills"] = SkillRegistry(
                SpecLoader.from_resources(resource_store, prefix=prefixes.skills)
            )
        if prefixes.mcp:
            kwargs["mcp_servers"] = MCPRegistry(
                SpecLoader.from_resources(resource_store, prefix=prefixes.mcp)
            )
        if prefixes.tools:
            kwargs["tool_policies"] = ToolRegistry(
                SpecLoader.from_resources(resource_store, prefix=prefixes.tools)
            )
        if packages_base is not None:
            from ..package.resolver import DirectoryEntrypointResolver, PackageRegistry

            kwargs["packages"] = PackageRegistry(packages_base)
            kwargs["package_resources"] = PackageRegistry(packages_base)
            kwargs["entrypoints"] = DirectoryEntrypointResolver(
                {pid: packages_base / pid for pid in _list_dir_names(packages_base)}
            )
        return cls(**kwargs)


def _list_dir_names(base: Any) -> "list[str]":
    from pathlib import Path

    base_path = Path(base)
    if not base_path.is_dir():
        return []
    return [p.name for p in base_path.iterdir() if p.is_dir()]
