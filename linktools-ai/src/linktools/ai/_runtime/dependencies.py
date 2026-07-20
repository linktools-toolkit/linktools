#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RuntimeDependencies: the typed spec-provider declarations handed to
``Runtime.build``. Holds the optional spec providers for each capability domain
plus the pre-built capability providers. Distinct from Storage (state),
CapabilityRuntimeOptions (policy), and SkillPrivateSubagentConfig (skill-private
wiring, injected separately).

The spec-provider Protocols live in their DOMAINS (AgentSpecProvider in
``agent.spec``, SwarmSpecProvider in ``swarm.spec``, etc.); this module only
aggregates them into the single declaration bundle. Re-exported publicly as
``linktools.ai.runtime.RuntimeDependencies`` (this module is an implementation
detail of the runtime build path, not a public path itself)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Mapping, TypeVar

from ..agent.spec import AgentSpecProvider
from ..extension.spec import ExtensionResourceProvider, ExtensionSpecProvider
from ..governance.policy.rule import ToolPolicyMetadataSource
from ..mcp.spec import MCPServerSpecProvider
from ..skill.models import SkillSpecProvider
from ..subagent.models import SubagentSpecProvider
from ..swarm.spec import SwarmSpecProvider

if TYPE_CHECKING:
    from ..capability.provider import CapabilityProvider
    from ..extension.resolver import EntrypointResolver

T = TypeVar("T")


@dataclass(frozen=True)
class ProviderPrefixes:
    """Asset-path prefixes for building default registries from a shared
    AssetStore. An empty/None prefix means that domain is not
    constructed."""

    agents: str = "specs/agents"
    skills: str = "specs/skills"
    mcp: str = "specs/mcp"
    tools: str = "specs/tools"
    extensions: "str | None" = (
        None  # extensions need a filesystem root (see from_resources)
    )


class MappingProvider(Generic[T]):
    """A simple in-memory spec provider backed by a ``{id: spec}`` mapping.
    Provides the ``list_ids`` + ``get`` surface every spec-provider Protocol
    requires; downstream is NOT required to inherit (it satisfies a Protocol
    structurally). Used for built-in registries and test fakes."""

    def __init__(self, specs: "Mapping[str, T]") -> None:
        self._specs = dict(specs)

    async def list_ids(self) -> "tuple[str, ...]":
        return tuple(self._specs.keys())

    async def get(self, spec_id: str) -> T:
        if spec_id not in self._specs:
            raise KeyError(spec_id)
        return self._specs[spec_id]


@dataclass(frozen=True)
class RuntimeDependencies:
    agents: "AgentSpecProvider | None" = None
    skills: "SkillSpecProvider | None" = None
    mcp_servers: "MCPServerSpecProvider | None" = None
    tool_policies: "ToolPolicyMetadataSource | None" = None
    swarms: "SwarmSpecProvider | None" = None
    subagents: "SubagentSpecProvider | None" = None
    extensions: "ExtensionSpecProvider | None" = None
    extension_resources: "ExtensionResourceProvider | None" = None
    entrypoints: "EntrypointResolver | None" = None
    capabilities: "tuple[CapabilityProvider, ...]" = ()
    # NOTE: skill-private-subagent wiring no longer lives here -- it flows
    # through a typed SkillPrivateSubagentConfig injected via
    # ``Runtime.build(skill_subagent=...)`` straight into the SubagentProvider.
    # RuntimeDependencies is now spec-providers + capabilities only.

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
                    self.extensions,
                    self.extension_resources,
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
        extensions_base: "Any | None" = None,
    ) -> "RuntimeDependencies":
        """Build a RuntimeDependencies of default registries from a shared
        AssetStore. Each Spec-backed registry (agents/skills/mcp/tools) is
        constructed via SpecLoader.from_resources under its prefix.
        ``extensions_base`` (a filesystem root Path) optionally builds a
        ExtensionRegistry + DirectoryEntrypointResolver; extensions are
        filesystem trees, not single Spec files, so they take a root rather
        than a prefix."""
        from ..agent.catalog import AgentCatalog
        from ..catalog.parsing import SpecLoader
        from ..mcp.catalog import MCPCatalog
        from ..skill.catalog import SkillCatalog
        from ..tool.catalog import ToolCatalog

        prefixes = prefixes or ProviderPrefixes()
        kwargs: "dict[str, Any]" = {}
        if prefixes.agents:
            kwargs["agents"] = AgentCatalog.from_specloader(
                SpecLoader.from_resources(resource_store, prefix=prefixes.agents)
            )
        if prefixes.skills:
            kwargs["skills"] = SkillCatalog.from_specloader(
                SpecLoader.from_resources(resource_store, prefix=prefixes.skills)
            )
        if prefixes.mcp:
            kwargs["mcp_servers"] = MCPCatalog.from_specloader(
                SpecLoader.from_resources(resource_store, prefix=prefixes.mcp)
            )
        if prefixes.tools:
            kwargs["tool_policies"] = ToolCatalog.from_specloader(
                SpecLoader.from_resources(resource_store, prefix=prefixes.tools)
            )
        if extensions_base is not None:
            from ..extension.resolver import (
                DirectoryEntrypointResolver,
                ExtensionRegistry,
            )

            kwargs["extensions"] = ExtensionRegistry(extensions_base)
            kwargs["extension_resources"] = ExtensionRegistry(extensions_base)
            kwargs["entrypoints"] = DirectoryEntrypointResolver(
                {eid: extensions_base / eid for eid in _list_dir_names(extensions_base)}
            )
        return cls(**kwargs)


def _list_dir_names(base: Any) -> "list[str]":
    from pathlib import Path

    base_path = Path(base)
    if not base_path.is_dir():
        return []
    return [p.name for p in base_path.iterdir() if p.is_dir()]


__all__: "list[str]" = [
    "MappingProvider",
    "ProviderPrefixes",
    "RuntimeDependencies",
]
