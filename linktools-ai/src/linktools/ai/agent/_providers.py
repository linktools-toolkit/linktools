#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asset-backed capability providers used by AgentCompiler."""

from ..asset import AssetRef, AssetRepository
from ..capability import (
    CapabilityBinding,
    CapabilityProvider,
    MCPRuntimeProvider,
    bind_mcp_capability,
    bind_skill_capability,
)
from ..errors import AIError, ErrorCode
from ..spec import AgentCapabilityRef, MCPServerSpec, SkillSpec


class AssetSkillProvider:
    """Resolve skills from the canonical AssetRepository directory layout."""

    provider = "skill"

    def __init__(self, assets: AssetRepository) -> None:
        self._assets = assets

    async def bind(self, refs: "tuple[AgentCapabilityRef, ...]") -> CapabilityBinding:
        values: list[SkillSpec | None] = []
        for ref in refs:
            try:
                resolved = await self._assets.resolve(AssetRef("skill", ref.id))
            except AIError as error:
                if error.code is ErrorCode.STORAGE_NOT_FOUND and not ref.required:
                    values.append(None)
                    continue
                raise
            if not isinstance(resolved.spec, SkillSpec):
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            values.append(resolved.spec)
        return bind_skill_capability(refs, values)


class AssetMCPProvider:
    """Resolve MCP declarations from AssetRepository and bind an execution provider."""

    provider = "mcp"

    def __init__(self, assets: AssetRepository, runtime: MCPRuntimeProvider) -> None:
        self._assets = assets
        self._runtime = runtime

    async def bind(self, refs: "tuple[AgentCapabilityRef, ...]") -> CapabilityBinding:
        values: list[MCPServerSpec | None] = []
        for ref in refs:
            try:
                resolved = await self._assets.resolve(AssetRef("mcp", ref.id))
            except AIError as error:
                if error.code is ErrorCode.STORAGE_NOT_FOUND and not ref.required:
                    values.append(None)
                    continue
                raise
            if not isinstance(resolved.spec, MCPServerSpec):
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            values.append(resolved.spec)
        return bind_mcp_capability(refs, values, self._runtime)


def build_asset_capability_providers(
    assets: AssetRepository,
    *,
    mcp_runtime: "MCPRuntimeProvider | None" = None,
) -> "tuple[CapabilityProvider, ...]":
    """Build declaration providers backed by the workspace AssetRepository."""
    providers: list[CapabilityProvider] = [AssetSkillProvider(assets)]
    if mcp_runtime is not None:
        providers.append(AssetMCPProvider(assets, mcp_runtime))
    return tuple(providers)


__all__ = ["AssetMCPProvider", "AssetSkillProvider", "build_asset_capability_providers"]
