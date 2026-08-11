#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-domain declaration loading and Agent binding composition."""

from collections.abc import Sequence
from types import MappingProxyType
from typing import Protocol

from linktools.core import environ

from ..agent import AgentBinder, AgentBinding
from ..asset import AssetRef, AssetRepository
from ..capability import (
    CapabilityBinding,
    CapabilityInjection,
    MCPRuntimeProvider,
    bind_mcp_capability,
    bind_skill_capability,
    group_capability_refs,
    unresolved_binding,
)
from ..core import canonical_sha256
from ..errors import AIError, ErrorCode
from ..spec import AgentCapabilityRef, AgentSpec, MCPServerSpec, PromptSpec, SkillSpec

_logger = environ.get_logger("ai.app.composition")


class CapabilityPreparer(Protocol):
    @property
    def provider(self) -> str: ...

    async def prepare(
        self,
        refs: "tuple[AgentCapabilityRef, ...]",
    ) -> CapabilityBinding: ...


class AgentBindingComposer:
    """Load declarations and compile one immutable AgentBinding."""

    def __init__(
        self,
        assets: AssetRepository,
        binder: AgentBinder,
        preparers: "Sequence[CapabilityPreparer]" = (),
    ) -> None:
        if assets is None or binder is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if not assets.ready:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        providers: dict[str, CapabilityPreparer] = {}
        for preparer in preparers:
            provider = preparer.provider
            if not isinstance(provider, str) or not provider.strip() or provider in providers:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            providers[provider] = preparer
        self._assets = assets
        self._binder = binder
        self._preparers = MappingProxyType(providers)

    async def compose(
        self,
        *,
        agent_id: str,
        prompt_id: str,
        injections: "Sequence[CapabilityInjection]" = (),
    ) -> AgentBinding:
        if not isinstance(agent_id, str) or not agent_id.strip() or not isinstance(prompt_id, str) or not prompt_id.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        _logger.debug(
            "agent binding composition started: agent_id_digest=%s prompt_id_digest=%s",
            _digest(agent_id),
            _digest(prompt_id),
        )
        agent = await self._assets.resolve(AssetRef("agent", agent_id))
        prompt = await self._assets.resolve(AssetRef("prompt", prompt_id))
        if type(agent.spec) is not AgentSpec or type(prompt.spec) is not PromptSpec:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        groups = group_capability_refs(agent.spec.capabilities)
        prepared: list[CapabilityBinding] = []
        for provider, refs in groups:
            preparer = self._preparers.get(provider)
            if preparer is None:
                if any(ref.required for ref in refs):
                    raise AIError(ErrorCode.CAPABILITY_PROVIDER_UNKNOWN)
                prepared.append(unresolved_binding(provider, refs))
                continue
            prepared.append(await preparer.prepare(refs))
        binding = self._binder.bind(
            agent.spec,
            prompt.spec,
            capabilities=tuple(prepared),
            injections=tuple(injections),
        )
        _logger.debug(
            "agent binding composition completed: agent_id_digest=%s prompt_id_digest=%s binding=%s providers=%s",
            _digest(agent_id),
            _digest(prompt_id),
            binding.manifest.digest,
            tuple(provider for provider, _ in groups),
        )
        return binding


def build_builtin_capability_preparers(
    assets: AssetRepository,
    mcp_runtime: "MCPRuntimeProvider | None",
) -> "tuple[CapabilityPreparer, ...]":
    """Create the built-in Asset declaration preparers for one composition."""
    preparers: list[CapabilityPreparer] = [_AssetSkillPreparer(assets)]
    if mcp_runtime is not None:
        preparers.append(_AssetMCPPreparer(assets, mcp_runtime))
    return tuple(preparers)


class _AssetSkillPreparer:
    provider = "skill"

    def __init__(self, assets: AssetRepository) -> None:
        self._assets = assets

    async def prepare(self, refs: "tuple[AgentCapabilityRef, ...]") -> CapabilityBinding:
        specifications: list[SkillSpec | None] = []
        for ref in refs:
            specifications.append(await _resolve_optional(self._assets, "skill", ref.id, SkillSpec))
        return bind_skill_capability(refs, tuple(specifications))


class _AssetMCPPreparer:
    provider = "mcp"

    def __init__(self, assets: AssetRepository, runtime: MCPRuntimeProvider) -> None:
        self._assets = assets
        self._runtime = runtime

    async def prepare(self, refs: "tuple[AgentCapabilityRef, ...]") -> CapabilityBinding:
        servers: list[MCPServerSpec | None] = []
        for ref in refs:
            servers.append(await _resolve_optional(self._assets, "mcp", ref.id, MCPServerSpec))
        return bind_mcp_capability(refs, tuple(servers), self._runtime)


async def _resolve_optional(
    assets: AssetRepository,
    kind: str,
    identifier: str,
    expected_type: type[SkillSpec] | type[MCPServerSpec],
) -> "SkillSpec | MCPServerSpec | None":
    try:
        resolved = await assets.resolve(AssetRef(kind, identifier))
    except AIError as error:
        if error.code is ErrorCode.STORAGE_NOT_FOUND:
            return None
        raise
    if type(resolved.spec) is not expected_type:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return resolved.spec


def _digest(value: str) -> str:
    return canonical_sha256(value)


__all__ = ["AgentBindingComposer", "CapabilityPreparer"]
