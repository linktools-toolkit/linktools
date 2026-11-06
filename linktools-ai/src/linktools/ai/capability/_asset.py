#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asset-backed declarations captured before Agent compilation."""

from dataclasses import replace
from typing import Generic, TypeVar

from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapperCapability

from ..asset import AssetStore
from ..errors import AIError, ErrorCode
from ..spec import AgentSpec, MCPServerSpec, RepositoryInstructionDocument
from ._context import AgentContext
from ._contribution import CapabilityContribution
from ._declaration import BuiltinDeclarationLoader
from ._loading import CapabilityLoadContext, CapabilityLoader
from ._skill import SkillDefinition
from ._skill_source import AssetSkillSource, SkillSourceRef

AppT = TypeVar("AppT")


class AssetCapability(AbstractCapability[AgentContext[AppT]], Generic[AppT]):
    """Load declarations from one versioned AssetStore.

    Runtime owns the declared store when consuming RuntimeFeatures. A prebuilt
    capture instead borrows its reader from an externally managed store.
    Declare a store for one Runtime lifetime; wrapping that same store again
    does not create an independent resource. Share a capture across Runtimes.
    RuntimeFeatures consumes this Runtime-scoped capability during capture;
    it does not participate in native Agent execution hooks.
    """

    id = "runtime.assets"

    def __init__(self, store: AssetStore) -> None:
        if not isinstance(store, AssetStore):
            raise TypeError("store must be AssetStore")
        self._store = store
        self._store_claimed = False
        self._loaders: dict[str, CapabilityLoader[AppT]] = {
            kind: BuiltinDeclarationLoader(kind) for kind in ("agent", "skill", "mcp", "rule")
        }

    def claim_store(self) -> AssetStore:
        """Reserve this declaration for one Runtime before initialization."""
        if self._store_claimed:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                          "Asset declaration already belongs to an active Runtime")
        self._store_claimed = True
        return self._store

    async def close_store(self) -> None:
        """Release a claimed store only after its close completes successfully."""
        if self._store_claimed:
            await self._store.close()
            self._store_claimed = False

    def for_agent(self, agent: object) -> "AssetCapability[AppT]":
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                      safe_details={"reason": "asset_declaration_requires_capture"})

    def loader(self, kind: str, loader: CapabilityLoader[AppT]) -> CapabilityLoader[AppT]:
        """Register the sole loader for one Asset kind, replacing its slot."""
        if not isinstance(kind, str) or not kind.strip():
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if not callable(getattr(loader, "load", None)):
            raise TypeError("loader must implement load")
        source_kind = getattr(loader, "source_kind", None)
        if source_kind is not None and source_kind != kind:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        self._loaders[kind] = loader
        return loader

    def skill_source(self, group_id: str) -> AssetSkillSource:
        """Bind current Skill reads to the declared loader and owned store."""
        loader = self._loaders["skill"]

        async def discover() -> tuple[SkillDefinition, ...]:
            context = await CapabilityLoadContext.read_current(group_id, self._store)
            skills: list[SkillDefinition] = []
            for value in await loader.load(context):
                _validate_skill_source(value, context)
                if isinstance(value, CapabilityContribution):
                    if value.kind != "skill":
                        continue
                    value = value.value
                elif isinstance(value, (AgentSpec, MCPServerSpec, RepositoryInstructionDocument)):
                    continue
                if not isinstance(value, SkillDefinition):
                    raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
                skills.append(_bind_skill_source(value, f"{group_id}/skills"))
            return tuple(skills)

        return AssetSkillSource(f"{group_id}/skills", self._store, discover=discover)

    async def capture(
        self, group_id: str,
    ) -> tuple[CapabilityLoadContext, tuple[CapabilityContribution[AppT] | RepositoryInstructionDocument, ...]]:
        """Capture all loader outputs against the same source revision."""
        loaders = tuple(self._loaders.items())
        context = await CapabilityLoadContext.capture(group_id, self._store)
        entries: list[CapabilityContribution[AppT] | RepositoryInstructionDocument] = []
        for kind, loader in loaders:
            for value in await loader.load(context):
                _validate_skill_source(value, context)
                if kind == "skill":
                    if isinstance(value, SkillDefinition):
                        value = _bind_skill_source(value, f"{group_id}/skills")
                    elif isinstance(value, CapabilityContribution) and value.kind == "skill":
                        value = CapabilityContribution.from_declaration(
                            _bind_skill_source(value.value, f"{group_id}/skills")
                        )
                if isinstance(value, (AgentSpec, SkillDefinition, MCPServerSpec)):
                    entries.append(CapabilityContribution.from_declaration(value))
                elif isinstance(value, (CapabilityContribution, RepositoryInstructionDocument)):
                    if isinstance(value, CapabilityContribution) and value.kind == "capability":
                        def reject_bootstrap(capability: AbstractCapability[object]) -> None:
                            if isinstance(capability, WrapperCapability):
                                reject_bootstrap(capability.wrapped)
                            elif isinstance(capability, CombinedCapability):
                                for child in capability.capabilities:
                                    reject_bootstrap(child)
                            elif isinstance(capability, AssetCapability):
                                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                                              "Asset loaders cannot introduce AssetCapability")

                        reject_bootstrap(value.value)
                        value.value.apply(reject_bootstrap)
                    entries.append(value)
                else:
                    raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        await context.verify_source_revision()
        return context, tuple(entries)


def _bind_skill_source(definition: SkillDefinition, group_id: str) -> SkillDefinition:
    source = definition.source_ref
    return replace(
        definition,
        source_ref=SkillSourceRef(group_id, definition.id) if source is None else replace(source, source_id=group_id),
    )


def _validate_skill_source(
    value: object,
    context: CapabilityLoadContext,
) -> None:
    if isinstance(value, CapabilityContribution):
        if value.kind != "skill" or not isinstance(value.value, SkillDefinition):
            return
        definition = value.value
    elif isinstance(value, SkillDefinition):
        definition = value
    else:
        return
    source_ref = definition.source_ref
    if source_ref is None:
        return
    if source_ref.source_id != context.group_id:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    versions = tuple(item.asset for item in source_ref.resources)
    try:
        captured = context.bind_versions(tuple(ref.key for ref in versions))
    except AIError as error:
        if error.code is ErrorCode.SNAPSHOT_CONFLICT:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
        raise
    if versions != captured:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


__all__ = ["AssetCapability"]
