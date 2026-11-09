#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve current Asset-backed declarations before Agent compilation."""

from typing import Generic, TypeVar

from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapperCapability

from ..asset import AssetStore, AssetStoreReader
from ..errors import AIError, ErrorCode
from ..spec import AgentSpec, MCPServerSpec, RepositoryInstructionDocument
from ._context import AgentContext
from ._contribution import CapabilityContribution
from ._declaration import BuiltinDeclarationLoader
from ._loading import CapabilityLoadContext, CapabilityLoader
from ._skill import SkillDefinition

AppT = TypeVar("AppT")


class AssetCapability(AbstractCapability[AgentContext[AppT]], Generic[AppT]):
    """Load declarations from one versioned AssetStore.

    Runtime owns the declared store when consuming RuntimeFeatures. An explicit live
    source view instead borrows an externally managed store.
    Declare a store for one Runtime lifetime; wrapping that same store again
    does not create an independent resource. Share a borrowed live source across Runtimes.
    RuntimeFeatures consumes this Runtime-scoped capability during source resolution;
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

    @property
    def reader(self) -> AssetStoreReader:
        """Return the version-reading source used by Runtime resource adapters."""
        return self._store

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
                      safe_details={"reason": "asset_declaration_requires_resolution"})

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

    async def load(
        self, group_id: str,
    ) -> tuple[CapabilityLoadContext, tuple[CapabilityContribution[AppT] | RepositoryInstructionDocument, ...]]:
        """Resolve the current loader slots against this call's Asset versions."""
        loaders = tuple(self._loaders.values())
        context = await CapabilityLoadContext.read_current(group_id, self._store)
        entries: list[CapabilityContribution[AppT] | RepositoryInstructionDocument] = []
        for loader in loaders:
            for value in await loader.load(context):
                _validate_skill_source(value, context)
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
        return context, tuple(entries)



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
    resources = (*source_ref.resources, *((source_ref.declaration,) if source_ref.declaration is not None else ()))
    versions = tuple(item.asset for item in resources)
    try:
        captured = context.bind_versions(tuple(ref.key for ref in versions))
    except AIError as error:
        if error.code is ErrorCode.SNAPSHOT_CONFLICT:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
        raise
    if versions != captured:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


__all__ = ["AssetCapability"]
