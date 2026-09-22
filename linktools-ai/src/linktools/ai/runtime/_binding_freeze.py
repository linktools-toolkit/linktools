#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze execution-owned Agent binding dependencies."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import cast

from ..agent import (
    AgentBinding,
    AgentBindingSnapshot,
    AgentCatalog,
    AgentCompiler,
    SemanticPin,
)
from ..capability import (
    SkillDefinition,
    SkillSourceRegistry,
    SnapshotSkillResourceSource,
)
from ..asset import AssetKey, AssetStore
from ..errors import AIError, ErrorCode
from ..spec import MCPServerSpec, MCPServerSpecCodec
from ..storage import ObjectRef, ObjectStore, StorageRevision
from ._mcp_resources import validate_resource_path, validate_resource_tree


class _RuntimeBindingFreezer:
    """Materialize the immutable dependencies required by one execution binding."""

    def __init__(
        self,
        catalog: AgentCatalog,
        compiler: AgentCompiler,
        skill_sources: SkillSourceRegistry,
        object_store: ObjectStore,
        *,
        freeze_dependencies: bool,
        mcp_assets: "Mapping[str, AssetStore] | None" = None,
    ) -> None:
        if not isinstance(catalog, AgentCatalog):
            raise TypeError("catalog must be AgentCatalog")
        if not isinstance(compiler, AgentCompiler):
            raise TypeError("compiler must be AgentCompiler")
        if not isinstance(skill_sources, SkillSourceRegistry):
            raise TypeError("skill_sources must be SkillSourceRegistry")
        self._catalog = catalog
        self._compiler = compiler
        self._skill_sources = skill_sources
        self._objects = object_store
        self._freeze_dependencies = freeze_dependencies
        self._mcp_assets = dict(mcp_assets or {})

    @property
    def root_ids(self) -> tuple[str, ...]:
        return self._catalog.root_ids

    async def freeze(self, binding: AgentBinding) -> AgentBinding:
        """Freeze one current binding before its first durable admission."""
        if not isinstance(binding, AgentBinding):
            raise TypeError("binding must be AgentBinding")
        if not self._freeze_dependencies and not _has_mcp_resources(
            binding.snapshot
        ):
            return binding
        snapshot = await self.freeze_snapshot(binding.snapshot)
        if snapshot == binding.snapshot:
            return binding
        return self._compiler.restore(snapshot)

    async def freeze_root(
        self,
        agent_id: str,
        *,
        skill_snapshots: "dict[tuple[str, str], ObjectRef] | None" = None,
    ) -> AgentBindingSnapshot:
        """Freeze one Agent as a root execution target and its direct children."""
        definition = self._catalog.root_definition(agent_id)
        return await self.freeze_snapshot(
            self._compiler.bind(definition, output=None).snapshot,
            skill_snapshots=skill_snapshots,
        )

    async def freeze_snapshot(
        self,
        snapshot: AgentBindingSnapshot,
        *,
        skill_snapshots: "dict[tuple[str, str], ObjectRef] | None" = None,
    ) -> AgentBindingSnapshot:
        """Freeze Skill resources and direct child bindings for one snapshot."""
        if not isinstance(snapshot, AgentBindingSnapshot):
            raise TypeError("snapshot must be AgentBindingSnapshot")
        if not self._freeze_dependencies and not _has_mcp_resources(snapshot):
            return snapshot
        cache = {} if skill_snapshots is None else skill_snapshots
        frozen = (
            await self._freeze_skills(snapshot, skill_snapshots=cache)
            if self._freeze_dependencies
            else snapshot
        )
        frozen = await self._freeze_mcp(frozen)
        if frozen.subagent_bindings:
            children = tuple(
                [
                    await self._freeze_mcp(
                        await self._freeze_skills(child, skill_snapshots=cache)
                        if self._freeze_dependencies
                        else child
                    )
                    for child in frozen.subagent_bindings
                ]
            )
        elif self._freeze_dependencies:
            children = tuple(
                [
                    await self._freeze_child(child_id, skill_snapshots=cache)
                    for child_id in frozen.subagent_ids
                ]
            )
        else:
            children = frozen.subagent_bindings
        return replace(frozen, subagent_bindings=children)

    async def _freeze_child(
        self,
        agent_id: str,
        *,
        skill_snapshots: "dict[tuple[str, str], ObjectRef]",
    ) -> AgentBindingSnapshot:
        definition = self._catalog.root_definition(agent_id)
        snapshot = self._compiler.bind_subagent(definition).snapshot
        return await self._freeze_mcp(
            await self._freeze_skills(
                snapshot,
                skill_snapshots=skill_snapshots,
            )
        )

    async def _freeze_mcp(
        self,
        snapshot: AgentBindingSnapshot,
    ) -> AgentBindingSnapshot:
        selected: list[SemanticPin] = []
        for pin in snapshot.selected:
            if pin.kind != "mcp":
                selected.append(pin)
                continue
            codec = MCPServerSpecCodec()
            server, resource_snapshot = codec.from_frozen_payload(
                cast("Mapping[str, object]", pin.contract)
            )
            if resource_snapshot is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if server.resource_root is None:
                selected.append(pin)
                continue
            store = self._mcp_assets.get(server.id)
            if store is None:
                raise AIError(
                    ErrorCode.CAPABILITY_REQUIRED_MISSING,
                    safe_details={"kind": "mcp_resource", "server_id": server.id},
                )
            reference = await _snapshot_mcp_resources(
                store,
                server.resource_root,
                server.args,
                object_store=self._objects,
            )
            selected.append(
                SemanticPin(
                    "mcp",
                    pin.id,
                    codec.to_frozen_payload(server, reference),
                )
            )
        return replace(snapshot, selected=tuple(selected))

    async def _freeze_skills(
        self,
        snapshot: AgentBindingSnapshot,
        *,
        skill_snapshots: "dict[tuple[str, str], ObjectRef]",
    ) -> AgentBindingSnapshot:
        selected: list[SemanticPin] = []
        for pin in snapshot.selected:
            if pin.kind != "skill":
                selected.append(pin)
                continue
            skill = SkillDefinition.from_semantic_contract(
                cast("Mapping[str, object]", pin.contract)
            )
            source_ref = skill.source_ref
            if source_ref is None or source_ref.snapshot is not None:
                selected.append(pin)
                continue
            source = self._skill_sources.resolve(source_ref.source_id)
            if not isinstance(source, SnapshotSkillResourceSource):
                raise AIError(
                    ErrorCode.CAPABILITY_REQUIRED_MISSING,
                    safe_details={
                        "kind": "skill_snapshot",
                        "skill_id": skill.id,
                        "source_id": source_ref.source_id,
                    },
                )
            source_key = (source_ref.source_id, source_ref.root)
            reference = skill_snapshots.get(source_key)
            if reference is None:
                revision = await source.current_revision(source_ref.root)
                if not isinstance(revision, StorageRevision):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                reference = await source.snapshot(
                    source_ref.root,
                    expected_revision=revision,
                    object_store=self._objects,
                )
                skill_snapshots[source_key] = reference
            frozen_skill = SkillDefinition(
                skill.spec,
                source_ref.with_snapshot(reference),
            )
            selected.append(
                SemanticPin(
                    "skill",
                    pin.id,
                    frozen_skill.semantic_contract,
                )
            )
        return replace(snapshot, selected=tuple(selected))


def _has_mcp_resources(snapshot: AgentBindingSnapshot) -> bool:
    codec = MCPServerSpecCodec()
    for pin in snapshot.selected:
        if pin.kind != "mcp":
            continue
        server, resource_snapshot = codec.from_frozen_payload(
            cast("Mapping[str, object]", pin.contract)
        )
        if resource_snapshot is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if server.resource_root is not None:
            return True
    return any(_has_mcp_resources(child) for child in snapshot.subagent_bindings)


__all__ = ["_RuntimeBindingFreezer"]


async def _snapshot_mcp_resources(
    store: AssetStore,
    root: AssetKey,
    args: Sequence[str],
    *,
    object_store: ObjectStore,
) -> ObjectRef:
    revision = await store.current_revision()
    infos = await store.metadata_snapshot()
    prefix = f"{root.id}/"
    selected = tuple(
        info.key
        for info in infos
        if info.key.kind == root.kind and info.key.id.startswith(prefix)
    )
    available = {key.id[len(prefix) :] for key in selected}
    validate_resource_tree(available)
    for argument in args:
        if not argument.startswith("resource:"):
            continue
        relative = argument[len("resource:") :]
        validate_resource_path(relative)
        if relative not in available:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return await store.snapshot(
        selected,
        object_store=object_store,
        expected_revision=revision,
    )
