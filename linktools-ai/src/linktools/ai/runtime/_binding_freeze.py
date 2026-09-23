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
    FrozenSkillResourceSource,
    SkillDefinition,
    SkillSourceRegistry,
    SnapshotSkillResourceSource,
    validate_resource_path,
    validate_resource_tree,
)
from ..asset import AssetKey, AssetStoreReader
from ..core import JsonValue, canonical_sha256
from ..errors import AIError, ErrorCode
from ..spec import MCPServerSpec, MCPServerSpecCodec
from ..storage import ObjectRef, ObjectStore, StorageRevision
from ..workspace import StdioSandbox, Workspace


class _RuntimeBindingFreezer:
    """Materialize the immutable dependencies required by one execution binding."""

    def __init__(
        self,
        catalog: AgentCatalog,
        compiler: AgentCompiler,
        skill_sources: SkillSourceRegistry,
        object_store: ObjectStore,
        *,
        workspace: Workspace | None,
        mcp_assets: "Mapping[str, AssetStoreReader] | None" = None,
        asset_sources: "Mapping[str, tuple[AssetStoreReader, StorageRevision]] | None" = None,
        mcp_revisions: "Mapping[str, StorageRevision] | None" = None,
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
        self._workspace = workspace
        self._mcp_assets = dict(mcp_assets or {})
        self._asset_sources = dict(asset_sources or {})
        self._mcp_revisions = dict(mcp_revisions or {})

    @property
    def root_ids(self) -> tuple[str, ...]:
        return self._catalog.root_ids

    async def freeze(self, binding: AgentBinding) -> AgentBinding:
        """Freeze one current binding before its first durable admission."""
        if not isinstance(binding, AgentBinding):
            raise TypeError("binding must be AgentBinding")
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
        cache = {} if skill_snapshots is None else skill_snapshots
        frozen = await self._freeze_skills(snapshot, skill_snapshots=cache)
        frozen = await self._freeze_mcp(frozen)
        if frozen.subagent_bindings:
            children = tuple(
                [
                    await self._freeze_mcp(
                        await self._freeze_skills(child, skill_snapshots=cache)
                    )
                    for child in frozen.subagent_bindings
                ]
            )
        else:
            children = tuple(
                [
                    await self._freeze_child(child_id, skill_snapshots=cache)
                    for child_id in frozen.subagent_ids
                ]
            )
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
        execution_policy: "Mapping[str, JsonValue] | None" = None
        selected: list[SemanticPin] = []
        for pin in snapshot.selected:
            if pin.kind != "mcp":
                selected.append(pin)
                continue
            codec = MCPServerSpecCodec()
            if execution_policy is None:
                execution_policy = _mcp_execution_policy(self._workspace)
            server, resource_snapshot = codec.from_frozen_payload(
                cast("Mapping[str, object]", pin.contract)
            )
            current_policy = dict(execution_policy)
            frozen_policy = pin.contract.get("execution_policy")
            if frozen_policy is not None and dict(frozen_policy) != current_policy:
                raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
            if resource_snapshot is not None:
                if resource_snapshot.store_id != self._objects.store_id:
                    raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                if frozen_policy is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                selected.append(pin)
                continue
            reference = None
            resource_semantic_digest = None
            if server.resource_root is not None:
                store = self._mcp_assets.get(server.id)
                if store is None:
                    raise AIError(
                        ErrorCode.CAPABILITY_REQUIRED_MISSING,
                        safe_details={
                            "kind": "mcp_resource",
                            "server_id": server.id,
                        },
                    )
                expected_revision = self._mcp_revisions.get(server.id)
                if (
                    expected_revision is not None
                    and await store.current_revision() != expected_revision
                ):
                    raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
                reference, resource_semantic_digest = (
                    await _snapshot_mcp_resources(
                        store,
                        server.resource_root,
                        server.args,
                        object_store=self._objects,
                        expected_revision=expected_revision,
                    )
                )
            elif any(argument.startswith("resource:") for argument in server.args):
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            selected.append(
                SemanticPin(
                    "mcp",
                    pin.id,
                    codec.to_frozen_payload(
                        server,
                        reference,
                        resource_semantic_digest=resource_semantic_digest,
                        execution_policy=execution_policy,
                    ),
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
            await self._verify_asset_source(source_ref.source_id)
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
            frozen_source = FrozenSkillResourceSource(
                source_ref.source_id,
                {source_ref.root: reference},
                self._objects,
            )
            resource_semantic_digest = await frozen_source.semantic_digest(
                source_ref.root
            )
            await self._verify_asset_source(source_ref.source_id)
            frozen_skill = SkillDefinition(
                skill.spec,
                source_ref.with_snapshot(reference, resource_semantic_digest),
            )
            selected.append(
                SemanticPin(
                    "skill",
                    pin.id,
                    frozen_skill.semantic_contract,
                )
            )
        return replace(snapshot, selected=tuple(selected))

    async def _verify_asset_source(self, source_id: str) -> None:
        asset_source = self._asset_sources.get(source_id)
        if asset_source is None:
            return
        store, expected_revision = asset_source
        if await store.current_revision() != expected_revision:
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)


def _mcp_execution_policy(
    workspace: Workspace | None,
) -> "Mapping[str, JsonValue]":
    if workspace is None:
        return {"version": 1, "boundary": "host-stdio"}
    backend = workspace.sandbox
    if not isinstance(backend, StdioSandbox):
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
    return backend.stdio_execution_policy()


__all__ = ["_RuntimeBindingFreezer"]


async def _snapshot_mcp_resources(
    store: AssetStoreReader,
    root: AssetKey,
    args: Sequence[str],
    *,
    object_store: ObjectStore,
    expected_revision: "StorageRevision | None" = None,
) -> tuple[ObjectRef, str]:
    revision = await store.current_revision()
    if expected_revision is not None and revision != expected_revision:
        raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
    infos = await store.metadata_snapshot()
    prefix = f"{root.id}/"
    selected_infos = tuple(
        info
        for info in infos
        if info.key.kind == root.kind and info.key.id.startswith(prefix)
        and info.key.id[len(prefix) :] not in {"mcp.json", "mcp.yaml"}
    )
    selected = tuple(info.key for info in selected_infos)
    available = {
        info.key.id[len(prefix) :]
        for info in selected_infos
    }
    validate_resource_tree(available)
    for argument in args:
        if not argument.startswith("resource:"):
            continue
        relative = argument[len("resource:") :]
        validate_resource_path(relative)
        if relative not in available:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    reference = await store.snapshot(
        selected,
        object_store=object_store,
        expected_revision=(
            revision if expected_revision is None else expected_revision
        ),
    )
    resource_files = [
        {
            "path": info.key.id[len(prefix) :],
            "sha256": info.etag,
        }
        for info in selected_infos
    ]
    resource_files.sort(key=lambda item: cast(str, item["path"]))
    resource_semantic_digest = canonical_sha256(
        {
            "version": 1,
            "kind": "mcp-resource-semantics",
            "files": resource_files,
        }
    )
    return reference, resource_semantic_digest
