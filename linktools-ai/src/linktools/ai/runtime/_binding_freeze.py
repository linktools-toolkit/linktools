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
    SkillSourceRef,
    SkillSourceRegistry,
    VersionedSkillResourceSource,
    validate_resource_path,
    validate_resource_tree,
)
from ..asset import AssetKey, AssetStoreReader, AssetVersionRef
from ..core import JsonValue, canonical_sha256
from ..errors import AIError, ErrorCode
from ..spec import MCPServerSpec, MCPServerSpecCodec
from ..workspace import StdioSandbox, Workspace


class _RuntimeBindingFreezer:
    """Materialize the immutable dependencies required by one execution binding."""

    def __init__(
        self,
        catalog: AgentCatalog,
        compiler: AgentCompiler,
        skill_sources: SkillSourceRegistry,
        *,
        workspace: Workspace | None,
        mcp_assets: "Mapping[str, tuple[str, AssetStoreReader]] | None" = None,
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
        self._workspace = workspace
        self._mcp_assets = dict(mcp_assets or {})

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
        skill_versions: "dict[tuple[str, str], SkillSourceRef] | None" = None,
    ) -> AgentBindingSnapshot:
        """Freeze one Agent as a root execution target and its direct children."""
        definition = self._catalog.root_definition(agent_id)
        return await self.freeze_snapshot(
            self._compiler.bind(definition, output=None).snapshot,
            skill_versions=skill_versions,
        )

    async def freeze_snapshot(
        self,
        snapshot: AgentBindingSnapshot,
        *,
        skill_versions: "dict[tuple[str, str], SkillSourceRef] | None" = None,
    ) -> AgentBindingSnapshot:
        """Freeze Skill resources and direct child bindings for one snapshot."""
        if not isinstance(snapshot, AgentBindingSnapshot):
            raise TypeError("snapshot must be AgentBindingSnapshot")
        cache = {} if skill_versions is None else skill_versions
        frozen = await self._freeze_skills(snapshot, skill_versions=cache)
        frozen = await self._freeze_mcp(frozen)
        if frozen.subagent_bindings:
            children = tuple(
                [
                    await self._freeze_mcp(
                        await self._freeze_skills(child, skill_versions=cache)
                    )
                    for child in frozen.subagent_bindings
                ]
            )
        else:
            children = tuple(
                [
                    await self._freeze_child(child_id, skill_versions=cache)
                    for child_id in frozen.subagent_ids
                ]
            )
        return replace(frozen, subagent_bindings=children)

    async def _freeze_child(
        self,
        agent_id: str,
        *,
        skill_versions: "dict[tuple[str, str], SkillSourceRef]",
    ) -> AgentBindingSnapshot:
        definition = self._catalog.root_definition(agent_id)
        snapshot = self._compiler.bind_subagent(definition).snapshot
        return await self._freeze_mcp(
            await self._freeze_skills(
                snapshot,
                skill_versions=skill_versions,
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
            server, resource_versions = codec.from_frozen_payload(
                cast("Mapping[str, object]", pin.contract)
            )
            current_policy = dict(execution_policy)
            frozen_policy = pin.contract.get("execution_policy")
            if frozen_policy is not None and dict(frozen_policy) != current_policy:
                raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
            if resource_versions is not None:
                if frozen_policy is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                selected.append(pin)
                continue
            frozen_versions = None
            resource_source_id = None
            resource_semantic_digest = None
            if server.resource_root is not None:
                asset_source = self._mcp_assets.get(server.id)
                if asset_source is None:
                    raise AIError(
                        ErrorCode.CAPABILITY_REQUIRED_MISSING,
                        safe_details={
                            "kind": "mcp_resource",
                            "server_id": server.id,
                        },
                    )
                resource_source_id, store = asset_source
                frozen_versions, resource_semantic_digest = (
                    await _resolve_mcp_resource_versions(
                        store,
                        server.resource_root,
                        server.args,
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
                        frozen_versions,
                        resource_source_id=resource_source_id,
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
        skill_versions: "dict[tuple[str, str], SkillSourceRef]",
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
            if source_ref is None:
                selected.append(pin)
                continue
            if source_ref.frozen:
                selected.append(pin)
                continue
            source = self._skill_sources.resolve(source_ref.source_id)
            if not isinstance(source, VersionedSkillResourceSource):
                raise AIError(
                    ErrorCode.CAPABILITY_REQUIRED_MISSING,
                    safe_details={
                        "kind": "skill_version_source",
                        "skill_id": skill.id,
                        "source_id": source_ref.source_id,
                    },
                )
            source_key = (source_ref.source_id, source_ref.root)
            frozen_ref = skill_versions.get(source_key)
            if frozen_ref is None:
                frozen_ref = await source.freeze(source_ref.root)
                if (
                    frozen_ref.source_id != source_ref.source_id
                    or frozen_ref.root != source_ref.root
                    or not frozen_ref.frozen
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                skill_versions[source_key] = frozen_ref
            frozen_skill = SkillDefinition(skill.spec, frozen_ref)
            selected.append(
                SemanticPin(
                    "skill",
                    pin.id,
                    frozen_skill.semantic_contract,
                )
            )
        return replace(snapshot, selected=tuple(selected))

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


async def _resolve_mcp_resource_versions(
    store: AssetStoreReader,
    root: AssetKey,
    args: Sequence[str],
) -> tuple[tuple[AssetVersionRef, ...], str]:
    infos = await store.metadata_snapshot()
    prefix = f"{root.id}/"
    selected_infos = tuple(
        info
        for info in infos
        if info.key.kind == root.kind
        and info.key.id.startswith(prefix)
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
    versions = await store.resolve_versions(selected)
    for info, version in zip(selected_infos, versions, strict=True):
        if (
            version.key != info.key
            or version.etag != info.etag
            or version.size != info.size
        ):
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
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
    return tuple(versions), resource_semantic_digest

