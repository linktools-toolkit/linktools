#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve execution-owned Agent binding dependencies to immutable Asset versions."""

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
from ..asset import AssetKey, AssetStoreReader, AssetVersionRef
from ..capability import (
    mcp_resource_path,
    validate_resource_path,
    validate_resource_tree,
)
from ..core import JsonValue, canonical_sha256
from ..errors import AIError, ErrorCode
from ..spec import MCPServerSpec, MCPServerSpecCodec
from ..workspace import Sandbox, StdioSandbox


class _RuntimeBindingResolver:
    """Resolve execution-owned MCP resources and direct child bindings."""

    def __init__(
        self,
        catalog: AgentCatalog,
        compiler: AgentCompiler,
        *,
        sandbox: Sandbox | None = None,
        mcp_assets: "Mapping[str, tuple[str, AssetStoreReader]] | None" = None,
    ) -> None:
        if not isinstance(catalog, AgentCatalog):
            raise TypeError("catalog must be AgentCatalog")
        if not isinstance(compiler, AgentCompiler):
            raise TypeError("compiler must be AgentCompiler")
        self._catalog = catalog
        self._compiler = compiler
        self._sandbox = sandbox
        self._mcp_assets = dict(mcp_assets or {})

    @property
    def root_ids(self) -> tuple[str, ...]:
        return self._catalog.root_ids

    async def resolve(self, binding: AgentBinding) -> AgentBinding:
        """Resolve one current binding before its first durable admission."""
        if not isinstance(binding, AgentBinding):
            raise TypeError("binding must be AgentBinding")
        snapshot = await self.resolve_snapshot(binding.snapshot)
        if snapshot == binding.snapshot:
            return binding
        return self._compiler.restore(snapshot)

    async def resolve_root(self, agent_id: str) -> AgentBindingSnapshot:
        """Resolve one Agent as a root execution target and its direct children."""
        definition = self._catalog.root_definition(agent_id)
        return await self.resolve_snapshot(
            self._compiler.bind(definition, output=None).snapshot
        )

    async def resolve_snapshot(
        self,
        snapshot: AgentBindingSnapshot,
    ) -> AgentBindingSnapshot:
        """Resolve MCP Asset versions and direct child bindings for one snapshot."""
        if not isinstance(snapshot, AgentBindingSnapshot):
            raise TypeError("snapshot must be AgentBindingSnapshot")
        resolved = await self._resolve_mcp(snapshot)
        if resolved.subagent_bindings:
            children = tuple(
                [await self._resolve_mcp(child) for child in resolved.subagent_bindings]
            )
        else:
            children = tuple(
                [await self._resolve_child(child_id) for child_id in resolved.subagent_ids]
            )
        return replace(resolved, subagent_bindings=children)

    async def _resolve_child(self, agent_id: str) -> AgentBindingSnapshot:
        definition = self._catalog.root_definition(agent_id)
        snapshot = self._compiler.bind_subagent(definition).snapshot
        return await self._resolve_mcp(snapshot)

    async def _resolve_mcp(
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
                execution_policy = _mcp_execution_policy(
                    self._sandbox,
                )
            server, resource_versions = codec.from_execution_payload(
                cast("Mapping[str, object]", pin.contract)
            )
            current_policy = dict(execution_policy)
            bound_policy = pin.contract.get("execution_policy")
            if bound_policy is not None and dict(bound_policy) != current_policy:
                raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
            if resource_versions is not None:
                if bound_policy is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                selected.append(pin)
                continue
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
                resource_versions, resource_semantic_digest = (
                    await _resolve_mcp_resource_versions(
                        store,
                        server.resource_root,
                        server.args,
                    )
                )
            selected.append(
                SemanticPin(
                    "mcp",
                    pin.id,
                    codec.to_execution_payload(
                        server,
                        resource_versions,
                        resource_source_id=resource_source_id,
                        resource_semantic_digest=resource_semantic_digest,
                        execution_policy=execution_policy,
                    ),
                )
            )
        return replace(snapshot, selected=tuple(selected))


def _mcp_execution_policy(
    sandbox: Sandbox | None,
) -> "Mapping[str, JsonValue]":
    if sandbox is None:
        return {"version": 1, "boundary": "host-stdio"}
    if not isinstance(sandbox, StdioSandbox):
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
    return sandbox.stdio_execution_policy()


__all__ = ["_RuntimeBindingResolver"]


async def _resolve_mcp_resource_versions(
    store: AssetStoreReader,
    root: AssetKey,
    args: Sequence[str],
) -> tuple[tuple[AssetVersionRef, ...], str]:
    infos = await store.metadata_snapshot()
    selected_infos = tuple(
        (info, relative)
        for info in infos
        if (relative := mcp_resource_path(info.key, root)) is not None
    )
    selected = tuple(info.key for info, _relative in selected_infos)
    available = {relative for _info, relative in selected_infos}
    validate_resource_tree(available)
    for argument in args:
        if not argument.startswith("resource:"):
            continue
        relative = argument[len("resource:") :]
        validate_resource_path(relative)
        if relative not in available:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    versions = await store.resolve_versions(selected)
    for (info, _relative), version in zip(selected_infos, versions, strict=True):
        if not version.matches_info(info):
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
    resource_files = [
        {
            "path": relative,
            "sha256": info.etag,
        }
        for info, relative in selected_infos
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
