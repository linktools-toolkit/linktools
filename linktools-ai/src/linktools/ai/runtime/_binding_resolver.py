#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve execution-owned Agent binding dependencies to immutable Asset versions."""

from collections.abc import Mapping
from dataclasses import replace
from typing import cast

from ..agent import (
    AgentBinding,
    AgentBindingSnapshot,
    AgentCatalog,
    AgentCompiler,
    CapabilityPin,
)
from ..core import JsonValue
from ..errors import AIError, ErrorCode
from ..spec import MCPServerSpec, MCPServerSpecCodec
from ..workspace import Sandbox
from ._mcp import _mcp_execution_policy


class _RuntimeBindingResolver:
    """Resolve execution-owned MCP resources and direct child bindings."""

    def __init__(
        self,
        catalog: AgentCatalog,
        compiler: AgentCompiler,
        *,
        sandbox: Sandbox | None = None,
    ) -> None:
        if not isinstance(catalog, AgentCatalog):
            raise TypeError("catalog must be AgentCatalog")
        if not isinstance(compiler, AgentCompiler):
            raise TypeError("compiler must be AgentCompiler")
        self._catalog = catalog
        self._compiler = compiler
        self._sandbox = sandbox

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
        compiled_agent = self._catalog.root_agent(agent_id)
        return await self.resolve_snapshot(
            self._compiler.bind(compiled_agent, output=None).snapshot
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
        compiled_agent = self._catalog.root_agent(agent_id)
        snapshot = self._compiler.bind_subagent(compiled_agent).snapshot
        return await self._resolve_mcp(snapshot)

    async def _resolve_mcp(
        self,
        snapshot: AgentBindingSnapshot,
    ) -> AgentBindingSnapshot:
        execution_policy: "Mapping[str, JsonValue] | None" = None
        selected: list[CapabilityPin] = []
        codec = MCPServerSpecCodec()
        for pin in snapshot.selected:
            if pin.kind != "mcp":
                selected.append(pin)
                continue
            if execution_policy is None:
                execution_policy = _mcp_execution_policy(self._sandbox)
            server, resource_versions = codec.from_execution_payload(
                cast("Mapping[str, object]", pin.contract)
            )
            current_policy = dict(execution_policy)
            bound_policy = pin.contract.get("execution_policy")
            if bound_policy is not None and dict(bound_policy) != current_policy:
                raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)

            resource_source_id = pin.contract.get("resource_source_id")
            if server.resource_root is None:
                if resource_versions is not None or resource_source_id is not None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                source_id = None
            else:
                if (
                    resource_versions is None
                    or not isinstance(resource_source_id, str)
                    or not resource_source_id
                ):
                    raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING)
                source_id = resource_source_id

            selected.append(
                CapabilityPin(
                    "mcp",
                    pin.id,
                    codec.to_execution_payload(
                        server,
                        resource_versions,
                        resource_source_id=source_id,
                        execution_policy=current_policy,
                    ),
                )
            )
        return replace(snapshot, selected=tuple(selected))


__all__ = ["_RuntimeBindingResolver"]
