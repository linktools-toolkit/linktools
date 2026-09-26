#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve execution-owned Agent binding dependencies to immutable Asset versions."""

from collections.abc import Mapping
from dataclasses import replace

from ..agent import (
    AgentBinding,
    AgentBindingContract,
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
        binding_contract = await self.resolve_contract(binding.binding_contract)
        if binding_contract == binding.binding_contract:
            return binding
        return self._compiler.restore(binding_contract)

    async def resolve_root(self, agent_id: str) -> AgentBindingContract:
        """Resolve one Agent as a root execution target and its direct children."""
        compiled_agent = self._catalog.root_agent(agent_id)
        return await self.resolve_contract(
            self._compiler.bind(compiled_agent, output=None).binding_contract
        )

    async def resolve_contract(
        self,
        binding_contract: AgentBindingContract,
    ) -> AgentBindingContract:
        """Resolve MCP Asset versions and direct child bindings for one contract."""
        if not isinstance(binding_contract, AgentBindingContract):
            raise TypeError("binding_contract must be AgentBindingContract")
        resolved = await self._resolve_mcp(binding_contract)
        if resolved.subagent_bindings:
            children = tuple(
                [await self._resolve_mcp(child) for child in resolved.subagent_bindings]
            )
        else:
            children = tuple(
                [await self._resolve_child(child_id) for child_id in resolved.subagent_ids]
            )
        return replace(resolved, subagent_bindings=children)

    async def _resolve_child(self, agent_id: str) -> AgentBindingContract:
        compiled_agent = self._catalog.root_agent(agent_id)
        binding_contract = self._compiler.bind_subagent(compiled_agent).binding_contract
        return await self._resolve_mcp(binding_contract)

    async def _resolve_mcp(
        self,
        binding_contract: AgentBindingContract,
    ) -> AgentBindingContract:
        current_binding = self._compiler.restore(binding_contract)
        current_servers = {
            server.id: server
            for server in current_binding.compiled_agent.mcp_servers
        }
        selected: list[CapabilityPin] = []
        codec = MCPServerSpecCodec()
        for pin in binding_contract.selected:
            if pin.kind != "mcp":
                selected.append(pin)
                continue
            server = current_servers.get(pin.id)
            if server is None:
                raise AIError(ErrorCode.AGENT_BINDING_UNAVAILABLE)
            resource_versions = codec.decode_execution_payload(
                pin.contract,
                declaration=server,
            )
            current_policy = dict(_mcp_execution_policy(server, self._sandbox))
            bound_policy = pin.contract.get("execution_policy")
            if bound_policy is not None and dict(bound_policy) != current_policy:
                raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
            asset_source_id = pin.contract.get("asset_source_id")
            if server.resource is None:
                if resource_versions is not None or asset_source_id is not None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                asset_source_id = None
            elif (
                resource_versions is None
                or not isinstance(asset_source_id, str)
                or not asset_source_id
            ):
                raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING)
            selected.append(
                CapabilityPin(
                    "mcp",
                    pin.id,
                    codec.to_execution_payload(
                        server,
                        resource_versions,
                        asset_source_id=asset_source_id,
                        execution_policy=current_policy,
                    ),
                )
            )
        return replace(binding_contract, selected=tuple(selected))


__all__ = ["_RuntimeBindingResolver"]
