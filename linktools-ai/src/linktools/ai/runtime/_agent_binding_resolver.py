#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve current Agent admissions and execution-owned hook and child contracts."""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from ..agent import (
    AgentBinding,
    AgentBindingContract,
    AgentCatalog,
    AgentCompiler,
    OutputBinding,
)
from ..model import ModelBinding
from ..errors import AIError, ErrorCode


if TYPE_CHECKING:
    from pydantic import BaseModel


class _AgentBindingResolver:
    """Use one current declaration observation for an admission or restore."""

    def __init__(
        self,
        catalog: AgentCatalog,
        compiler: AgentCompiler,
        *,
        current_compiler: Callable[[], Awaitable[AgentCompiler]] | None = None,
    ) -> None:
        if not isinstance(catalog, AgentCatalog):
            raise TypeError("catalog must be AgentCatalog")
        if not isinstance(compiler, AgentCompiler):
            raise TypeError("compiler must be AgentCompiler")
        self._catalog = catalog
        self._compiler = compiler
        if current_compiler is not None and not callable(current_compiler):
            raise TypeError("current_compiler must be callable or None")
        self._current_compiler = current_compiler

    @property
    def root_ids(self) -> tuple[str, ...]:
        return self._catalog.root_ids

    async def _compiler_for_resolution(self) -> AgentCompiler:
        compiler = (
            self._compiler
            if self._current_compiler is None
            else await self._current_compiler()
        )
        if not isinstance(compiler, AgentCompiler):
            raise TypeError("current_compiler must return AgentCompiler")
        return compiler

    async def resolve_agent(
        self,
        agent_id: str,
        derivations: Sequence[Mapping[str, object]] = (),
        output: "type[BaseModel] | OutputBinding | None" = None,
    ) -> AgentBinding:
        """Resolve one new admission from a single current source observation."""
        compiler = await self._compiler_for_resolution()
        compiled = compiler.compile_registered(agent_id, derivations)
        binding = compiler.bind(compiled, output=output)
        return await self._resolve_binding(binding, compiler)

    async def resolve(self, binding: AgentBinding) -> AgentBinding:
        """Resolve dependencies without reinterpreting an existing declaration."""
        if not isinstance(binding, AgentBinding):
            raise TypeError("binding must be AgentBinding")
        compiler = await self._compiler_for_resolution()
        return await self._resolve_binding(binding, compiler)

    async def _resolve_binding(
        self, binding: AgentBinding, compiler: AgentCompiler,
    ) -> AgentBinding:
        contract = await self._resolve_contract(
            binding.binding_contract, compiler,
        )
        # Resolution adds dependency facts; it does not replace admitted metadata
        # or Python output validation with a decoded copy of its durable schema.
        return binding if contract == binding.binding_contract else replace(
            binding, binding_contract=contract,
        )

    async def restore(
        self, contract: AgentBindingContract, *, model: ModelBinding | None = None,
    ) -> AgentBinding:
        """Keep admitted Agent facts and hooks while resolving the current directory."""
        compiler = await self._compiler_for_resolution()
        binding = compiler.restore(contract, model=model)
        return await self._resolve_binding(binding, compiler)

    async def resolve_root(self, agent_id: str) -> AgentBindingContract:
        """Resolve one current Agent as a root execution target and its children."""
        return (await self.resolve_agent(agent_id)).binding_contract

    async def resolve_contract(
        self,
        binding_contract: AgentBindingContract,
    ) -> AgentBindingContract:
        """Resolve declared dependencies while retaining admitted contract facts."""
        compiler = await self._compiler_for_resolution()
        return await self._resolve_contract(binding_contract, compiler)

    async def _resolve_contract(
        self,
        binding_contract: AgentBindingContract,
        compiler: AgentCompiler,
    ) -> AgentBindingContract:
        if not isinstance(binding_contract, AgentBindingContract):
            raise TypeError("binding_contract must be AgentBindingContract")
        if binding_contract.subagent_bindings:
            return binding_contract
        children = tuple(
            [await self._resolve_child(ref.id, ref.revision, compiler) for ref in binding_contract.subagents]
        )
        return replace(binding_contract, subagent_bindings=children)

    async def _resolve_child(
        self, agent_id: str, revision: int, compiler: AgentCompiler,
    ) -> AgentBindingContract:
        compiled = compiler.compile_registered(agent_id)
        if compiled.spec.revision != revision:
            raise AIError(ErrorCode.AGENT_BINDING_UNAVAILABLE, safe_details={"agent_id": agent_id})
        binding = compiler.bind_subagent(compiled)
        return binding.binding_contract


__all__ = ["_AgentBindingResolver"]
