#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve current contributions through native model-step toolsets."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import TracebackType
from typing import Any, TYPE_CHECKING

import anyio
from pydantic_ai.messages import InstructionPart
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset, DynamicToolset, FunctionToolset, ToolsetTool
from pydantic_ai.tools import RunContext

from ..agent import CompiledAgent
from ..asset import AssetMaterializer, AssetStoreReader
from ..capability import AgentContext, CapabilityContribution, tool_class_from_metadata, tool_effect_policy_from_metadata
from ..core import ToolEffects
from ..errors import AIError, ErrorCode
from ..spec import CapabilityRef, MCPServerSpecCodec
from ._budget import RunBudgetContext
from ._mcp import (
    _MCPBinding, _mcp_execution_policy, close_mcp_resources,
    materialize_mcp_capabilities, prepare_mcp_projections,
)
from ._tool import ToolOperationBridge
from ._tool_boundary import BoundaryToolset, managed_tool_descriptor_from_metadata, require_tool_effects
from ._tool_metrics import ToolMetricContext
from ._tool_response_fixture import _ToolResponseManifest

if TYPE_CHECKING:
    from pydantic_ai.capabilities import AbstractCapability
    from pydantic_ai.workspaces import WorkspaceBackend
    from ..capability import RunCallbacks


def create_current_toolset(
    provider: Callable[[], Awaitable[CompiledAgent]],
    *,
    asset_readers: Mapping[str, AssetStoreReader],
    workspace_backend: WorkspaceBackend | None,
    tool_operations: ToolOperationBridge | None,
    tool_metrics: ToolMetricContext | None = None,
    budget: RunBudgetContext | None = None,
    tool_effects: ToolEffects = "live",
    response_fixture: _ToolResponseManifest | None = None,
    result_callbacks: RunCallbacks | None = None,
    reserved_tool_names: Sequence[str] = (),
    register_cleanup: Callable[[Callable[[bool], Awaitable[None]]], None] | None = None,
    workspace_provider: Callable[[], Awaitable[WorkspaceBackend | None]] | None = None,
) -> AbstractToolset[AgentContext[object]]:
    """Use one current provider and register one run-owned MCP cleanup callback.

    The callback accepts whether the run's workspace has successfully closed.
    This proof allows disposal of materializations even when client cleanup fails.
    """
    return _CurrentToolsets(
        provider, asset_readers, workspace_backend, tool_operations, tool_metrics, budget,
        response_fixture, result_callbacks, tuple(reserved_tool_names), register_cleanup, workspace_provider,
        tool_effects,
    )


@dataclass
class _CurrentToolsets(AbstractToolset[AgentContext[object]]):
    provider: Callable[[], Awaitable[CompiledAgent]]
    asset_readers: Mapping[str, AssetStoreReader]
    workspace_backend: WorkspaceBackend | None
    tool_operations: ToolOperationBridge | None
    tool_metrics: ToolMetricContext | None
    budget: RunBudgetContext | None
    response_fixture: _ToolResponseManifest | None
    result_callbacks: RunCallbacks | None
    reserved_tool_names: tuple[str, ...]
    register_cleanup: Callable[[Callable[[bool], Awaitable[None]]], None] | None
    workspace_provider: Callable[[], Awaitable[WorkspaceBackend | None]] | None
    tool_effects: ToolEffects
    _compiled: CompiledAgent | None = field(default=None, init=False)
    _owned: _MCPToolsetScope | None = field(default=None, init=False)
    _candidate: _MCPToolsetScope | None = field(default=None, init=False)
    _inner: AbstractToolset[AgentContext[object]] = field(init=False)

    def __post_init__(self) -> None:
        self._inner = CombinedToolset((
            DynamicToolset(self.business_tools, per_run_step=True, id="linktools.current-business"),
            DynamicToolset(self.mcp_tools, per_run_step=True, id="linktools.current-mcp"),
        ))

    @property
    def id(self) -> str:
        return "linktools.current-tools"

    async def for_run(self, ctx: RunContext[AgentContext[object]]) -> _CurrentToolsets:
        bound = replace(self)
        bound._inner = await bound._inner.for_run(ctx)
        if bound.register_cleanup is not None:
            bound.register_cleanup(bound.close_resources)
        return bound

    async def for_run_step(self, ctx: RunContext[AgentContext[object]]) -> _CurrentToolsets:
        self._compiled = await self.provider()
        if self.tool_effects != "live":
            for candidate in self._compiled.selected_mcp:
                require_tool_effects(self.tool_effects, kind=candidate.kind, effect_policy=None)
            for candidate in self._compiled.selected_tools:
                require_tool_effects(
                    self.tool_effects, kind=candidate.kind,
                    effect_policy=tool_effect_policy_from_metadata(candidate.contract.get("metadata"), require=True),
                )
        if (self.workspace_provider is not None and self.response_fixture is None
                and any(server.transport == "stdio" for server in self._compiled.mcp_servers)):
            self.workspace_backend = await self.workspace_provider()
        self._inner = await self._inner.for_run_step(ctx)
        return self

    async def business_tools(self, ctx: RunContext[AgentContext[object]]) -> AbstractToolset[AgentContext[object]] | None:
        assert self._compiled is not None
        tools, descriptors = [], {}
        for candidate in self._compiled.selected_tools:
            metadata = candidate.contract.get("metadata")
            if not isinstance(metadata, Mapping):
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            tool_class = tool_class_from_metadata(metadata)
            if tool_class in {"memory", "filesystem.read", "filesystem.write", "shell"}:
                continue
            if tool_class != "business" or candidate.id in self.reserved_tool_names:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            tools.append(candidate.materialize_tool())
            descriptors[candidate.id] = replace(
                managed_tool_descriptor_from_metadata(metadata),
                capability_ref=CapabilityRef("tool", candidate.id, candidate.revision),
            )
        if not tools:
            return None
        boundary = BoundaryToolset(
            (FunctionToolset(tools, id="linktools.business"),), descriptors, id="linktools.business",
            tool_operations=self.tool_operations, tool_metrics=self.tool_metrics, budget=self.budget,
            tool_effects=self.tool_effects,
            result_callbacks=self.result_callbacks,
        )
        bound = await boundary.for_run(ctx)
        return await bound.for_run_step(ctx)

    async def mcp_tools(self, ctx: RunContext[AgentContext[object]]) -> AbstractToolset[AgentContext[object]] | None:
        assert self._compiled is not None
        if not self._compiled.mcp_servers:
            return None
        if self._candidate is not None and self._candidate.can_reuse(self._compiled, self.workspace_backend):
            return await self._candidate.for_run_step(ctx)
        self._candidate = _MCPToolsetScope(self, self._compiled, ctx, self.workspace_backend)
        return self._candidate

    def claim_resources(self, scope: _MCPToolsetScope) -> None:
        if self._owned is not None:
            raise AIError(ErrorCode.MCP_CLEANUP_FAILED, safe_details={"reason": "previous_resources_unclosed"})
        self._owned = scope

    def release_resources(self, scope: _MCPToolsetScope) -> None:
        if self._owned is scope:
            self._owned = None

    async def close_resources(self, workspace_closed: bool = False) -> None:
        if self._owned is not None:
            await self._owned.close_resources(workspace_closed=workspace_closed)

    async def __aenter__(self) -> _CurrentToolsets:
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        try:
            return await self._inner.__aexit__(*args)
        except BaseException as error:
            if args[1] is not None and args[1] is not error:
                raise args[1] from error
            raise

    async def get_tools(self, ctx: RunContext[AgentContext[object]]) -> dict[str, ToolsetTool[AgentContext[object]]]:
        return await self._inner.get_tools(ctx)

    async def get_instructions(self, ctx: RunContext[AgentContext[object]]) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        return await self._inner.get_instructions(ctx)

    async def call_tool(self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentContext[object]], tool: ToolsetTool[AgentContext[object]]) -> Any:
        return await self._inner.call_tool(name, tool_args, ctx, tool)


def _candidate_mcp_binding(
    candidate: CapabilityContribution[object], workspace_backend: WorkspaceBackend | None,
    response_fixture: _ToolResponseManifest | None,
) -> _MCPBinding:
    server, contract = candidate.value, candidate.contract
    codec = MCPServerSpecCodec()
    versions = codec.decode_binding_payload(contract, declaration=server)
    source_id = contract.get("asset_source_id")
    if source_id is not None and not isinstance(source_id, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    current_policy = (
        {"version": 1, "boundary": "host-network"} if response_fixture is not None
        else dict(_mcp_execution_policy(server, workspace_backend))
    )
    retained_policy = contract.get("execution_policy")
    if retained_policy is None:
        policy = {**current_policy, "tool_effect_policy": "replay_safe"}
    elif isinstance(retained_policy, Mapping):
        policy = codec.normalize_execution_policy(retained_policy)
        if (response_fixture is None and
                {key: value for key, value in policy.items() if key != "tool_effect_policy"} != current_policy):
            raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
    else:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return _MCPBinding(versions, source_id, policy)


class _MCPToolsetScope(AbstractToolset[AgentContext[object]]):
    """Native MCP context with one run-owned fallback for failed cleanup."""

    def __init__(self, owner: _CurrentToolsets, compiled: CompiledAgent, ctx: RunContext[AgentContext[object]], workspace_backend: WorkspaceBackend | None) -> None:
        self._owner, self._compiled, self._ctx, self._workspace_backend = owner, compiled, ctx, workspace_backend
        self._inner: AbstractToolset[AgentContext[object]] | None = None
        self._mcp: tuple[AbstractCapability[AgentContext[object]], ...] = ()
        self._materializer: AssetMaterializer | None = None

    @property
    def id(self) -> str:
        return "linktools.current-mcp-scope"

    def can_reuse(self, compiled: CompiledAgent, workspace_backend: WorkspaceBackend | None) -> bool:
        if (self._inner is None or self._workspace_backend is not workspace_backend
                or self._compiled.mcp_policy != compiled.mcp_policy):
            return False
        previous, current = self._compiled.selected_mcp, compiled.selected_mcp
        codec = MCPServerSpecCodec()
        # Transport equality only decides whether an entered connection can be reused.
        return len(previous) == len(current) and all(
            old.contract == new.contract and codec.to_wire_payload(old.value) == codec.to_wire_payload(new.value)
            for old, new in zip(previous, current)
        )

    async def for_run_step(self, ctx: RunContext[AgentContext[object]]) -> _MCPToolsetScope:
        if self._inner is not None:
            self._inner = await self._inner.for_run_step(ctx)
        return self

    async def __aenter__(self) -> _MCPToolsetScope:
        self._owner.claim_resources(self)
        try:
            self._materializer = AssetMaterializer()
            owner = self._owner
            bindings = {candidate.id: _candidate_mcp_binding(candidate, self._workspace_backend, owner.response_fixture)
                        for candidate in self._compiled.selected_mcp}
            projections = await prepare_mcp_projections(
                self._compiled.mcp_servers if owner.response_fixture is None else (), bindings,
                asset_readers=owner.asset_readers, sandboxed=self._workspace_backend is not None,
                materializer=self._materializer,
            )
            self._mcp = await materialize_mcp_capabilities(
                self._compiled.mcp_servers, self._compiled.mcp_policy,
                workspace_backend=self._workspace_backend, host_cwd=None, bindings=bindings, projections=projections,
                tool_operations=owner.tool_operations, tool_metrics=owner.tool_metrics, budget=owner.budget,
                tool_effects=owner.tool_effects,
                response_fixture=owner.response_fixture, result_callbacks=owner.result_callbacks,
            )
            inner = await CombinedToolset(tuple(capability.get_toolset() for capability in self._mcp)).for_run(self._ctx)
            self._inner = await inner.for_run_step(self._ctx)
            await self._inner.__aenter__()
        except BaseException as primary:
            await self.__aexit__(type(primary), primary, primary.__traceback__)
            raise
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        primary = exc_value
        if self._inner is not None:
            inner, self._inner = self._inner, None
            try:
                await inner.__aexit__(exc_type, exc_value, traceback)
            except BaseException as error:
                if primary is None:
                    primary = error
                elif error is not primary:
                    primary.__cause__ = error
        try:
            await self.close_resources()
        except BaseException as error:
            if primary is not None and primary is not error:
                raise primary from error
            raise
        if primary is not None and primary is not exc_value:
            raise primary

    async def close_resources(self, *, workspace_closed: bool = False) -> None:
        task = asyncio.create_task(self._close_resources(workspace_closed=workspace_closed))
        cancelled = None
        with anyio.CancelScope(shield=True):
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError as error:
                    cancelled = error
                except BaseException:
                    break
        try:
            task.result()
        except BaseException as error:
            if cancelled is not None:
                raise cancelled from error
            raise
        if cancelled is not None:
            raise cancelled

    async def _close_resources(self, *, workspace_closed: bool) -> None:
        failure: BaseException | None = None
        try:
            await close_mcp_resources(self._mcp)
        except BaseException as error:
            failure = error
        else:
            self._mcp = ()
        if failure is None or workspace_closed:
            try:
                if self._materializer is not None:
                    await self._materializer.close()
                    self._materializer = None
            except BaseException as error:
                if failure is not None:
                    raise failure from error
                raise
        if self._materializer is None and not self._mcp:
            self._owner.release_resources(self)
        if failure is not None:
            raise failure

    async def get_tools(self, ctx: RunContext[AgentContext[object]]) -> dict[str, ToolsetTool[AgentContext[object]]]:
        if self._inner is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await self._inner.get_tools(ctx)

    async def get_instructions(self, ctx: RunContext[AgentContext[object]]) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        return None if self._inner is None else await self._inner.get_instructions(ctx)

    async def call_tool(self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentContext[object]], tool: ToolsetTool[AgentContext[object]]) -> Any:
        if self._inner is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await self._inner.call_tool(name, tool_args, ctx, tool)


__all__ = ["create_current_toolset"]
