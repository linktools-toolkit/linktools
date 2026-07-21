#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime: the top-level integration surface.
Runtime.build() assembles Storage + AgentCompiler + AgentEngine + SwarmRunner +
ModelRouter; Runtime.run(spec, prompt) compiles the spec, resolves (or creates)
a Session, mints a RunContext, and delegates to AgentEngine (AgentSpec) or
SwarmRunner (SwarmSpec).

Capability Runtime: build() accepts a RuntimeDependencies + CapabilityRuntimeOptions,
from which it builds a CapabilityAssembler wired into AgentEngine. Declared
AgentSpec.tools are then resolved into prompt sections + toolsets via the
registered capability providers. Runtime is an async context manager that
releases MCP connections on close.

Runtime keeps only build/inspect/close + MCP-lifecycle ownership; every
run-lifecycle method (run/run_stream/cancel/approve/reject/resume) delegates
to :class:`~linktools.ai.run.coordinator.RunCoordinator`, the single
application service that owns create/pause/approve/reject/resume/cancel/
commit/terminal-convergence."""

from typing import TYPE_CHECKING, Any, Mapping

# AsyncIterator is a typing-only alias used to annotate the streaming
# generator below; the function itself is an ``async def`` that ``yield``s.
if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from ._runtime.build import (
    RuntimeBuildConfig,
    RuntimeComponents,
    RuntimeSettings,
    build_runtime_components,
)
from ._runtime.dependencies import RuntimeDependencies
from .agent.spec import AgentSpec
from .capability.models import CapabilityRuntimeOptions
from .execution.protocols import ExecutionBackend
from .middleware.pipeline import MiddlewarePipeline
from .model.router import ModelRouter
from .mcp.client import MCPConnectionManager
from .observability.metrics import ObservabilityMetrics
from .run.commit import RunCommitCoordinator
from .run.coordinator import RunCoordinator
from .run.options import RuntimeCancellationOptions
from .run.requirements import RuntimeRequirements, RuntimeTopology
from .run.schema_registry import OutputSchemaRegistry
from .storage.facade import Storage
from .swarm.spec import SwarmSpec
from .tool.executor import GovernedToolInvoker

if TYPE_CHECKING:
    from .capability.models import CapabilityInspection
    from .retrieval.retriever import Retriever
    from .identity.principal import PrincipalContext
    from .subagent.config import SkillPrivateSubagentConfig


class Runtime:
    def __init__(
        self,
        *,
        components: RuntimeComponents,
    ) -> None:
        self._components = components
        self._coordinator = RunCoordinator(components)

    @classmethod
    def build(
        cls,
        *,
        storage: Storage,
        commit_coordinator: "RunCommitCoordinator | None" = None,
        topology: RuntimeTopology = RuntimeTopology.SINGLE_PROCESS,
        model_router: "ModelRouter | None" = None,
        middleware_pipeline: "MiddlewarePipeline | None" = None,
        retriever: "Retriever | None" = None,
        execution: "ExecutionBackend | None" = None,
        tool_executor: "GovernedToolInvoker | None" = None,
        mcp_connection_manager: "MCPConnectionManager | None" = None,
        providers: "RuntimeDependencies | None" = None,
        options: "CapabilityRuntimeOptions | None" = None,
        allow_mcp_wildcard: bool = False,
        security: Any = None,
        local_trusted_mode: bool = False,
        multi_tenant: bool = False,
        cancellation: "RuntimeCancellationOptions | None" = None,
        schema_registry: "OutputSchemaRegistry | None" = None,
        metrics: "ObservabilityMetrics | None" = None,
        authorization: Any = None,
        skill_subagent: "SkillPrivateSubagentConfig | None" = None,
        requirements: "RuntimeRequirements | None" = None,
    ) -> "Runtime":
        """Assemble a Runtime from optional sub-components + a RuntimeDependencies.

        Capability providers come exclusively via ``providers`` (a
        RuntimeDependencies); the direct ``Runtime.run(spec, ...)`` path stays the
        shortest and needs no providers configured.

        ``commit_coordinator`` (REQUIRED at the build-kernel level): the build
        kernel no longer selects a coordinator from Storage type -- the caller
        (the composition root) constructs the concrete coordinator for its
        Storage and injects it. ``None`` is accepted at this layer only so a
        caller that fails to inject one gets the build kernel's clear fail-fast
        error rather than a ``TypeError``; production code always passes a real
        coordinator.

        ``topology`` (default SINGLE_PROCESS) declares the shape of the
        process graph; the build kernel uses it to derive default capability
        minimums when ``requirements`` is not supplied. ``requirements``, when
        passed, takes precedence.

        ``local_trusted_mode`` (default False): when False, cancel /
        resume reject a missing ``principal`` (production-safe); when True the
        Runtime is explicitly single-tenant / local and a missing principal is
        allowed (with a deprecation warning)."""
        config = RuntimeBuildConfig(
            storage=storage,
            commit_coordinator=commit_coordinator,
            providers=providers or RuntimeDependencies(),
            model_router=model_router,
            middleware_pipeline=middleware_pipeline,
            retriever=retriever,
            execution=execution,
            tool_executor=tool_executor,
            security=security,
            capability_options=options,
            mcp_connection_manager=mcp_connection_manager,
            settings=RuntimeSettings(
                allow_mcp_wildcard=allow_mcp_wildcard,
                local_trusted_mode=local_trusted_mode,
                multi_tenant=multi_tenant,
                cancellation=cancellation or RuntimeCancellationOptions(),
                topology=topology,
            ),
            schema_registry=schema_registry,
            metrics=metrics,
            authorization=authorization,
            skill_subagent=skill_subagent,
            requirements=requirements,
        )
        c = build_runtime_components(config)
        return cls(components=c)

    async def inspect(self, spec: AgentSpec) -> "CapabilityInspection":
        """A stable, immutable view of what ``spec`` resolves to: the exposed
        tool descriptors, merged prompt sections, and any warnings. Leaks no
        mutable internal state (no handlers); a capability that degrades during
        resolution surfaces as a warning. See :mod:`linktools.ai._runtime.inspection`."""
        from ._runtime.inspection import inspect_capabilities

        return await inspect_capabilities(
            assembler=self._components.capability_assembler,
            options=self._components.options,
            spec=spec,
            execution=self._components.execution,
        )

    async def __aenter__(self) -> "Runtime":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Release runtime-owned resources (MCP connections). Idempotent."""
        if self._components.mcp_connection_manager is not None:
            await self._components.mcp_connection_manager.close()

    async def run(
        self,
        spec: "AgentSpec | SwarmSpec",
        prompt: str,
        *,
        session_id: "str | None" = None,
        run_id: "str | None" = None,
        user_id: "str | None" = None,
        tenant_id: "str | None" = None,
        agents: "Mapping[str, AgentSpec] | None" = None,
        context_metadata: "Mapping[str, Any] | None" = None,
    ):
        """Non-streaming run entry point. See :meth:`RunCoordinator.run`."""
        return await self._coordinator.run(
            spec,
            prompt,
            session_id=session_id,
            run_id=run_id,
            user_id=user_id,
            tenant_id=tenant_id,
            agents=agents,
            context_metadata=context_metadata,
        )

    async def cancel(
        self,
        run_id: str,
        *,
        principal: "PrincipalContext | None" = None,
        reason: "str | None" = None,
    ) -> None:
        """Cancel an in-flight Run. See :meth:`RunCoordinator.cancel`."""
        await self._coordinator.cancel(run_id, principal=principal, reason=reason)

    async def run_stream(
        self,
        spec: "AgentSpec | SwarmSpec",
        prompt: str,
        *,
        session_id: "str | None" = None,
        run_id: "str | None" = None,
        user_id: "str | None" = None,
        tenant_id: "str | None" = None,
        context_metadata: "Mapping[str, Any] | None" = None,
    ) -> "AsyncIterator[dict]":
        """Streaming variant of :meth:`run`. See :meth:`RunCoordinator.run_stream`."""
        async for event in self._coordinator.run_stream(
            spec,
            prompt,
            session_id=session_id,
            run_id=run_id,
            user_id=user_id,
            tenant_id=tenant_id,
            context_metadata=context_metadata,
        ):
            yield event

    async def approve(
        self,
        approval_id: str,
        *,
        principal: "PrincipalContext",
        expected_version: int,
    ):
        """Approve through the Principal-bound service. See :meth:`RunCoordinator.approve`."""
        return await self._coordinator.approve(
            approval_id, principal=principal, expected_version=expected_version
        )

    async def reject(
        self,
        approval_id: str,
        *,
        principal: "PrincipalContext",
        expected_version: int,
        reason: "str | None" = None,
    ):
        """Reject through the Principal-bound service. See :meth:`RunCoordinator.reject`."""
        return await self._coordinator.reject(
            approval_id,
            principal=principal,
            expected_version=expected_version,
            reason=reason,
        )

    async def resume(
        self,
        run_id: str,
        *,
        principal: "PrincipalContext | None" = None,
    ) -> "AsyncIterator[dict]":
        """Resume a paused Run from its persisted definition. See
        :meth:`RunCoordinator.resume` for the full 14-step protocol."""
        async for event in self._coordinator.resume(run_id, principal=principal):
            yield event


# re-export for tooling that imports Runtime alongside these types
__all__ = ["Runtime"]
