#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime: the top-level integration surface.
Runtime.build() assembles Storage + AgentCompiler + AgentRunner + SwarmRunner +
ModelRouter; Runtime.run(spec, prompt) compiles the spec, resolves (or creates)
a Session, mints a RunContext, and delegates to AgentRunner (AgentSpec) or
SwarmRunner (SwarmSpec).

Capability Runtime: build() accepts a ProviderBundle + CapabilityRuntimeOptions,
from which it builds a CapabilityAssembler wired into AgentRunner. Declared
AgentSpec.tools are then resolved into prompt sections + toolsets via the
registered capability providers. Runtime is an async context manager that
releases MCP connections on close."""

from typing import TYPE_CHECKING, Any, Mapping

# AsyncIterator is a typing-only alias used to annotate the streaming
# generator below; the function itself is an ``async def`` that ``yield``s.
if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from ._runtime.build import (
    RuntimeBuildConfig,
    RuntimeComponents,
    build_runtime_components,
)
from .agent.spec import AgentSpec
from .capability.models import CapabilityRuntimeOptions
from .errors import SwarmError
from .execution.protocols import ExecutionBackend
from .middleware.pipeline import MiddlewarePipeline
from .model.router import ModelRouter
from .mcp.client import MCPConnectionManager
from .providers.bundle import ProviderBundle
from .run.models import RunInput
from .storage.facade import Storage
from .swarm.spec import SwarmSpec
from .tool.executor import ToolExecutor

if TYPE_CHECKING:
    from .capability.models import CapabilityInspection
    from .knowledge.retriever import Retriever


class Runtime:
    def __init__(
        self,
        *,
        components: RuntimeComponents,
    ) -> None:
        self._components = components

    @classmethod
    def build(
        cls,
        *,
        storage: Storage,
        model_router: "ModelRouter | None" = None,
        middleware_pipeline: "MiddlewarePipeline | None" = None,
        retriever: "Retriever | None" = None,
        execution: "ExecutionBackend | None" = None,
        tool_executor: "ToolExecutor | None" = None,
        pause_on_approval: bool = False,
        mcp_connection_manager: "MCPConnectionManager | None" = None,
        providers: "ProviderBundle | None" = None,
        options: "CapabilityRuntimeOptions | None" = None,
        allow_mcp_wildcard: bool = False,
        security: Any = None,
    ) -> "Runtime":
        """Assemble a Runtime from optional sub-components + a ProviderBundle.

        Capability providers come exclusively via ``providers`` (a
        ProviderBundle); the direct ``Runtime.run(spec, ...)`` path stays the
        shortest and needs no providers configured."""
        config = RuntimeBuildConfig(
            storage=storage,
            providers=providers or ProviderBundle(),
            model_router=model_router,
            middleware_pipeline=middleware_pipeline,
            retriever=retriever,
            execution=execution,
            tool_executor=tool_executor,
            security=security,
            capability_options=options,
            pause_on_approval=pause_on_approval,
            allow_mcp_wildcard=allow_mcp_wildcard,
            mcp_connection_manager=mcp_connection_manager,
        )
        c = build_runtime_components(config)
        return cls(components=c)

    async def inspect(
        self, spec: AgentSpec
    ) -> "CapabilityInspection":
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
    ):
        from .run.lifecycle import prepare_run

        prepared = await prepare_run(
            storage=self._components.storage, spec=spec, session_id=session_id,
            run_id=run_id, user_id=user_id, tenant_id=tenant_id,
        )

        if isinstance(spec, SwarmSpec):
            if agents is None:
                raise SwarmError("agents mapping is required to run a SwarmSpec")
            return await self._components.swarm_runner.run(
                spec, RunInput(prompt=prompt), prepared.context, agents=agents
            )

        compiled = await self._components.compiler.compile(spec)
        return await self._components.runner.run(
            compiled, RunInput(prompt=prompt), prepared.context
        )

    async def cancel(self, run_id: str) -> None:
        """Cancel an in-flight Run.

        Two paths, depending on whether a live asyncio.Task is registered with
        the RunController:

        * **In-flight task registered** -- the run is actually being driven by
          AgentRunner.execute(). Transition the store to CANCELLING
          (distinguishes "cancel requested" from "actually cancelled"), then
          call ``run_controller.cancel(run_id)`` which (a) sets the
          CancellationToken so the runner's next execution-point check raises
          CancelledError, and (b) calls ``task.cancel()`` so any hanging await
          inside the model call also unblocks.

          If the record is already CANCELLING, skip straight to
          ``run_controller.cancel(run_id)`` -- re-transitioning is not a legal
          edge, so repeated cancel() calls are idempotent.

        * **No in-flight task** -- there is nothing to actually stop, so the
          store goes directly to CANCELLED.

        Idempotent: a Run already in a terminal status is a no-op. Raises
        :class:`RunNotFoundError` when the run does not exist."""
        from .errors import RunConflictError, RunNotFoundError
        from .run.models import RunStatus

        storage = self._components.storage
        controller = self._components.run_controller
        record = await storage.runs.get(run_id)
        if record is None:
            raise RunNotFoundError(f"run not found: {run_id}")
        if record.status in (
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        ):
            return

        in_flight = (
            controller is not None and controller.get_token(run_id) is not None
        )
        if in_flight:
            if record.status == RunStatus.CANCELLING:
                await controller.cancel(run_id)
                return

            try:
                await storage.runs.transition(
                    run_id,
                    RunStatus.CANCELLING,
                    expected_version=record.version,
                )
            except RunConflictError:
                fresh = await storage.runs.get(run_id)
                if fresh is None:
                    raise RunNotFoundError(f"run not found: {run_id}")
                if fresh.status in (
                    RunStatus.SUCCEEDED,
                    RunStatus.FAILED,
                    RunStatus.CANCELLED,
                ):
                    return
                if fresh.status == RunStatus.CANCELLING:
                    await controller.cancel(run_id)
                    return
                raise

            await controller.cancel(run_id)
        else:
            await storage.runs.transition(
                run_id,
                RunStatus.CANCELLED,
                expected_version=record.version,
            )

    async def run_stream(
        self,
        spec: "AgentSpec | SwarmSpec",
        prompt: str,
        *,
        session_id: "str | None" = None,
        run_id: "str | None" = None,
        user_id: "str | None" = None,
        tenant_id: "str | None" = None,
    ) -> "AsyncIterator[dict]":
        """Streaming variant of :meth:`run`. Only ``AgentSpec`` is supported --
        a ``SwarmSpec`` raises :class:`SwarmError` because swarm streaming is not
        implemented. Session resolution mirrors :meth:`run` exactly."""
        from .run.lifecycle import prepare_run

        if isinstance(spec, SwarmSpec):
            raise SwarmError("run_stream does not support SwarmSpec")

        prepared = await prepare_run(
            storage=self._components.storage, spec=spec, session_id=session_id,
            run_id=run_id, user_id=user_id, tenant_id=tenant_id,
        )
        compiled = await self._components.compiler.compile(spec)
        async for event in self._components.runner.run_stream(
            compiled, RunInput(prompt=prompt), prepared.context
        ):
            yield event

    async def resume(
        self,
        run_id: str,
        spec: "AgentSpec",
        *,
        user_id: "str | None" = None,
        tenant_id: "str | None" = None,
    ) -> "AsyncIterator[dict]":
        """Resume a paused Run. Loads the paused RunRecord,
        deserializes its checkpoint's message history, transitions
        WAITING_APPROVAL -> RUNNING, and re-enters :meth:`AgentRunner.run_stream`.

        Yields ``{"type": "resumed", "run_id": run_id}`` first, then the same
        dict-event shape ``run_stream`` yields. Raises :class:`RunNotFoundError`
        when the run/checkpoint does not exist; :class:`InvalidRunTransitionError`
        when the run is not WAITING_APPROVAL."""
        from .agent.checkpoint import deserialize_messages
        from .errors import InvalidRunTransitionError, RunNotFoundError
        from .run.models import RunStatus

        storage = self._components.storage
        record = await storage.runs.get(run_id)
        if record is None:
            raise RunNotFoundError(f"run not found: {run_id}")
        if record.status != RunStatus.WAITING_APPROVAL:
            raise InvalidRunTransitionError(
                f"cannot resume run in status {record.status}"
            )
        checkpoint = await storage.checkpoints.latest(run_id)
        if checkpoint is None:
            raise RunNotFoundError(f"no checkpoint for run: {run_id}")
        messages = deserialize_messages(checkpoint.payload)
        await storage.runs.transition(
            run_id,
            RunStatus.RUNNING,
            expected_version=record.version,
        )
        compiled = await self._components.compiler.compile(spec)
        from ._runtime.lifecycle import create_run_context

        context = create_run_context(
            run_id=run_id,
            session_id=record.session_id,
            runnable_id=record.runnable_id,
            runnable_type=record.runnable_type,
            user_id=user_id,
            tenant_id=tenant_id,
            root_run_id=record.root_run_id,
            parent_run_id=record.parent_run_id,
        )
        yield {"type": "resumed", "run_id": run_id}
        async for event in self._components.runner.run_stream(
            compiled,
            RunInput(prompt=""),
            context,
            message_history=messages,
        ):
            yield event


# re-export for tooling that imports Runtime alongside these types
__all__ = ["Runtime"]
