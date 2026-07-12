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

import uuid
from typing import TYPE_CHECKING, Any, Mapping

# AsyncIterator is a typing-only alias used to annotate the streaming
# generator below; the function itself is an ``async def`` that ``yield``s.
if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from ._runtime.build import RuntimeBuildConfig, build_runtime_components
from .agent.compiler import AgentCompiler
from .agent.runner import AgentRunner
from .agent.spec import AgentSpec
from .capability.assembler import CapabilityAssembler
from .capability.models import CapabilityRuntimeOptions
from .errors import SwarmError
from .execution.protocols import ExecutionBackend
from .middleware.pipeline import MiddlewarePipeline
from .model.router import ModelRouter
from .mcp.client import MCPConnectionManager
from .providers.bundle import ProviderBundle
from .run.controller import RunController
from .run.models import RunInput, RunnableType
from .storage.facade import Storage
from .swarm.runner import SwarmRunner
from .swarm.spec import SwarmSpec
from .tool.executor import ToolExecutor

if TYPE_CHECKING:
    from .capability.models import CapabilityInspection
    from .knowledge.retriever import Retriever


class Runtime:
    def __init__(
        self,
        *,
        storage: Storage,
        compiler: AgentCompiler,
        runner: AgentRunner,
        swarm_runner: SwarmRunner,
        model_router: ModelRouter,
        run_controller: "RunController | None" = None,
        capability_assembler: "CapabilityAssembler | None" = None,
        mcp_connection_manager: "MCPConnectionManager | None" = None,
        options: "CapabilityRuntimeOptions | None" = None,
        provider_bundle: "ProviderBundle | None" = None,
    ) -> None:
        self.storage = storage
        self.compiler = compiler
        self.runner = runner
        self.swarm_runner = swarm_runner
        self.model_router = model_router
        # Tracks in-flight asyncio.Tasks so cancel(run_id)
        # can actually stop a running task, not just flip the DB status.
        self.run_controller = run_controller
        # Capability Runtime wiring. When non-None, AgentRunner uses
        # this assembler to resolve declared AgentSpec.tools into toolsets.
        self._capability_assembler = capability_assembler
        self._mcp_connection_manager = mcp_connection_manager
        self._options = options or CapabilityRuntimeOptions()
        # The declaration bundle wired at build time; held so the runtime can
        # resolve capabilities against the configured providers.
        self._provider_bundle = provider_bundle

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
        return cls(
            storage=c.storage,
            compiler=c.compiler,
            runner=c.runner,
            swarm_runner=c.swarm_runner,
            model_router=c.model_router,
            run_controller=c.run_controller,
            capability_assembler=c.capability_assembler,
            mcp_connection_manager=c.mcp_connection_manager,
            options=c.options,
            provider_bundle=c.provider_bundle,
        )

    async def inspect(
        self, spec: AgentSpec, *, execution: "ExecutionBackend | None"
    ) -> "CapabilityInspection":
        """A stable, immutable view of what ``spec`` resolves to: the exposed
        tool descriptors, merged prompt sections, and any warnings. Leaks no
        mutable internal state (no handlers); a capability that degrades during
        resolution surfaces as a warning. See :mod:`linktools.ai._runtime.inspection`."""
        from ._runtime.inspection import inspect_capabilities

        return await inspect_capabilities(
            assembler=self._capability_assembler,
            options=self._options,
            spec=spec,
            execution=execution,
        )

    async def __aenter__(self) -> "Runtime":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Release runtime-owned resources (MCP connections). Idempotent."""
        if self._mcp_connection_manager is not None:
            await self._mcp_connection_manager.close()
            self._mcp_connection_manager = None

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
        from ._runtime.lifecycle import create_run_context, resolve_session

        resolved_session_id = await resolve_session(self.storage, session_id)
        resolved_run_id = run_id or str(uuid.uuid4())

        if isinstance(spec, SwarmSpec):
            if agents is None:
                raise SwarmError("agents mapping is required to run a SwarmSpec")
            context = create_run_context(
                run_id=resolved_run_id,
                session_id=resolved_session_id,
                runnable_id=spec.id,
                runnable_type=RunnableType.SWARM,
                user_id=user_id,
                tenant_id=tenant_id,
            )
            return await self.swarm_runner.run(
                spec,
                RunInput(prompt=prompt),
                context,
                agents=agents,
            )

        compiled = await self.compiler.compile(spec)
        context = create_run_context(
            run_id=resolved_run_id,
            session_id=resolved_session_id,
            runnable_id=spec.id,
            runnable_type=RunnableType.AGENT,
            user_id=user_id,
            tenant_id=tenant_id,
        )
        return await self.runner.run(compiled, RunInput(prompt=prompt), context)

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

        record = await self.storage.runs.get(run_id)
        if record is None:
            raise RunNotFoundError(f"run not found: {run_id}")
        if record.status in (
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        ):
            return

        in_flight = (
            self.run_controller is not None
            and self.run_controller.get_token(run_id) is not None
        )
        if in_flight:
            if record.status == RunStatus.CANCELLING:
                await self.run_controller.cancel(run_id)
                return

            try:
                await self.storage.runs.transition(
                    run_id,
                    RunStatus.CANCELLING,
                    expected_version=record.version,
                )
            except RunConflictError:
                fresh = await self.storage.runs.get(run_id)
                if fresh is None:
                    raise RunNotFoundError(f"run not found: {run_id}")
                if fresh.status in (
                    RunStatus.SUCCEEDED,
                    RunStatus.FAILED,
                    RunStatus.CANCELLED,
                ):
                    return
                if fresh.status == RunStatus.CANCELLING:
                    await self.run_controller.cancel(run_id)
                    return
                raise

            await self.run_controller.cancel(run_id)
        else:
            await self.storage.runs.transition(
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
        from ._runtime.lifecycle import create_run_context, resolve_session

        resolved_session_id = await resolve_session(self.storage, session_id)
        resolved_run_id = run_id or str(uuid.uuid4())

        if isinstance(spec, SwarmSpec):
            raise SwarmError("run_stream does not support SwarmSpec")

        compiled = await self.compiler.compile(spec)
        context = create_run_context(
            run_id=resolved_run_id,
            session_id=resolved_session_id,
            runnable_id=spec.id,
            runnable_type=RunnableType.AGENT,
            user_id=user_id,
            tenant_id=tenant_id,
        )
        async for event in self.runner.run_stream(
            compiled, RunInput(prompt=prompt), context
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

        record = await self.storage.runs.get(run_id)
        if record is None:
            raise RunNotFoundError(f"run not found: {run_id}")
        if record.status != RunStatus.WAITING_APPROVAL:
            raise InvalidRunTransitionError(
                f"cannot resume run in status {record.status}"
            )
        checkpoint = await self.storage.checkpoints.latest(run_id)
        if checkpoint is None:
            raise RunNotFoundError(f"no checkpoint for run: {run_id}")
        messages = deserialize_messages(checkpoint.payload)
        await self.storage.runs.transition(
            run_id,
            RunStatus.RUNNING,
            expected_version=record.version,
        )
        compiled = await self.compiler.compile(spec)
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
        async for event in self.runner.run_stream(
            compiled,
            RunInput(prompt=""),
            context,
            message_history=messages,
        ):
            yield event


# re-export for tooling that imports Runtime alongside these types
__all__ = ["Runtime"]
