#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime: the top-level integration surface.
Runtime.build() assembles Storage + AgentCompiler + AgentRunner + SwarmRunner +
ModelRouter; Runtime.run(spec, prompt) compiles the spec, resolves (or creates)
a Session, mints a RunContext, and delegates to AgentRunner (AgentSpec) or
SwarmRunner (SwarmSpec).

Capability Runtime: build() also accepts spec Providers (a
ProviderBundle or the expanded agent/skill/mcp/... params) + CapabilityRuntimeOptions,
from which it builds a CapabilityAssembler wired into AgentRunner. Declared
AgentSpec.tools are then resolved into prompt sections + toolsets via the
registered capability providers; an unconfigured spec keeps the legacy default
builtin toolset behavior. Runtime is an async context manager that releases MCP
connections on close."""

import asyncio
import contextlib
import dataclasses
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping

# AsyncIterator is a typing-only alias used to annotate the streaming
# generator below; the function itself is an ``async def`` that ``yield``s.
if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from .agent.compiler import AgentCompiler
from .agent.runner import AgentRunner
from .agent.spec import AgentSpec
from .capability.assembler import CapabilityAssembler
from .capability.builtin import BuiltinProvider
from .capability.options import CapabilityRuntimeOptions
from .errors import SessionError, SwarmError
from .execution.protocols import ExecutionBackend
from .middleware.pipeline import MiddlewarePipeline
from .model.router import ModelRouter
from .mcp.client import MCPConnectionManager
from .mcp.provider import MCPProvider
from .package.capability_provider import PackageProvider
from .package.resolver import EntrypointResolver
from .providers.bundle import ProviderBundle
from .providers.mcp import MCPServerSpecProvider
from .providers.subagent import SubagentSpecProvider
from .run.context import RunContext
from .run.controller import RunController
from .run.models import RunInput, RunnableType
from .session.models import SessionRecord, SessionStatus
from .skill.provider import SkillProvider
from .storage.facade import Storage
from .subagent.models import SubagentResult
from .subagent.provider import SubagentProvider
from .swarm.runner import SwarmRunner
from .swarm.spec import SwarmSpec
from .tool.executor import ToolExecutor

if TYPE_CHECKING:
    from .knowledge.retriever import Retriever


def _build_capability_providers(
    bundle: ProviderBundle,
    execution: "ExecutionBackend | None",
    options: CapabilityRuntimeOptions,
    mcp_manager: "MCPConnectionManager | None",
    subagent_executor: Any = None,
) -> "dict[str, Any]":
    """Map the declaration bundle onto the kind -> CapabilityProvider dict the
    assembler consumes. Builtin is registered only when an execution backend
    exists (it cannot resolve without one). The subagent executor is passed in
    so both SubagentProvider and PackageProvider receive it at construction."""
    providers: "dict[str, Any]" = {}
    if execution is not None:
        providers["builtin"] = BuiltinProvider()
    if bundle.skills is not None:
        providers["skill"] = SkillProvider(bundle.skills)
    if bundle.mcp_servers is not None:
        providers["mcp"] = MCPProvider(
            bundle.mcp_servers, mcp_manager,
            allow_mcp_wildcard=bool(options.allow_mcp_wildcard),
        )
    if bundle.entrypoints is not None or bundle.subagents is not None:
        providers["subagent"] = SubagentProvider(
            subagent_provider=bundle.subagents,
            entrypoint_resolver=bundle.entrypoints,
            executor=subagent_executor,
        )
    if bundle.package_resources is not None or bundle.entrypoints is not None:
        # PackageProvider handles three tool-ref kinds; one instance registered
        # under each so package-resource / package-entrypoint refs resolve.
        pkg = PackageProvider(
            resource_provider=bundle.package_resources,
            entrypoint_resolver=bundle.entrypoints,
            entrypoint_executor=subagent_executor,
        )
        for k in ("package", "package-resource", "package-entrypoint"):
            providers[k] = pkg
    return providers


class Runtime:
    def __init__(self, *, storage: Storage, compiler: AgentCompiler,
                 runner: AgentRunner, swarm_runner: SwarmRunner,
                 model_router: ModelRouter,
                 run_controller: "RunController | None" = None,
                 capability_assembler: "CapabilityAssembler | None" = None,
                 mcp_connection_manager: "MCPConnectionManager | None" = None,
                 options: "CapabilityRuntimeOptions | None" = None,
                 provider_bundle: "ProviderBundle | None" = None) -> None:
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
        # The declaration bundle: consumed by resolve_swarm /
        # resolve_agent for by-id lookups so the swarm/agent providers are not
        # dead inputs.
        self._provider_bundle = provider_bundle

    @classmethod
    def build(cls, *, storage: Storage,
              model_router: "ModelRouter | None" = None,
              middleware_pipeline: "MiddlewarePipeline | None" = None,
              retriever: "Retriever | None" = None,
              execution: "ExecutionBackend | None" = None,
              tool_executor: "ToolExecutor | None" = None,
              pause_on_approval: bool = False,
              agents: "Any | None" = None,
              skills: "Any | None" = None,
              mcp_servers: "MCPServerSpecProvider | None" = None,
              tool_policies: "Any | None" = None,
              swarms: "Any | None" = None,
              subagents: "SubagentSpecProvider | None" = None,
              packages: "Any | None" = None,
              package_resources: "Any | None" = None,
              providers: "ProviderBundle | None" = None,
              options: "CapabilityRuntimeOptions | None" = None,
              allow_mcp_wildcard: bool = False,
              security: Any = None) -> "Runtime":
        """Assemble a Runtime from optional sub-components + capability providers.

        Capability providers may be passed either as a ``ProviderBundle`` or as
        the expanded ``agents``/``skills``/``mcp_servers``/... params -- never
        both (mixing raises ValueError to avoid silent override).
        All provider params are optional; the direct ``Runtime.run(spec, ...)``
        path stays the shortest and needs no providers configured."""
        expanded = (agents, skills, mcp_servers, tool_policies, swarms,
                    subagents, packages, package_resources)
        if providers is not None and any(v is not None for v in expanded):
            raise ValueError(
                "pass either `providers=...` or the expanded provider params "
                "(agents/skills/mcp_servers/...), not both"
            )
        resolved_options = options or CapabilityRuntimeOptions()
        # The build-level allow_mcp_wildcard flag is the documented mcp:* opt-in
        # Fold it into the options so MCPProvider honors it.
        if allow_mcp_wildcard and not resolved_options.allow_mcp_wildcard:
            resolved_options = dataclasses.replace(resolved_options, allow_mcp_wildcard=True)

        bundle = providers if providers is not None else ProviderBundle(
            agents=agents, skills=skills, mcp_servers=mcp_servers,
            tool_policies=tool_policies, swarms=swarms, subagents=subagents,
            packages=packages, package_resources=package_resources,
        )

        router = model_router or ModelRouter()
        resolved_executor = tool_executor
        if tool_executor is None and pause_on_approval:
            from .policy.engine import PolicyEngine
            from .security.baseline import SecurityBaseline

            baseline = security if security is not None else SecurityBaseline()
            rules: "list[Any]" = []
            if baseline.enabled and baseline.command_policy is not None:
                from .policy.command import CommandRule
                rules.append(CommandRule(
                    denied_patterns=baseline.command_policy.denied_patterns))
            resolved_executor = ToolExecutor(
                policy=PolicyEngine(rules=tuple(rules)),
                approval_store=storage.approvals,
                pause_on_approval=True,
            )
        compiler = AgentCompiler(
            model_router=router,
            middleware_pipeline=middleware_pipeline,
            tool_executor=resolved_executor,
        )
        uow_factory = (
            storage.transaction
            if storage.capabilities.cross_store_transactions else None
        )
        run_controller = RunController()
        # Resolve the SecurityBaseline + pipeline for the runner.
        from .security.baseline import SecurityBaseline
        baseline = security if security is not None else SecurityBaseline()
        runner_pipeline = getattr(baseline, "pipeline", None)
        from .tool.policy import ResolvedToolPolicy
        runner_baseline_policy = ResolvedToolPolicy() if baseline.enabled else None

        # Wrap the old-style tool policy provider (get_metadata_map) into the
        # new ToolPolicyProvider Protocol (resolve descriptor -> ResolvedToolPolicy).
        runner_policy_provider = None
        if bundle.tool_policies is not None:
            from .tool.policy_adapter import MetadataBackedPolicyProvider
            runner_policy_provider = MetadataBackedPolicyProvider(bundle.tool_policies)

        runner = AgentRunner(
            run_store=storage.runs, session_store=storage.sessions,
            event_store=storage.events, checkpoint_store=storage.checkpoints,
            middleware_pipeline=middleware_pipeline,
            memory_store=storage.memories,
            retriever=retriever,
            uow_factory=uow_factory,
            run_controller=run_controller,
            execution=execution,
            approval_store=storage.approvals,
            capability_options=resolved_options,
            security_pipeline=runner_pipeline,
            baseline_policy=runner_baseline_policy,
            tool_policy_provider=runner_policy_provider,
        )
        # Wire the ToolExecutor so ManagedToolsetWrapper can delegate policy/
        # approval checks for managed tool calls.
        runner._tool_executor_for_managed = resolved_executor
        swarm_runner = SwarmRunner(
            swarm_store=storage.swarms,
            run_store=storage.runs,
            session_store=storage.sessions,
            event_store=storage.events,
            compiler=compiler,
            agent_runner=runner,
            run_controller=run_controller,
        )

        # Capability providers + assembler. The subagent executor captures the
        # runner/compiler/storage just built, so build it first and pass it into
        # _build_capability_providers so SubagentProvider + PackageProvider
        # receive it at construction (no post-mutation).
        mcp_manager = MCPConnectionManager() if bundle.mcp_servers is not None else None
        sub_executor = None
        if bundle.entrypoints is not None or bundle.subagents is not None:
            sub_executor = _make_runtime_subagent_executor(
                storage=storage, compiler=compiler, runner=runner,
            )
        capability_providers = _build_capability_providers(
            bundle, execution, resolved_options, mcp_manager, sub_executor,
        )
        assembler = (
            CapabilityAssembler(capability_providers) if capability_providers else None
        )
        # AgentRunner reads the assembler at execute() time; set it now that
        # capability providers (which needed the runner) are wired.
        runner._capability_assembler = assembler

        return cls(
            storage=storage, compiler=compiler, runner=runner,
            swarm_runner=swarm_runner, model_router=router,
            run_controller=run_controller,
            capability_assembler=assembler,
            mcp_connection_manager=mcp_manager,
            options=resolved_options,
            provider_bundle=bundle,
        )

    @property
    def providers(self) -> "ProviderBundle | None":
        """The declaration bundle wired at build time. Canonical entry point for
        spec-by-id lookups (``runtime.providers.agents.get(id)``)."""
        return self._provider_bundle

    @property
    def capability_assembler(self) -> "CapabilityAssembler | None":
        """The CapabilityAssembler used internally. Prefer
        ``capability_assembler.assemble(spec, context)`` over the deprecated
        :meth:`assemble`."""
        return self._capability_assembler

    async def assemble(self, spec: AgentSpec, *, execution: "ExecutionBackend | None") -> Any:
        """Convenience wrapper around ``runtime.capability_assembler.assemble(spec, context)``.
        ."""
        from .capability.provider import CapabilityContext

        if self._capability_assembler is None:
            from .capability.bundle import CapabilityBundle
            return CapabilityBundle.empty()
        context = CapabilityContext(
            agent_id=spec.id,
            exposure_policy=self._options.tool_exposure,
            execution=execution,
        )
        return await self._capability_assembler.assemble(spec, context)

    async def resolve_swarm(self, swarm_id: str) -> "SwarmSpec":
        """Convenience wrapper around ``runtime.providers.swarms.get(swarm_id)``.
        ."""
        bundle = self._provider_bundle
        if bundle is None or bundle.swarms is None:
            raise SwarmError("no SwarmSpecProvider configured")
        return await bundle.swarms.get(swarm_id)

    async def resolve_agent(self, agent_id: str) -> AgentSpec:
        """Convenience wrapper around ``runtime.providers.agents.get(agent_id)``.
        ."""
        bundle = self._provider_bundle
        if bundle is None or bundle.agents is None:
            raise SwarmError("no AgentSpecProvider configured")
        return await bundle.agents.get(agent_id)

    async def __aenter__(self) -> "Runtime":
        return self
    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Release runtime-owned resources (MCP connections). Idempotent."""
        if self._mcp_connection_manager is not None:
            await self._mcp_connection_manager.close()
            self._mcp_connection_manager = None

    async def run(self, spec: "AgentSpec | SwarmSpec", prompt: str, *,
                  session_id: "str | None" = None,
                  run_id: "str | None" = None,
                  user_id: "str | None" = None,
                  tenant_id: "str | None" = None,
                  agents: "Mapping[str, AgentSpec] | None" = None):
        resolved_session_id = session_id or str(uuid.uuid4())
        if session_id is not None:
            existing = await self.storage.sessions.get(session_id)
            if existing is None:
                raise SessionError(f"session not found: {session_id}")
        else:
            now = datetime.now(timezone.utc)
            await self.storage.sessions.create(SessionRecord(
                id=resolved_session_id, parent_id=None, status=SessionStatus.ACTIVE, version=1,
                created_at=now, updated_at=now))

        resolved_run_id = run_id or str(uuid.uuid4())

        if isinstance(spec, SwarmSpec):
            if agents is None:
                raise SwarmError(
                    "agents mapping is required to run a SwarmSpec"
                )
            context = RunContext(
                run_id=resolved_run_id, root_run_id=resolved_run_id, parent_run_id=None,
                session_id=resolved_session_id, runnable_id=spec.id, runnable_type=RunnableType.SWARM,
                user_id=user_id, tenant_id=tenant_id, workspace=None)
            return await self.swarm_runner.run(
                spec, RunInput(prompt=prompt), context, agents=agents,
            )

        compiled = await self.compiler.compile(spec)
        context = RunContext(
            run_id=resolved_run_id, root_run_id=resolved_run_id, parent_run_id=None,
            session_id=resolved_session_id, runnable_id=spec.id, runnable_type=RunnableType.AGENT,
            user_id=user_id, tenant_id=tenant_id, workspace=None)
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
            RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED,
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
                    run_id, RunStatus.CANCELLING,
                    expected_version=record.version,
                )
            except RunConflictError:
                fresh = await self.storage.runs.get(run_id)
                if fresh is None:
                    raise RunNotFoundError(f"run not found: {run_id}")
                if fresh.status in (
                    RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED,
                ):
                    return
                if fresh.status == RunStatus.CANCELLING:
                    await self.run_controller.cancel(run_id)
                    return
                raise

            await self.run_controller.cancel(run_id)
        else:
            await self.storage.runs.transition(
                run_id, RunStatus.CANCELLED, expected_version=record.version,
            )

    async def run_stream(
        self, spec: "AgentSpec | SwarmSpec", prompt: str, *,
        session_id: "str | None" = None,
        run_id: "str | None" = None,
        user_id: "str | None" = None,
        tenant_id: "str | None" = None,
    ) -> "AsyncIterator[dict]":
        """Streaming variant of :meth:`run`. Only ``AgentSpec`` is supported --
        a ``SwarmSpec`` raises :class:`SwarmError` because swarm streaming is not
        implemented. Session resolution mirrors :meth:`run` exactly."""
        resolved_session_id = session_id or str(uuid.uuid4())
        if session_id is not None:
            existing = await self.storage.sessions.get(session_id)
            if existing is None:
                raise SessionError(f"session not found: {session_id}")
        else:
            now = datetime.now(timezone.utc)
            await self.storage.sessions.create(SessionRecord(
                id=resolved_session_id, parent_id=None, status=SessionStatus.ACTIVE, version=1,
                created_at=now, updated_at=now))

        resolved_run_id = run_id or str(uuid.uuid4())

        if isinstance(spec, SwarmSpec):
            raise SwarmError("run_stream does not support SwarmSpec")

        compiled = await self.compiler.compile(spec)
        context = RunContext(
            run_id=resolved_run_id, root_run_id=resolved_run_id, parent_run_id=None,
            session_id=resolved_session_id, runnable_id=spec.id, runnable_type=RunnableType.AGENT,
            user_id=user_id, tenant_id=tenant_id, workspace=None)
        async for event in self.runner.run_stream(compiled, RunInput(prompt=prompt), context):
            yield event

    async def resume(
        self, run_id: str, spec: "AgentSpec", *,
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
        from .agent.checkpoint_io import deserialize_messages
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
            run_id, RunStatus.RUNNING, expected_version=record.version,
        )
        compiled = await self.compiler.compile(spec)
        context = RunContext(
            run_id=run_id, root_run_id=record.root_run_id,
            parent_run_id=record.parent_run_id, session_id=record.session_id,
            runnable_id=record.runnable_id, runnable_type=record.runnable_type,
            user_id=user_id, tenant_id=tenant_id, workspace=None,
        )
        yield {"type": "resumed", "run_id": run_id}
        async for event in self.runner.run_stream(
            compiled, RunInput(prompt=""), context, message_history=messages,
        ):
            yield event


def _make_runtime_subagent_executor(*, storage: Storage, compiler: AgentCompiler,
                                    runner: AgentRunner):
    """Build a SubagentExecutor that runs a resolved child AgentSpec under a
    parent run: creates a child session (parent_id = parent session), mints a
    child run recording parent_run_id / root_run_id, delegates to the runner
    (bounded by timeout_seconds), and returns a structured SubagentResult.

    Depth is tracked via the ``_CURRENT_DEPTH`` contextvar: the child run sees
    parent_depth + 1, so SubagentProvider.enforce_depth bounds the full chain
    (the token is reset on return so parallel/sibling calls are unaffected)."""
    from .subagent.runner import _CURRENT_DEPTH

    async def execute(*, agent_spec, task, context, parent_run_id, root_run_id,
                      parent_session_id, scope, timeout_seconds,
                      user_id=None, tenant_id=None, workspace=None):
        child_session = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        await storage.sessions.create(SessionRecord(
            id=child_session, parent_id=parent_session_id,
            status=SessionStatus.ACTIVE, version=1,
            created_at=now, updated_at=now,
        ))
        child_run = str(uuid.uuid4())
        effective_root = root_run_id or parent_run_id or child_run
        run_ctx = RunContext(
            run_id=child_run, root_run_id=effective_root, parent_run_id=parent_run_id,
            session_id=child_session, runnable_id=agent_spec.id,
            runnable_type=RunnableType.AGENT, user_id=user_id, tenant_id=tenant_id,
            workspace=workspace,
        )
        scope_dict = None
        if scope is not None:
            scope_dict = {
                "package_id": scope.package_id,
                "package_kind": scope.package_kind,
            }

        async def _drive():
            compiled = await compiler.compile(agent_spec)
            return await runner.run(compiled, RunInput(prompt=task), run_ctx)

        # Increment depth for the child run; reset on return so siblings/parents
        # keep their own depth value.
        from .events.payloads import SubagentCompleted, SubagentErrored, SubagentStarted

        async def _evt(payload):
            await storage.events.append(
                stream_id=child_run, run_id=child_run, root_run_id=effective_root,
                parent_run_id=parent_run_id, session_id=child_session,
                runnable_id=agent_spec.id, payload=payload,
            )

        token = _CURRENT_DEPTH.set(_CURRENT_DEPTH.get() + 1)
        await _evt(SubagentStarted(
            agent_id=agent_spec.id, parent_run_id=parent_run_id,
            scope=scope_dict.get("package_id") if scope_dict else None,
        ))
        try:
            if timeout_seconds is not None:
                result = await asyncio.wait_for(_drive(), timeout=timeout_seconds)
            else:
                result = await _drive()
            await _evt(SubagentCompleted(agent_id=agent_spec.id, run_id=child_run, status="succeeded"))
            return SubagentResult(
                agent_id=agent_spec.id, scope=scope_dict,
                session_id=child_session, run_id=child_run,
                status="succeeded", output=getattr(result, "output", None),
            )
        except asyncio.TimeoutError:
            await _evt(SubagentErrored(agent_id=agent_spec.id, reason=f"timeout after {timeout_seconds}s"))
            return SubagentResult(
                agent_id=agent_spec.id, scope=scope_dict,
                session_id=child_session, run_id=child_run,
                status="failed", error={"reason": f"timeout after {timeout_seconds}s"},
            )
        except Exception as exc:  # child failures surface as structured errors
            await _evt(SubagentErrored(agent_id=agent_spec.id, reason=str(exc)))
            return SubagentResult(
                agent_id=agent_spec.id, scope=scope_dict,
                session_id=child_session, run_id=child_run,
                status="failed", error={"reason": str(exc)},
            )
        finally:
            _CURRENT_DEPTH.reset(token)

    return execute


# re-export for tooling that imports Runtime alongside these types
__all__ = ["Runtime"]
