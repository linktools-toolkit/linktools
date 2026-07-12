#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The single Runtime build kernel (spec §9.2-§9.4).

``build_runtime_components(RuntimeBuildConfig) -> RuntimeComponents`` is the one
place the Runtime's sub-components are assembled. Runtime.build is a thin wrapper
that constructs the config, calls this, and unpacks the result; nothing else
constructs the compiler/runner/assembler/executor graph."""

import asyncio
import dataclasses
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from ..agent.compiler import AgentCompiler
from ..agent.runner import AgentRunner
from ..capability.assembler import CapabilityAssembler
from ..capability.builtin import BuiltinProvider
from ..capability.models import CapabilityRuntimeOptions
from ..execution.protocols import ExecutionBackend
from ..middleware.pipeline import MiddlewarePipeline
from ..model.router import ModelRouter
from ..mcp.client import MCPConnectionManager
from ..mcp.provider import MCPProvider
from ..package.capability_provider import PackageProvider
from ..providers.bundle import ProviderBundle
from ..run.controller import RunController
from ..run.models import RunnableType
from ..session.models import SessionRecord, SessionStatus
from ..skill.provider import SkillProvider
from ..storage.facade import Storage
from ..subagent.models import SubagentResult
from ..subagent.provider import SubagentProvider
from ..swarm.runner import SwarmRunner
from ..tool.executor import ToolExecutor

if TYPE_CHECKING:
    from ..agent.compiler import AgentCompiler as _AgentCompiler
    from ..agent.runner import AgentRunner as _AgentRunner
    from ..capability.assembler import CapabilityAssembler as _CapabilityAssembler
    from ..knowledge.retriever import Retriever
    from ..swarm.runner import SwarmRunner as _SwarmRunner


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeComponents:
    """The fully-wired sub-components a Runtime drives. Built once by
    :func:`build_runtime_components`; Runtime unpacks these onto itself."""

    storage: Storage
    provider_bundle: ProviderBundle
    options: CapabilityRuntimeOptions
    model_router: ModelRouter
    compiler: "_AgentCompiler"
    runner: "_AgentRunner"
    swarm_runner: "_SwarmRunner"
    run_controller: RunController
    capability_assembler: "_CapabilityAssembler | None"
    tool_executor: ToolExecutor
    mcp_connection_manager: "MCPConnectionManager | None"


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeBuildConfig:
    """The final set of inputs Runtime.build accepts (spec §9.3). Capability
    providers come exclusively via ``providers``."""

    storage: Storage
    providers: ProviderBundle
    model_router: "ModelRouter | None" = None
    middleware_pipeline: "MiddlewarePipeline | None" = None
    retriever: "Retriever | None" = None
    execution: "ExecutionBackend | None" = None
    tool_executor: "ToolExecutor | None" = None
    security: Any = None
    capability_options: "CapabilityRuntimeOptions | None" = None
    pause_on_approval: bool = False
    allow_mcp_wildcard: bool = False
    mcp_connection_manager: "MCPConnectionManager | None" = None


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
            bundle.mcp_servers,
            mcp_manager,
            allow_mcp_wildcard=bool(options.allow_mcp_wildcard),
        )
    if bundle.entrypoints is not None or bundle.subagents is not None:
        providers["subagent"] = SubagentProvider(
            subagent_provider=bundle.subagents,
            entrypoint_resolver=bundle.entrypoints,
            executor=subagent_executor,
        )
    if bundle.package_resources is not None or bundle.entrypoints is not None:
        # PackageProvider declares every kind it handles via supported_kinds;
        # register the one instance under all of them (no manual alias hack).
        from ..capability.provider import provider_kinds

        pkg = PackageProvider(
            resource_provider=bundle.package_resources,
            entrypoint_resolver=bundle.entrypoints,
            entrypoint_executor=subagent_executor,
        )
        for k in provider_kinds(pkg):
            providers[k] = pkg
    # Pre-built capability providers (e.g. a custom MCPProvider wired with a
    # fake connection manager) override the bundle-constructed ones for every
    # kind they support -- the single registration path for custom providers.
    if bundle.capabilities:
        from ..capability.provider import provider_kinds

        for provider in bundle.capabilities:
            for k in provider_kinds(provider):
                providers[k] = provider
    return providers


def _make_runtime_subagent_executor(
    *,
    storage: Storage,
    compiler: "_AgentCompiler",
    runner_provider: "Callable[[], _AgentRunner]",
):
    """Build a SubagentExecutor that runs a resolved child AgentSpec under a
    parent run. ``runner_provider`` resolves the runner LAZILY (called inside
    ``execute``) so the build kernel can construct the runner with its assembler
    before the subagent executor references it -- breaking the
    runner→assembler→provider→executor→runner cycle without post-build private
    mutation."""
    from ..run.context import RunContext
    from ..run.models import RunInput

    async def execute(*, agent_spec, task, context, parent, scope, timeout_seconds):
        parent_run_id = parent.run_id if parent is not None else None
        parent_session_id = parent.session_id if parent is not None else None
        child_session = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        await storage.sessions.create(
            SessionRecord(
                id=child_session,
                parent_id=parent_session_id,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
        child_run = str(uuid.uuid4())
        effective_root = (
            (parent.root_run_id if parent is not None else None)
            or parent_run_id
            or child_run
        )
        run_ctx = RunContext(
            run_id=child_run,
            root_run_id=effective_root,
            parent_run_id=parent_run_id,
            session_id=child_session,
            runnable_id=agent_spec.id,
            runnable_type=RunnableType.AGENT,
            user_id=parent.user_id if parent is not None else None,
            tenant_id=parent.tenant_id if parent is not None else None,
            workspace=parent.workspace if parent is not None else None,
        )
        scope_dict = None
        if scope is not None:
            scope_dict = {
                "package_id": scope.package_id,
                "package_kind": scope.package_kind,
            }

        async def _drive():
            compiled = await compiler.compile(agent_spec)
            return await runner_provider().run(compiled, RunInput(prompt=task), run_ctx)

        from ..events.payloads import (
            SubagentCompleted,
            SubagentErrored,
            SubagentStarted,
        )

        async def _evt(payload):
            from ..events.context import EventContext, append_event

            await append_event(
                storage.events,
                EventContext(
                    stream_id=child_run,
                    run_id=child_run,
                    root_run_id=effective_root,
                    parent_run_id=parent_run_id,
                    session_id=child_session,
                    runnable_id=agent_spec.id,
                ),
                payload,
            )

        from ..subagent.runner import _CURRENT_DEPTH

        token = _CURRENT_DEPTH.set(_CURRENT_DEPTH.get() + 1)
        await _evt(
            SubagentStarted(
                agent_id=agent_spec.id,
                parent_run_id=parent_run_id,
                scope=scope_dict.get("package_id") if scope_dict else None,
            )
        )
        try:
            if timeout_seconds is not None:
                result = await asyncio.wait_for(_drive(), timeout=timeout_seconds)
            else:
                result = await _drive()
            await _evt(
                SubagentCompleted(
                    agent_id=agent_spec.id, run_id=child_run, status="succeeded"
                )
            )
            return SubagentResult(
                agent_id=agent_spec.id,
                scope=scope_dict,
                session_id=child_session,
                run_id=child_run,
                status="succeeded",
                output=getattr(result, "output", None),
            )
        except asyncio.TimeoutError:
            await _evt(
                SubagentErrored(
                    agent_id=agent_spec.id, reason=f"timeout after {timeout_seconds}s"
                )
            )
            return SubagentResult(
                agent_id=agent_spec.id,
                scope=scope_dict,
                session_id=child_session,
                run_id=child_run,
                status="failed",
                error={"reason": f"timeout after {timeout_seconds}s"},
            )
        except Exception as exc:  # child failures surface as structured errors
            await _evt(SubagentErrored(agent_id=agent_spec.id, reason=str(exc)))
            return SubagentResult(
                agent_id=agent_spec.id,
                scope=scope_dict,
                session_id=child_session,
                run_id=child_run,
                status="failed",
                error={"reason": str(exc)},
            )
        finally:
            _CURRENT_DEPTH.reset(token)

    return execute


def build_runtime_components(config: RuntimeBuildConfig) -> RuntimeComponents:
    """Assemble the compiler/runner/assembler/executor graph from a
    RuntimeBuildConfig. Internal splits: resolve security + exposure -> build
    executor -> build capability assembler -> build compiler -> build runner ->
    build swarm runner."""
    base_options = config.capability_options or CapabilityRuntimeOptions()
    from ..security.baseline import SecurityBaseline

    baseline = config.security if config.security is not None else SecurityBaseline()
    from ..capability.exposure import CapabilityToolExposurePolicy

    if (
        config.capability_options is not None
        and config.capability_options.tool_exposure is not None
    ):
        effective_exposure = config.capability_options.tool_exposure
    elif baseline.enabled and baseline.tool_exposure_policy is not None:
        effective_exposure = baseline.tool_exposure_policy
    else:
        effective_exposure = CapabilityToolExposurePolicy()
    resolved_options = dataclasses.replace(
        base_options,
        tool_exposure=effective_exposure,
        allow_mcp_wildcard=(
            base_options.allow_mcp_wildcard or config.allow_mcp_wildcard
        ),
    )

    bundle = config.providers
    router = config.model_router or ModelRouter()

    resolved_executor = config.tool_executor
    if resolved_executor is None:
        from ..policy.engine import PolicyEngine

        rules: "list[Any]" = []
        if baseline.enabled and baseline.command_policy is not None:
            from ..policy.command import CommandRule

            rules.append(
                CommandRule(denied_patterns=baseline.command_policy.denied_patterns)
            )
        resolved_executor = ToolExecutor(
            policy=PolicyEngine(rules=tuple(rules)),
            approval_store=config.storage.approvals,
            event_store=config.storage.events,
            idempotency_store=config.storage.idempotency,
            pause_on_approval=config.pause_on_approval,
            security_audit_failure_mode=getattr(
                baseline, "audit_failure_mode", "fail_closed"
            ),
        )
    compiler = AgentCompiler(
        model_router=router,
        middleware_pipeline=config.middleware_pipeline,
        tool_executor=resolved_executor,
    )
    uow_factory = (
        config.storage.transaction
        if config.storage.capabilities.cross_store_transactions
        else None
    )
    run_controller = RunController()
    runner_pipeline = getattr(baseline, "pipeline", None)
    from ..tool.policy import ResolvedToolPolicy

    runner_baseline_policy = ResolvedToolPolicy() if baseline.enabled else None

    runner_policy_provider = None
    if bundle.tool_policies is not None:
        from ..tool.policy import MetadataBackedPolicyProvider

        runner_policy_provider = MetadataBackedPolicyProvider(bundle.tool_policies)

    mcp_manager = None
    if bundle.mcp_servers is not None:
        mcp_manager = config.mcp_connection_manager or MCPConnectionManager()
    sub_executor = None
    if bundle.entrypoints is not None or bundle.subagents is not None:
        # ``runner`` is assigned below; the lambda resolves it at call time
        # (only during a real subagent execute, long after build returns).
        sub_executor = _make_runtime_subagent_executor(
            storage=config.storage,
            compiler=compiler,
            runner_provider=lambda: runner,
        )
    capability_providers = _build_capability_providers(
        bundle,
        config.execution,
        resolved_options,
        mcp_manager,
        sub_executor,
    )
    assembler = (
        CapabilityAssembler(capability_providers) if capability_providers else None
    )

    runner = AgentRunner(
        run_store=config.storage.runs,
        session_store=config.storage.sessions,
        event_store=config.storage.events,
        checkpoint_store=config.storage.checkpoints,
        middleware_pipeline=config.middleware_pipeline,
        memory_store=config.storage.memories,
        retriever=config.retriever,
        uow_factory=uow_factory,
        run_controller=run_controller,
        execution=config.execution,
        approval_store=config.storage.approvals,
        capability_options=resolved_options,
        capability_assembler=assembler,
        security_pipeline=runner_pipeline,
        baseline_policy=runner_baseline_policy,
        tool_policy_provider=runner_policy_provider,
        managed_tool_executor=resolved_executor,
        security_audit_failure_mode=getattr(
            baseline, "audit_failure_mode", "fail_closed"
        ),
    )
    swarm_runner = SwarmRunner(
        swarm_store=config.storage.swarms,
        run_store=config.storage.runs,
        session_store=config.storage.sessions,
        event_store=config.storage.events,
        compiler=compiler,
        agent_runner=runner,
        run_controller=run_controller,
    )

    return RuntimeComponents(
        storage=config.storage,
        provider_bundle=bundle,
        options=resolved_options,
        model_router=router,
        compiler=compiler,
        runner=runner,
        swarm_runner=swarm_runner,
        run_controller=run_controller,
        capability_assembler=assembler,
        tool_executor=resolved_executor,
        mcp_connection_manager=mcp_manager,
    )
