#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The single Runtime build kernel.

``build_runtime_components(RuntimeBuildConfig) -> RuntimeComponents`` is the one
place the Runtime's sub-components are assembled. Runtime.build is a thin wrapper
that constructs the config, calls this, and unpacks the result; nothing else
constructs the compiler/runner/assembler/executor graph."""

import asyncio
import dataclasses
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..agent.compiler import AgentCompiler
from ..agent.runner import AgentEngine
from ..capability.assembler import CapabilityAssembler
from ..capability.registry import CapabilityProviderRegistry
from ..capability.builtin import BuiltinProvider
from ..subagent.config import SkillPrivateSubagentConfig
from ..capability.models import CapabilityRuntimeOptions
from ..execution.protocols import ExecutionBackend
from ..execution.protocols import ExecutionIsolationLevel
from ..middleware.pipeline import MiddlewarePipeline
from ..model.router import ModelRouter
from ..mcp.client import MCPConnectionManager
from ..mcp.provider import MCPProvider
from ..extension.capability_provider import ExtensionProvider
from .dependencies import RuntimeDependencies
from ..run.controller import RunController
from ..run.models import RunnableType
from ..run.options import RuntimeCancellationOptions
from ..run.schema_registry import OutputSchemaRegistry
from ..observability.metrics import InMemoryMetrics
from ..session.models import SessionRecord, SessionStatus
from ..skill.provider import SkillProvider
from ..storage.facade import Storage
from ..storage.features import TransactionScope
from ..subagent.models import SubagentResult
from ..subagent.provider import SubagentProvider
from ..swarm.runner import SwarmRunner
from ..tool.executor import GovernedToolInvoker

if TYPE_CHECKING:
    from ..agent.compiler import AgentCompiler as _AgentCompiler
    from ..agent.runner import AgentEngine as _AgentEngine
    from ..capability.assembler import CapabilityAssembler as _CapabilityAssembler
    from ..retrieval.retriever import Retriever
    from ..run.dispatch import RunDispatcher, RunDispatchRequest
    from ..run.models import RunResult
    from ..run.requirements import RuntimeRequirements
    from ..swarm.runner import SwarmRunner as _SwarmRunner


class _LateBoundRunDispatcher:
    """A RunDispatcher bound to its real target after construction.

    The subagent executor is built before the AgentEngine it will eventually
    delegate to exists -- the runner depends on the capability assembler,
    which depends on the subagent executor: a genuine self-reference, not an
    accidental cycle. This handle confines that one-time forward reference to
    a single bind-once seam instead of a bare closure; every caller only ever
    sees the narrow ``RunDispatcher`` Protocol, never the runner or builder
    internals directly."""

    def __init__(self) -> None:
        self._target: "RunDispatcher | None" = None

    def bind(self, target: "RunDispatcher") -> None:
        self._target = target

    async def dispatch(self, request: "RunDispatchRequest") -> "RunResult":
        if self._target is None:
            raise RuntimeError(
                "_LateBoundRunDispatcher.dispatch() called before bind() -- "
                "the build kernel must bind the real dispatcher before any "
                "subagent execution can occur"
            )
        return await self._target.dispatch(request)


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Serializable runtime configuration values -- distinct from
    RuntimeDependencies (the process-in-memory object graph: storage,
    catalogs, capability providers, execution backend, etc.). Plan §4.1's
    "配置是可序列化值；依赖是进程内对象" split, applied to the cleanly-classifiable
    pure-value fields only. ``SecurityBaseline`` (can carry a live
    ``SecurityPipeline``), ``CapabilityRuntimeOptions`` (its ``memory_policy``
    / ``retrieval_policy`` / etc. fields are ``Any`` and may hold live
    objects), ``OutputSchemaRegistry`` (stateful), ``metrics``, and
    ``authorization`` are mixed bags or live services -- they stay
    dependency-shaped on RuntimeBuildConfig rather than moving here."""

    allow_mcp_wildcard: bool = False
    # Default False (strict): sensitive APIs reject a missing principal. When
    # True the Runtime is explicitly single-tenant / local-trusted and a
    # missing principal is allowed (with a deprecation warning). Never
    # default True: do not ship loose.
    local_trusted_mode: bool = False
    multi_tenant: bool = False
    cancellation: RuntimeCancellationOptions = dataclasses.field(
        default_factory=RuntimeCancellationOptions
    )


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeComponents:
    """The fully-wired sub-components a Runtime drives. Built once by
    :func:`build_runtime_components`; Runtime unpacks these onto itself."""

    storage: Storage
    providers: RuntimeDependencies
    options: CapabilityRuntimeOptions
    model_router: ModelRouter
    compiler: "_AgentCompiler"
    runner: "_AgentEngine"
    swarm_runner: "_SwarmRunner"
    run_controller: RunController
    capability_assembler: "_CapabilityAssembler | None"
    tool_executor: GovernedToolInvoker
    execution: "ExecutionBackend | None"
    mcp_connection_manager: "MCPConnectionManager | None"
    commit_coordinator: Any = None
    settings: RuntimeSettings = dataclasses.field(default_factory=RuntimeSettings)
    schema_registry: OutputSchemaRegistry = dataclasses.field(default_factory=OutputSchemaRegistry)
    metrics: Any = dataclasses.field(default_factory=InMemoryMetrics)
    authorization: Any = None


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeBuildConfig:
    """The final set of inputs Runtime.build accepts. Capability
    providers come exclusively via ``providers``."""

    storage: Storage
    providers: RuntimeDependencies
    # Typed skill-private-subagent wiring (was 5 Any fields on RuntimeDependencies).
    # Injected straight into the SubagentProvider / SkillProvider; never flows
    # through RuntimeDependencies (which would cycle providers <-> skill/subagent).
    skill_subagent: "SkillPrivateSubagentConfig | None" = None
    model_router: "ModelRouter | None" = None
    middleware_pipeline: "MiddlewarePipeline | None" = None
    retriever: "Retriever | None" = None
    execution: "ExecutionBackend | None" = None
    tool_executor: "GovernedToolInvoker | None" = None
    security: Any = None
    capability_options: "CapabilityRuntimeOptions | None" = None
    mcp_connection_manager: "MCPConnectionManager | None" = None
    settings: RuntimeSettings = dataclasses.field(default_factory=RuntimeSettings)
    schema_registry: OutputSchemaRegistry | None = None
    metrics: Any = None
    authorization: Any = None
    # Optional capability minimums the topology declares it needs; the
    # capability gate (enforce_storage_capability_gate) refuses a Storage whose
    # StorageFeatures fall below these at build time. None = no gate.
    requirements: "RuntimeRequirements | None" = None


def _build_file_commit_coordinator(storage):
    """Build a FilesystemRunCommitCoordinator from a File-backed Storage."""
    from ..storage.filesystem.commit import FilesystemRunCommitCoordinator

    return FilesystemRunCommitCoordinator(
        approval_store=storage.approvals,
        checkpoint_store=storage.checkpoints,
        run_store=storage.runs,
        session_store=storage.sessions,
        event_store=storage.events,
        transactions_root=storage.root / "transactions",
    )


def _capability_attribute_from_features(features) -> "dict[str, str]":
    """Map a StorageFeatures snapshot to a stable low-cardinality capability
    label for the build-failure metrics. Each flag becomes the attribute value
    it contributes to the gate's shortfall; the actual gate message is the
    authority on which capability fell short, but the metric only needs the
    bounded set of capability names."""
    try:
        return {
            "coordination": getattr(features.coordination, "value", "unknown"),
            "transactions": getattr(features.transactions, "value", "unknown"),
        }
    except Exception:  # noqa: BLE001 - attributes are observability-only
        return {}


def _build_commit_coordinator(storage):
    """Build the storage-appropriate RunCommitCoordinator.

    SQL-backed storage (database-scoped transactions) gets the atomic
    SqlAlchemyRunCommitCoordinator -- pause/complete share one transaction so
    the cross-store commit is all-or-nothing. File-backed storage gets the
    sequential FilesystemRunCommitCoordinator (no cross-store txn available)."""
    if storage.features.transactions is TransactionScope.DATABASE:
        from ..storage.sqlalchemy.commit import SqlAlchemyRunCommitCoordinator

        return SqlAlchemyRunCommitCoordinator(storage)
    return _build_file_commit_coordinator(storage)


def _build_capability_registry(
    bundle: RuntimeDependencies,
    execution: "ExecutionBackend | None",
    options: CapabilityRuntimeOptions,
    mcp_manager: "MCPConnectionManager | None",
    subagent_executor: Any = None,
    skill_subagent: "SkillPrivateSubagentConfig | None" = None,
) -> "CapabilityProviderRegistry | None":
    """Map the declaration bundle onto the runtime CapabilityProviderRegistry
    the assembler dispatches over. Builtin is registered only when an execution
    backend exists (it cannot resolve without one). The subagent executor is
    passed in so both SubagentProvider and ExtensionProvider receive it at
    construction. ``skill_subagent`` carries the typed skill-private-subagent
    wiring (formerly 5 Any fields on RuntimeDependencies) into the SkillProvider /
    SubagentProvider. Returns None when no provider is wired (so the runner
    treats the run as having no capability resolution)."""
    registry = CapabilityProviderRegistry()
    sk = skill_subagent or SkillPrivateSubagentConfig.empty()
    if execution is not None:
        registry.replace(BuiltinProvider())
    if bundle.skills is not None:
        registry.replace(
            SkillProvider(bundle.skills, active_skill_lookup=sk.active_skill_lookup)
        )
    if bundle.mcp_servers is not None:
        registry.replace(
            MCPProvider(
                bundle.mcp_servers,
                mcp_manager,
                allow_mcp_wildcard=bool(options.allow_mcp_wildcard),
            )
        )
    if bundle.entrypoints is not None or bundle.subagents is not None:
        registry.replace(
            SubagentProvider(
                subagent_provider=bundle.subagents,
                entrypoint_resolver=bundle.entrypoints,
                executor=subagent_executor,
                skill_resolver=sk.skill_resolver,
                active_skill_provider=sk.active_skill_provider,
                child_model_policy=sk.child_model_policy,
                parent_delegated_tools=sk.parent_delegated_tools,
            )
        )
    if bundle.extension_resources is not None or bundle.entrypoints is not None:
        # ExtensionProvider declares every kind it handles via supported_kinds;
        # replace() registers the one instance under all of them.
        pkg = ExtensionProvider(
            resource_provider=bundle.extension_resources,
            entrypoint_resolver=bundle.entrypoints,
            entrypoint_executor=subagent_executor,
        )
        registry.replace(pkg)
    # Pre-built capability providers (e.g. a custom MCPProvider wired with a
    # fake connection manager) override the bundle-constructed ones for every
    # kind they support -- the single registration path for custom providers.
    if bundle.capabilities:
        for provider in bundle.capabilities:
            registry.replace(provider)
    return registry or None


def _make_runtime_subagent_executor(
    *,
    storage: Storage,
    compiler: "_AgentCompiler",
    dispatcher: "RunDispatcher",
):
    """Build a SubagentExecutor that runs a resolved child AgentSpec under a
    parent run. ``dispatcher`` is a :class:`LateBoundRunDispatcher` bound to
    the real runner only after the runner is constructed -- the runner needs
    the capability assembler, which needs this executor, which needs a way to
    drive a child run: a genuine self-reference. The bind-once handle confines
    that indirection to a single named seam instead of a bare closure, and
    every caller here only ever sees the narrow ``RunDispatcher`` Protocol."""
    from ..run.context import RunContext
    from ..run.dispatch import RunDispatchRequest
    from ..run.models import RunInput
    from ..run.preparation import RunPreparationCoordinator

    # A child agent run (subagent / extension entrypoint) gets the same
    # resumable snapshot as a top-level run: if one of its tools pauses on
    # approval, Runtime.resume(child_run_id) can restore its spec + identity.
    preparation = RunPreparationCoordinator(storage.run_definitions)

    async def execute(*, agent_spec, task, context, parent, scope, timeout_seconds):
        parent_run_id = parent.run_id if parent is not None else None
        parent_session_id = parent.session_id if parent is not None else None
        child_session = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        await storage.sessions.create(
            SessionRecord(
                id=child_session,
                parent_id=parent_session_id,
                # A subagent child session inherits its parent's principal, so a
                # worker pause/resume stays within the same ownership domain.
                user_id=parent.user_id if parent is not None else None,
                tenant_id=parent.tenant_id if parent is not None else None,
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
                "extension_id": scope.extension_id,
                "extension_kind": scope.extension_kind,
            }

        async def _drive():
            # A child run starts OUTSIDE any skill: clear the parent's active
            # skill for the duration of the child so a subagent cannot address
            # the parent's skill via call_subagent(instruction_path=...) (skill
            # isolation). Imported lazily to avoid a build-time import cycle.
            from ..skill.private import reset_active_skill, set_active_skill

            skill_token = set_active_skill(None)
            try:
                await preparation.prepare_agent_run(spec=agent_spec, context=run_ctx)
                compiled = await compiler.compile(agent_spec)
                return await dispatcher.dispatch(
                    RunDispatchRequest(
                        agent=compiled, input=RunInput(prompt=task), context=run_ctx
                    )
                )
            finally:
                reset_active_skill(skill_token)

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
                scope=scope_dict.get("extension_id") if scope_dict else None,
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
            from ..governance.security.redact import redact_exception

            safe_error = redact_exception(exc)
            await _evt(SubagentErrored(agent_id=agent_spec.id, reason=safe_error))
            return SubagentResult(
                agent_id=agent_spec.id,
                scope=scope_dict,
                session_id=child_session,
                run_id=child_run,
                status="failed",
                error={"error_type": type(exc).__name__, "reason": safe_error},
            )
        finally:
            _CURRENT_DEPTH.reset(token)

    return execute


def build_runtime_components(config: RuntimeBuildConfig) -> RuntimeComponents:
    """Assemble the compiler/runner/assembler/executor graph from a
    RuntimeBuildConfig. Internal splits: resolve security + exposure -> build
    executor -> build capability assembler -> build compiler -> build runner ->
    build swarm runner."""
    if config.settings.multi_tenant and config.execution is not None:
        # Protocol runtime checks cannot distinguish a trusted local backend;
        # use its declared isolation level when available.
        if getattr(config.execution, "isolation_level", None) == ExecutionIsolationLevel.TRUSTED_LOCAL and not config.settings.local_trusted_mode:
            from ..errors import UnsafeExecutionBackendError

            raise UnsafeExecutionBackendError(
                "LocalExecutionBackend is trusted-only and cannot serve multi-tenant runs"
            )
    if config.storage.run_definitions is None:
        # Fail fast at build time, not when a subagent/worker tool first
        # pauses on approval and Runtime.resume(child_run_id) cannot find a
        # snapshot. Resumability is a required capability, not opt-in.
        from ..errors import RuntimeInitializationError

        raise RuntimeInitializationError(
            "RunDefinitionStore is required for resumable runs"
        )
    # Pure-capability gate: when the caller declares RuntimeRequirements,
    # refuse a Storage whose StorageFeatures fall below them -- fail fast
    # rather than silently degrading to a weaker scope.
    from ..errors import StorageRequirementsNotMetError
    from ..run.requirements import enforce_storage_capability_gate

    build_metrics = config.metrics
    try:
        enforce_storage_capability_gate(config.storage.features, config.requirements)
    except StorageRequirementsNotMetError:
        # The gate's message identifies which capability fell short. Record it
        # under both the generic build-failure counter (any runtime build
        # failure) and the capability-specific counter (gate-driven). The
        # capability name is a stable low-cardinality attribute (coordination
        # / transactions / leasing / fencing / ...) extracted from the gate's
        # own message rather than the caller-supplied requirements object so
        # the counter never carries free-form identifiers.
        if build_metrics is not None:
            attr = _capability_attribute_from_features(config.storage.features)
            build_metrics.counter(
                "runtime_build_failure_total", attributes=attr
            )
            build_metrics.counter(
                "storage_capability_validation_failure_total", attributes=attr
            )
        raise
    base_options = config.capability_options or CapabilityRuntimeOptions()
    from ..governance.security.baseline import SecurityBaseline

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
            base_options.allow_mcp_wildcard or config.settings.allow_mcp_wildcard
        ),
    )

    bundle = config.providers
    router = config.model_router or ModelRouter()

    resolved_executor = config.tool_executor
    if resolved_executor is None:
        from ..governance.policy.engine import PolicyEngine
        from ..storage.artifact_backends import build_artifact_store_from_assets

        rules: "list[Any]" = []
        if baseline.enabled and baseline.command_policy is not None:
            from ..governance.policy.command import CommandRule

            rules.append(
                CommandRule(denied_patterns=baseline.command_policy.denied_patterns)
            )
        resolved_executor = GovernedToolInvoker(
            policy=PolicyEngine(rules=tuple(rules)),
            approval_store=config.storage.approvals,
            idempotency_store=config.storage.idempotency,
            metrics=config.metrics or InMemoryMetrics(),
            receipt_store=build_artifact_store_from_assets(config.storage.assets),
            tenant_id_resolver=lambda context: context.tenant_id,
        )
    compiler = AgentCompiler(
        model_router=router,
        middleware_pipeline=config.middleware_pipeline,
        tool_executor=resolved_executor,
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
    dispatcher_handle = None
    if bundle.entrypoints is not None or bundle.subagents is not None:
        # ``runner`` (the real dispatcher) is bound below, after it exists.
        dispatcher_handle = _LateBoundRunDispatcher()
        sub_executor = _make_runtime_subagent_executor(
            storage=config.storage,
            compiler=compiler,
            dispatcher=dispatcher_handle,
        )
    capability_registry = _build_capability_registry(
        bundle,
        config.execution,
        resolved_options,
        mcp_manager,
        sub_executor,
        config.skill_subagent,
    )
    assembler = (
        CapabilityAssembler(capability_registry) if capability_registry else None
    )

    runner = AgentEngine(
        run_store=config.storage.runs,
        session_store=config.storage.sessions,
        event_store=config.storage.events,
        checkpoint_store=config.storage.checkpoints,
        middleware_pipeline=config.middleware_pipeline,
        memory_store=config.storage.memories,
        retriever=config.retriever,
        run_controller=run_controller,
        execution=config.execution,
        capability_options=resolved_options,
        capability_assembler=assembler,
        security_pipeline=runner_pipeline,
        baseline_policy=runner_baseline_policy,
        tool_policy_provider=runner_policy_provider,
        managed_tool_executor=resolved_executor,
        security_audit_failure_mode=getattr(
            baseline, "audit_failure_mode", "fail_closed"
        ),
        commit_coordinator=_build_commit_coordinator(config.storage),
    )
    if dispatcher_handle is not None:
        dispatcher_handle.bind(runner)
    swarm_runner = SwarmRunner(
        swarm_store=config.storage.swarms,
        run_store=config.storage.runs,
        session_store=config.storage.sessions,
        event_store=config.storage.events,
        compiler=compiler,
        dispatcher=runner,
        run_controller=run_controller,
        run_definitions=config.storage.run_definitions,
    )

    if config.authorization is None:
        from ..governance.security.authorization import DenyAllAuthorization, ScopeAuthorization

        authorization = ScopeAuthorization() if config.settings.local_trusted_mode else DenyAllAuthorization()
    else:
        authorization = config.authorization

    return RuntimeComponents(
        storage=config.storage,
        providers=bundle,
        options=resolved_options,
        model_router=router,
        compiler=compiler,
        runner=runner,
        swarm_runner=swarm_runner,
        run_controller=run_controller,
        capability_assembler=assembler,
        tool_executor=resolved_executor,
        execution=config.execution,
        mcp_connection_manager=mcp_manager,
        commit_coordinator=runner._commit_coordinator,
        settings=config.settings,
        schema_registry=config.schema_registry or OutputSchemaRegistry(),
        metrics=config.metrics or InMemoryMetrics(),
        authorization=authorization,
    )
