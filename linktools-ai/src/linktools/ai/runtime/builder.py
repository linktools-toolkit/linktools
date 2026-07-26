#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The single Runtime build kernel.

``build_runtime_components(RuntimeBuildConfig) -> RuntimeComponents`` is the one
place the Runtime's sub-components are assembled. build_runtime is a thin wrapper
that constructs the config, calls this, and unpacks the result; nothing else
constructs the compiler/runner/resolver/executor graph."""

import dataclasses
from typing import TYPE_CHECKING, Any

from ..agent.compiler import AgentCompiler
from ..agent.engine import AgentEngine
from ..capability.resolver import CapabilityResolver
from ..capability.registry import CapabilityProviderRegistry
from ..capability.builtin import BuiltinProvider
from ..subagent.config import SkillPrivateSubagentConfig
from ..subagent.executor import SubagentExecutor
from ..capability.models import CapabilityRuntimeOptions
from ..sandbox.protocols import Sandbox
from ..sandbox.protocols import ExecutionIsolationLevel
from ..middleware.pipeline import MiddlewarePipeline
from ..model.resolver import ModelResolver
from ..mcp.client import MCPConnectionPool
from ..mcp.provider import MCPProvider
from ..extension.capability_provider import ExtensionProvider
from .dependencies import RuntimeDependencies
from .dispatcher import LateBoundRunDispatcher
from ..run.commit import RunCommitCoordinator
from ..run.controller import RunController
from ..run.options import RuntimeCancellationOptions
from ..run.requirements import (
    RuntimeRequirements,
    RuntimeTopology,
    derive_runtime_requirements,
    merge_runtime_requirements,
)
from ..run.schema_registry import OutputSchemaRegistry
from ..observability.metrics import InMemoryMetrics
from ..skill.provider import SkillProvider
from .persistence.facade import Storage
from ..subagent.provider import SubagentProvider
from ..swarm.engine import SwarmEngine
from ..tool.executor import GovernedToolInvoker

if TYPE_CHECKING:
    from ..agent.compiler import AgentCompiler as _AgentCompiler
    from ..agent.engine import AgentEngine as _AgentEngine
    from ..capability.resolver import CapabilityResolver as _CapabilityResolver
    from ..retrieval.retriever import Retriever
    from ..run.dispatch import RunDispatcher, RunDispatchRequest
    from ..run.models import RunResult
    from ..swarm.engine import SwarmEngine as _SwarmEngine


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Serializable runtime configuration values -- distinct from
    RuntimeDependencies (the process-in-memory object graph: storage,
    catalogs, capability providers, sandbox backend, etc.). The
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
    # The shape of the process graph this Runtime is assembled for. The build
    # kernel derives default capability minimums from this when the caller does
    # not pass explicit RuntimeRequirements; the capability gate stays the
    # single source of truth. Carried on RuntimeSettings (a serializable value)
    # so the topology travels with the rest of the configuration, not as a
    # loose build-time-only field.
    topology: RuntimeTopology = RuntimeTopology.SINGLE_PROCESS


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeComponents:
    """The fully-wired sub-components a Runtime drives. Built once by
    :func:`build_runtime_components`; Runtime unpacks these onto itself."""

    storage: Storage
    providers: RuntimeDependencies
    options: CapabilityRuntimeOptions
    model_resolver: ModelResolver
    compiler: "_AgentCompiler"
    runner: "_AgentEngine"
    swarm_engine: "_SwarmEngine"
    run_controller: RunController
    capability_resolver: "_CapabilityResolver | None"
    tool_executor: GovernedToolInvoker
    sandbox: "Sandbox | None"
    mcp_connection_pool: "MCPConnectionPool | None"
    commit_coordinator: Any = None
    run_coordinator: Any = None
    settings: RuntimeSettings = dataclasses.field(default_factory=RuntimeSettings)
    schema_registry: OutputSchemaRegistry = dataclasses.field(default_factory=OutputSchemaRegistry)
    metrics: Any = dataclasses.field(default_factory=InMemoryMetrics)
    authorization: Any = None


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeBuildConfig:
    """The final set of inputs build_runtime accepts. Capability
    providers come exclusively via ``providers``."""

    storage: Storage
    providers: RuntimeDependencies
    # The cross-store commit coordinator. REQUIRED (no default): the build
    # kernel no longer selects one based on Storage type -- the composition
    # root (the caller that constructs this config) picks the concrete
    # coordinator and injects it. ``build_runtime_components`` fail-closes if
    # this is None rather than silently degrading.
    commit_coordinator: "RunCommitCoordinator | None"
    # Optional FencedRunEventWriter. When None, build_runtime_components
    # constructs the backend-appropriate writer from the Storage if the
    # Storage exposes one (FilesystemStorage / SqlAlchemyStorage do); callers
    # that want to disable fenced security-event appending pass None
    # explicitly via a sentinel (rare -- tests/local mode).
    fenced_event_writer: Any = None
    # REQUIRED SwarmCommitCoordinator. The composition root (build_runtime or
    # the direct caller of build_runtime_components) constructs the concrete
    # coordinator from its concrete Storage and injects it; the build kernel
    # never branches on Storage type to dispatch a backend.
    swarm_commit_coordinator: Any = None
    # Typed skill-private-subagent wiring (was 5 Any fields on RuntimeDependencies).
    # Injected straight into the SubagentProvider / SkillProvider; never flows
    # through RuntimeDependencies (which would cycle providers <-> skill/subagent).
    skill_subagent: "SkillPrivateSubagentConfig | None" = None
    model_resolver: "ModelResolver | None" = None
    middleware_pipeline: "MiddlewarePipeline | None" = None
    retriever: "Retriever | None" = None
    sandbox: "Sandbox | None" = None
    tool_executor: "GovernedToolInvoker | None" = None
    security: Any = None
    capability_options: "CapabilityRuntimeOptions | None" = None
    mcp_connection_pool: "MCPConnectionPool | None" = None
    settings: RuntimeSettings = dataclasses.field(default_factory=RuntimeSettings)
    schema_registry: OutputSchemaRegistry | None = None
    metrics: Any = None
    authorization: Any = None
    # Optional capability minimums the topology declares it needs; the
    # capability gate (enforce_storage_capability_gate) refuses a Storage whose
    # StorageFeatures fall below these at build time. None = derive from
    # ``settings.topology`` (single-process -> no minimums; multi-worker ->
    # distributed coordination + leasing + fencing).
    requirements: "RuntimeRequirements | None" = None


def _capability_attribute_from_features(features) -> "dict[str, str]":
    """Map a StorageFeatures snapshot to a stable low-cardinality capability
    label for the build-failure metrics. Each flag becomes the attribute value
    it contributes to the gate's shortfall; the actual gate message is the
    authority on which capability fell short, but the metric only needs the
    bounded set of capability names."""
    try:
        return {
            "coordination": getattr(features.coordination_scope, "value", "unknown"),
            "transactions": getattr(features.transaction_scope, "value", "unknown"),
        }
    except Exception:  # noqa: BLE001 - attributes are observability-only
        return {}


def _build_capability_registry(
    bundle: RuntimeDependencies,
    sandbox: "Sandbox | None",
    options: CapabilityRuntimeOptions,
    mcp_manager: "MCPConnectionPool | None",
    subagent_executor: Any = None,
    skill_subagent: "SkillPrivateSubagentConfig | None" = None,
) -> "CapabilityProviderRegistry | None":
    """Map the declaration bundle onto the runtime CapabilityProviderRegistry
    the resolver dispatches over. Builtin is registered only when an sandbox
    backend exists (it cannot resolve without one). The subagent executor is
    passed in so both SubagentProvider and ExtensionProvider receive it at
    construction. ``skill_subagent`` carries the typed skill-private-subagent
    wiring (formerly 5 Any fields on RuntimeDependencies) into the SkillProvider /
    SubagentProvider. Returns None when no provider is wired (so the runner
    treats the run as having no capability resolution)."""
    registry = CapabilityProviderRegistry()
    sk = skill_subagent or SkillPrivateSubagentConfig.empty()
    if sandbox is not None:
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
    if bundle.extension_content is not None or bundle.entrypoints is not None:
        # ExtensionProvider declares every kind it handles via supported_kinds;
        # replace() registers the one instance under all of them.
        pkg = ExtensionProvider(
            content_source=bundle.extension_content,
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


def _resolve_swarm_commit_coordinator(config: "RuntimeBuildConfig") -> Any:
    """Return the SwarmCommitCoordinator to wire into SwarmEngine. The
    composition root constructs the concrete coordinator from its concrete
    Storage and passes it via RuntimeBuildConfig.swarm_commit_coordinator;
    this resolver just reads that field. The build kernel never branches on
    Storage type -- doing so would import concrete backend modules as a side
    effect of assembly, breaking the no-concrete-backends invariant.

    Fails the build if no coordinator was injected: SwarmEngine requires one
    and a silent None would defeat commit-log idempotency."""
    if config.swarm_commit_coordinator is None:
        from ..errors import SwarmCommitCoordinatorUnavailableError

        raise SwarmCommitCoordinatorUnavailableError(
            "RuntimeBuildConfig.swarm_commit_coordinator is None -- the "
            "composition root must construct the backend-appropriate "
            "SwarmCommitCoordinator from its Storage and inject it. The "
            "build kernel never dispatches a backend itself."
        )
    return config.swarm_commit_coordinator


def build_runtime_components(config: RuntimeBuildConfig) -> RuntimeComponents:
    """Assemble the compiler/runner/resolver/executor graph from a
    RuntimeBuildConfig. Internal splits: resolve security + exposure -> build
    executor -> build capability resolver -> build compiler -> build runner ->
    build swarm runner."""
    if config.commit_coordinator is None:
        # The build kernel no longer selects a coordinator based on Storage
        # type -- the composition root must inject one. Fail fast at build
        # time rather than silently degrading to a no-op commit path.
        from ..errors import RuntimeInitializationError

        raise RuntimeInitializationError(
            "RunCommitCoordinator must be injected; the build kernel no "
            "longer selects one"
        )
    if config.settings.multi_tenant and config.sandbox is not None:
        # Protocol runtime checks cannot distinguish a trusted local backend;
        # use its declared isolation level when available.
        if getattr(config.sandbox, "isolation_level", None) == ExecutionIsolationLevel.TRUSTED_LOCAL and not config.settings.local_trusted_mode:
            from ..errors import UnsafeSandboxError

            raise UnsafeSandboxError(
                "LocalSandbox is trusted-only and cannot serve multi-tenant runs"
            )
    if config.storage.run_definitions is None:
        # Fail fast at build time, not when a subagent/worker tool first
        # pauses on approval and Runtime.resume(child_run_id) cannot find a
        # snapshot. Resumability is a required capability, not opt-in.
        from ..errors import RuntimeInitializationError

        raise RuntimeInitializationError(
            "RunDefinitionStore is required for resumable runs"
        )
    # Pure-capability gate: ALWAYS derive the topology baseline first
    # (single-process -> no minimums; multi-worker -> distributed coordination +
    # leasing + fencing), then MERGE any explicit requirements on top so the
    # effective requirements meet-or-exceed BOTH -- an explicit declaration can
    # only STRENGTHEN what the topology demands, never replace/weaken it (the old
    # replace semantics let ``MULTI_WORKER + RuntimeRequirements()`` bypass the
    # distributed-coordination/leasing/fencing the topology requires). The gate
    # then refuses a Storage whose StorageFeatures fall below the merged result.
    from ..errors import StorageRequirementsNotMetError
    from ..run.requirements import (
        enforce_storage_capability_gate,
        enforce_storage_feature_consistency,
    )

    baseline = derive_runtime_requirements(
        settings=config.settings,
        # The gate reads dependencies.storage / .run_commit_coordinator for
        # real-object checks; populate them from the composition root so the
        # effective dependencies carry the wired objects, not the
        # spec-providers-only bundle the caller may have passed.
        dependencies=dataclasses.replace(
            config.providers,
            storage=config.storage,
            run_commit_coordinator=config.commit_coordinator,
        ),
    )
    effective_requirements = merge_runtime_requirements(
        baseline,
        config.requirements
        if config.requirements is not None
        else RuntimeRequirements(),
    )
    build_metrics = config.metrics
    try:
        enforce_storage_capability_gate(
            config.storage.features, effective_requirements
        )
        # Beyond feature FLAGS, verify the Storage wired the real OBJECTS behind
        # them (a backend declaring streaming_blobs without an ArtifactStore, or
        # transactions without a manager, would otherwise AttributeError at first
        # use).
        enforce_storage_feature_consistency(config.storage)
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
    # The resolver owns the whole ModelPolicy resolution: it walks primary +
    # fallbacks once and wraps multiple registered candidates in a pydantic-ai
    # FallbackModel (request-layer fallback). There is no separate gateway --
    # retry/fallback no longer live at the registry-lookup layer.
    model_resolver = config.model_resolver or ModelResolver()

    resolved_executor = config.tool_executor
    if resolved_executor is None:
        from ..governance.policy.engine import PolicyEngine

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
            receipt_store=config.storage.artifacts,
            tenant_id_resolver=lambda context: context.tenant_id,
        )
    compiler = AgentCompiler(
        model_resolver=model_resolver,
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
        mcp_manager = config.mcp_connection_pool or MCPConnectionPool()
    # The LateBoundRunDispatcher is always created so BOTH SubagentExecutor and
    # SwarmEngine resolve child-run dispatch to the same CoordinatorRunDispatcher
    # (bound after RunCoordinator is constructed below). It is no longer bound
    # to the AgentEngine -- AgentEngine is a pure execution loop with no
    # dispatch/run lifecycle entry of its own.
    dispatcher_handle = LateBoundRunDispatcher()
    sub_executor = None
    if bundle.entrypoints is not None or bundle.subagents is not None:
        sub_executor = SubagentExecutor(
            storage=config.storage,
            compiler=compiler,
            dispatcher=dispatcher_handle,
        )
    capability_registry = _build_capability_registry(
        bundle,
        config.sandbox,
        resolved_options,
        mcp_manager,
        sub_executor,
        config.skill_subagent,
    )
    resolver = (
        CapabilityResolver(capability_registry) if capability_registry else None
    )

    runner = AgentEngine(
        middleware_pipeline=config.middleware_pipeline,
        memory_store=config.storage.memories,
        retriever=config.retriever,
        sandbox=config.sandbox,
        capability_options=resolved_options,
        capability_resolver=resolver,
        security_pipeline=runner_pipeline,
        baseline_policy=runner_baseline_policy,
        tool_policy_provider=runner_policy_provider,
        managed_tool_executor=resolved_executor,
        security_audit_failure_mode=getattr(
            baseline, "audit_failure_mode", "fail_closed"
        ),
    )
    swarm_engine = SwarmEngine(
        swarm_store=config.storage.swarms,
        compiler=compiler,
        dispatcher=dispatcher_handle,
        swarm_commit_coordinator=_resolve_swarm_commit_coordinator(config),
    )

    if config.authorization is None:
        from ..governance.security.authorization import DenyAllAuthorization, ScopeAuthorization

        authorization = ScopeAuthorization() if config.settings.local_trusted_mode else DenyAllAuthorization()
    else:
        authorization = config.authorization

    # RunCoordinator (sole Run-lifecycle owner) with explicit deps, built after
    # agent_engine + swarm_engine so it can wrap both. CoordinatorRunDispatcher
    # then resolves the LateBoundRunDispatcher so Swarm/Subagent child-run
    # dispatch goes through the Coordinator (not AgentEngine).
    from ..clock import SystemClock
    from ..run.coordinator import RunCoordinator
    from ..run.live_events import RunLiveEventHub
    from ..run.preparation import RunPreparationCoordinator
    from ..session.reader import SessionReader
    from .dispatcher import CoordinatorRunDispatcher

    schema_registry = config.schema_registry or OutputSchemaRegistry()
    metrics = config.metrics or InMemoryMetrics()
    run_coordinator = RunCoordinator(
        storage=config.storage,
        compiler=compiler,
        agent_engine=runner,
        swarm_engine=swarm_engine,
        commit_coordinator=config.commit_coordinator,
        run_controller=run_controller,
        live_event_hub=RunLiveEventHub(),
        session_reader=SessionReader(config.storage.sessions),
        preparation=RunPreparationCoordinator(config.storage.run_definitions),
        clock=SystemClock(),
        authorization=authorization,
        settings=config.settings,
        schema_registry=schema_registry,
        metrics=metrics,
        model_resolver=model_resolver,
        fenced_event_writer=config.fenced_event_writer,
    )
    dispatcher_handle.bind(CoordinatorRunDispatcher(run_coordinator))

    return RuntimeComponents(
        storage=config.storage,
        providers=bundle,
        options=resolved_options,
        model_resolver=model_resolver,
        compiler=compiler,
        runner=runner,
        swarm_engine=swarm_engine,
        run_controller=run_controller,
        capability_resolver=resolver,
        tool_executor=resolved_executor,
        sandbox=config.sandbox,
        mcp_connection_pool=mcp_manager,
        commit_coordinator=config.commit_coordinator,
        run_coordinator=run_coordinator,
        settings=config.settings,
        schema_registry=schema_registry,
        metrics=metrics,
        authorization=authorization,
    )
