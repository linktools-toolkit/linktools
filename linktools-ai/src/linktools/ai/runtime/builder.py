"""The only runtime composition root."""

from dataclasses import replace

from ..agent.assembly.assembler import AgentAssembler
from ..agent.codec import AgentSpecCodec
from ..agent.compiler import AgentCompiler
from ..agent.engine import AgentEngine
from ..agent.tool.pydantic_ai import PydanticAIToolAdapter
from ..agent.tool.schema import JsonSchemaToolValidator
from ..agent.tool.service import ToolExecutionService
from ..agent.tool.exposure import ToolAssembler
from ..agent.builtin import BuiltinToolProvider
from ..errors import RuntimeInitializationError, StorageFeatureSupportError
from ..execution.live_events import NoopRunLiveEventSink, NoopSecurityEventSink
from ..execution.query import ExecutionQueryService
from ..execution.service import ExecutionService
from ..execution import trace_codec
from ..model.resolver import ModelResolver
from ..storage.database import CoordinationScope
from .dependencies import RuntimeDependencies
from .facade import Runtime
from .requirements import RuntimeRequirements, RuntimeTopology
from .storage import RuntimeStorage


def build_runtime(
    *,
    storage: RuntimeStorage,
    dependencies: RuntimeDependencies | None = None,
    requirements: RuntimeRequirements = RuntimeRequirements(),
    model_resolver: ModelResolver | None = None,
) -> Runtime:
    if requirements.topology is RuntimeTopology.MULTI_PROCESS:
        coordinated = {
            "execution": storage.execution,
            "tools": storage.tools,
            "tasks": storage.tasks,
        }
        for name, store in coordinated.items():
            if store is not None and (
                store.coordination_scope is not CoordinationScope.SHARED_DATABASE
            ):
                raise StorageFeatureSupportError(
                    f"{name} storage does not support multi-process coordination"
                )
    dependencies = dependencies or RuntimeDependencies(
        model_resolver=model_resolver or ModelResolver()
    )
    if requirements.tools and storage.tools is None:
        raise StorageFeatureSupportError(
            "tools are required but no tool state store was configured"
        )
    if requirements.tools and dependencies is not None and dependencies.tool_policy is None:
        raise RuntimeInitializationError(
            "tools are required but no ToolPolicyResolver was configured"
        )

    registry = dependencies.feature_registry
    if registry.get("builtin") is None:
        registry.register(BuiltinToolProvider())
    providers = (
        dependencies.skill_provider,
        dependencies.subagent_provider,
        dependencies.extension_provider,
        (
            replace(dependencies.mcp_provider, policy=dependencies.mcp_policy)
            if dependencies.mcp_provider is not None
            else None
        ),
    )
    for provider in providers:
        if provider is not None:
            registry.register(provider)
    registry.freeze()

    security_events = dependencies.security_events or NoopSecurityEventSink()
    tool_execution = ToolExecutionService(
        state=storage.tools,
        policy=dependencies.tool_policy,
        policy_engine=dependencies.tool_policy_engine,
        security=dependencies.tool_security,
        policy_revision=dependencies.tool_policy_revision,
    )
    tool_adapter = PydanticAIToolAdapter(tool_execution)
    tool_assembler = ToolAssembler(
        exposure=requirements.tool_exposure,
        schema_validator=JsonSchemaToolValidator(),
    )
    assembler = AgentAssembler(
        registry=registry,
        tool_assembler=tool_assembler,
        events=security_events,
    )
    compiler = AgentCompiler(
        model_resolver=dependencies.model_resolver,
        tool_executor=tool_execution,
        middleware_pipeline=dependencies.middleware,
        output_types=dependencies.output_types,
    )
    engine = AgentEngine(
        assembler=assembler,
        tool_adapter=tool_adapter,
        middleware_pipeline=dependencies.middleware,
        memory_store=dependencies.memory,
        retriever=dependencies.retrieval,
        sandbox=dependencies.sandbox,
        security_pipeline=dependencies.tool_security,
        observability=dependencies.observability,
        metrics=dependencies.metrics,
        pricing_provider=dependencies.pricing,
        session_window=dependencies.session_window,
        trace_codec=trace_codec,
    )
    codec = AgentSpecCodec(output_types=dependencies.output_types)
    execution = ExecutionService(
        storage.execution,
        compiler,
        engine=engine,
        assembler=assembler,
        tool_execution_ready=(
            storage.tools is not None
            and dependencies.tool_policy is not None
        ),
        sandbox=dependencies.sandbox,
        spec_codec=codec,
        authorization=dependencies.authorization,
        live_events=dependencies.live_events or NoopRunLiveEventSink(),
        security_events=security_events,
    )
    return Runtime(
        execution=execution,
        query=ExecutionQueryService(
            storage.execution,
            authorization=dependencies.authorization,
        ),
        assembler=assembler,
        tool_execution_ready=(
            storage.tools is not None
            and dependencies.tool_policy is not None
        ),
        sandbox=dependencies.sandbox,
        mcp_connections=(
            dependencies.mcp_provider.connections
            if dependencies.mcp_provider is not None
            else None
        ),
    )


__all__ = ["build_runtime"]
