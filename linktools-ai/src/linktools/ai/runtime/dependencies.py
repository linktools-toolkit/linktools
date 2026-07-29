"""Typed inputs to the runtime composition root."""

from dataclasses import dataclass, field

from ..agent.assembly.registry import AgentFeatureRegistry
from ..agent.codec import OutputTypeRegistry
from ..agent.integrations.mcp.models import MCPRuntimePolicy
from ..agent.integrations.mcp.tool_provider import MCPToolProvider
from ..agent.extension.provider import ExtensionProvider
from ..agent.skill.provider import SkillProvider
from ..agent.subagent.provider import SubagentProvider
from ..agent.middleware.pipeline import MiddlewarePipeline
from ..agent.prompt.window import SessionWindowPolicy
from ..agent.memory.store import MemoryStore
from ..agent.retrieval.retriever import Retriever
from ..agent.tool.sandbox.protocols import Sandbox
from ..agent.tool.policy.resolver import ToolPolicyResolver
from ..execution.live_events import RunLiveEventSink, SecurityEventSink
from ..governance.authorization import (
    AuthorizationPolicy,
    OwnershipAuthorizationPolicy,
)
from ..governance.policy.engine import PolicyEngine
from ..governance.security.pipeline import SecurityPipeline
from ..model.pricing import ModelPricingProvider
from ..model.resolver import ModelResolver
from ..observability.metrics import ObservabilityMetrics
from ..observability.tracing import ObservabilitySink


@dataclass(frozen=True, slots=True)
class RuntimeDependencies:
    model_resolver: ModelResolver
    output_types: OutputTypeRegistry = field(default_factory=OutputTypeRegistry)
    feature_registry: AgentFeatureRegistry = field(default_factory=AgentFeatureRegistry)
    authorization: AuthorizationPolicy = field(
        default_factory=OwnershipAuthorizationPolicy
    )
    skill_provider: SkillProvider | None = None
    subagent_provider: SubagentProvider | None = None
    extension_provider: ExtensionProvider | None = None
    mcp_provider: MCPToolProvider | None = None
    middleware: MiddlewarePipeline | None = None
    session_window: SessionWindowPolicy | None = None
    memory: MemoryStore | None = None
    retrieval: Retriever | None = None
    sandbox: Sandbox | None = None
    tool_policy: ToolPolicyResolver | None = None
    tool_policy_engine: PolicyEngine | None = None
    tool_security: SecurityPipeline | None = None
    tool_policy_revision: str = "default"
    live_events: RunLiveEventSink | None = None
    security_events: SecurityEventSink | None = None
    observability: ObservabilitySink | None = None
    metrics: ObservabilityMetrics | None = None
    pricing: ModelPricingProvider | None = None
    mcp_policy: MCPRuntimePolicy = field(default_factory=MCPRuntimePolicy)


__all__ = ["RuntimeDependencies"]
