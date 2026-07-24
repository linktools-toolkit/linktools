#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strongly-typed event payloads. Each
payload carries the minimum data meaningful for that event type -- the spec
mandates which payload TYPES must exist, not their exact fields."""

from dataclasses import dataclass, field

from typing import Any, ClassVar, Mapping, Union

from .criticality import EventCriticality


@dataclass(frozen=True, slots=True)
class RunStarted:
    event_type: ClassVar[str] = 'RunStarted'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    run_id: str
    runnable_id: str


@dataclass(frozen=True, slots=True)
class RunCompleted:
    event_type: ClassVar[str] = 'RunCompleted'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    run_id: str
    result_summary: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunFailed:
    event_type: ClassVar[str] = 'RunFailed'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    run_id: str
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class RunPaused:
    event_type: ClassVar[str] = 'RunPaused'
    criticality: ClassVar[EventCriticality] = EventCriticality.STATE_CRITICAL
    run_id: str
    reason: "str | None" = None


@dataclass(frozen=True, slots=True)
class RunResumed:
    event_type: ClassVar[str] = 'RunResumed'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    run_id: str


@dataclass(frozen=True, slots=True)
class RunCancelled:
    event_type: ClassVar[str] = 'RunCancelled'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    run_id: str
    reason: "str | None" = None


@dataclass(frozen=True, slots=True)
class ModelStarted:
    event_type: ClassVar[str] = 'ModelStarted'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    model_type: str


@dataclass(frozen=True, slots=True)
class ModelCompleted:
    event_type: ClassVar[str] = 'ModelCompleted'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    model_type: str
    token_usage: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelFailed:
    event_type: ClassVar[str] = 'ModelFailed'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    model_type: str
    error_message: str


@dataclass(frozen=True, slots=True)
class ToolStarted:
    event_type: ClassVar[str] = 'ToolStarted'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    tool_name: str
    tool_call_id: str


@dataclass(frozen=True, slots=True)
class ToolCompleted:
    event_type: ClassVar[str] = 'ToolCompleted'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    tool_name: str
    tool_call_id: str
    success: bool
    execution_success: "bool | None" = None
    result_action: str = "returned"


@dataclass(frozen=True, slots=True)
class ToolFailed:
    event_type: ClassVar[str] = 'ToolFailed'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    tool_name: str
    tool_call_id: str
    error_message: str


@dataclass(frozen=True, slots=True)
class ApprovalRequested:
    event_type: ClassVar[str] = 'ApprovalRequested'
    criticality: ClassVar[EventCriticality] = EventCriticality.STATE_CRITICAL
    approval_id: str
    tool_name: str
    reason: str


@dataclass(frozen=True, slots=True)
class ApprovalApproved:
    event_type: ClassVar[str] = 'ApprovalApproved'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    approval_id: str
    resolved_by: "str | None" = None


@dataclass(frozen=True, slots=True)
class ApprovalRejected:
    event_type: ClassVar[str] = 'ApprovalRejected'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    approval_id: str
    resolved_by: "str | None" = None
    reason: "str | None" = None


@dataclass(frozen=True, slots=True)
class SwarmStarted:
    event_type: ClassVar[str] = 'SwarmStarted'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    swarm_run_id: str
    swarm_id: str


@dataclass(frozen=True, slots=True)
class SwarmRoundStarted:
    event_type: ClassVar[str] = 'SwarmRoundStarted'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    swarm_run_id: str
    round: int


@dataclass(frozen=True, slots=True)
class SwarmRoundCompleted:
    event_type: ClassVar[str] = 'SwarmRoundCompleted'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    swarm_run_id: str
    round: int


@dataclass(frozen=True, slots=True)
class SwarmStepCreated:
    event_type: ClassVar[str] = 'SwarmStepCreated'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    swarm_run_id: str
    task_id: str
    description: str


@dataclass(frozen=True, slots=True)
class SwarmStepClaimed:
    event_type: ClassVar[str] = 'SwarmStepClaimed'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    swarm_run_id: str
    task_id: str
    assigned_agent_id: str


@dataclass(frozen=True, slots=True)
class SwarmStepCompleted:
    event_type: ClassVar[str] = 'SwarmStepCompleted'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    swarm_run_id: str
    task_id: str


@dataclass(frozen=True, slots=True)
class SwarmStepFailed:
    event_type: ClassVar[str] = 'SwarmStepFailed'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    swarm_run_id: str
    task_id: str
    error_message: str


@dataclass(frozen=True, slots=True)
class SwarmCompleted:
    event_type: ClassVar[str] = 'SwarmCompleted'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    swarm_run_id: str


@dataclass(frozen=True, slots=True)
class SwarmFailed:
    event_type: ClassVar[str] = 'SwarmFailed'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    swarm_run_id: str
    error: str


@dataclass(frozen=True, slots=True)
class SwarmCancelled:
    event_type: ClassVar[str] = 'SwarmCancelled'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    swarm_run_id: str


@dataclass(frozen=True, slots=True)
class AssetChanged:
    event_type: ClassVar[str] = 'AssetChanged'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    path: str
    revision: int


# --- Capability Runtime lifecycle events ---
# These let a downstream EventStore observe the capability/skill/mcp/subagent/
# package/prompt/tool-exposure lifecycle without coupling to internal classes.


@dataclass(frozen=True, slots=True)
class CapabilityResolveStarted:
    event_type: ClassVar[str] = 'CapabilityResolveStarted'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    agent_id: str
    capability_ref: str


@dataclass(frozen=True, slots=True)
class CapabilityResolveCompleted:
    event_type: ClassVar[str] = 'CapabilityResolveCompleted'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    agent_id: str
    capability_ref: str
    tool_count: int


@dataclass(frozen=True, slots=True)
class SkillListed:
    event_type: ClassVar[str] = 'SkillListed'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    agent_id: "str | None" = None
    query: "str | None" = None
    count: int = 0


@dataclass(frozen=True, slots=True)
class SkillRead:
    event_type: ClassVar[str] = 'SkillRead'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    agent_id: "str | None" = None
    skill_id: str = ""
    allowed: bool = True


@dataclass(frozen=True, slots=True)
class ExtensionContentListed:
    event_type: ClassVar[str] = 'ExtensionContentListed'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    extension_id: str = ""
    path: str = ""
    count: int = 0


@dataclass(frozen=True, slots=True)
class ExtensionContentRead:
    event_type: ClassVar[str] = 'ExtensionContentRead'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    extension_id: str = ""
    path: str = ""
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class ExtensionEntrypointListed:
    event_type: ClassVar[str] = 'ExtensionEntrypointListed'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    extension_id: str = ""
    kind: "str | None" = None
    count: int = 0


@dataclass(frozen=True, slots=True)
class ExtensionEntrypointResolved:
    event_type: ClassVar[str] = 'ExtensionEntrypointResolved'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    extension_id: str = ""
    kind: str = ""
    name: str = ""


@dataclass(frozen=True, slots=True)
class McpConnectStarted:
    event_type: ClassVar[str] = 'McpConnectStarted'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    server_id: str = ""


@dataclass(frozen=True, slots=True)
class McpConnectCompleted:
    event_type: ClassVar[str] = 'McpConnectCompleted'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    server_id: str = ""
    tool_count: "int | None" = None


@dataclass(frozen=True, slots=True)
class McpToolCallStarted:
    event_type: ClassVar[str] = 'McpToolCallStarted'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    server_id: str = ""
    tool_name: str = ""


@dataclass(frozen=True, slots=True)
class McpToolCallCompleted:
    event_type: ClassVar[str] = 'McpToolCallCompleted'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    server_id: str = ""
    tool_name: str = ""
    success: bool = True


@dataclass(frozen=True, slots=True)
class SubagentStarted:
    event_type: ClassVar[str] = 'SubagentStarted'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    agent_id: str = ""
    parent_run_id: "str | None" = None
    scope: "str | None" = None


@dataclass(frozen=True, slots=True)
class SubagentCompleted:
    event_type: ClassVar[str] = 'SubagentCompleted'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    agent_id: str = ""
    run_id: str = ""
    status: str = ""


@dataclass(frozen=True, slots=True)
class SubagentErrored:
    event_type: ClassVar[str] = 'SubagentErrored'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    agent_id: str = ""
    reason: str = ""


@dataclass(frozen=True, slots=True)
class PromptCatalogInjected:
    event_type: ClassVar[str] = 'PromptCatalogInjected'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    agent_id: "str | None" = None
    section: str = ""


@dataclass(frozen=True, slots=True)
class PromptWindowApplied:
    event_type: ClassVar[str] = 'PromptWindowApplied'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    policy: str = ""
    before: int = 0
    after: int = 0


@dataclass(frozen=True, slots=True)
class ToolExposureApplied:
    event_type: ClassVar[str] = 'ToolExposureApplied'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY
    agent_id: "str | None" = None
    total_tools: int = 0


@dataclass(frozen=True, slots=True)
class ToolExposureDenied:
    event_type: ClassVar[str] = 'ToolExposureDenied'
    criticality: ClassVar[EventCriticality] = EventCriticality.SECURITY_CRITICAL
    agent_id: "str | None" = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class SecurityDegraded:
    """Emitted when a security-relevant component fails and the system falls
    back to a safer-but-degraded posture rather than failing open -- e.g. a
    ToolPolicyProvider error caught and replaced with a fail-closed policy."""
    event_type: ClassVar[str] = 'SecurityDegraded'
    criticality: ClassVar[EventCriticality] = EventCriticality.SECURITY_CRITICAL

    run_id: "str | None" = None
    component: str = ""
    reason: str = ""
    error_code: str | None = None
    server_id: str | None = None
    connection_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class TruncatedSecurityEvent:
    """Replaces an oversized security event so the audit store still receives a
    valid dataclass payload (FilesystemEventStore persists via dataclasses.asdict and
    reconstructs by class name -- a plain dict would TypeError there). Carries
    only the original type name and the measured size; the original payload is
    deliberately dropped so a too-large event can never re-bloat the store."""
    event_type: ClassVar[str] = 'TruncatedSecurityEvent'
    criticality: ClassVar[EventCriticality] = EventCriticality.OBSERVABILITY

    original_event_type: str
    reason: str = "payload_too_large"
    original_size_bytes: int = 0


@dataclass(frozen=True, slots=True)
class ToolPolicyResolved:
    """tool.policy.resolved: the finalized policy that governs one tool call
    (enabled/timeout/retries/idempotent/approval/risk), for audit."""
    event_type: ClassVar[str] = 'ToolPolicyResolved'
    criticality: ClassVar[EventCriticality] = EventCriticality.SECURITY_CRITICAL

    run_id: "str | None" = None
    tool_name: str = ""
    enabled: bool = True
    timeout_seconds: "float | None" = None
    max_retries: int = 0
    idempotent: bool = False
    require_approval: bool = False
    risk: str = "medium"


@dataclass(frozen=True, slots=True)
class ToolPipelineBefore:
    """tool.pipeline.before: a SecurityPipeline's before_tool was consulted for
    a tool call."""
    event_type: ClassVar[str] = 'ToolPipelineBefore'
    criticality: ClassVar[EventCriticality] = EventCriticality.SECURITY_CRITICAL

    run_id: "str | None" = None
    tool_name: str = ""
    call_id: "str | None" = None


@dataclass(frozen=True, slots=True)
class ToolPipelineDecision:
    """tool.pipeline.decision: the decision a pipeline returned (allow/deny/
    require_approval/modify) for a tool call."""
    event_type: ClassVar[str] = 'ToolPipelineDecision'
    criticality: ClassVar[EventCriticality] = EventCriticality.SECURITY_CRITICAL

    run_id: "str | None" = None
    tool_name: str = ""
    call_id: "str | None" = None
    action: str = "allow"
    reason: str = ""
    stage: str = "before"


@dataclass(frozen=True, slots=True)
class ToolPipelineAfter:
    """tool.pipeline.after: a SecurityPipeline's after_tool was consulted for a
    completed tool call."""
    event_type: ClassVar[str] = 'ToolPipelineAfter'
    criticality: ClassVar[EventCriticality] = EventCriticality.SECURITY_CRITICAL

    run_id: "str | None" = None
    tool_name: str = ""
    call_id: "str | None" = None
    success: bool = True


# Union of every event payload type. This is the type of the ``payload`` field
# EventStore.append accepts -- callers pass a concrete
# payload instance and the store wraps it in an EventEnvelope.
EventPayload = Union[
    RunStarted,
    RunCompleted,
    RunFailed,
    RunPaused,
    RunResumed,
    RunCancelled,
    ModelStarted,
    ModelCompleted,
    ModelFailed,
    ToolStarted,
    ToolCompleted,
    ToolFailed,
    ApprovalRequested,
    ApprovalApproved,
    ApprovalRejected,
    SwarmStarted,
    SwarmRoundStarted,
    SwarmRoundCompleted,
    SwarmStepCreated,
    SwarmStepClaimed,
    SwarmStepCompleted,
    SwarmStepFailed,
    SwarmCompleted,
    AssetChanged,
    CapabilityResolveStarted,
    CapabilityResolveCompleted,
    SkillListed,
    SkillRead,
    ExtensionContentListed,
    ExtensionContentRead,
    ExtensionEntrypointListed,
    ExtensionEntrypointResolved,
    McpConnectStarted,
    McpConnectCompleted,
    McpToolCallStarted,
    McpToolCallCompleted,
    SubagentStarted,
    SubagentCompleted,
    SubagentErrored,
    PromptCatalogInjected,
    PromptWindowApplied,
    ToolExposureApplied,
    ToolExposureDenied,
    SecurityDegraded,
    TruncatedSecurityEvent,
    ToolPolicyResolved,
    ToolPipelineBefore,
    ToolPipelineDecision,
    ToolPipelineAfter,
]
