#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strongly-typed event payloads. Each
payload carries the minimum data meaningful for that event type -- the spec
mandates which payload TYPES must exist, not their exact fields."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Union


@dataclass(frozen=True, slots=True)
class RunStarted:
    run_id: str
    runnable_id: str


@dataclass(frozen=True, slots=True)
class RunCompleted:
    run_id: str
    result_summary: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunFailed:
    run_id: str
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class RunPaused:
    run_id: str
    reason: "str | None" = None


@dataclass(frozen=True, slots=True)
class RunResumed:
    run_id: str


@dataclass(frozen=True, slots=True)
class RunCancelled:
    run_id: str
    reason: "str | None" = None


@dataclass(frozen=True, slots=True)
class ModelStarted:
    model_type: str


@dataclass(frozen=True, slots=True)
class ModelCompleted:
    model_type: str
    token_usage: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelFailed:
    model_type: str
    error_message: str


@dataclass(frozen=True, slots=True)
class ToolStarted:
    tool_name: str
    tool_call_id: str


@dataclass(frozen=True, slots=True)
class ToolCompleted:
    tool_name: str
    tool_call_id: str
    success: bool
    execution_success: "bool | None" = None
    result_action: str = "returned"


@dataclass(frozen=True, slots=True)
class ToolFailed:
    tool_name: str
    tool_call_id: str
    error_message: str


@dataclass(frozen=True, slots=True)
class ApprovalRequested:
    approval_id: str
    tool_name: str
    reason: str


@dataclass(frozen=True, slots=True)
class ApprovalApproved:
    approval_id: str
    resolved_by: "str | None" = None


@dataclass(frozen=True, slots=True)
class ApprovalRejected:
    approval_id: str
    resolved_by: "str | None" = None
    reason: "str | None" = None


@dataclass(frozen=True, slots=True)
class SwarmStarted:
    swarm_run_id: str
    swarm_id: str


@dataclass(frozen=True, slots=True)
class SwarmRoundStarted:
    swarm_run_id: str
    round: int


@dataclass(frozen=True, slots=True)
class SwarmRoundCompleted:
    swarm_run_id: str
    round: int


@dataclass(frozen=True, slots=True)
class SwarmTaskCreated:
    swarm_run_id: str
    task_id: str
    description: str


@dataclass(frozen=True, slots=True)
class SwarmTaskClaimed:
    swarm_run_id: str
    task_id: str
    assigned_agent_id: str


@dataclass(frozen=True, slots=True)
class SwarmTaskCompleted:
    swarm_run_id: str
    task_id: str


@dataclass(frozen=True, slots=True)
class SwarmTaskFailed:
    swarm_run_id: str
    task_id: str
    error_message: str


@dataclass(frozen=True, slots=True)
class SwarmCompleted:
    swarm_run_id: str


@dataclass(frozen=True, slots=True)
class ResourceChanged:
    path: str
    revision: int


# --- Capability Runtime lifecycle events ---
# These let a downstream EventStore observe the capability/skill/mcp/subagent/
# package/prompt/tool-exposure lifecycle without coupling to internal classes.


@dataclass(frozen=True, slots=True)
class CapabilityResolveStarted:
    agent_id: str
    capability_ref: str


@dataclass(frozen=True, slots=True)
class CapabilityResolveCompleted:
    agent_id: str
    capability_ref: str
    tool_count: int


@dataclass(frozen=True, slots=True)
class SkillListed:
    agent_id: "str | None" = None
    query: "str | None" = None
    count: int = 0


@dataclass(frozen=True, slots=True)
class SkillRead:
    agent_id: "str | None" = None
    skill_id: str = ""
    allowed: bool = True


@dataclass(frozen=True, slots=True)
class PackageResourceListed:
    package_id: str = ""
    path: str = ""
    count: int = 0


@dataclass(frozen=True, slots=True)
class PackageResourceRead:
    package_id: str = ""
    path: str = ""
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class PackageEntrypointListed:
    package_id: str = ""
    kind: "str | None" = None
    count: int = 0


@dataclass(frozen=True, slots=True)
class PackageEntrypointResolved:
    package_id: str = ""
    kind: str = ""
    name: str = ""


@dataclass(frozen=True, slots=True)
class McpConnectStarted:
    server_id: str = ""


@dataclass(frozen=True, slots=True)
class McpConnectCompleted:
    server_id: str = ""
    tool_count: "int | None" = None


@dataclass(frozen=True, slots=True)
class McpToolCallStarted:
    server_id: str = ""
    tool_name: str = ""


@dataclass(frozen=True, slots=True)
class McpToolCallCompleted:
    server_id: str = ""
    tool_name: str = ""
    success: bool = True


@dataclass(frozen=True, slots=True)
class SubagentStarted:
    agent_id: str = ""
    parent_run_id: "str | None" = None
    scope: "str | None" = None


@dataclass(frozen=True, slots=True)
class SubagentCompleted:
    agent_id: str = ""
    run_id: str = ""
    status: str = ""


@dataclass(frozen=True, slots=True)
class SubagentErrored:
    agent_id: str = ""
    reason: str = ""


@dataclass(frozen=True, slots=True)
class PromptCatalogInjected:
    agent_id: "str | None" = None
    section: str = ""


@dataclass(frozen=True, slots=True)
class PromptWindowApplied:
    policy: str = ""
    before: int = 0
    after: int = 0


@dataclass(frozen=True, slots=True)
class ToolExposureApplied:
    agent_id: "str | None" = None
    total_tools: int = 0


@dataclass(frozen=True, slots=True)
class ToolExposureDenied:
    agent_id: "str | None" = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class SecurityDegraded:
    """Emitted when a security-relevant component fails and the system falls
    back to a safer-but-degraded posture rather than failing open -- e.g. a
    ToolPolicyProvider error caught and replaced with a fail-closed policy."""

    run_id: "str | None" = None
    component: str = ""
    reason: str = ""
    error_code: str | None = None
    server_id: str | None = None
    connection_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class TruncatedSecurityEvent:
    """Replaces an oversized security event so the audit store still receives a
    valid dataclass payload (FileEventStore persists via dataclasses.asdict and
    reconstructs by class name -- a plain dict would TypeError there). Carries
    only the original type name and the measured size; the original payload is
    deliberately dropped so a too-large event can never re-bloat the store."""

    original_event_type: str
    reason: str = "payload_too_large"
    original_size_bytes: int = 0


@dataclass(frozen=True, slots=True)
class ToolPolicyResolved:
    """tool.policy.resolved: the finalized policy that governs one tool call
    (enabled/timeout/retries/idempotent/approval/risk), for audit."""

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

    run_id: "str | None" = None
    tool_name: str = ""
    call_id: "str | None" = None


@dataclass(frozen=True, slots=True)
class ToolPipelineDecision:
    """tool.pipeline.decision: the decision a pipeline returned (allow/deny/
    require_approval/modify) for a tool call."""

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
    SwarmTaskCreated,
    SwarmTaskClaimed,
    SwarmTaskCompleted,
    SwarmTaskFailed,
    SwarmCompleted,
    ResourceChanged,
    CapabilityResolveStarted,
    CapabilityResolveCompleted,
    SkillListed,
    SkillRead,
    PackageResourceListed,
    PackageResourceRead,
    PackageEntrypointListed,
    PackageEntrypointResolved,
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
