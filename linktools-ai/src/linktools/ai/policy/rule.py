#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Policy enums, ToolPolicyMetadata, and the PolicyRule/PolicyDecision/ToolRequest/
ToolContext types they revolve around. Section 25 defines the rule protocol; the
rule modules live in permission.py / risk.py / path.py / network.py / approval.py /
command.py, and PolicyEngine (engine.py) composes them. The policy-relevant slice
of a tool's declaration (section 26.1) is ToolPolicyMetadata below -- the full
ToolSpec lands in registry/tool.py (Task 11)."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable


class Permission(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    ADMIN = "admin"


class RiskLevel(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class SideEffectKind(str, Enum):
    NONE = "none"
    READ_ONLY = "read_only"
    NAMESPACE_MUTATING = "namespace_mutating"
    DESTRUCTIVE = "destructive"


class ApprovalMode(str, Enum):
    NEVER = "never"
    ON_RISK = "on_risk"
    ALWAYS = "always"


@dataclass(frozen=True, slots=True)
class ToolPolicyMetadata:
    permissions: "frozenset[Permission]"
    risk: RiskLevel
    side_effect: SideEffectKind
    approval: ApprovalMode


class PolicyDecisionKind(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    kind: PolicyDecisionKind
    rule_id: str
    reason: "str | None"
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolRequest:
    tool_name: str
    arguments: "Mapping[str, Any]"


@dataclass(frozen=True, slots=True)
class ToolContext:
    run_id: str
    session_id: str
    # pydantic-ai ToolCallPart.tool_call_id, threaded through by PolicyCapability
    # so ToolExecutor can key ApprovalRequest.tool_call_id on the SAME id the
    # model's message history uses -- the linchpin of resume (a re-driven call
    # after approve() must find the matching approval). None when the context
    # is constructed outside a real pydantic-ai call (executor falls back to uuid).
    tool_call_id: "str | None" = None
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


@runtime_checkable
class PolicyRule(Protocol):
    async def evaluate(self, request: ToolRequest, context: ToolContext) -> PolicyDecision:
        ...
