#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Policy enums, ToolPolicyMetadata, and the PolicyRule/PolicyDecision/ToolRequest/
ToolContext types they revolve around. defines the rule protocol; the
rule modules live in permission.py / risk.py / path.py / network.py / approval.py /
command.py, and PolicyEngine (engine.py) composes them. The policy-relevant slice
of a tool's declaration is ToolPolicyMetadata below -- the full
ToolSpec lands in tool/spec.py."""

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
    # Carried through from ToolSpec so the policy adapter can map them into
    # ResolvedToolPolicy instead of silently dropping them. Defaults keep
    # existing constructions (tests, older providers) working unchanged.
    idempotent: bool = False
    timeout_seconds: "float | None" = None
    schema_version: str = "1"
    enabled: bool = True
    max_retries: "int | None" = None
    idempotency_strategy: "str | None" = None
    idempotency_key_field: "str | None" = None
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


@runtime_checkable
class ToolPolicyMetadataSource(Protocol):
    """Provides a tool-name -> ToolPolicyMetadata map from any source (a YAML
    ToolRegistry, a DB, any business source). The Runtime consumes the map to
    enforce Permission/Risk/Approval rules."""

    async def get_metadata_map(self) -> "Mapping[str, ToolPolicyMetadata]": ...


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
    # Descriptor-sourced classification, when available (e.g. from
    # ManagedToolAdapter, which always knows its ToolDescriptor, or from
    # PolicyCapability when a per-run descriptor lookup is wired). Rules like
    # CommandRule match on these so a tool rename cannot silently evade a
    # category-based policy the way a tool_name string match could.
    category: "str | None" = None
    risk: "str | None" = None
    mutating: "bool | None" = None
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolContext:
    run_id: str
    session_id: str
    # pydantic-ai ToolCallPart.tool_call_id, threaded through by PolicyCapability
    # so GovernedToolInvoker can key ApprovalRequest.tool_call_id on the SAME id the
    # model's message history uses -- the linchpin of resume (a re-driven call
    # after approve() must find the matching approval). None when the context
    # is constructed outside a real pydantic-ai call (executor falls back to uuid).
    tool_call_id: "str | None" = None
    # Optional trusted principal propagated by Runtime into managed tools. A
    # missing principal is retained for local single-tenant compatibility, but
    # must never widen an idempotency scope.
    principal: Any = None
    tenant_id: str | None = None
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


@runtime_checkable
class PolicyRule(Protocol):
    async def evaluate(
        self, request: ToolRequest, context: ToolContext
    ) -> PolicyDecision: ...
