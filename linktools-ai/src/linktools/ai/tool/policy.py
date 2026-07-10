#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ResolvedToolPolicy + ToolPolicyProvider Protocol + ToolInvocationContext.

ResolvedToolPolicy is the merged policy that governs a single tool invocation.
ToolPolicyProvider resolves a descriptor + run context into a policy.
ToolInvocationContext carries everything the governance chain needs."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

from ..security.descriptor import ToolDescriptor

if TYPE_CHECKING:
    from ..run.context import RunContext


@dataclass(frozen=True, slots=True)
class ResolvedToolPolicy:
    """Merged policy for a single tool invocation. Each field must have a real
    consumer in the execution chain."""
    enabled: bool = True
    timeout_seconds: "float | None" = None
    max_retries: int = 0
    idempotent: bool = False
    require_approval: bool = False
    risk: "str | None" = None
    metadata: "Mapping[str, Any]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0 or None, got {self.timeout_seconds}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")


@dataclass(frozen=True, slots=True)
class ToolInvocationContext:
    """Everything the governance chain (pipeline, executor, middleware, events)
    needs for one tool call."""
    descriptor: ToolDescriptor
    arguments: "Mapping[str, Any]"
    run_context: "RunContext"
    policy: ResolvedToolPolicy
    call_id: str


@runtime_checkable
class ToolPolicyProvider(Protocol):
    """Resolves a tool descriptor + run context into a ResolvedToolPolicy. When
    absent, the runtime uses ResolvedToolPolicy() (defaults) + SecurityBaseline."""

    async def resolve(
        self,
        descriptor: ToolDescriptor,
        context: "RunContext",
    ) -> ResolvedToolPolicy:
        ...


# --- Policy merge ---

def merge_policies(
    descriptor_default: "ResolvedToolPolicy | None",
    baseline: "ResolvedToolPolicy | None",
    provider: "ResolvedToolPolicy | None" = None,
) -> ResolvedToolPolicy:
    """Merge policies layer by layer. Rules:
    - enabled: any layer False -> False
    - timeout: smallest non-None value
    - max_retries: smallest value
    - idempotent: only True if all layers confirm
    - require_approval: any layer True -> True
    - risk: highest level
    """
    layers = [l for l in (descriptor_default, baseline, provider) if l is not None]
    if not layers:
        return ResolvedToolPolicy()

    enabled = all(l.enabled for l in layers)
    timeouts = [l.timeout_seconds for l in layers if l.timeout_seconds is not None]
    timeout = min(timeouts) if timeouts else None
    max_retries = min(l.max_retries for l in layers)
    idempotent = all(l.idempotent for l in layers)
    require_approval = any(l.require_approval for l in layers)

    risk_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    risks = [l.risk for l in layers if l.risk]
    risk = max(risks, key=lambda r: risk_order.get(r, 2)) if risks else None

    return ResolvedToolPolicy(
        enabled=enabled,
        timeout_seconds=timeout,
        max_retries=max_retries,
        idempotent=idempotent,
        require_approval=require_approval,
        risk=risk,
    )
