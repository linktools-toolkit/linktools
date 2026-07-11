#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ResolvedToolPolicy + ToolPolicyProvider Protocol + ToolInvocationContext.

ResolvedToolPolicy is a tri-state policy layer: every field is Optional so
"not declared by this layer" (None) is distinguishable from "explicitly
declared 0/False". merge_policies() combines layers under that distinction;
finalize_policy() then collapses the merged tri-state result into a concrete
EffectiveToolPolicy the execution chain consumes. ToolPolicyProvider resolves
a descriptor + run context into a (tri-state) policy layer.
ToolInvocationContext carries everything the governance chain needs."""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

from ..security.descriptor import ToolDescriptor
from ..utils.freeze import freeze_value


class IdempotencyStrategy(str, Enum):
    EXACT_CALL = "exact_call"
    BUSINESS_KEY = "business_key"

if TYPE_CHECKING:
    from ..run.context import RunContext


@dataclass(frozen=True, slots=True)
class ResolvedToolPolicy:
    """One policy layer for a single tool invocation. Every field is
    tri-state (``None`` = not declared by this layer) so a layer that omits a
    field never overrides a more specific layer's explicit 0/False -- only
    finalize_policy() collapses the merged result to concrete values."""
    enabled: "bool | None" = None
    timeout_seconds: "float | None" = None
    max_retries: "int | None" = None
    idempotent: "bool | None" = None
    require_approval: "bool | None" = None
    risk: "str | None" = None
    schema_version: "str | None" = None
    idempotency_strategy: "IdempotencyStrategy | None" = None
    idempotency_key_field: "str | None" = None
    metadata: "Mapping[str, Any]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_value(dict(self.metadata)))

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0 or None, got {self.timeout_seconds}")
        if self.max_retries is not None and self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")


@dataclass(frozen=True, slots=True)
class EffectiveToolPolicy:
    """The merged tri-state ResolvedToolPolicy collapsed to concrete values --
    what the execution chain (ManagedToolAdapter, ToolExecutor) actually acts
    on. Produced only by finalize_policy()."""
    enabled: bool = True
    timeout_seconds: "float | None" = None
    max_retries: int = 0
    idempotent: bool = False
    require_approval: bool = False
    risk: str = "medium"
    schema_version: str = "1"
    idempotency_strategy: IdempotencyStrategy = IdempotencyStrategy.EXACT_CALL
    idempotency_key_field: "str | None" = None
    metadata: "Mapping[str, Any]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_value(dict(self.metadata)))


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
    """Merge tri-state policy layers into a (still tri-state) ResolvedToolPolicy.
    A field left undeclared (None) by every layer stays None -- only
    finalize_policy() applies a concrete default. Rules (only over layers that
    declared the field; an undeclared layer never contributes):
    - enabled: any declared False -> False; else True if any layer declared it,
      else None
    - timeout: smallest declared value, else None
    - max_retries: smallest declared value, else None
    - idempotent: True only if every layer that declared it declared True
      (and at least one did); any declared False -> False; else None
    - require_approval: any declared True -> True; else False if any layer
      declared it, else None
    - risk: highest declared level, else None
    - metadata: shallow-merged across layers (later layers win per key) so no
      layer's metadata is silently dropped
    """
    layers = [l for l in (descriptor_default, baseline, provider) if l is not None]
    if not layers:
        return ResolvedToolPolicy()

    def _merge_bool_any_true(values: "list[bool]") -> "bool | None":
        if not values:
            return None
        return True if any(values) else False

    declared_enabled = [l.enabled for l in layers if l.enabled is not None]
    enabled = None
    if declared_enabled:
        enabled = False if any(v is False for v in declared_enabled) else True

    timeouts = [l.timeout_seconds for l in layers if l.timeout_seconds is not None]
    timeout = min(timeouts) if timeouts else None

    retries = [l.max_retries for l in layers if l.max_retries is not None]
    max_retries = min(retries) if retries else None

    declared_idempotent = [l.idempotent for l in layers if l.idempotent is not None]
    idempotent = None
    if declared_idempotent:
        idempotent = all(declared_idempotent)

    declared_approval = [l.require_approval for l in layers if l.require_approval is not None]
    require_approval = _merge_bool_any_true(declared_approval)

    risk_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    risks = [l.risk for l in layers if l.risk]
    risk = max(risks, key=lambda r: risk_order.get(r, 2)) if risks else None

    # schema_version: explicit priority, NOT string sort. Layers are ordered
    # (descriptor, baseline, provider) -- the most specific layer (provider,
    # last) wins. Only the provider layer should set schema_version; baseline
    # never does. No max()/min() -- that would do lexicographic string sort
    # ("10" < "2") which is semantically wrong for versions.
    schema_version = None
    for l in layers:
        if l.schema_version:
            schema_version = l.schema_version

    metadata: "dict[str, Any]" = {}
    for l in layers:
        if l.metadata:
            metadata.update(l.metadata)

    return ResolvedToolPolicy(
        enabled=enabled,
        timeout_seconds=timeout,
        max_retries=max_retries,
        idempotent=idempotent,
        require_approval=require_approval,
        risk=risk,
        schema_version=schema_version,
        idempotency_strategy=next((l.idempotency_strategy for l in reversed(layers)
                                   if l.idempotency_strategy is not None), None),
        idempotency_key_field=next((l.idempotency_key_field for l in reversed(layers)
                                    if l.idempotency_key_field is not None), None),
        metadata=metadata,
    )


def finalize_policy(resolved: "ResolvedToolPolicy | None") -> EffectiveToolPolicy:
    """Collapse a (possibly partially tri-state) merged ResolvedToolPolicy into
    the concrete EffectiveToolPolicy the execution chain acts on. Undeclared
    fields get the safe default: enabled defaults open (True) since an absent
    policy layer must not silently disable every tool; idempotent/
    require_approval default closed (False) since "not declared" must never be
    read as an implicit safety opt-in."""
    if resolved is None:
        return EffectiveToolPolicy()
    return EffectiveToolPolicy(
        enabled=True if resolved.enabled is None else resolved.enabled,
        timeout_seconds=resolved.timeout_seconds,
        max_retries=0 if resolved.max_retries is None else resolved.max_retries,
        idempotent=bool(resolved.idempotent),
        require_approval=bool(resolved.require_approval),
        risk=resolved.risk or "medium",
        schema_version=resolved.schema_version or "1",
        idempotency_strategy=resolved.idempotency_strategy or IdempotencyStrategy.EXACT_CALL,
        idempotency_key_field=resolved.idempotency_key_field,
        metadata=resolved.metadata,
    )
