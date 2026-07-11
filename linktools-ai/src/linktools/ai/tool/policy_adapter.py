#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapter bridging the existing ToolSpec/ToolPolicyMetadata-based provider to
the new ToolPolicyProvider Protocol (resolve descriptor -> ResolvedToolPolicy).
Maps ALL fields from the existing ToolSpec that have runtime consumers."""

from typing import TYPE_CHECKING, Any

from ..errors import ToolPolicyResolutionError
from .policy import ResolvedToolPolicy, parse_idempotency_strategy

if TYPE_CHECKING:
    from ..security.descriptor import ToolDescriptor
    from ..run.context import RunContext


class MetadataBackedPolicyProvider:
    """Wraps an old-style ``get_metadata_map()`` provider and resolves a
    ToolDescriptor into a ResolvedToolPolicy by looking up the tool's metadata.
    Tools not in the metadata map get default policy (enabled, no approval).
    Provider errors fail closed."""

    def __init__(self, metadata_provider: Any) -> None:
        self._provider = metadata_provider

    async def resolve(self, descriptor: "ToolDescriptor", context: "RunContext") -> ResolvedToolPolicy:
        # Fail closed: if the underlying metadata source is unavailable, raise
        # so the ManagedToolAdapter emits a SecurityDegraded event and denies
        # the call -- never run a tool ungoverned because its policy couldn't
        # be resolved.
        try:
            metadata_map = await self._provider.get_metadata_map()
        except Exception as exc:
            raise ToolPolicyResolutionError(
                f"tool policy metadata source unavailable for {descriptor.name!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        meta = metadata_map.get(descriptor.name)
        if meta is None:
            return ResolvedToolPolicy()
        # Map all ToolSpec/ToolPolicyMetadata fields that have consumers.
        risk_val = getattr(meta, "risk", None)
        if isinstance(risk_val, str):
            risk = risk_val.lower()
        elif risk_val is not None:
            # RiskLevel is an enum -- use .name so RiskLevel.HIGH -> "high",
            # not str() which yields "risklevel.high".
            risk = risk_val.name.lower()
        else:
            risk = "medium"
        approval = getattr(meta, "approval", None)
        # Compare the enum VALUE, not str(enum): on Python 3.11+
        # str(ApprovalMode.NEVER) is "ApprovalMode.NEVER" (not "never"), so a
        # str().upper()-based check would wrongly require approval for NEVER.
        from ..policy.rule import ApprovalMode
        require_approval = approval not in (None, ApprovalMode.NEVER)
        side_effect = getattr(meta, "side_effect", None)
        idempotent = getattr(meta, "idempotent", None)
        timeout = getattr(meta, "timeout_seconds", None)
        # Carry every ToolSpec field with a runtime consumer into namespaced
        # metadata so none is silently dropped. schema_version folds into
        # idempotency hashing downstream; the rest are auditable context.
        meta_extra = dict(getattr(meta, "metadata", {}) or {})
        namespaced: "dict[str, Any]" = {
            "permissions": [str(p) for p in getattr(meta, "permissions", frozenset())],
            "side_effect": str(side_effect) if side_effect is not None else "read_only",
        }
        if meta_extra:
            namespaced["source_metadata"] = meta_extra
        return ResolvedToolPolicy(
            enabled=getattr(meta, "enabled", True),
            timeout_seconds=float(timeout) if timeout is not None else None,
            max_retries=getattr(meta, "max_retries", None),
            idempotent=idempotent,
            require_approval=require_approval,
            risk=risk,
            schema_version=str(getattr(meta, "schema_version", "1")) or None,
            idempotency_strategy=parse_idempotency_strategy(
                getattr(meta, "idempotency_strategy", None)),
            idempotency_key_field=getattr(meta, "idempotency_key_field", None),
            metadata=namespaced,
        )
