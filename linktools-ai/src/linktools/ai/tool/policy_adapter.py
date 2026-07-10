#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapter bridging the existing ToolSpec/ToolPolicyMetadata-based provider to
the new ToolPolicyProvider Protocol (resolve descriptor -> ResolvedToolPolicy).
Maps ALL fields from the existing ToolSpec that have runtime consumers."""

from typing import TYPE_CHECKING, Any

from .policy import ResolvedToolPolicy

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
        try:
            metadata_map = await self._provider.get_metadata_map()
        except Exception:
            return ResolvedToolPolicy(enabled=True, require_approval=True, risk="high")
        meta = metadata_map.get(descriptor.name)
        if meta is None:
            return ResolvedToolPolicy()
        # Map all ToolSpec/ToolPolicyMetadata fields that have consumers.
        risk = str(getattr(meta, "risk", "medium")).lower()
        approval = getattr(meta, "approval", None)
        require_approval = approval is not None and str(approval).upper() != "NEVER"
        side_effect = getattr(meta, "side_effect", None)
        idempotent = bool(getattr(meta, "idempotent", False))
        timeout = getattr(meta, "timeout_seconds", None)
        return ResolvedToolPolicy(
            enabled=True,
            timeout_seconds=float(timeout) if timeout is not None else None,
            max_retries=0,
            idempotent=idempotent,
            require_approval=require_approval,
            risk=risk,
            metadata={
                "permissions": [str(p) for p in getattr(meta, "permissions", frozenset())],
                "side_effect": str(side_effect) if side_effect is not None else "read_only",
            },
        )
