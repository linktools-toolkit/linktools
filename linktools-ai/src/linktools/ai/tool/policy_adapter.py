#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapter bridging the existing ToolPolicyMetadata-based provider to the new
ToolPolicyProvider Protocol (resolve descriptor -> ResolvedToolPolicy)."""

from typing import TYPE_CHECKING

from .policy import ResolvedToolPolicy

if TYPE_CHECKING:
    from ..security.descriptor import ToolDescriptor
    from ..run.context import RunContext


class MetadataBackedPolicyProvider:
    """Wraps an old-style ``get_metadata_map()`` provider and resolves a
    ToolDescriptor into a ResolvedToolPolicy by looking up the tool's metadata.
    Tools not in the metadata map get default policy (enabled, no approval)."""

    def __init__(self, metadata_provider: Any) -> None:
        self._provider = metadata_provider

    async def resolve(self, descriptor: "ToolDescriptor", context: "RunContext") -> ResolvedToolPolicy:
        try:
            metadata_map = await self._provider.get_metadata_map()
        except Exception:
            # Fail closed: if the provider errors, return a restrictive default.
            return ResolvedToolPolicy(enabled=True, require_approval=True, risk="high")
        meta = metadata_map.get(descriptor.name)
        if meta is None:
            return ResolvedToolPolicy()
        return ResolvedToolPolicy(
            enabled=True,
            require_approval=getattr(meta, "approval", None) is not None
                             and str(getattr(meta, "approval", "")).upper() != "NEVER",
            risk=str(getattr(meta, "risk", "medium")).lower(),
        )


from typing import Any  # noqa: E402
