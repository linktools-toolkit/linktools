#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CapabilityInspection: a stable, immutable view of what an AgentSpec resolves
to. Returned by Runtime.inspect so downstream tooling never depends on the
mutable internal CapabilityBundle / raw handlers."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..security.descriptor import ToolDescriptor
from ..utils.freeze import freeze_value
from .policy import CapabilityToolExposurePolicy


@dataclass(frozen=True)
class CapabilityInspection:
    """Immutable snapshot of a resolved capability bundle."""
    tools: "tuple[ToolDescriptor, ...]" = ()
    prompt_sections: "Mapping[str, str]" = field(default_factory=dict)
    warnings: "tuple[str, ...]" = ()
    exposure_policy: CapabilityToolExposurePolicy = field(
        default_factory=CapabilityToolExposurePolicy)

    @classmethod
    def from_bundle(cls, bundle: Any, *, exposure_policy: CapabilityToolExposurePolicy | None = None) -> "CapabilityInspection":
        """Build an inspection from a CapabilityBundle without leaking its
        mutable internals: tools come from the per-tool definitions (and/or
        legacy descriptors), prompt sections are copied."""
        tools: "list[ToolDescriptor]" = []
        for c in bundle.tool_contributions:
            if getattr(c, "tools", None):
                tools.extend(md.descriptor for md in c.tools)
            else:
                tools.extend(c.descriptors)
        # Deduplicate by name while preserving order.
        seen: "set[str]" = set()
        unique: "list[ToolDescriptor]" = []
        for d in tools:
            if d.name not in seen:
                seen.add(d.name)
                unique.append(d)
        return cls(
            tools=tuple(unique),
            prompt_sections=freeze_value(dict(bundle.prompt_sections)),
            exposure_policy=exposure_policy or CapabilityToolExposurePolicy(),
        )
