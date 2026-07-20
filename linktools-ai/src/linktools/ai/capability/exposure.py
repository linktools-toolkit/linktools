#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CapabilityToolExposurePolicy + centralized exposure enforcement. The single
place a descriptor's category/mutating flag decides whether it reaches the
model -- Providers supply complete descriptors; they do not each implement
their own exposure judgment.

Defaults are conservative: prompt catalog + read-only discovery tools are on;
execution tools are OFF until a caller opts in. Per-capability and total tool
counts are capped so adding skills/MCP/extensions cannot balloon the tool schema
unboundedly."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tool.models import ToolDescriptor


@dataclass(frozen=True, slots=True)
class CapabilityToolExposurePolicy:
    expose_prompt_catalog: bool = True
    expose_discovery_tools: bool = True
    expose_execution_tools: bool = False
    max_tools_total: int = 64
    max_tools_per_capability: int = 16
    max_resources_per_list: int = 50
    max_read_bytes: int = 65536
    max_entrypoints_per_extension: int = 20
    allowed_entrypoint_kinds: "tuple[str, ...]" = ("agent",)
    require_explicit_entrypoint_allowlist: bool = True


def is_descriptor_exposable(
    descriptor: "ToolDescriptor",
    policy: "CapabilityToolExposurePolicy",
) -> bool:
    """True iff ``descriptor`` may reach the model under ``policy``.
    Discovery-category tools are gated by ``expose_discovery_tools``; any
    mutating tool (write/terminal/subagent/extension-execute/...) is gated by
    ``expose_execution_tools``. Everything else (non-discovery, non-mutating
    reads) is exposed unconditionally."""
    if descriptor.category == "discovery":
        return policy.expose_discovery_tools
    if descriptor.mutating:
        return policy.expose_execution_tools
    return True
