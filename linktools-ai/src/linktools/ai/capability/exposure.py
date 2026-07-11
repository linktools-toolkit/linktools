#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Centralized ToolExposurePolicy enforcement. The single place a descriptor's
category/mutating flag decides whether it reaches the model -- Providers
supply complete descriptors; they do not each implement their own exposure
judgment."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..security.descriptor import ToolDescriptor
    from .policy import CapabilityToolExposurePolicy


def is_descriptor_exposable(
    descriptor: "ToolDescriptor",
    policy: "CapabilityToolExposurePolicy",
) -> bool:
    """True iff ``descriptor`` may reach the model under ``policy``.
    Discovery-category tools are gated by ``expose_discovery_tools``; any
    mutating tool (write/terminal/subagent/package-execute/...) is gated by
    ``expose_execution_tools``. Everything else (non-discovery, non-mutating
    reads) is exposed unconditionally."""
    if descriptor.category == "discovery":
        return policy.expose_discovery_tools
    if descriptor.mutating:
        return policy.expose_execution_tools
    return True
