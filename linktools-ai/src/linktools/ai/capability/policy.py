#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CapabilityToolExposurePolicy: the single knob set that bounds how many tools
a Capability Runtime exposes and which risk tiers are auto-enabled (spec §11.4).

Defaults are conservative: prompt catalog + read-only discovery tools are on;
execution tools are OFF until a caller opts in. Per-capability and total tool
counts are capped so adding skills/MCP/packages cannot balloon the tool schema
unboundedly."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CapabilityToolExposurePolicy:
    expose_prompt_catalog: bool = True
    expose_discovery_tools: bool = True
    expose_execution_tools: bool = False
    max_tools_total: int = 64
    max_tools_per_capability: int = 16
    max_resources_per_list: int = 50
    max_read_bytes: int = 65536
    max_entrypoints_per_package: int = 20
    allowed_entrypoint_kinds: "tuple[str, ...]" = ("agent",)
    require_explicit_entrypoint_allowlist: bool = True
