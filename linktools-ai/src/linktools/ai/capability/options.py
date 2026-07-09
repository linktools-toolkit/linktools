#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CapabilityRuntimeOptions (spec §17.9): the runtime-policy bundle -- distinct
from Storage (state) and ProviderBundle (declarations). Holds the tool-exposure
policy, optional prompt/memory/subagent policies, and the MCP wildcard gate."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .policy import CapabilityToolExposurePolicy

if TYPE_CHECKING:
    from ..prompt.window import SessionWindowPolicy


@dataclass(frozen=True)
class CapabilityRuntimeOptions:
    tool_exposure: CapabilityToolExposurePolicy = field(default_factory=CapabilityToolExposurePolicy)
    # Optional policies. None means "not wired" (effective Noop). Memory and
    # subagent-context policies are reserved slots -- their Protocols land with
    # the memory / subagent subsystems; the Runtime applies them when present.
    session_window_policy: "SessionWindowPolicy | None" = None
    memory_policy: Any = None
    subagent_context_policy: Any = None
    allow_mcp_wildcard: bool = False
