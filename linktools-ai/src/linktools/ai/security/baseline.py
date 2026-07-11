#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SecurityBaseline: the default domain-agnostic safety configuration shipped
with linktools-ai. Enabled by default; callers can override individual rules,
inject a custom pipeline, or disable entirely via ``SecurityBaseline(enabled=False)``.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..policy.command import DEFAULT_DENIED_COMMAND_PATTERNS

if TYPE_CHECKING:
    from ..capability.policy import CapabilityToolExposurePolicy
    from .pipeline import SecurityPipeline


@dataclass(frozen=True, slots=True)
class CommandPolicy:
    """Minimal domain-agnostic command denylist patterns. Reuses
    policy.command.DEFAULT_DENIED_COMMAND_PATTERNS -- the single source of
    truth for the default denylist -- rather than maintaining a second,
    independently-drifting pattern set here."""
    denied_patterns: "tuple[str, ...]" = DEFAULT_DENIED_COMMAND_PATTERNS


@dataclass(frozen=True)
class SecurityBaseline:
    """Default safety baseline. Enabled by default; closable/overridable."""
    enabled: bool = True
    command_policy: "CommandPolicy | None" = field(default_factory=CommandPolicy)
    tool_exposure_policy: "CapabilityToolExposurePolicy | None" = None
    pipeline: "SecurityPipeline | None" = None
