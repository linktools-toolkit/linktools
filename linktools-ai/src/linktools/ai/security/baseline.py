#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SecurityBaseline: the default domain-agnostic safety configuration shipped
with linktools-ai. Enabled by default; callers can override individual rules,
inject a custom pipeline, or disable entirely via ``SecurityBaseline(enabled=False)``.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..capability.policy import CapabilityToolExposurePolicy
    from .pipeline import SecurityPipeline


@dataclass(frozen=True, slots=True)
class CommandPolicy:
    """Minimal domain-agnostic command denylist patterns."""
    denied_patterns: "tuple[str, ...]" = (
        r"\brm\s+-rf\s+/\b",
        r"\bmkfs\b",
        r"\bdd\s+if=/dev/zero\s+of=/dev/",
    )


@dataclass(frozen=True)
class SecurityBaseline:
    """Default safety baseline. Enabled by default; closable/overridable."""
    enabled: bool = True
    command_policy: "CommandPolicy | None" = field(default_factory=CommandPolicy)
    tool_exposure_policy: "CapabilityToolExposurePolicy | None" = None
    pipeline: "SecurityPipeline | None" = None
