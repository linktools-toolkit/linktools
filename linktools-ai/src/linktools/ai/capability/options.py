#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CapabilityRuntimeOptions: the runtime-policy bundle -- distinct
from Storage (state) and ProviderBundle (declarations). Holds the tool-exposure
policy, optional prompt/memory/subagent policies, and the MCP wildcard gate."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .policy import CapabilityToolExposurePolicy

_UNSET = object()

if TYPE_CHECKING:
    from ..prompt.window import SessionWindowPolicy


@dataclass(frozen=True)
class CapabilityRuntimeOptions:
    tool_exposure: CapabilityToolExposurePolicy | None = field(default=_UNSET)
    # Optional policies. None means "use the runner's default" (which preserves
    # historical behavior) or "not wired" (Noop). Each is substitutable.
    session_window_policy: "SessionWindowPolicy | None" = None
    memory_policy: Any = None
    retrieval_policy: Any = None
    prompt_context_formatter: Any = None
    subagent_context_policy: Any = None
    allow_mcp_wildcard: bool = False
    _tool_exposure_explicit: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        # The public default remains a usable policy for callers that inspect
        # the options object, while Runtime can distinguish it from an
        # explicitly supplied exposure policy when merging a baseline.
        explicit = self.tool_exposure is not _UNSET
        if not explicit:
            object.__setattr__(self, "tool_exposure", CapabilityToolExposurePolicy())
        object.__setattr__(self, "_tool_exposure_explicit", explicit)
