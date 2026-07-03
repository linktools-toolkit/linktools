#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single per-call pydantic_ai.AgentCapability: instructions/toolset/model_settings
construction + failure-diagnostics extraction (Section 2).

This capability now covers file/terminal tools only (HookedBuiltinToolset, built in
base.py and handed in via get_toolset()). skill_view, call_subagent, and each MCP
server are separate AbstractCapability instances (SkillCapability, SubagentCapability,
HookedMCPCapability) with their own wrap_tool_execute, each attributing on `self`
(server name, hardcoded "builtin", etc.) — granular-by-capability sidesteps the
ToolDefinition/RunContext toolset-origin-identity gap that a single shared hook
across a combined toolset would have hit.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability

if TYPE_CHECKING:
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.toolsets import AbstractToolset


@dataclass
class RuntimeRunCapability(AbstractCapability[None]):
    instructions: str = ""
    toolset: "AbstractToolset[None] | None" = None
    model_settings: "ModelSettings | None" = None
    last_error_detail: "dict[str, Any] | None" = field(default=None, compare=False)

    def get_instructions(self) -> "str | None":
        return self.instructions or None

    def get_toolset(self) -> "AbstractToolset[None] | None":
        return self.toolset

    def get_model_settings(self) -> "ModelSettings | None":
        return self.model_settings

    async def on_run_error(self, ctx: Any, *, error: BaseException) -> Any:
        """Extract diagnostic detail once, then re-raise the ORIGINAL error unchanged.

        Does not convert exception types — generate()/stream() still own that
        (UsageLimitExceeded -> ModelTurnLimitExceeded, etc.); this only centralizes
        the `getattr(exc, ...)` diagnostics extraction that was duplicated per except
        branch in both call sites.
        """
        if isinstance(error, BaseException) and error.__class__.__name__ == "CancelledError":
            self.last_error_detail = {"reason": "worker_timeout_or_parent_cancelled"}
            raise error
        detail = getattr(error, "body", None)
        if detail is None:
            detail = getattr(error, "diagnostics", None)
        self.last_error_detail = detail if detail is not None else {}
        raise error
