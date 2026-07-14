#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RunContext: the read-only context threaded through one Run's execution.
`workspace` is `WorkspaceRef | None` (WorkspaceRef is defined in
execution/workspace.py)."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from .models import RunnableType

if TYPE_CHECKING:
    from ..execution.workspace import WorkspaceRef


@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: str
    root_run_id: str
    parent_run_id: "str | None"
    session_id: str
    runnable_id: str
    runnable_type: RunnableType
    user_id: "str | None"
    tenant_id: "str | None"
    workspace: "WorkspaceRef | None"
    metadata: "Mapping[str, Any]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("RunContext.run_id must be a non-empty string")
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("RunContext.session_id must be a non-empty string")
        if not isinstance(self.runnable_id, str) or not self.runnable_id:
            raise ValueError("RunContext.runnable_id must be a non-empty string")
        from ..utils.freeze import freeze_value

        object.__setattr__(self, "metadata", freeze_value(dict(self.metadata)))
