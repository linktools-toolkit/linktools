#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SubagentResult: the structured return value of a delegated
subagent run. Carries the child session/run ids, terminal status, output or
structured error, and accounting fields."""

from typing import Any, Literal

from pydantic import BaseModel, Field

SubagentStatus = Literal["succeeded", "failed", "cancelled"]


class SubagentResult(BaseModel):
    agent_id: str
    scope: "dict[str, Any] | None" = None
    session_id: str
    run_id: str
    status: SubagentStatus
    output: Any = None
    error: "dict[str, Any] | None" = None
    token_usage: "dict[str, Any] | None" = None
    duration_ms: "int | None" = None
    metadata: "dict[str, Any]" = Field(default_factory=dict)
