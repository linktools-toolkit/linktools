#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Versioned Temporal workflow run context."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..core import canonical_json_bytes


class TemporalRunContext(BaseModel):
    model_config = ConfigDict(frozen=True)
    version: int = Field(default=1, ge=1)
    deps: "dict[str, Any] | None" = None
    execution_id: str
    run_id: str
    conversation_id: "str | None" = None
    run_step: int = Field(default=0, ge=0)
    usage: "dict[str, int]" = Field(default_factory=dict)
    usage_limits: "dict[str, int]" = Field(default_factory=dict)
    tool_call_id: "str | None" = None
    tool_name: "str | None" = None
    approval_id: "str | None" = None
    partial_output: "str | None" = None

    def serialize_run_context(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))

    @classmethod
    def deserialize_run_context(cls, content: bytes) -> "TemporalRunContext":
        return cls.model_validate_json(content)


RunContext = TemporalRunContext


__all__ = ["TemporalRunContext", "RunContext"]
