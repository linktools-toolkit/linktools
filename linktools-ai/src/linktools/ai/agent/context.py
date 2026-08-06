#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Versioned, serializable Temporal run context."""

from pydantic import BaseModel, ConfigDict, Field

from ..foundation.json import canonical_json_bytes
from .deps import AgentDeps


class LinktoolsTemporalRunContext(BaseModel):
    model_config = ConfigDict(frozen=True)
    version: int = Field(default=1, ge=1)
    deps: "AgentDeps | None" = None
    execution_id: str
    run_id: str
    conversation_id: "str | None" = None
    run_step: int = Field(default=0, ge=0)
    usage: "dict[str, int]" = Field(default_factory=dict)
    usage_limits: "dict[str, int]" = Field(default_factory=dict)
    tool_call_id: "str | None" = None
    tool_name: "str | None" = None
    approval_id: "str | None" = None
    partial_output: "object | None" = None

    def serialize_run_context(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))

    @classmethod
    def deserialize_run_context(cls, content: bytes) -> "LinktoolsTemporalRunContext":
        return cls.model_validate_json(content)


RunContext = LinktoolsTemporalRunContext


__all__ = ["LinktoolsTemporalRunContext", "RunContext"]
