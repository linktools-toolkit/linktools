#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Versioned, serializable Temporal run context."""

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from ..core.ids import canonical_sha256
from ..spec.model import AgentSpec, PromptSpec
from ..core.json import canonical_json_bytes
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
    partial_output: "str | None" = None

    def serialize_run_context(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))

    @classmethod
    def deserialize_run_context(cls, content: bytes) -> "LinktoolsTemporalRunContext":
        return cls.model_validate_json(content)


RunContext = LinktoolsTemporalRunContext


@dataclass(frozen=True, slots=True)
class AgentBinding:
    spec: AgentSpec
    prompt: PromptSpec
    spec_fingerprint: str
    prompt_fingerprint: str
    model_registry_revision: int
    output_schema_fingerprint: str
    capability_manifest_digest: str

    def __post_init__(self) -> None:
        if (
            self.model_registry_revision < 0
            or not self.spec_fingerprint
            or not self.prompt_fingerprint
            or not self.output_schema_fingerprint
            or not self.capability_manifest_digest
        ):
            raise ValueError("Agent binding is incomplete")

    @property
    def digest(self) -> str:
        return canonical_sha256(
            {
                "agent_id": self.spec.id,
                "agent_revision": self.spec.revision,
                "prompt_id": self.prompt.id,
                "prompt_revision": self.prompt.revision,
                "spec_fingerprint": self.spec_fingerprint,
                "prompt_fingerprint": self.prompt_fingerprint,
                "model_registry_revision": self.model_registry_revision,
                "output_schema_fingerprint": self.output_schema_fingerprint,
                "capability_manifest_digest": self.capability_manifest_digest,
            }
        )


__all__ = ["AgentBinding", "LinktoolsTemporalRunContext", "RunContext"]
