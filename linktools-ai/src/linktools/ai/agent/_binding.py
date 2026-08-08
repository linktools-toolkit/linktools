#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolved Agent definition binding and stable behavior digest."""

from dataclasses import dataclass

from ..core import canonical_sha256
from ..spec import AgentSpec, PromptSpec


@dataclass(frozen=True, slots=True)
class AgentBinding:
    spec: AgentSpec
    prompt: PromptSpec
    spec_fingerprint: str
    prompt_fingerprint: str
    model_registry_revision: int
    output_schema_fingerprint: str
    capability_manifest_digest: str
    tool_policy_fingerprint: str
    sandbox_fingerprint: str
    middleware_fingerprint: str

    def __post_init__(self) -> None:
        if (
            self.model_registry_revision < 0
            or not self.spec_fingerprint
            or not self.prompt_fingerprint
            or not self.output_schema_fingerprint
            or not self.capability_manifest_digest
            or not self.tool_policy_fingerprint
            or not self.sandbox_fingerprint
            or not self.middleware_fingerprint
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
                "output_schema_id": self.spec.output_schema,
                "output_schema_revision": self.spec.output_schema_revision,
                "capability_manifest_digest": self.capability_manifest_digest,
                "tool_policy_fingerprint": self.tool_policy_fingerprint,
                "sandbox_fingerprint": self.sandbox_fingerprint,
                "middleware_fingerprint": self.middleware_fingerprint,
            }
        )


__all__ = ["AgentBinding"]
