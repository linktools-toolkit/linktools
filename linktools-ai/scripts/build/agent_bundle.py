#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic build-time Agent bundle."""

from dataclasses import dataclass

from linktools.ai.core import canonical_sha256
from linktools.ai.spec import AgentSpec, PromptSpec


@dataclass(frozen=True, slots=True)
class AgentBundle:
    bundle_id: str
    bundle_digest: str
    agent_id: str
    agent_revision: int
    model_route: str
    toolset_ids: "tuple[str, ...]"
    capability_ids: "tuple[str, ...]"
    output_schema_id: str
    output_schema_revision: int
    output_schema_fingerprint: str
    codec_manifest_digest: str
    harness_version: str
    pydantic_ai_version: str
    spec_fingerprint: str
    prompt_fingerprint: str
    capability_manifest_digest: str
    instructions: "tuple[str, ...]" = ()

    @property
    def digest(self) -> str:
        return self.bundle_digest


def build_bundle(
    spec: AgentSpec,
    prompt: PromptSpec,
    capability_manifest_digest: str,
    *,
    codec_manifest_digest: str = "",
    output_schema_fingerprint: str = "",
    harness_version: str = "0.13.0",
    pydantic_ai_version: str = "2.27.0",
) -> AgentBundle:
    if not spec.model.strip() or spec.model.lower() in {"test", "testmodel"}:
        raise ValueError("production bundles require a released model route")
    if harness_version != "0.13.0" or pydantic_ai_version != "2.27.0":
        raise ValueError("bundles require the locked Harness and Pydantic AI versions")
    spec_fingerprint = canonical_sha256(
        {
            "id": spec.id,
            "revision": spec.revision,
            "model": spec.model,
            "capabilities": [
                {
                    "provider": capability.provider,
                    "id": capability.id,
                    "revision": capability.revision,
                    "required": capability.required,
                    "config": dict(capability.config),
                }
                for capability in spec.capabilities
            ],
            "output_schema": spec.output_schema,
            "output_schema_revision": spec.output_schema_revision,
            "metadata": dict(spec.metadata),
            "instructions": list(spec.instructions),
        }
    )
    prompt_fingerprint = canonical_sha256({"id": prompt.id, "revision": prompt.revision, "system": prompt.system, "instructions": list(prompt.instructions)})
    capability_ids = tuple(f"{capability.provider}:{capability.id}" for capability in spec.capabilities)
    toolset_ids: tuple[str, ...] = ()
    bundle_digest = canonical_sha256(
        {
            "agent_id": spec.id,
            "agent_revision": spec.revision,
            "model_route": spec.model,
            "toolset_ids": list(toolset_ids),
            "capability_ids": list(capability_ids),
            "output_schema_id": spec.output_schema,
            "output_schema_revision": spec.output_schema_revision,
            "output_schema_fingerprint": output_schema_fingerprint,
            "codec_manifest_digest": codec_manifest_digest,
            "harness_version": harness_version,
            "pydantic_ai_version": pydantic_ai_version,
            "spec_fingerprint": spec_fingerprint,
            "prompt_fingerprint": prompt_fingerprint,
            "capability_manifest_digest": capability_manifest_digest,
        }
    )
    return AgentBundle(
        f"b{bundle_digest[:12]}",
        bundle_digest,
        spec.id,
        spec.revision,
        spec.model,
        toolset_ids,
        capability_ids,
        spec.output_schema,
        spec.output_schema_revision,
        output_schema_fingerprint,
        codec_manifest_digest,
        harness_version,
        pydantic_ai_version,
        spec_fingerprint,
        prompt_fingerprint,
        capability_manifest_digest,
        (prompt.system, *prompt.instructions, *spec.instructions),
    )


__all__ = ["AgentBundle", "build_bundle"]
