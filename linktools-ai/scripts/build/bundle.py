#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Deterministic build-time Agent Bundle compilation."""

from dataclasses import dataclass

from linktools.ai.core import canonical_sha256
from linktools.ai.spec import AgentSpec, PromptSpec
from .agent_bundle import AgentBundle, build_bundle


@dataclass(frozen=True, slots=True)
class AssemblyInput:
    agent_id: str
    revision: int
    capabilities: "tuple[str, ...]" = ()


@dataclass(frozen=True, slots=True)
class BundleCompilation:
    bundle: AgentBundle
    module_name: str
    source: str


@dataclass(frozen=True, slots=True)
class CapabilityAssemblyEntry:
    feature_name: str
    assembly_mode: str
    serialization_name: "str | None"
    order: int
    config_digest: str


@dataclass(frozen=True, slots=True)
class CapabilityAssemblyPlan:
    entries: "tuple[CapabilityAssemblyEntry, ...]"

    @property
    def digest(self) -> str:
        return canonical_sha256(
            {
                "entries": [
                    {
                        "feature_name": entry.feature_name,
                        "assembly_mode": entry.assembly_mode,
                        "serialization_name": entry.serialization_name,
                        "order": entry.order,
                        "config_digest": entry.config_digest,
                    }
                    for entry in self.entries
                ]
            }
        )


class AgentBundleCompiler:
    """Compile an immutable bundle and its importable source module."""

    def compile(
        self,
        spec: AgentSpec,
        capabilities: 'tuple[str, ...]' = (),
        *,
        capability_modes: 'dict[str, str] | None' = None,
        model_name: 'str | None' = None,
        prompt: 'PromptSpec | None' = None,
        codec_manifest_digest: str = "",
        output_schema_fingerprint: str = "",
    ) -> BundleCompilation:
        ordered = tuple(dict.fromkeys(capabilities))
        if len(ordered) != len(capabilities):
            raise ValueError("bundle capabilities must be unique")
        selected_model = model_name or spec.model
        if not selected_model.strip() or selected_model.lower() in {"test", "testmodel"}:
            raise ValueError("bundle model name must not be empty")
        modes = capability_modes or {}
        entries: list[CapabilityAssemblyEntry] = []
        for index, capability in enumerate(ordered, start=1):
            mode = modes.get(capability, "python_injected")
            if mode not in {"spec_serializable", "python_injected", "building_block"}:
                raise ValueError(f"unsupported capability assembly mode: {mode}")
            entries.append(
                CapabilityAssemblyEntry(
                    capability,
                    mode,
                    capability if mode == "spec_serializable" else None,
                    index,
                    canonical_sha256(capability),
                )
            )
        selected_prompt = prompt or PromptSpec("bundle", 1, "", (), ())
        bundle = build_bundle(
            spec,
            selected_prompt,
            CapabilityAssemblyPlan(tuple(entries)).digest,
            codec_manifest_digest=codec_manifest_digest,
            output_schema_fingerprint=output_schema_fingerprint,
        )
        return BundleCompilation(bundle, bundle.bundle_id, _bundle_source(bundle, selected_model))


def _bundle_source(bundle: AgentBundle, model_name: str) -> str:
    return (
        "#!/usr/bin/env python3\n"
        "# -*- coding: utf-8 -*-\n\n"
        "from pydantic_ai import Agent\n\n"
        f"BUNDLE_ID = {bundle.bundle_id!r}\n"
        f"BUNDLE_DIGEST = {bundle.bundle_digest!r}\n"
        f"AGENT_ID = {bundle.agent_id!r}\n"
        f"AGENT_REVISION = {bundle.agent_revision!r}\n"
        f"MODEL_ROUTE = {model_name!r}\n"
        f"TOOLSET_IDS = {bundle.toolset_ids!r}\n"
        f"CAPABILITY_IDS = {bundle.capability_ids!r}\n"
        f"OUTPUT_SCHEMA_ID = {bundle.output_schema_id!r}\n"
        f"OUTPUT_SCHEMA_REVISION = {bundle.output_schema_revision!r}\n"
        f"OUTPUT_SCHEMA_FINGERPRINT = {bundle.output_schema_fingerprint!r}\n"
        f"CODEC_MANIFEST_DIGEST = {bundle.codec_manifest_digest!r}\n"
        f"HARNESS_VERSION = {bundle.harness_version!r}\n"
        f"PYDANTIC_AI_VERSION = {bundle.pydantic_ai_version!r}\n"
        f"INSTRUCTIONS = {bundle.instructions!r}\n"
        "AGENT = Agent(MODEL_ROUTE, name=AGENT_ID, instructions=INSTRUCTIONS, output_type=str, defer_model_check=True)\n"
    )


__all__ = [
    "AgentBundleCompiler",
    "AssemblyInput",
    "BundleCompilation",
    "CapabilityAssemblyEntry",
    "CapabilityAssemblyPlan",
]
