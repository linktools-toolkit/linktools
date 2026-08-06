#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build-time capability assembly input."""

from dataclasses import dataclass

from ..domain.agent import AgentBundleDescriptor, AgentRelease
from ..foundation.digest import sha256_digest
from ..foundation.json import canonical_json_bytes

@dataclass(frozen=True, slots=True)
class AssemblyInput:
    agent_id: str
    revision: int
    capabilities: "tuple[str, ...]" = ()


@dataclass(frozen=True, slots=True)
class BundleCompilation:
    descriptor: AgentBundleDescriptor
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
        values = tuple(
            (entry.feature_name, entry.assembly_mode, entry.serialization_name, entry.order, entry.config_digest)
            for entry in self.entries
        )
        return sha256_digest(canonical_json_bytes(values))


class AgentBundleCompiler:
    """Compile a release into deterministic, import-time-pure bundle source."""

    def compile(
        self,
        release: AgentRelease,
        capabilities: "tuple[str, ...]" = (),
        *,
        capability_modes: "dict[str, str] | None" = None,
        model_name: str,
    ) -> BundleCompilation:
        ordered = tuple(dict.fromkeys(capabilities))
        if len(ordered) != len(capabilities):
            raise ValueError("bundle capabilities must be unique")
        if not model_name.strip():
            raise ValueError("bundle model name must not be empty")
        modes = capability_modes or {}
        allowed_modes = {"spec_serializable", "python_injected", "building_block", "upstream_pending"}
        entries = []
        for index, capability in enumerate(ordered, start=1):
            mode = modes.get(capability, "python_injected")
            if mode not in allowed_modes:
                raise ValueError(f"unsupported capability assembly mode: {mode}")
            if mode == "upstream_pending":
                raise ValueError(f"capability is not released: {capability}")
            entries.append(
                CapabilityAssemblyEntry(
                    feature_name=capability,
                    assembly_mode=mode,
                    serialization_name=capability if mode == "spec_serializable" else None,
                    order=index,
                    config_digest=sha256_digest(capability.encode("utf-8")),
                )
            )
        plan = CapabilityAssemblyPlan(tuple(entries))
        toolset_ids = tuple(
            f"lt.{release.agent_id}.r{release.revision}.{capability}.{index}"
            for index, capability in enumerate(ordered, start=1)
        )
        identity = {
            "agent_id": release.agent_id,
            "agent_revision": release.revision,
            "spec_sha256": release.spec_sha256,
            "policy_id": release.policy_id,
            "output_contract_id": release.output_contract_id,
            "deps_contract_id": release.deps_contract_id,
            "toolset_ids": toolset_ids,
            "assembly_digest": plan.digest,
            "model_name": model_name,
        }
        build_id = sha256_digest(canonical_json_bytes(identity))
        descriptor = AgentBundleDescriptor(
            agent_id=release.agent_id,
            agent_revision=release.revision,
            toolset_ids=toolset_ids,
            output_contract_id=release.output_contract_id,
            deps_contract_id=release.deps_contract_id,
            build_id=build_id,
        )
        module_name = f"bundle_{build_id}"
        source = (
            "#!/usr/bin/env python3\n"
            "# -*- coding: utf-8 -*-\n\n"
            "from typing import Any, Final\n"
            "from pydantic_ai import Agent\n"
            "from pydantic_ai.durable_exec.temporal import TemporalDurability\n"
            "from ...agent.deps import AgentDeps\n\n"
            f"BUNDLE_ID: Final[str] = {build_id!r}\n"
            f"AGENT_NAME: Final[str] = {f'lt.{release.agent_id}.r{release.revision}'!r}\n"
            f"TOOLSET_IDS: Final[tuple[str, ...]] = {toolset_ids!r}\n"
            f"ASSEMBLY_DIGEST: Final[str] = {plan.digest!r}\n"
            f"MODEL_NAME: Final[str] = {model_name!r}\n"
            "agent: \"Agent[AgentDeps, Any]\" = Agent(\n"
            "    MODEL_NAME,\n"
            "    name=AGENT_NAME,\n"
            "    deps_type=AgentDeps,\n"
            "    output_type=Any,\n"
            "    capabilities=[TemporalDurability(name=AGENT_NAME, deps_type=AgentDeps)],\n"
            ")\n"
        )
        return BundleCompilation(descriptor, module_name, source)


__all__ = ["AgentBundleCompiler", "AssemblyInput", "BundleCompilation", "CapabilityAssemblyEntry", "CapabilityAssemblyPlan"]
