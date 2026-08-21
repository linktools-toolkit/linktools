#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent Asset binding used as one-level subagent authorization."""

from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability

from ..asset import AssetRef, AssetRepository
from ..core import canonical_sha256
from ..errors import AIError, ErrorCode
from ..spec import AgentSpec
from ._contract import CapabilityBinding, CapabilityMaterializationContext, CapabilityRefResolution


@dataclass(frozen=True, slots=True)
class SubagentCapabilityBinding:
    resolutions: "tuple[CapabilityRefResolution, ...]"
    agent_ids: "tuple[str, ...]"
    fingerprint: str

    @property
    def id(self) -> str:
        return "agent"

    @property
    def provider(self) -> str:
        return "agent"

    async def materialize(
        self,
        context: CapabilityMaterializationContext,
    ) -> "tuple[AbstractCapability[None], ...]":
        del context
        return ()


class SubagentCapabilityProvider:
    provider = "agent"
    value_type = AgentSpec

    async def bind(
        self,
        refs: "tuple[AssetRef, ...]",
        *,
        assets: AssetRepository,
    ) -> CapabilityBinding:
        resolutions: list[CapabilityRefResolution] = []
        ids: list[str] = []
        seen_ids: set[str] = set()
        for ref in refs:
            resolved = await assets.resolve(ref)
            if type(resolved.spec) is not AgentSpec or resolved.spec.id != ref.id:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            if resolved.spec.id in seen_ids:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            seen_ids.add(resolved.spec.id)
            ids.append(resolved.spec.id)
            resolutions.append(
                CapabilityRefResolution(
                    ref,
                    resolved.spec.revision,
                    canonical_sha256(
                        {
                            "id": resolved.spec.id,
                            "revision": resolved.spec.revision,
                            "model": resolved.spec.model,
                            "system_prompt": resolved.spec.system_prompt,
                            "instructions": list(resolved.spec.instructions),
                            "allow_tools": list(resolved.spec.allow_tools),
                            "metadata": dict(resolved.spec.metadata),
                        }
                    ),
                )
            )
        ordered = tuple(sorted(zip(resolutions, ids), key=lambda item: (item[0].ref.kind, item[0].ref.id)))
        return _binding(tuple(item[0] for item in ordered), tuple(item[1] for item in ordered))

    def select(
        self,
        binding: CapabilityBinding,
        refs: "tuple[AssetRef, ...]",
    ) -> CapabilityBinding:
        if not isinstance(binding, SubagentCapabilityBinding) or not refs:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        values = {
            resolution.ref: agent_id
            for resolution, agent_id in zip(binding.resolutions, binding.agent_ids)
        }
        ordered = tuple(sorted(refs, key=lambda ref: (ref.kind, ref.id)))
        if len(ordered) != len(set(ordered)) or any(ref not in values for ref in ordered):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        return _binding(
            tuple(next(item for item in binding.resolutions if item.ref == ref) for ref in ordered),
            tuple(values[ref] for ref in ordered),
        )


def _binding(
    resolutions: "tuple[CapabilityRefResolution, ...]",
    agent_ids: "tuple[str, ...]",
) -> SubagentCapabilityBinding:
    return SubagentCapabilityBinding(
        resolutions,
        agent_ids,
        canonical_sha256(
            {
                "provider": "agent",
                "resolutions": [
                    {
                        "kind": item.ref.kind,
                        "id": item.ref.id,
                        "revision": item.resolved_revision,
                        "fingerprint": item.fingerprint,
                    }
                    for item in resolutions
                ],
            }
        ),
    )


__all__ = ["SubagentCapabilityBinding", "SubagentCapabilityProvider"]
