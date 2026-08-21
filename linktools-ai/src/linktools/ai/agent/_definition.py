#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable executable Agent definitions."""

from dataclasses import dataclass

from pydantic import BaseModel

from ..capability import CapabilityBinding, validate_fingerprint
from ..errors import AIError, ErrorCode
from ..model import ModelBinding
from ..spec import AgentSpec
from ._binding import AgentBindingSnapshot
from ._output import OutputBinding


@dataclass(frozen=True, slots=True)
class AgentDefinition:
    """Freeze all declaration, model, output, and capability execution inputs."""

    digest: str
    spec: AgentSpec
    model: ModelBinding
    output_binding: OutputBinding
    effective_capabilities: "tuple[CapabilityBinding, ...]"
    binding_snapshot: AgentBindingSnapshot

    def __post_init__(self) -> None:
        validate_fingerprint(self.digest)
        if not isinstance(self.output_binding, OutputBinding):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        if not isinstance(self.binding_snapshot, AgentBindingSnapshot):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            self.binding_snapshot.binding_digest != self.digest
            or self.binding_snapshot.agent_spec != self.spec
            or self.binding_snapshot.output_schema_id != self.output_binding.schema_id
            or self.binding_snapshot.output_schema_revision != self.output_binding.schema_revision
            or self.binding_snapshot.output_schema_fingerprint != self.output_binding.schema_fingerprint
            or self.binding_snapshot.output_type_module != self.output_type.__module__
            or self.binding_snapshot.output_type_qualname != self.output_type.__qualname__
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if any(capability is None for capability in self.effective_capabilities):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        try:
            identities = tuple((capability.provider, capability.id) for capability in self.effective_capabilities)
        except (AttributeError, TypeError) as error:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
        if len(set(identities)) != len(identities):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)

    @property
    def output_type(self) -> type[BaseModel]:
        return self.output_binding.value_type

    @property
    def output_schema_fingerprint(self) -> str:
        return self.output_binding.schema_fingerprint


__all__ = ["AgentDefinition"]
