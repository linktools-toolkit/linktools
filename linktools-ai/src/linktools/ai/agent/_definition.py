#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable executable Agent definitions."""

from dataclasses import dataclass

from ..capability import CapabilityBinding, validate_fingerprint
from ..errors import AIError, ErrorCode
from ..model import ModelBinding
from ..spec import AgentSpec
from ._output import OutputBinding


@dataclass(frozen=True, slots=True)
class AgentDefinition:
    """Freeze all declaration, model, output, and capability execution inputs."""

    digest: str
    spec: AgentSpec
    model: ModelBinding
    output_binding: OutputBinding
    effective_capabilities: "tuple[CapabilityBinding, ...]"

    def __post_init__(self) -> None:
        validate_fingerprint(self.digest)
        if not isinstance(self.output_binding, OutputBinding):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        if any(capability is None for capability in self.effective_capabilities):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        try:
            identities = tuple((capability.provider, capability.id) for capability in self.effective_capabilities)
        except (AttributeError, TypeError) as error:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
        if len(set(identities)) != len(identities):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)

    @property
    def output_type(self):
        return self.output_binding.value_type

    @property
    def output_schema_fingerprint(self) -> str:
        return self.output_binding.schema_fingerprint


__all__ = ["AgentDefinition"]
