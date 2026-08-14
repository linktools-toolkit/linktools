#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable executable Agent definitions."""

from dataclasses import dataclass

from pydantic import BaseModel

from ..capability import CapabilityBinding
from ..capability.validation_api import validate_fingerprint
from ..errors import AIError, ErrorCode
from ..model import ModelBinding
from ..spec import AgentSpec, PromptSpec


@dataclass(frozen=True, slots=True)
class AgentDefinition:
    """Freeze all declaration, model, output, and capability execution inputs."""

    digest: str
    spec: AgentSpec
    prompt: PromptSpec
    model: ModelBinding
    output_type: "type[BaseModel]"
    output_schema_fingerprint: str
    effective_capabilities: "tuple[CapabilityBinding, ...]"

    def __post_init__(self) -> None:
        validate_fingerprint(self.digest)
        validate_fingerprint(self.output_schema_fingerprint)
        if any(capability is None for capability in self.effective_capabilities):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        try:
            identities = tuple((capability.provider, capability.id) for capability in self.effective_capabilities)
        except (AttributeError, TypeError) as error:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
        if len(set(identities)) != len(identities):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)


__all__ = ["AgentDefinition"]
