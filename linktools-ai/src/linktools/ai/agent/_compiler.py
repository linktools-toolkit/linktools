#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compile declarations into immutable executable Agent definitions."""

from collections.abc import Sequence

from linktools.core import environ
from pydantic import BaseModel

from ..capability import (
    CapabilityBinding,
    CapabilityRefResolution,
    RuntimeCapability,
    validate_fingerprint,
)
from ..core import canonical_sha256, validate_agent_id, validate_capability_provider
from ..errors import AIError, ErrorCode
from ..model import ModelResolver
from ..spec import AgentSpec
from ._definition import AgentDefinition
from ._output import bind_output

_logger = environ.get_logger("ai.agent.compiler")


class AgentCompiler:
    """Pure compiler over already-frozen Runtime composition inputs."""

    def __init__(
        self,
        *,
        model_resolver: ModelResolver,
        capabilities: "Sequence[CapabilityBinding]" = (),
        runtime_fingerprint: str,
    ) -> None:
        if model_resolver is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        validate_fingerprint(runtime_fingerprint)
        global_capabilities = tuple(capabilities)
        _validate_bindings(global_capabilities)
        self._model_resolver = model_resolver
        self._capabilities = global_capabilities
        self._runtime_fingerprint = runtime_fingerprint

    def compile(
        self,
        spec: AgentSpec,
        *,
        capabilities: "Sequence[RuntimeCapability]" = (),
        output: "type[BaseModel] | None" = None,
    ) -> AgentDefinition:
        validate_agent_id(spec.id)
        local_capabilities = tuple(capabilities)
        if any(not isinstance(capability, RuntimeCapability) for capability in local_capabilities):
            raise TypeError("agent-local capabilities must contain RuntimeCapability values")
        if any(not capability.durable for capability in local_capabilities):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        effective: tuple[CapabilityBinding, ...] = (*self._capabilities, *local_capabilities)
        _validate_bindings(effective)
        model = self._model_resolver.resolve(spec.model)
        output_binding = bind_output(output)
        digest = _definition_digest(
            spec,
            model.fingerprint,
            output_binding.schema_fingerprint,
            effective,
            self._runtime_fingerprint,
        )
        definition = AgentDefinition(
            digest,
            spec,
            model,
            output_binding,
            effective,
        )
        _logger.debug(
            "agent definition compiled: agent=%s digest=%s capabilities=%s",
            spec.id,
            digest,
            tuple((capability.provider, capability.id) for capability in effective),
        )
        return definition


def _validate_bindings(bindings: "Sequence[CapabilityBinding]") -> None:
    identities: list[tuple[str, str]] = []
    for binding in bindings:
        _validate_binding_shape(binding)
        identities.append((binding.provider, binding.id))
    if len(identities) != len(set(identities)):
        raise AIError(ErrorCode.CAPABILITY_CONFLICT)


def _validate_binding_shape(binding: CapabilityBinding) -> None:
    try:
        provider = binding.provider
        binding_id = binding.id
        resolutions = binding.resolutions
        fingerprint = binding.fingerprint
    except (AttributeError, TypeError) as error:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
    try:
        validate_capability_provider(provider)
    except AIError as error:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
    if not isinstance(binding_id, str) or not binding_id.strip() or not isinstance(resolutions, tuple):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    validate_fingerprint(fingerprint)
    if any(not isinstance(resolution, CapabilityRefResolution) for resolution in resolutions):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


def _definition_digest(
    spec: AgentSpec,
    model_fingerprint: str,
    output_fingerprint: str,
    capabilities: "Sequence[CapabilityBinding]",
    runtime_fingerprint: str,
) -> str:
    return canonical_sha256(
        {
            "version": 7,
            "agent": {
                "id": spec.id,
                "revision": spec.revision,
                "model": spec.model,
                "system_prompt": spec.system_prompt,
                "instructions": list(spec.instructions),
                "allow_tools": list(spec.allow_tools),
                "metadata": dict(spec.metadata),
                "usage_limits": None
                if spec.usage_limits is None
                else {
                    "model_requests": spec.usage_limits.model_requests,
                    "tool_calls": spec.usage_limits.tool_calls,
                    "input_tokens": spec.usage_limits.input_tokens,
                    "output_tokens": spec.usage_limits.output_tokens,
                    "total_tokens": spec.usage_limits.total_tokens,
                },
            },
            "model_fingerprint": model_fingerprint,
            "output_schema_fingerprint": output_fingerprint,
            "capabilities": [_binding_payload(binding) for binding in capabilities],
            "runtime_fingerprint": runtime_fingerprint,
        }
    )


def _binding_payload(binding: CapabilityBinding) -> "dict[str, object]":
    return {
        "id": binding.id,
        "provider": binding.provider,
        "fingerprint": binding.fingerprint,
        "resolutions": [
            {
                "kind": resolution.ref.kind,
                "id": resolution.ref.id,
                "resolved_revision": resolution.resolved_revision,
                "fingerprint": resolution.fingerprint,
            }
            for resolution in binding.resolutions
        ],
    }


__all__ = ["AgentCompiler"]
