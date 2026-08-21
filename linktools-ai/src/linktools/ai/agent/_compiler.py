#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compile declarations into immutable executable Agent definitions."""

from collections.abc import Sequence

from linktools.core import environ
from pydantic import BaseModel
from pydantic_ai.capabilities import Thinking
from pydantic_ai_harness.planning import Planning

from ..capability import CapabilityBinding, CapabilityRefResolution, RuntimeCapability, validate_fingerprint
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
        execution_profile_fingerprint: str,
    ) -> None:
        if model_resolver is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        validate_fingerprint(execution_profile_fingerprint)
        global_capabilities = tuple(capabilities)
        _validate_bindings(global_capabilities)
        self._model_resolver = model_resolver
        self._capabilities = global_capabilities
        self._execution_profile_fingerprint = execution_profile_fingerprint

    def compile(
        self,
        spec: AgentSpec,
        *,
        capabilities: "Sequence[RuntimeCapability]" = (),
        output: "type[BaseModel] | None" = None,
        planning: bool = False,
        thinking: bool = False,
    ) -> AgentDefinition:
        validate_agent_id(spec.id)
        if not isinstance(planning, bool) or not isinstance(thinking, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        local_capabilities = tuple(capabilities) + _mode_capabilities(
            planning=planning,
            thinking=thinking,
        )
        _validate_bindings(local_capabilities)
        effective = self._capabilities + local_capabilities
        _validate_bindings(effective)
        model = self._model_resolver.resolve(spec.model)
        output_binding = bind_output(output)
        digest = _definition_digest(
            spec,
            model,
            output_binding.fingerprint,
            effective,
            self._execution_profile_fingerprint,
        )
        definition = AgentDefinition(
            digest,
            spec,
            model,
            output_binding.value_type,
            output_binding.fingerprint,
            effective,
        )
        _logger.debug(
            "agent definition compiled: agent=%s digest=%s capabilities=%s planning=%s thinking=%s",
            spec.id,
            digest,
            tuple(capability.id for capability in effective),
            planning,
            thinking,
        )
        return definition

    def derive_subagent(self, definition: AgentDefinition) -> AgentDefinition:
        effective = tuple(
            capability
            for capability in definition.effective_capabilities
            if not isinstance(capability, RuntimeCapability)
            or capability.inherit_to_subagents
        )
        if effective == definition.effective_capabilities:
            return definition
        digest = _definition_digest(
            definition.spec,
            definition.model,
            definition.output_schema_fingerprint,
            effective,
            self._execution_profile_fingerprint,
        )
        return AgentDefinition(
            digest,
            definition.spec,
            definition.model,
            definition.output_type,
            definition.output_schema_fingerprint,
            effective,
        )


def _mode_capabilities(*, planning: bool, thinking: bool) -> "tuple[RuntimeCapability, ...]":
    values: list[RuntimeCapability] = []
    if planning:
        values.append(RuntimeCapability("planning", Planning(), revision=1))
    if thinking:
        values.append(RuntimeCapability("thinking", Thinking(), revision=1))
    return tuple(values)


def _validate_bindings(bindings: "Sequence[CapabilityBinding]") -> None:
    for binding in bindings:
        _validate_binding_shape(binding)
    identities = [(binding.provider, binding.id) for binding in bindings]
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
    model: object,
    output_fingerprint: str,
    capabilities: "Sequence[CapabilityBinding]",
    execution_profile_fingerprint: str,
) -> str:
    return canonical_sha256(
        {
            "version": 4,
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
            "model_fingerprint": model.fingerprint,
            "output_schema_fingerprint": output_fingerprint,
            "capabilities": [_binding_payload(binding) for binding in capabilities],
            "execution_profile_fingerprint": execution_profile_fingerprint,
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
