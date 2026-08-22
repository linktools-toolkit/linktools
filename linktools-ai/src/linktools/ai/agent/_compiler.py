#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compile declarations and exact output bindings from frozen Runtime inputs."""

from collections.abc import Mapping, Sequence

from linktools.core import environ
from pydantic import BaseModel

from ..capability import (
    CapabilityBinding,
    CapabilityRefResolution,
    RuntimeCapability,
    validate_fingerprint,
)
from ..core import JsonValue, canonical_sha256, validate_agent_id, validate_capability_provider
from ..errors import AIError, ErrorCode
from ..model import ModelResolver
from ..spec import AgentSpec
from ._binding import AgentBinding, AgentBindingSnapshot
from ._definition import AgentDefinition
from ._output import bind_output, restore_output

_logger = environ.get_logger("ai.agent.compiler")


class AgentCompiler:
    """Pure compiler over already-frozen Runtime composition inputs."""

    def __init__(
        self,
        *,
        model_resolver: ModelResolver,
        capabilities: "Sequence[CapabilityBinding]" = (),
        platform_capabilities: "Sequence[CapabilityBinding]" = (),
        runtime_fingerprint: str,
        trusted_tool_classes: "Mapping[str, str] | None" = None,
        trusted_mcp_selectors: "Sequence[str]" = (),
    ) -> None:
        if model_resolver is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        validate_fingerprint(runtime_fingerprint)
        global_capabilities = tuple(capabilities)
        platform = tuple(platform_capabilities)
        _validate_bindings((*global_capabilities, *platform))
        trusted = tuple(sorted((trusted_tool_classes or {}).items()))
        selectors = tuple(sorted(trusted_mcp_selectors))
        if len(selectors) != len(set(selectors)) or any(
            not isinstance(selector, str)
            or not selector.startswith("mcp__")
            or selector == "mcp__"
            or "__" in selector[5:]
            for selector in selectors
        ):
            raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
        self._model_resolver = model_resolver
        self._capabilities = global_capabilities
        self._platform_capabilities = platform
        self._runtime_fingerprint = runtime_fingerprint
        self._trusted_tool_classes = trusted
        self._trusted_mcp_selectors = selectors

    def compile(
        self,
        spec: AgentSpec,
        *,
        capabilities: "Sequence[RuntimeCapability]" = (),
    ) -> AgentDefinition:
        validate_agent_id(spec.id)
        local_capabilities = tuple(capabilities)
        if any(
            not isinstance(capability, RuntimeCapability)
            for capability in local_capabilities
        ):
            raise TypeError(
                "agent-local capabilities must contain RuntimeCapability values"
            )
        if any(not capability.durable for capability in local_capabilities):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        restored_locals: list[RuntimeCapability] = []
        local_descriptors: list[Mapping[str, JsonValue]] = []
        for capability in local_capabilities:
            descriptor = capability.descriptor
            if descriptor is None:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            try:
                restored = RuntimeCapability.restore(descriptor)
            except AIError as error:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
            restored_descriptor = restored.descriptor
            if restored_descriptor is None or restored_descriptor != descriptor:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            restored_locals.append(restored)
            local_descriptors.append(restored_descriptor)
        effective: tuple[CapabilityBinding, ...] = (
            *self._capabilities,
            *restored_locals,
            *self._platform_capabilities,
        )
        _validate_bindings(effective)
        model = self._model_resolver.resolve(spec.model)
        digest = _definition_digest(
            spec,
            model.fingerprint,
            effective,
            self._runtime_fingerprint,
        )
        definition = AgentDefinition(
            digest,
            spec,
            model,
            effective,
            tuple(local_descriptors),
            self._trusted_tool_classes,
            self._trusted_mcp_selectors,
        )
        _logger.debug(
            "agent definition compiled: agent=%s digest=%s capabilities=%s",
            spec.id,
            digest,
            tuple(
                (capability.provider, capability.id)
                for capability in effective
            ),
        )
        return definition

    def bind(
        self,
        definition: AgentDefinition,
        *,
        output: "type[BaseModel] | None" = None,
    ) -> AgentBinding:
        if not isinstance(definition, AgentDefinition):
            raise TypeError("definition must be AgentDefinition")
        output_binding = bind_output(output)
        binding_digest = canonical_sha256(
            {
                "version": 1,
                "agent_digest": definition.digest,
                "output_binding_fingerprint": output_binding.fingerprint,
            }
        )
        snapshot = AgentBindingSnapshot(
            version=1,
            agent_spec=definition.spec,
            agent_digest=definition.digest,
            output_type_module=output_binding.value_type.__module__,
            output_type_qualname=output_binding.value_type.__qualname__,
            output_schema_id=output_binding.schema_id,
            output_schema_revision=output_binding.schema_revision,
            output_schema_fingerprint=output_binding.schema_fingerprint,
            local_runtime_capability_descriptors=definition.local_runtime_capability_descriptors,
            binding_digest=binding_digest,
        )
        binding = AgentBinding(
            binding_digest,
            definition,
            output_binding,
            snapshot,
        )
        _logger.debug(
            "agent binding compiled: agent=%s agent_digest=%s binding_digest=%s",
            definition.spec.id,
            definition.digest,
            binding_digest,
        )
        return binding

    def restore(self, snapshot: AgentBindingSnapshot) -> AgentBinding:
        if not isinstance(snapshot, AgentBindingSnapshot):
            raise TypeError("snapshot must be AgentBindingSnapshot")
        try:
            local_capabilities = tuple(
                RuntimeCapability.restore(descriptor)
                for descriptor in snapshot.local_runtime_capability_descriptors
            )
            output_binding = restore_output(
                {
                    "version": 1,
                    "schema_id": snapshot.output_schema_id,
                    "schema_revision": snapshot.output_schema_revision,
                    "schema_fingerprint": snapshot.output_schema_fingerprint,
                    "module": snapshot.output_type_module,
                    "qualname": snapshot.output_type_qualname,
                }
            )
            definition = self.compile(
                snapshot.agent_spec,
                capabilities=local_capabilities,
            )
            binding = self.bind(
                definition,
                output=output_binding.value_type,
            )
        except AIError as error:
            if error.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE:
                raise
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE) from error
        if (
            definition.digest != snapshot.agent_digest
            or binding.digest != snapshot.binding_digest
            or binding.snapshot != snapshot
        ):
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
        return binding


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
    if (
        not isinstance(binding_id, str)
        or not binding_id.strip()
        or not isinstance(resolutions, tuple)
    ):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    validate_fingerprint(fingerprint)
    if any(
        not isinstance(resolution, CapabilityRefResolution)
        for resolution in resolutions
    ):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


def _definition_digest(
    spec: AgentSpec,
    model_fingerprint: str,
    capabilities: "Sequence[CapabilityBinding]",
    runtime_fingerprint: str,
) -> str:
    return canonical_sha256(
        {
            "version": 1,
            "agent": {
                "id": spec.id,
                "revision": spec.revision,
                "model": spec.model,
                "system_prompt": spec.system_prompt,
                "instructions": list(spec.instructions),
                "allow_tools": list(spec.allow_tools),
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
            "capabilities": [
                _binding_payload(binding) for binding in capabilities
            ],
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
