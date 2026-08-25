#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compile declarations and output bindings from frozen Runtime inputs."""

from collections.abc import Mapping, Sequence

from linktools.core import environ
from pydantic import BaseModel

from ..capability import (
    CapabilityBinding,
    CapabilityRefResolution,
    RuntimeCapability,
    validate_fingerprint,
)
from ..core import (
    JsonValue,
    canonical_sha256,
    validate_agent_id,
    validate_capability_provider,
)
from ..errors import AIError, ErrorCode
from ..model import ModelResolver
from ..spec import AgentSpec, AgentSpecCodec
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
        local_capabilities = self._restore_local_capabilities(capabilities)
        return self._compile_definition(spec, local_capabilities)

    def _compile_definition(
        self,
        spec: AgentSpec,
        local_capabilities: "tuple[RuntimeCapability, ...]",
        *,
        global_runtime_overrides: "Mapping[str, RuntimeCapability] | None" = None,
        global_runtime_descriptors: "tuple[Mapping[str, JsonValue], ...] | None" = None,
    ) -> AgentDefinition:
        validate_agent_id(spec.id)
        globals_effective: list[CapabilityBinding] = []
        overrides = dict(global_runtime_overrides or {})
        for capability in self._capabilities:
            if isinstance(capability, RuntimeCapability) and capability.durable:
                replacement = overrides.pop(capability.id, None)
                globals_effective.append(capability if replacement is None else replacement)
            else:
                globals_effective.append(capability)
        if overrides:
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
        effective: tuple[CapabilityBinding, ...] = (
            *globals_effective,
            *local_capabilities,
            *self._platform_capabilities,
        )
        _validate_bindings(effective)
        local_descriptors = tuple(
            _required_runtime_descriptor(capability)
            for capability in local_capabilities
        )
        if global_runtime_descriptors is None:
            selected_global_descriptors = tuple(
                _required_runtime_descriptor(capability)
                for capability in globals_effective
                if isinstance(capability, RuntimeCapability) and capability.durable
            )
        else:
            selected_global_descriptors = global_runtime_descriptors
        model = self._model_resolver.resolve(spec.model)
        digest = _definition_digest(
            spec,
            model.fingerprint,
            effective,
            self._runtime_fingerprint,
        )
        definition = AgentDefinition(
            digest=digest,
            spec=spec,
            model=model,
            effective_capabilities=effective,
            local_runtime_capability_descriptors=local_descriptors,
            trusted_tool_classes=self._trusted_tool_classes,
            trusted_mcp_selectors=self._trusted_mcp_selectors,
            global_runtime_capability_descriptors=selected_global_descriptors,
        )
        _logger.debug(
            "agent definition compiled: agent=%s digest=%s capabilities=%s",
            spec.id,
            digest,
            tuple((capability.provider, capability.id) for capability in effective),
        )
        return definition

    def _restore_local_capabilities(
        self,
        capabilities: "Sequence[RuntimeCapability]",
    ) -> "tuple[RuntimeCapability, ...]":
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
        restored: list[RuntimeCapability] = []
        for capability in local_capabilities:
            descriptor = _required_runtime_descriptor(capability)
            try:
                value = RuntimeCapability.restore(descriptor)
            except AIError as error:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
            if (
                type(value.capability) is not type(capability.capability)
                or value.descriptor != descriptor
            ):
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            restored.append(value)
        return tuple(restored)

    def bind(
        self,
        definition: AgentDefinition,
        *,
        output: "type[BaseModel] | None" = None,
    ) -> AgentBinding:
        if not isinstance(definition, AgentDefinition):
            raise TypeError("definition must be AgentDefinition")
        output_binding = bind_output(output)
        binding_digest = _binding_digest(definition.digest, output_binding.fingerprint)
        snapshot = AgentBindingSnapshot(
            version=1,
            agent_spec=definition.spec,
            agent_digest=definition.digest,
            output_schema_id=output_binding.schema_id,
            output_schema_revision=output_binding.schema_revision,
            output_schema_fingerprint=output_binding.schema_fingerprint,
            local_runtime_capability_descriptors=definition.local_runtime_capability_descriptors,
            binding_digest=binding_digest,
            global_runtime_capability_descriptors=definition.global_runtime_capability_descriptors,
            output_schema_definition=output_binding.schema_definition,
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
            global_overrides, global_descriptors = self._restore_global_capabilities(
                snapshot
            )
            output_descriptor: dict[str, JsonValue] = {
                "version": 1,
                "schema_id": snapshot.output_schema_id,
                "schema_revision": snapshot.output_schema_revision,
                "schema_fingerprint": snapshot.output_schema_fingerprint,
                "schema_definition": snapshot.output_schema_definition,
            }
            output_binding = restore_output(output_descriptor)
            definition = self._compile_definition(
                snapshot.agent_spec,
                local_capabilities,
                global_runtime_overrides=global_overrides,
                global_runtime_descriptors=global_descriptors,
            )
        except AIError as error:
            if error.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE:
                raise
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE) from error
        binding_digest = _binding_digest(definition.digest, output_binding.fingerprint)
        if (
            definition.digest != snapshot.agent_digest
            or binding_digest != snapshot.binding_digest
        ):
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
        return AgentBinding(
            snapshot.binding_digest,
            definition,
            output_binding,
            snapshot,
        )

    def _restore_global_capabilities(
        self,
        snapshot: AgentBindingSnapshot,
    ) -> "tuple[dict[str, RuntimeCapability], tuple[Mapping[str, JsonValue], ...]]":
        current = tuple(
            capability
            for capability in self._capabilities
            if isinstance(capability, RuntimeCapability) and capability.durable
        )
        current_by_id = {capability.id: capability for capability in current}
        if len(current_by_id) != len(current):
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
        if len(snapshot.global_runtime_capability_descriptors) != len(current):
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
        restored_by_id: dict[str, RuntimeCapability] = {}
        for descriptor in snapshot.global_runtime_capability_descriptors:
            identity = descriptor.get("id")
            if not isinstance(identity, str) or identity in restored_by_id:
                raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
            current_capability = current_by_id.get(identity)
            if current_capability is None:
                raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
            if _runtime_capability_semantics(descriptor) != _runtime_capability_semantics(
                current_capability
            ):
                raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
            restored_by_id[identity] = current_capability
        if set(restored_by_id) != set(current_by_id):
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
        return restored_by_id, snapshot.global_runtime_capability_descriptors


def _runtime_capability_semantics(
    capability: "RuntimeCapability | Mapping[str, JsonValue]",
) -> "tuple[str, int, str, JsonValue]":
    descriptor = (
        _required_runtime_descriptor(capability)
        if isinstance(capability, RuntimeCapability)
        else capability
    )
    identity = descriptor.get("id")
    revision = descriptor.get("revision")
    serialization_name = descriptor.get("serialization_name")
    config = descriptor.get("config")
    fingerprint = descriptor.get("fingerprint")
    if (
        not isinstance(identity, str)
        or not isinstance(revision, int)
        or isinstance(revision, bool)
        or not isinstance(serialization_name, str)
        or not isinstance(config, dict)
        or not isinstance(fingerprint, str)
        or canonical_sha256(
            {
                "id": identity,
                "revision": revision,
                "serialization_name": serialization_name,
                "config": config,
            }
        )
        != fingerprint
    ):
        raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
    return (
        identity,
        revision,
        serialization_name,
        config,
    )


def _required_runtime_descriptor(
    capability: RuntimeCapability,
) -> Mapping[str, JsonValue]:
    descriptor = capability.descriptor
    if descriptor is None:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return descriptor


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
            "agent": AgentSpecCodec().to_payload(spec),
            "model_fingerprint": model_fingerprint,
            "capabilities": [
                _binding_payload(binding) for binding in capabilities
            ],
            "runtime_fingerprint": runtime_fingerprint,
        }
    )


def _binding_digest(agent_digest: str, output_fingerprint: str) -> str:
    return canonical_sha256(
        {
            "version": 1,
            "agent_digest": agent_digest,
            "output_binding_fingerprint": output_fingerprint,
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
