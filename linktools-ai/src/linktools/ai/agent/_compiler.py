#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compile declarations into immutable executable agent definitions."""

from collections.abc import Sequence
from types import MappingProxyType

from linktools.core import environ

from ..asset import AssetRepository
from ..capability import (
    CapabilityBinding,
    CapabilityProvider,
    CapabilityRefResolution,
    RuntimeCapability,
    group_capability_refs,
    unresolved_binding,
    validate_fingerprint,
)
from ..core import canonical_sha256, validate_agent_id, validate_capability_provider
from ..errors import AIError, ErrorCode
from ..model import ModelResolver
from ..spec import AgentCapabilityRef, AgentSpec
from ._definition import AgentDefinition
from ._output import OutputTypeRegistry

_logger = environ.get_logger("ai.agent.compiler")


class AgentCompiler:
    def __init__(
        self,
        assets: AssetRepository,
        *,
        model_resolver: ModelResolver,
        output_types: OutputTypeRegistry,
        capability_providers: "Sequence[CapabilityProvider]" = (),
        capabilities: "Sequence[RuntimeCapability]" = (),
        execution_profile_fingerprint: str,
    ) -> None:
        if assets is None or not assets.ready or model_resolver is None or output_types is None or not output_types.frozen:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        validate_fingerprint(execution_profile_fingerprint)
        providers: "dict[str, CapabilityProvider]" = {}
        for provider in capability_providers:
            name = provider.provider
            if not isinstance(name, str) or not name.strip() or name in providers:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            providers[name] = provider
        direct_capabilities = tuple(capabilities)
        _validate_bindings(direct_capabilities)
        if any(capability.provider in providers for capability in direct_capabilities):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        if any(provider_name == "runtime" for provider_name in providers):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        self._assets = assets
        self._model_resolver = model_resolver
        self._output_types = output_types
        self._providers = MappingProxyType(providers)
        self._capabilities = direct_capabilities
        self._execution_profile_fingerprint = execution_profile_fingerprint

    async def compile(self, spec: AgentSpec) -> AgentDefinition:
        return await self._compile(spec=spec, direct_capabilities=self._capabilities)

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

    async def _compile(
        self,
        *,
        spec: AgentSpec,
        direct_capabilities: "Sequence[RuntimeCapability]",
    ) -> AgentDefinition:
        validate_agent_id(spec.id)
        direct_providers = {capability.provider for capability in direct_capabilities}
        if any(ref.provider in direct_providers for ref in spec.capabilities):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        declarative: "list[CapabilityBinding]" = []
        for provider_name, refs in group_capability_refs(spec.capabilities):
            provider = self._providers.get(provider_name)
            if provider is None:
                if any(ref.required for ref in refs):
                    raise AIError(ErrorCode.CAPABILITY_PROVIDER_UNKNOWN)
                binding = unresolved_binding(provider_name, refs)
            else:
                binding = await provider.bind(refs, assets=self._assets)
                _validate_binding(provider_name, refs, binding)
            declarative.append(binding)
        effective = tuple(declarative) + tuple(direct_capabilities)
        _validate_bindings(effective)
        model = self._model_resolver.resolve(spec.model)
        output_type = self._output_types.resolve(spec.output_schema, spec.output_schema_revision)
        output_fingerprint = self._output_types.fingerprint(spec.output_schema, spec.output_schema_revision)
        digest = _definition_digest(spec, model, output_fingerprint, effective, self._execution_profile_fingerprint)
        definition = AgentDefinition(digest, spec, model, output_type, output_fingerprint, effective)
        _logger.debug(
            "agent definition compiled: agent=%s digest=%s capabilities=%s",
            spec.id,
            digest,
            tuple(capability.id for capability in effective),
        )
        return definition

def _validate_binding(provider: str, refs: "tuple[AgentCapabilityRef, ...]", binding: CapabilityBinding) -> None:
    _validate_binding_shape(binding)
    if binding.provider != provider or len(binding.resolutions) != len(refs):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    for ref, resolution in zip(refs, binding.resolutions):
        if resolution.id != ref.id or resolution.requested_revision != ref.revision or resolution.required != ref.required:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if ref.required and resolution.status != "resolved":
            raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING)


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
    return canonical_sha256({
        "version": 3,
        "agent": {
            "id": spec.id,
            "revision": spec.revision,
            "model": spec.model,
            "capabilities": [_ref_payload(ref) for ref in spec.capabilities],
            "output_schema": spec.output_schema,
            "output_schema_revision": spec.output_schema_revision,
            "system_prompt": spec.system_prompt,
            "instructions": list(spec.instructions),
            "allow_tools": spec.allow_tools,
            "allow_skills": spec.allow_skills,
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
    })


def _ref_payload(ref: AgentCapabilityRef) -> "dict[str, object]":
    return {"provider": ref.provider, "id": ref.id, "revision": ref.revision, "required": ref.required, "config": dict(ref.config)}


def _binding_payload(binding: CapabilityBinding) -> "dict[str, object]":
    payload = {"id": binding.id, "provider": binding.provider, "fingerprint": binding.fingerprint, "resolutions": []}
    payload["resolutions"] = [{"id": resolution.id, "requested_revision": resolution.requested_revision, "resolved_revision": resolution.resolved_revision, "required": resolution.required, "status": resolution.status, "fingerprint": resolution.fingerprint} for resolution in binding.resolutions]
    return payload


__all__ = ["AgentCompiler"]
