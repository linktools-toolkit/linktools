#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compile declarations into immutable executable Agent definitions."""

from collections.abc import Mapping, Sequence

from linktools.core import environ
from pydantic import BaseModel

from ..asset import AssetRef, AssetTypeRegistrySnapshot
from ..capability import (
    CapabilityBinding,
    CapabilityProvider,
    CapabilityRef,
    CapabilityRefResolution,
    RuntimeCapability,
    validate_fingerprint,
)
from ..core import canonical_sha256, validate_agent_id, validate_capability_provider
from ..errors import AIError, ErrorCode
from ..model import ModelResolver
from ..spec import AgentSpec, SkillSpec
from ._definition import AgentDefinition
from ._output import bind_output

_logger = environ.get_logger("ai.agent.compiler")

CapabilitySelection = CapabilityRef | AssetRef | RuntimeCapability


class AgentCompiler:
    """Pure compiler over already-frozen Runtime composition inputs."""

    def __init__(
        self,
        *,
        model_resolver: ModelResolver,
        capabilities: "Sequence[CapabilityBinding]" = (),
        execution_profile_fingerprint: str,
        asset_registry: "AssetTypeRegistrySnapshot | None" = None,
        capability_providers: "Sequence[CapabilityProvider]" = (),
        capability_bindings: "Mapping[str, CapabilityBinding] | None" = None,
    ) -> None:
        if model_resolver is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        validate_fingerprint(execution_profile_fingerprint)
        global_capabilities = tuple(capabilities)
        _validate_bindings(global_capabilities)
        providers = tuple(capability_providers)
        providers_by_name: dict[str, CapabilityProvider] = {}
        providers_by_type: dict[type[object], CapabilityProvider] = {}
        for provider in providers:
            name = provider.provider
            if name in providers_by_name or provider.value_type in providers_by_type:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            providers_by_name[name] = provider
            providers_by_type[provider.value_type] = provider
        bindings = dict(capability_bindings or {})
        if set(bindings) - set(providers_by_name):
            raise AIError(ErrorCode.CAPABILITY_PROVIDER_UNKNOWN)
        for name, binding in bindings.items():
            if binding.provider != name:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
            _validate_binding_shape(binding)
        self._model_resolver = model_resolver
        self._capabilities = global_capabilities
        self._execution_profile_fingerprint = execution_profile_fingerprint
        self._asset_registry = asset_registry
        self._providers_by_name = providers_by_name
        self._providers_by_type = providers_by_type
        self._family_bindings = bindings

    def compile(
        self,
        spec: AgentSpec,
        *,
        capabilities: "Sequence[CapabilitySelection]" = (),
        output: "type[BaseModel] | None" = None,
    ) -> AgentDefinition:
        validate_agent_id(spec.id)
        defaults = self._default_capabilities(spec)
        local_capabilities = self._select_capabilities((*spec.capabilities, *tuple(capabilities)))
        effective = defaults + local_capabilities
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
            "agent definition compiled: agent=%s digest=%s capabilities=%s",
            spec.id,
            digest,
            tuple((capability.provider, capability.id) for capability in effective),
        )
        return definition

    def derive_subagent(self, definition: AgentDefinition) -> AgentDefinition:
        effective = tuple(
            capability
            for capability in definition.effective_capabilities
            if capability.provider != "agent"
            and (
                not isinstance(capability, RuntimeCapability)
                or capability.inherit_to_subagents
            )
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

    def _default_capabilities(self, spec: AgentSpec) -> "tuple[CapabilityBinding, ...]":
        selected: list[CapabilityBinding] = []
        for binding in self._capabilities:
            if binding.provider != "skill":
                selected.append(binding)
                continue
            skill = self._select_skill_binding(binding, spec.allow_skills)
            if skill is not None:
                selected.append(skill)
        return tuple(selected)

    def _select_skill_binding(
        self,
        binding: CapabilityBinding,
        allow_skills: "tuple[str, ...]",
    ) -> "CapabilityBinding | None":
        if not allow_skills:
            return None
        if "*" in allow_skills:
            return binding
        allowed = frozenset(allow_skills)
        refs = tuple(
            resolution.ref
            for resolution in binding.resolutions
            if resolution.ref.id in allowed
        )
        if not refs:
            return None
        provider = self._providers_by_name.get("skill")
        if provider is None or provider.value_type is not SkillSpec:
            raise AIError(ErrorCode.CAPABILITY_PROVIDER_UNKNOWN)
        return provider.select(binding, refs)

    def _select_capabilities(
        self,
        selections: "tuple[CapabilitySelection, ...]",
    ) -> "tuple[CapabilityBinding, ...]":
        direct: list[RuntimeCapability] = []
        aggregate: set[str] = set()
        selected_refs: dict[str, list[AssetRef]] = {}
        seen_refs: set[AssetRef] = set()
        for selection in selections:
            if isinstance(selection, RuntimeCapability):
                direct.append(selection)
                continue
            if isinstance(selection, CapabilityRef):
                provider = self._providers_by_name.get(selection.provider)
                if provider is None:
                    raise AIError(ErrorCode.CAPABILITY_PROVIDER_UNKNOWN)
                if provider.value_type is SkillSpec:
                    raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
                binding = self._family_bindings.get(selection.provider)
                if selection.id is None:
                    if selection.provider in aggregate or selected_refs.get(selection.provider):
                        raise AIError(ErrorCode.CAPABILITY_CONFLICT)
                    if binding is None:
                        if selection.required:
                            raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING)
                        continue
                    aggregate.add(selection.provider)
                    continue
                if selection.provider in aggregate:
                    raise AIError(ErrorCode.CAPABILITY_CONFLICT)
                if binding is None:
                    if selection.required:
                        raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING)
                    continue
                matches = tuple(
                    resolution
                    for resolution in binding.resolutions
                    if resolution.ref.id == selection.id
                )
                if len(matches) > 1:
                    raise AIError(ErrorCode.CAPABILITY_CONFLICT)
                if not matches:
                    if selection.required:
                        raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING)
                    continue
                match = matches[0]
                if selection.revision is not None and match.resolved_revision != selection.revision:
                    raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
                self._append_ref(selection.provider, match.ref, selected_refs, seen_refs)
                continue
            if isinstance(selection, AssetRef):
                provider = self._provider_for_asset(selection)
                if provider.value_type is SkillSpec:
                    raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
                if provider.provider in aggregate:
                    raise AIError(ErrorCode.CAPABILITY_CONFLICT)
                binding = self._family_bindings.get(provider.provider)
                if binding is None or selection not in {item.ref for item in binding.resolutions}:
                    raise AIError(ErrorCode.STORAGE_NOT_FOUND)
                self._append_ref(provider.provider, selection, selected_refs, seen_refs)
                continue
            raise TypeError("capabilities must contain CapabilityRef, AssetRef, or RuntimeCapability")

        resolved: list[CapabilityBinding] = []
        for provider_name in sorted(aggregate):
            resolved.append(self._family_bindings[provider_name])
        for provider_name in sorted(selected_refs):
            provider = self._providers_by_name[provider_name]
            binding = self._family_bindings.get(provider_name)
            if binding is None:
                raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING)
            refs = tuple(sorted(selected_refs[provider_name], key=lambda ref: (ref.kind, ref.id)))
            resolved.append(provider.select(binding, refs))
        resolved.extend(direct)
        _validate_bindings(resolved)
        return tuple(resolved)

    def _provider_for_asset(self, ref: AssetRef) -> CapabilityProvider:
        registry = self._asset_registry
        if registry is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        try:
            value_type = registry.binding(ref.kind).value_type
        except AIError:
            raise
        except (KeyError, ValueError) as error:
            raise AIError(ErrorCode.ASSET_CODEC_UNKNOWN) from error
        provider = self._providers_by_type.get(value_type)
        if provider is None:
            raise AIError(ErrorCode.CAPABILITY_PROVIDER_UNKNOWN)
        return provider

    @staticmethod
    def _append_ref(
        provider: str,
        ref: AssetRef,
        selected_refs: "dict[str, list[AssetRef]]",
        seen_refs: "set[AssetRef]",
    ) -> None:
        if ref in seen_refs:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        seen_refs.add(ref)
        selected_refs.setdefault(provider, []).append(ref)


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
            "version": 6,
            "agent": {
                "id": spec.id,
                "revision": spec.revision,
                "model": spec.model,
                "system_prompt": spec.system_prompt,
                "instructions": list(spec.instructions),
                "allow_tools": list(spec.allow_tools),
                "allow_skills": list(spec.allow_skills),
                "capabilities": [
                    {
                        "provider": item.provider,
                        "id": item.id,
                        "revision": item.revision,
                        "required": item.required,
                    }
                    for item in spec.capabilities
                ],
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


__all__ = ["AgentCompiler", "CapabilitySelection"]
