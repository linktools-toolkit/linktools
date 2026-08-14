#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compile declarations into immutable executable agent definitions."""

from collections.abc import Sequence
from types import MappingProxyType

from linktools.core import environ

from ..asset import AssetRef, AssetRepository, ResolvedAsset
from ..capability import CapabilityBinding, CapabilityGrant, CapabilityProvider, CapabilityRefResolution
from ..capability.validation_api import group_capability_refs, unresolved_binding, validate_fingerprint
from ..core import canonical_sha256, validate_capability_provider
from ..errors import AIError, ErrorCode
from ..model import ModelResolver
from ..spec import AgentCapabilityRef, AgentSpec, PromptSpec
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
        capability_grants: "Sequence[CapabilityGrant]" = (),
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
        grants = tuple(capability_grants)
        _validate_bindings(grants)
        if any(grant.provider in providers for grant in grants):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        self._assets = assets
        self._model_resolver = model_resolver
        self._output_types = output_types
        self._providers = MappingProxyType(providers)
        self._grants = grants
        self._execution_profile_fingerprint = execution_profile_fingerprint

    async def compile(self, *, agent_id: str, prompt_id: str) -> AgentDefinition:
        return await self._compile(agent_id=agent_id, prompt_id=prompt_id, direct_grants=self._grants, missing_definition_asset_code=None)

    async def compile_subagent(self, *, agent_id: str, prompt_id: str) -> AgentDefinition:
        return await self._compile(
            agent_id=agent_id,
            prompt_id=prompt_id,
            direct_grants=tuple(grant for grant in self._grants if grant.inherit_to_subagents),
            missing_definition_asset_code=ErrorCode.AGENT_NOT_FOUND,
        )

    async def _compile(
        self,
        *,
        agent_id: str,
        prompt_id: str,
        direct_grants: "Sequence[CapabilityGrant]",
        missing_definition_asset_code: ErrorCode | None,
    ) -> AgentDefinition:
        if not agent_id.strip() or not prompt_id.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        agent = await self._resolve_definition_asset(AssetRef("agent", agent_id), missing_definition_asset_code)
        prompt = await self._resolve_definition_asset(AssetRef("prompt", prompt_id), missing_definition_asset_code)
        if type(agent.spec) is not AgentSpec or type(prompt.spec) is not PromptSpec:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        spec = agent.spec
        prompt_spec = prompt.spec
        grants = tuple(direct_grants)
        grant_providers = {grant.provider for grant in grants}
        if any(ref.provider in grant_providers for ref in spec.capabilities):
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
        effective = tuple(declarative) + grants
        _validate_bindings(effective)
        model = self._model_resolver.resolve(spec.model)
        output_type = self._output_types.resolve(spec.output_schema, spec.output_schema_revision)
        output_fingerprint = self._output_types.fingerprint(spec.output_schema, spec.output_schema_revision)
        digest = _definition_digest(spec, prompt_spec, model, output_fingerprint, effective, self._execution_profile_fingerprint)
        definition = AgentDefinition(digest, spec, prompt_spec, model, output_type, output_fingerprint, effective)
        _logger.debug("agent definition compiled: agent=%s prompt=%s digest=%s capabilities=%s", agent_id, prompt_id, digest, tuple(capability.id for capability in effective))
        return definition

    async def _resolve_definition_asset(self, ref: AssetRef, missing_code: ErrorCode | None) -> ResolvedAsset:
        try:
            return await self._assets.resolve(ref)
        except AIError as error:
            if missing_code is not None and error.code is ErrorCode.STORAGE_NOT_FOUND:
                raise AIError(
                    missing_code,
                    "agent definition asset is unavailable",
                    safe_details={"kind": ref.kind, "id": ref.id},
                ) from error
            raise


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
    prompt: PromptSpec,
    model: object,
    output_fingerprint: str,
    capabilities: "Sequence[CapabilityBinding]",
    execution_profile_fingerprint: str,
) -> str:
    return canonical_sha256({
        "agent": {
            "id": spec.id,
            "revision": spec.revision,
            "model": spec.model,
            "capabilities": [_ref_payload(ref) for ref in spec.capabilities],
            "output_schema": spec.output_schema,
            "output_schema_revision": spec.output_schema_revision,
            "instructions": list(spec.instructions),
            "allow_tools": spec.allow_tools,
            "allow_skills": spec.allow_skills,
            "metadata": dict(spec.metadata),
        },
        "prompt": {"id": prompt.id, "revision": prompt.revision, "system": prompt.system, "instructions": list(prompt.instructions)},
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
