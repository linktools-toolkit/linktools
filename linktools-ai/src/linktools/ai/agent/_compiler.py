#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compile Asset declarations and authorized grants into AgentDefinitions."""

from collections.abc import Sequence
from types import MappingProxyType

from linktools.core import environ

from ..asset import AssetRef, AssetRepository
from ..capability import (
    CapabilityBinding,
    CapabilityProvider,
    CapabilityRefResolution,
    group_capability_refs,
    unresolved_binding,
    validate_fingerprint,
)
from ..core import canonical_sha256, validate_capability_provider
from ..errors import AIError, ErrorCode
from ..model import (
    ModelConnectionConfig,
    ModelConnectionRegistry,
    ModelResolver,
    ModelRoute,
)
from ..spec import AgentCapabilityRef, AgentSpec, PromptSpec
from ._definition import AgentDefinition
from ._output import OutputTypeRegistry

_logger = environ.get_logger("ai.agent.compiler")


class AgentCompiler:
    """Own the single declaration-to-executable-definition boundary."""

    def __init__(
        self,
        assets: AssetRepository,
        *,
        model_resolver: ModelResolver,
        model_connections: ModelConnectionRegistry,
        output_types: OutputTypeRegistry,
        capability_providers: "Sequence[CapabilityProvider]" = (),
        capability_grants: "Sequence[CapabilityBinding]" = (),
        execution_profile_fingerprint: str,
    ) -> None:
        if assets is None or not assets.ready or model_resolver is None or model_connections is None or output_types is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if not output_types.frozen:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        validate_fingerprint(execution_profile_fingerprint)
        providers: dict[str, CapabilityProvider] = {}
        for provider in capability_providers:
            try:
                name = provider.provider
            except (AttributeError, TypeError) as error:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
            if not isinstance(name, str) or not name.strip() or name in providers:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            providers[name] = provider
        grants = tuple(capability_grants)
        _validate_bindings(grants)
        if any(grant.provider in providers for grant in grants):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        self._assets = assets
        self._model_resolver = model_resolver
        self._model_connections = model_connections
        self._output_types = output_types
        self._providers = MappingProxyType(providers)
        self._grants = grants
        self._execution_profile_fingerprint = execution_profile_fingerprint

    async def compile(
        self,
        *,
        agent_id: str,
        prompt_id: str,
    ) -> AgentDefinition:
        if not agent_id.strip() or not prompt_id.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        agent = await self._assets.resolve(AssetRef("agent", agent_id))
        prompt = await self._assets.resolve(AssetRef("prompt", prompt_id))
        if type(agent.spec) is not AgentSpec or type(prompt.spec) is not PromptSpec:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        spec = agent.spec
        prompt_spec = prompt.spec
        grant_providers = {grant.provider for grant in self._grants}
        if any(ref.provider in grant_providers for ref in spec.capabilities):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        groups = group_capability_refs(spec.capabilities)
        declarative: list[CapabilityBinding] = []
        for provider_name, refs in groups:
            provider = self._providers.get(provider_name)
            if provider is None:
                if any(ref.required for ref in refs):
                    raise AIError(ErrorCode.CAPABILITY_PROVIDER_UNKNOWN)
                binding = unresolved_binding(provider_name, refs)
            else:
                binding = await provider.bind(refs)
                _validate_binding(provider_name, refs, binding)
            declarative.append(binding)
        effective = tuple(declarative) + self._grants
        _validate_bindings(effective)
        route = self._model_resolver.resolve(spec.model)
        connection = self._model_connections.resolve_optional(route.connection_id)
        output_type = self._output_types.resolve(spec.output_schema, spec.output_schema_revision)
        output_fingerprint = self._output_types.fingerprint(spec.output_schema, spec.output_schema_revision)
        digest = _definition_digest(
            spec,
            prompt_spec,
            route,
            connection,
            output_fingerprint,
            effective,
            self._execution_profile_fingerprint,
        )
        definition = AgentDefinition(
            digest,
            spec,
            prompt_spec,
            route,
            connection,
            output_type,
            output_fingerprint,
            effective,
        )
        _logger.debug(
            "agent definition compiled: agent=%s prompt=%s digest=%s capabilities=%s",
            agent_id,
            prompt_id,
            digest,
            tuple(capability.id for capability in effective),
        )
        return definition


def _validate_binding(provider: str, refs: tuple[AgentCapabilityRef, ...], binding: CapabilityBinding) -> None:
    _validate_binding_shape(binding)
    if binding.provider != provider or len(binding.resolutions) != len(refs):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    validate_fingerprint(binding.fingerprint)
    for ref, resolution in zip(refs, binding.resolutions):
        if not isinstance(resolution, CapabilityRefResolution):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if (
            resolution.id != ref.id
            or resolution.requested_revision != ref.revision
            or resolution.required != ref.required
            or resolution.status == "resolved"
            and ref.revision is not None
            and resolution.resolved_revision != ref.revision
        ):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if ref.required and resolution.status != "resolved":
            raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING)


def _validate_bindings(bindings: Sequence[CapabilityBinding]) -> None:
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
        inherit_to_subagents = binding.inherit_to_subagents
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
        or not isinstance(inherit_to_subagents, bool)
    ):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    validate_fingerprint(fingerprint)
    if any(not isinstance(resolution, CapabilityRefResolution) for resolution in resolutions):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


def _definition_digest(
    spec: AgentSpec,
    prompt: PromptSpec,
    route: ModelRoute,
    connection: "ModelConnectionConfig | None",
    output_fingerprint: str,
    capabilities: Sequence[CapabilityBinding],
    execution_profile_fingerprint: str,
) -> str:
    route_payload = {
        "route_id": route.route_id,
        "provider": route.provider,
        "model": route.model,
        "connection_id": route.connection_id,
    }
    connection_payload = None if connection is None else {
        "connection_id": connection.connection_id,
        "base_url": connection.base_url,
        "timeout_seconds": connection.timeout_seconds,
        "credential_id": connection.credential_id,
    }
    return canonical_sha256(
        {
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
                "allow_subagents": spec.allow_subagents,
                "metadata": dict(spec.metadata),
            },
            "prompt": {
                "id": prompt.id,
                "revision": prompt.revision,
                "system": prompt.system,
                "instructions": list(prompt.instructions),
            },
            "route": route_payload,
            "connection": connection_payload,
            "output_schema_fingerprint": output_fingerprint,
            "capabilities": [_binding_payload(binding) for binding in capabilities],
            "execution_profile_fingerprint": execution_profile_fingerprint,
        }
    )


def _ref_payload(ref: AgentCapabilityRef) -> dict[str, object]:
    return {
        "provider": ref.provider,
        "id": ref.id,
        "revision": ref.revision,
        "required": ref.required,
        "config": dict(ref.config),
    }


def _binding_payload(binding: CapabilityBinding) -> dict[str, object]:
    return {
        "id": binding.id,
        "provider": binding.provider,
        "fingerprint": binding.fingerprint,
        "inherit_to_subagents": binding.inherit_to_subagents,
        "resolutions": [
            {
                "id": resolution.id,
                "requested_revision": resolution.requested_revision,
                "resolved_revision": resolution.resolved_revision,
                "required": resolution.required,
                "status": resolution.status,
                "fingerprint": resolution.fingerprint,
            }
            for resolution in binding.resolutions
        ],
    }


__all__ = ["AgentCompiler"]
