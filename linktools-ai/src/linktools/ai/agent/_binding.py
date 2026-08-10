#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable Agent binding and process-local executable binding registry."""

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from ..capability import (
    CapabilityBinding,
    CapabilityInjection,
    CapabilityRefResolution,
    CapabilityResolver,
    CapabilityResolverRegistry,
    unresolved_binding,
    validate_fingerprint,
)
from ..core import canonical_sha256
from ..errors import AIError, ErrorCode
from ..model import (
    ModelConnectionConfig,
    ModelConnectionRegistry,
    ModelResolver,
    ModelRoute,
)
from ..spec import AgentCapabilityRef, AgentSpec, PromptSpec
from ._output import OutputTypeRegistry


@dataclass(frozen=True, slots=True)
class AgentBindingManifest:
    agent_id: str
    agent_revision: int
    prompt_id: str
    prompt_revision: int
    spec_fingerprint: str
    prompt_fingerprint: str
    model_route_fingerprint: str
    model_connection_fingerprint: str
    output_schema_fingerprint: str
    capabilities_fingerprint: str
    execution_profile_fingerprint: str

    def __post_init__(self) -> None:
        if not self.agent_id.strip() or self.agent_revision < 1 or not self.prompt_id.strip() or self.prompt_revision < 1:
            raise ValueError("agent binding identity is incomplete")
        for fingerprint in (
            self.spec_fingerprint,
            self.prompt_fingerprint,
            self.model_route_fingerprint,
            self.model_connection_fingerprint,
            self.output_schema_fingerprint,
            self.capabilities_fingerprint,
            self.execution_profile_fingerprint,
        ):
            validate_fingerprint(fingerprint)

    @property
    def digest(self) -> str:
        return canonical_sha256(
            {
                "agent_id": self.agent_id,
                "agent_revision": self.agent_revision,
                "prompt_id": self.prompt_id,
                "prompt_revision": self.prompt_revision,
                "spec_fingerprint": self.spec_fingerprint,
                "prompt_fingerprint": self.prompt_fingerprint,
                "model_route_fingerprint": self.model_route_fingerprint,
                "model_connection_fingerprint": self.model_connection_fingerprint,
                "output_schema_fingerprint": self.output_schema_fingerprint,
                "capabilities_fingerprint": self.capabilities_fingerprint,
                "execution_profile_fingerprint": self.execution_profile_fingerprint,
            }
        )


@dataclass(frozen=True, slots=True)
class AgentBinding:
    manifest: AgentBindingManifest
    spec: AgentSpec
    prompt: PromptSpec
    model_route: ModelRoute
    model_connection: "ModelConnectionConfig | None"
    output_type: "type[BaseModel]"
    capability_bindings: "tuple[CapabilityBinding, ...]"
    injections: "tuple[CapabilityInjection, ...]"


class AgentBindingRegistry:
    def __init__(self) -> None:
        self._bindings: dict[str, AgentBinding] = {}

    def register(self, binding: AgentBinding) -> None:
        digest = binding.manifest.digest
        previous = self._bindings.get(digest)
        if previous is not None:
            if previous.manifest != binding.manifest:
                raise AIError(ErrorCode.BINDING_CONFLICT)
            return
        self._bindings[digest] = binding

    def resolve(self, digest: str) -> AgentBinding:
        try:
            return self._bindings[digest]
        except KeyError as error:
            raise AIError(ErrorCode.BINDING_NOT_REGISTERED) from error


class AgentBinder:
    def __init__(
        self,
        *,
        model_resolver: ModelResolver,
        model_connections: ModelConnectionRegistry,
        output_types: OutputTypeRegistry,
        capability_resolvers: CapabilityResolverRegistry,
        execution_profile_fingerprint: str,
    ) -> None:
        validate_fingerprint(execution_profile_fingerprint)
        self._model_resolver = model_resolver
        self._model_connections = model_connections
        self._output_types = output_types
        self._capability_resolvers = capability_resolvers
        self._execution_profile_fingerprint = execution_profile_fingerprint

    def bind(
        self,
        spec: AgentSpec,
        prompt: PromptSpec,
        *,
        injections: tuple[CapabilityInjection, ...] = (),
    ) -> AgentBinding:
        route = self._model_resolver.resolve(spec.model)
        connection = self._model_connections.resolve_optional(route.connection_id)
        output_type = self._output_types.resolve(spec.output_schema, spec.output_schema_revision)
        groups = _group_capabilities(spec.capabilities)
        bindings: list[CapabilityBinding] = []
        capability_groups: list[dict[str, object]] = []
        for provider, refs in groups:
            resolver = self._capability_resolvers.get(provider)
            if resolver is None:
                if any(ref.required for ref in refs):
                    raise AIError(ErrorCode.CAPABILITY_PROVIDER_UNKNOWN)
                binding = unresolved_binding(provider, refs)
                resolver_fingerprint = None
            else:
                binding = _resolve_binding(resolver, refs)
                resolver_fingerprint = resolver.fingerprint
            bindings.append(binding)
            capability_groups.append(
                {
                    "provider": provider,
                    "resolver_fingerprint": resolver_fingerprint,
                    "binding_fingerprint": binding.fingerprint,
                    "inherit_to_subagents": binding.inherit_to_subagents,
                    "resolutions": [_resolution_payload(item) for item in binding.resolutions],
                }
            )
        _validate_injections(injections)
        capabilities_fingerprint = canonical_sha256(
            {
                "declarative": capability_groups,
                "injections": [
                    {
                        "id": injection.id,
                        "fingerprint": injection.fingerprint,
                        "inherit_to_subagents": injection.inherit_to_subagents,
                    }
                    for injection in injections
                ],
            }
        )
        manifest = AgentBindingManifest(
            spec.id,
            spec.revision,
            prompt.id,
            prompt.revision,
            _spec_fingerprint(spec),
            _prompt_fingerprint(prompt),
            canonical_sha256(
                {
                    "route_id": route.route_id,
                    "provider": route.provider,
                    "model": route.model,
                    "connection_id": route.connection_id,
                }
            ),
            canonical_sha256({"connection_id": None} if connection is None else {
                "connection_id": connection.connection_id,
                "base_url": connection.base_url,
                "timeout_seconds": connection.timeout_seconds,
                "credential_id": connection.credential_id,
            }),
            self._output_types.fingerprint(spec.output_schema, spec.output_schema_revision),
            capabilities_fingerprint,
            self._execution_profile_fingerprint,
        )
        binding = AgentBinding(manifest, spec, prompt, route, connection, output_type, tuple(bindings), tuple(injections))
        return binding


def _group_capabilities(refs: Sequence[AgentCapabilityRef]) -> "tuple[tuple[str, tuple[AgentCapabilityRef, ...]], ...]":
    grouped: dict[str, list[AgentCapabilityRef]] = {}
    order: list[str] = []
    for ref in refs:
        if ref.provider not in grouped:
            grouped[ref.provider] = []
            order.append(ref.provider)
        grouped[ref.provider].append(ref)
    return tuple((provider, tuple(grouped[provider])) for provider in order)


def _resolve_binding(resolver: CapabilityResolver, refs: tuple[AgentCapabilityRef, ...]) -> CapabilityBinding:
    try:
        binding = resolver.resolve(refs)
        provider = binding.provider
        resolutions = binding.resolutions
        fingerprint = binding.fingerprint
    except AIError:
        raise
    except Exception as error:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
    if provider != resolver.provider or len(resolutions) != len(refs):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    validate_fingerprint(fingerprint)
    for ref, resolution in zip(refs, resolutions):
        if not isinstance(resolution, CapabilityRefResolution) or resolution.id != ref.id or resolution.requested_revision != ref.revision:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if resolution.required != ref.required:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return binding


def _validate_injections(injections: Sequence[CapabilityInjection]) -> None:
    ids = [injection.id for injection in injections]
    if len(set(ids)) != len(ids):
        raise AIError(ErrorCode.CAPABILITY_CONFLICT)
    for injection in injections:
        validate_fingerprint(injection.fingerprint)


def _spec_fingerprint(spec: AgentSpec) -> str:
    return canonical_sha256(
        {
            "id": spec.id,
            "revision": spec.revision,
            "model": spec.model,
            "capabilities": [
                {
                    "provider": ref.provider,
                    "id": ref.id,
                    "revision": ref.revision,
                    "required": ref.required,
                    "config": dict(ref.config),
                }
                for ref in spec.capabilities
            ],
            "output_schema": spec.output_schema,
            "output_schema_revision": spec.output_schema_revision,
            "instructions": list(spec.instructions),
            "metadata": dict(spec.metadata),
        }
    )


def _prompt_fingerprint(prompt: PromptSpec) -> str:
    return canonical_sha256(
        {
            "id": prompt.id,
            "revision": prompt.revision,
            "system": prompt.system,
            "instructions": list(prompt.instructions),
        }
    )


def _resolution_payload(resolution: CapabilityRefResolution) -> "dict[str, object]":
    return {
        "id": resolution.id,
        "requested_revision": resolution.requested_revision,
        "resolved_revision": resolution.resolved_revision,
        "required": resolution.required,
        "status": resolution.status,
        "fingerprint": resolution.fingerprint,
    }


__all__ = ["AgentBinder", "AgentBinding", "AgentBindingManifest", "AgentBindingRegistry"]
