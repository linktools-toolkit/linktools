#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conformance tests for explicit non-workspace pipeline composition."""

import json
import re
import sys
from dataclasses import MISSING, dataclass, fields, replace
from pathlib import Path

import pytest
from linktools.ai.agent import AgentBinder, AgentBindingRegistry, OutputTypeRegistry
from linktools.ai.app import (
    RuntimeDependencies,
    RuntimePersistenceConfig,
    build_local_runtime_services,
    build_runtime,
    open_runtime_resources,
)
from linktools.ai.capability import (
    CapabilityInjection,
    CapabilityRefResolution,
    CapabilityResolverRegistry,
    CapabilityRuntimeContext,
)
from linktools.ai.core import (
    AuthorizationAction,
    Principal,
    PrincipalKind,
    ResourceKind,
    ResourceRef,
    TenantAuthorizationPolicy,
    canonical_sha256,
    service_principal,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import (
    ModelConnectionConfig,
    ModelConnectionRegistry,
    ModelMaterializer,
    ModelRegistry,
    ModelRoute,
    SnapshotModelResolver,
)
from linktools.ai.observe import MiddlewarePipeline
from linktools.ai.runtime import (
    AllowAllToolPolicy,
    ExecutionRequest,
    ToolAuthorization,
    ToolDescriptor,
)
from linktools.ai.spec import AgentCapabilityRef, AgentSpec, PromptSpec, PromptSpecCodec
from linktools.ai.workspace import DisabledSandbox
from pydantic import BaseModel
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel


class _Answer(BaseModel):
    value: str


class _Materializer(ModelMaterializer):
    def materialize(self, route: ModelRoute, connection: "ModelConnectionConfig | None") -> Model:
        del route, connection
        return TestModel(call_tools=[], custom_output_args={"value": "ok"})


@dataclass(frozen=True, slots=True)
class _CapabilityBinding:
    provider: str
    resolutions: "tuple[CapabilityRefResolution, ...]"
    fingerprint: str
    inherit_to_subagents: bool = True

    async def materialize(self, context: CapabilityRuntimeContext) -> tuple[()]:
        del context
        return ()


class _CapabilityResolver:
    provider = "custom"
    fingerprint = canonical_sha256("custom-resolver")

    def resolve(self, refs: "tuple[AgentCapabilityRef, ...]") -> _CapabilityBinding:
        return _CapabilityBinding(
            self.provider,
            tuple(
                CapabilityRefResolution(
                    ref.id,
                    ref.revision,
                    1,
                    ref.required,
                    "resolved",
                    canonical_sha256(ref.id),
                )
                for ref in refs
            ),
            canonical_sha256([ref.id for ref in refs]),
        )


def _binder() -> AgentBinder:
    registry = ModelRegistry()
    snapshot = registry.prime({"route": ModelRoute("route", "test", "test")})
    output_types = OutputTypeRegistry()
    output_types.register("answer", 1, _Answer)
    return AgentBinder(
        model_resolver=SnapshotModelResolver(snapshot),
        model_connections=ModelConnectionRegistry(),
        output_types=output_types,
        capability_resolvers=CapabilityResolverRegistry(()),
        execution_profile_fingerprint=canonical_sha256("non-workspace-profile"),
    )


@pytest.mark.asyncio
async def test_disabled_sandbox_is_fail_closed_without_side_effects(tmp_path: Path) -> None:
    sandbox = DisabledSandbox()
    target = tmp_path / "should-not-exist"
    command = f'{sys.executable} -c "from pathlib import Path; Path({str(target)!r}).write_text(\'created\')"'
    for operation in (
        sandbox.read_file(str(target)),
        sandbox.write_file(str(target), "content"),
        sandbox.run(command),
    ):
        with pytest.raises(AIError) as error:
            await operation
        assert error.value.code is ErrorCode.SANDBOX_UNAVAILABLE
    assert not target.exists()


@pytest.mark.asyncio
async def test_allow_all_policy_is_explicit_and_deterministic() -> None:
    policy = AllowAllToolPolicy()
    dependency_fields = {item.name: item for item in fields(RuntimeDependencies)}
    assert dependency_fields["sandbox"].default is MISSING
    assert dependency_fields["tool_policy"].default is MISSING
    principal = service_principal("tenant-a", "worker-a")
    other_principal = Principal("user-a", "tenant-a")
    execution = ResourceRef(ResourceKind.EXECUTION, "execution-a", "tenant-a")
    tool = ToolDescriptor("temporary-tool", replay_safe=True)
    for current_principal, digest in ((principal, "digest-a"), (other_principal, "digest-b")):
        assert await policy.authorize_tool(current_principal, execution, tool, digest) is ToolAuthorization.ALLOW
    assert re.fullmatch(r"[0-9a-f]{64}", DisabledSandbox().fingerprint)
    assert re.fullmatch(r"[0-9a-f]{64}", policy.fingerprint)
    assert policy.fingerprint == AllowAllToolPolicy().fingerprint
    assert policy.fingerprint != DisabledSandbox().fingerprint


@pytest.mark.asyncio
async def test_service_principal_keeps_tenant_authorization() -> None:
    principal = service_principal("tenant-a", "worker-a")
    assert principal.kind == PrincipalKind.SERVICE.value
    assert principal.tenant_id == "tenant-a"
    assert principal.principal_id == "worker-a"
    for tenant_id, principal_id in (("", "worker"), ("tenant", "")):
        with pytest.raises(AIError) as error:
            service_principal(tenant_id, principal_id)
        assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID

    policy = TenantAuthorizationPolicy()
    owned = ResourceRef(ResourceKind.EXECUTION, "execution-a", "tenant-a", owner_principal_id="worker-a")
    await policy.authorize(principal, AuthorizationAction.EXECUTION_RUN, owned)
    with pytest.raises(AIError) as error:
        await policy.authorize(principal, AuthorizationAction.EXECUTION_RUN, ResourceRef(ResourceKind.EXECUTION, "execution-b", "tenant-b"))
    assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
    with pytest.raises(AIError) as error:
        await policy.authorize(principal, AuthorizationAction.EXECUTION_RUN, ResourceRef(ResourceKind.EXECUTION, "execution-c", "tenant-a", owner_principal_id="other"))
    assert error.value.code is ErrorCode.AUTHORIZATION_DENIED


def test_prompt_spec_variables_are_removed_and_legacy_payloads_fail() -> None:
    prompt = PromptSpec("prompt", 1, "system", ("instruction",))
    codec = PromptSpecCodec()
    encoded = codec.encode(prompt)
    assert "variables" not in json.loads(encoded)
    assert codec.decode(encoded) == prompt
    with pytest.raises(TypeError):
        PromptSpec("prompt", 1, "system", (), variables=("name",))
    legacy = json.dumps(
        {
            "id": "prompt",
            "revision": 1,
            "system": "system",
            "instructions": [],
            "variables": ["name"],
        }
    ).encode("utf-8")
    with pytest.raises(AIError) as error:
        codec.decode(legacy)
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


def test_prompt_fingerprint_tracks_only_active_prompt_fields() -> None:
    binder = _binder()
    spec = AgentSpec("agent", 1, "route", (), "answer", 1)
    base = binder.bind(spec, PromptSpec("prompt", 1, "system", ()))
    same = binder.bind(spec, PromptSpec("prompt", 1, "system", ()))
    changed_system = binder.bind(spec, PromptSpec("prompt", 1, "other", ()))
    changed_instructions = binder.bind(spec, PromptSpec("prompt", 1, "system", ("one",)))
    assert base.manifest.prompt_fingerprint == same.manifest.prompt_fingerprint
    assert base.manifest.digest == same.manifest.digest
    assert base.manifest.prompt_fingerprint != changed_system.manifest.prompt_fingerprint
    assert base.manifest.prompt_fingerprint != changed_instructions.manifest.prompt_fingerprint


@pytest.mark.asyncio
async def test_non_workspace_pipeline_reuses_composition_and_supports_per_bind_extensions() -> None:
    authorization = TenantAuthorizationPolicy()
    principal = service_principal("tenant-a", "worker-a")
    routes = ModelRegistry()
    snapshot = routes.prime({"default": ModelRoute("default", "test", "test")})
    output_types = OutputTypeRegistry()
    output_types.register("answer", 1, _Answer)
    output_types.freeze()
    model_connections = ModelConnectionRegistry()
    async with open_runtime_resources(RuntimePersistenceConfig.in_memory(namespace="non-workspace-test")) as resources:
        local = build_local_runtime_services(resources, authorization, grant_key=b"non-workspace-key", materializer=_Materializer())
        dependencies = RuntimeDependencies(
            model_resolver=SnapshotModelResolver(snapshot),
            capability_resolvers=(),
            model_connections=model_connections,
            middleware=MiddlewarePipeline(),
            sandbox=DisabledSandbox(),
            tool_policy=AllowAllToolPolicy(),
            output_types=output_types,
            local=local,
        )
        spec_a = AgentSpec("pipeline-agent-a", 1, "default", (), "answer", 1)
        prompt_a = PromptSpec("pipeline-prompt-a", 1, "system-a", ())
        runtime_a = build_runtime(spec_a, prompt_a, dependencies=dependencies)
        first = await runtime_a.run(ExecutionRequest("first", principal, idempotency_key="job-a-1"))
        runtime_a_again = build_runtime(spec_a, prompt_a, dependencies=dependencies)
        second = await runtime_a_again.run(ExecutionRequest("second", principal, idempotency_key="job-a-2"))
        runtime_b = build_runtime(
            AgentSpec("pipeline-agent-b", 1, "default", (), "answer", 1),
            PromptSpec("pipeline-prompt-b", 1, "system-b", ()),
            dependencies=dependencies,
        )
        third = await runtime_b.run(ExecutionRequest("third", principal, idempotency_key="job-b-1"))
        history = await runtime_a.execution.history(first.execution_id, principal=principal)

        assert dependencies.local is local
        assert runtime_a.binding.manifest.digest == runtime_a_again.binding.manifest.digest
        assert runtime_a.binding.manifest.digest != runtime_b.binding.manifest.digest
        assert first.status.value == second.status.value == third.status.value == "SUCCEEDED"
        assert _Answer.model_validate(first.output).value == "ok"
        assert _Answer.model_validate(second.output).value == "ok"
        assert _Answer.model_validate(third.output).value == "ok"
        assert history.items

        injected = CapabilityInjection("temporary", canonical_sha256("temporary-v1"), lambda _context: None)
        injected_runtime = build_runtime(spec_a, prompt_a, dependencies=dependencies, capability_injections=(injected,))
        injected_again = build_runtime(spec_a, prompt_a, dependencies=dependencies, capability_injections=(injected,))
        changed_injection = build_runtime(
            spec_a,
            prompt_a,
            dependencies=dependencies,
            capability_injections=(CapabilityInjection("temporary", canonical_sha256("temporary-v2"), lambda _context: None),),
        )
        assert injected_runtime.binding.manifest.digest == injected_again.binding.manifest.digest
        assert injected_runtime.binding.manifest.digest != runtime_a.binding.manifest.digest
        assert changed_injection.binding.manifest.digest != injected_runtime.binding.manifest.digest

        custom_spec = AgentSpec("custom-agent", 1, "default", (AgentCapabilityRef("custom", "tool"),), "answer", 1)
        with pytest.raises(AIError) as error:
            build_runtime(custom_spec, prompt_a, dependencies=dependencies)
        assert error.value.code is ErrorCode.CAPABILITY_PROVIDER_UNKNOWN
        custom_runtime = build_runtime(
            custom_spec,
            prompt_a,
            dependencies=dependencies,
            additional_capability_resolvers=(_CapabilityResolver(),),
        )
        custom_with_injection = build_runtime(
            custom_spec,
            prompt_a,
            dependencies=dependencies,
            additional_capability_resolvers=(_CapabilityResolver(),),
            capability_injections=(injected,),
        )
        no_ref_runtime = build_runtime(
            spec_a,
            prompt_a,
            dependencies=dependencies,
            additional_capability_resolvers=(_CapabilityResolver(),),
        )
        assert custom_runtime.binding.capability_bindings[0].provider == "custom"
        assert custom_with_injection.binding.capability_bindings[0].provider == "custom"
        assert no_ref_runtime.binding.capability_bindings == ()
        with pytest.raises(AIError) as error:
            build_runtime(
                custom_spec,
                prompt_a,
                dependencies=replace(dependencies, capability_resolvers=(_CapabilityResolver(),)),
                additional_capability_resolvers=(_CapabilityResolver(),),
            )
        assert error.value.code is ErrorCode.CAPABILITY_CONFLICT


def test_capability_registry_keeps_duplicate_providers_rejected() -> None:
    with pytest.raises(AIError) as error:
        CapabilityResolverRegistry((_CapabilityResolver(), _CapabilityResolver()))
    assert error.value.code is ErrorCode.CAPABILITY_CONFLICT


def test_binding_registry_reuses_the_same_manifest() -> None:
    registry = AgentBindingRegistry()
    binder = _binder()
    binding = binder.bind(AgentSpec("agent", 1, "route", (), "answer", 1), PromptSpec("prompt", 1, "system", ()))
    registry.register(binding)
    registry.register(binding)
    assert registry.resolve(binding.manifest.digest) is binding
