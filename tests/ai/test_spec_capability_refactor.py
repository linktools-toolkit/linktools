#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conformance tests for declaration, capability and binding ownership."""

import asyncio
from dataclasses import dataclass

import pytest
from linktools.ai.agent import AgentBinder, AgentBindingRegistry, OutputTypeRegistry
from linktools.ai.app import LocalRuntimeServices, RuntimeDependencies, build_runtime
from linktools.ai.capability import (
    CapabilityInjection,
    CapabilityRefResolution,
    CapabilityResolverRegistry,
    CapabilityRuntimeContext,
    MCPServerCapabilityResolver,
    SkillCapability,
    SkillCapabilityResolver,
)
from linktools.ai.core import Principal, ResourceKind, ResourceRef, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import (
    ModelConnectionConfig,
    ModelConnectionRegistry,
    ModelRegistry,
    ModelRoute,
    OpenAIModelMaterializer,
    SnapshotModelResolver,
    StaticModelCredentialProvider,
)
from linktools.ai.runtime import RuntimeBackend, RuntimeServiceIdentity, RuntimeServices
from linktools.ai.spec import (
    AgentCapabilityRef,
    AgentSpec,
    AgentSpecCodec,
    MCPServerSpec,
    MCPServerSpecCodec,
    PromptSpec,
    PromptSpecCodec,
    SkillSpec,
    SkillSpecCodec,
)
from pydantic import BaseModel
from pydantic_ai.toolsets import FunctionToolset


class _Output(BaseModel):
    value: str


@dataclass(frozen=True, slots=True)
class _Binding:
    provider: str
    resolutions: "tuple[CapabilityRefResolution, ...]"
    fingerprint: str
    inherit_to_subagents: bool = True

    async def materialize(self, context: CapabilityRuntimeContext) -> tuple[()]:
        del context
        return ()


class _Resolver:
    provider = "custom"
    fingerprint = canonical_sha256("custom-resolver")

    def resolve(self, refs: "tuple[AgentCapabilityRef, ...]") -> _Binding:
        return _Binding(
            self.provider,
            tuple(CapabilityRefResolution(ref.id, ref.revision, 1, ref.required, "resolved", canonical_sha256(ref.id)) for ref in refs),
            canonical_sha256([ref.id for ref in refs]),
        )


def _binder(*, connections: tuple[ModelConnectionConfig, ...] = ()) -> AgentBinder:
    routes = ModelRegistry()
    snapshot = routes.prime({"route": ModelRoute("route", "test", "test")})
    output_types = OutputTypeRegistry()
    output_types.register("output", 1, _Output)
    return AgentBinder(
        model_resolver=SnapshotModelResolver(snapshot),
        model_connections=ModelConnectionRegistry(connections),
        output_types=output_types,
        capability_resolvers=CapabilityResolverRegistry((_Resolver(),)),
        execution_profile_fingerprint=canonical_sha256("execution-profile"),
    )


def test_agent_snapshot_and_codec_inputs_are_deeply_immutable() -> None:
    metadata = {
        "depends_on": ["base-agent"],
        "token_budget": 4096,
        "credential_status": "verified",
        "password_policy": "nist",
        "api_tokenization_mode": "strict",
    }
    spec = AgentSpec(
        "agent",
        1,
        "route",
        (AgentCapabilityRef("custom", "one", config={"nested": {"values": [1]}}),),
        "output",
        1,
        metadata=metadata,
    )
    nested = spec.capabilities[0].config["nested"]
    assert nested == {"values": [1]}
    assert spec.capabilities[0].config["nested"] is not nested
    assert dict(spec.metadata) == metadata
    binding = _binder().bind(spec, PromptSpec("prompt", 1, "system", (), ()))
    binding_digest = binding.manifest.digest
    metadata["depends_on"].append("other-agent")
    assert spec.metadata["depends_on"] == ["base-agent"]
    assert binding.manifest.digest == binding_digest
    with pytest.raises(AIError) as error:
        AgentSpec(
            "agent",
            1,
            "route",
            (AgentCapabilityRef("custom", "one"), AgentCapabilityRef("custom", "one", required=False)),
            "output",
            1,
        )
    assert error.value.code is ErrorCode.CAPABILITY_CONFLICT


def test_binder_groups_resolvers_and_registry_uses_only_manifest_identity() -> None:
    binder = _binder()
    prompt = PromptSpec("prompt", 1, "system", (), ())
    spec = AgentSpec(
        "agent",
        1,
        "route",
        (AgentCapabilityRef("custom", "a"), AgentCapabilityRef("custom", "b")),
        "output",
        1,
    )
    binding = binder.bind(spec, prompt)
    registry = AgentBindingRegistry()
    registry.register(binding)
    assert registry.resolve(binding.manifest.digest) is binding
    changed = binder.bind(AgentSpec("agent", 1, "route", spec.capabilities, "output", 1, metadata={"label": "changed"}), prompt)
    assert changed.manifest.digest != binding.manifest.digest
    registry.register(changed)


def test_unknown_provider_and_injection_contracts_are_explicit() -> None:
    binder = _binder()
    prompt = PromptSpec("prompt", 1, "system", (), ())
    with pytest.raises(AIError) as error:
        binder.bind(AgentSpec("agent", 1, "route", (AgentCapabilityRef("unknown", "x"),), "output", 1), prompt)
    assert error.value.code is ErrorCode.CAPABILITY_PROVIDER_UNKNOWN
    optional = binder.bind(AgentSpec("agent", 1, "route", (AgentCapabilityRef("unknown", "x", required=False),), "output", 1), prompt)
    assert optional.capability_bindings[0].resolutions[0].status == "unresolved"
    resolution = optional.capability_bindings[0].resolutions[0]
    assert optional.manifest.capabilities_fingerprint == canonical_sha256(
        {
            "declarative": [
                {
                    "provider": "unknown",
                    "resolver_fingerprint": None,
                    "binding_fingerprint": optional.capability_bindings[0].fingerprint,
                    "inherit_to_subagents": False,
                    "resolutions": [
                        {
                            "id": resolution.id,
                            "requested_revision": resolution.requested_revision,
                            "resolved_revision": resolution.resolved_revision,
                            "required": resolution.required,
                            "status": resolution.status,
                            "fingerprint": resolution.fingerprint,
                        }
                    ],
                }
            ],
            "injections": [],
        }
    )
    injection = CapabilityInjection("direct", canonical_sha256("injection"), lambda _context: None)
    with pytest.raises(AIError) as conflict:
        binder.bind(AgentSpec("agent", 1, "route", (), "output", 1), prompt, injections=(injection, injection))
    assert conflict.value.code is ErrorCode.CAPABILITY_CONFLICT


def test_declaration_codecs_round_trip_each_owned_spec() -> None:
    agent = AgentSpec(
        "agent",
        1,
        "route",
        (AgentCapabilityRef("custom", "one", config={"nested": [1, "two"]}),),
        "output",
        1,
        ("instruction",),
        {"labels": ["a", "b"]},
    )
    decoded_agent = AgentSpecCodec().decode(AgentSpecCodec().encode(agent))
    assert decoded_agent.id == agent.id
    assert dict(decoded_agent.capabilities[0].config) == dict(agent.capabilities[0].config)
    assert dict(decoded_agent.metadata) == dict(agent.metadata)

    prompt = PromptSpec("prompt", 2, "system", ("instruction",), ("name",))
    decoded_prompt = PromptSpecCodec().decode(PromptSpecCodec().encode(prompt))
    assert decoded_prompt == prompt

    skill = SkillSpec("skill", 3, "content")
    decoded_skill = SkillSpecCodec().decode(SkillSpecCodec().encode(skill))
    assert decoded_skill == skill

    server = MCPServerSpec("server", 4, "command", ("--flag",))
    decoded_server = MCPServerSpecCodec().decode(MCPServerSpecCodec().encode(server))
    assert decoded_server == server


def test_model_connection_is_stable_and_secret_free() -> None:
    connection = ModelConnectionConfig("primary", "https://example.test/v1/", 30.0, "credential")
    assert connection.base_url == "https://example.test/v1"
    assert len(connection.fingerprint) == 64
    with pytest.raises(ValueError):
        ModelConnectionConfig("unsafe", "https://user:password@example.test")
    with pytest.raises(AIError) as error:
        ModelConnectionRegistry((connection, ModelConnectionConfig("primary", "https://other.test")))
    assert error.value.code is ErrorCode.MODEL_CONNECTION_CONFLICT


def test_openai_model_materializer_applies_connection_configuration() -> None:
    materializer = OpenAIModelMaterializer(StaticModelCredentialProvider({"credential": "secret"}))
    route = ModelRoute("primary", "openai", "openai:gpt-test")
    connection = ModelConnectionConfig("primary-connection", "https://example.test/v1", 12.5, "credential")
    model = materializer.materialize(route, connection)
    assert model.model_name == "gpt-test"
    assert str(model.provider.client.base_url) == "https://example.test/v1/"
    assert model.settings["timeout"] == 12.5

    with pytest.raises(AIError) as unsupported:
        materializer.materialize(ModelRoute("custom", "custom", "custom-model"), None)
    assert unsupported.value.code is ErrorCode.MODEL_CONNECTION_UNSUPPORTED

    missing = OpenAIModelMaterializer(StaticModelCredentialProvider())
    with pytest.raises(AIError) as unavailable:
        missing.materialize(route, ModelConnectionConfig("missing", credential_id="unknown"))
    assert unavailable.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
    assert unavailable.value.safe_details == {"credential_id": "unknown"}


def test_runtime_dependencies_keep_one_local_service_and_binding_registry() -> None:
    routes = ModelRegistry()
    snapshot = routes.prime({"route": ModelRoute("route", "test", "test")})
    output_types = OutputTypeRegistry()
    output_types.register("output", 1, _Output)
    identity = RuntimeServiceIdentity("service", canonical_sha256("persistence"), RuntimeBackend.IN_MEMORY)
    services = RuntimeServices(identity, object(), object(), object(), object(), object(), object(), object())
    local = LocalRuntimeServices(services, AgentBindingRegistry())
    dependencies = RuntimeDependencies(
        model_resolver=SnapshotModelResolver(snapshot),
        capability_resolvers=(_Resolver(),),
        model_connections=ModelConnectionRegistry(),
        middleware=type("Middleware", (), {"fingerprint": "middleware"})(),
        sandbox=type("Sandbox", (), {"fingerprint": "sandbox"})(),
        tool_policy=type("ToolPolicy", (), {"fingerprint": "tool-policy"})(),
        output_types=output_types,
        local=local,
    )
    runtime = build_runtime(
        AgentSpec("agent", 1, "route", (AgentCapabilityRef("custom", "capability"),), "output", 1),
        PromptSpec("prompt", 1, "system", (), ()),
        dependencies=dependencies,
    )
    assert runtime.service_identity is services.identity
    assert local.binding_registry.resolve(runtime.binding.manifest.digest) is runtime.binding
    assert not hasattr(dependencies, "services")
    assert not hasattr(dependencies, "binding_registry")
    assert not hasattr(dependencies, "agent_catalog")


def test_skill_aggregates_and_mcp_materializes_only_with_runtime_context() -> None:
    class Skills:
        def manifest(self) -> str:
            return canonical_sha256("skills")

        def resolve_ref(self, skill_id: str, revision: int | None = None) -> SkillSpec:
            return SkillSpec(skill_id, revision or 1, skill_id)

    class Mcp:
        def __init__(self) -> None:
            self.calls = 0

        def manifest(self) -> str:
            return canonical_sha256("mcp")

        def resolve_ref(self, server_id: str, revision: int | None = None) -> MCPServerSpec:
            return MCPServerSpec(server_id, revision or 1, "server")

        async def connect(self, server_id: str) -> object:
            del server_id
            raise AssertionError("connect is not part of binding")

        async def toolsets(self, servers, *, principal, execution):
            del principal, execution
            self.calls += 1
            return (FunctionToolset(id=servers[0].id),)

    async def run() -> None:
        refs = (AgentCapabilityRef("skill", "a"), AgentCapabilityRef("skill", "b"))
        skill_binding = SkillCapabilityResolver(Skills()).resolve(refs)
        values = await skill_binding.materialize(CapabilityRuntimeContext(Principal("p", "t"), ResourceRef(ResourceKind.EXECUTION, "e", "t")))
        assert len(values) == 1 and isinstance(values[0], SkillCapability)
        mcp = Mcp()
        mcp_binding = MCPServerCapabilityResolver(mcp).resolve((AgentCapabilityRef("mcp", "server"),))
        assert mcp.calls == 0
        await mcp_binding.materialize(CapabilityRuntimeContext(Principal("p", "t"), ResourceRef(ResourceKind.EXECUTION, "e", "t")))
        assert mcp.calls == 1

    asyncio.run(run())
