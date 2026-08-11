#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conformance tests for Asset-backed declaration composition."""

import asyncio
import importlib
from dataclasses import dataclass

import pytest
from linktools.ai.agent import AgentBinder, AgentBindingRegistry, OutputTypeRegistry
from linktools.ai.app import (
    BOUND_RUNTIME_PROFILE_FINGERPRINT,
    build_agent_binding_composer,
    build_asset_repository,
)
from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.capability import (
    CapabilityRefResolution,
    CapabilityRuntimeContext,
    MCPServerCapabilityBinding,
    SkillCapability,
    SkillCapabilityBinding,
    SkillCatalogSnapshot,
    bind_mcp_capability,
    bind_skill_capability,
    unresolved_binding,
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
from linktools.ai.storage import StorageOverlay
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


def _binder() -> AgentBinder:
    routes = ModelRegistry()
    snapshot = routes.prime({"route": ModelRoute("route", "test", "test")})
    output_types = OutputTypeRegistry()
    output_types.register("output", 1, _Output)
    return AgentBinder(
        model_resolver=SnapshotModelResolver(snapshot),
        model_connections=ModelConnectionRegistry(),
        output_types=output_types,
        execution_profile_fingerprint=BOUND_RUNTIME_PROFILE_FINGERPRINT,
    )


async def _asset_store() -> AssetStore:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    return store


def test_agent_snapshot_and_codec_inputs_are_deeply_immutable() -> None:
    metadata = {"depends_on": ["base-agent"], "token_budget": 4096}
    spec = AgentSpec(
        "agent",
        1,
        "route",
        (AgentCapabilityRef("custom", "one", config={"nested": {"values": [1]}}),),
        "output",
        1,
        metadata=metadata,
    )
    assert spec.capabilities[0].config["nested"] == {"values": [1]}
    assert spec.capabilities[0].config["nested"] is not spec.capabilities[0].config["nested"]
    binding = _binder().bind(spec, PromptSpec("prompt", 1, "system", ()), capabilities=(
        _Binding(
            "custom",
            (CapabilityRefResolution("one", None, 1, True, "resolved", canonical_sha256("one")),),
            canonical_sha256("custom"),
        ),
    ))
    digest = binding.manifest.digest
    metadata["depends_on"].append("other-agent")
    assert binding.manifest.digest == digest
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


@pytest.mark.parametrize(
    "prepared",
    [
        (),
        (_Binding("other", (), canonical_sha256("other")),),
    ],
)
def test_binder_requires_exact_prepared_capability_groups(prepared: tuple[_Binding, ...]) -> None:
    spec = AgentSpec("agent", 1, "route", (AgentCapabilityRef("custom", "one"),), "output", 1)
    with pytest.raises(AIError) as error:
        _binder().bind(spec, PromptSpec("prompt", 1, "system", ()), capabilities=prepared)
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


def test_optional_unknown_provider_is_prepared_by_composer_contract() -> None:
    refs = (AgentCapabilityRef("unknown", "x", required=False),)
    binding = unresolved_binding("unknown", refs)
    assert binding.resolutions[0].status == "unresolved"
    agent = AgentSpec("agent", 1, "route", refs, "output", 1)
    compiled = _binder().bind(agent, PromptSpec("prompt", 1, "system", ()), capabilities=(binding,))
    assert compiled.capability_bindings[0].fingerprint == binding.fingerprint


def test_declaration_codecs_round_trip_each_owned_spec() -> None:
    agent = AgentSpec("agent", 1, "route", (AgentCapabilityRef("custom", "one"),), "output", 1)
    assert AgentSpecCodec().decode(AgentSpecCodec().encode(agent)) == agent
    prompt = PromptSpec("prompt", 2, "system", ("instruction",))
    assert PromptSpecCodec().decode(PromptSpecCodec().encode(prompt)) == prompt
    skill = SkillSpec("skill", 3, "content")
    assert SkillSpecCodec().decode(SkillSpecCodec().encode(skill)) == skill
    server = MCPServerSpec("server", 4, "command", ("--flag",))
    assert MCPServerSpecCodec().decode(MCPServerSpecCodec().encode(server)) == server


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


def test_local_runtime_services_is_not_a_public_constructor() -> None:
    app = importlib.import_module("linktools.ai.app")
    assert "LocalRuntimeServices" not in app.__all__
    assert not hasattr(app, "LocalRuntimeServices")


def test_skill_binding_owns_an_immutable_catalog_snapshot() -> None:
    async def run() -> None:
        binding = bind_skill_capability(
            (AgentCapabilityRef("skill", "python"),),
            (SkillSpec("python", 2, "instructions"),),
        )
        assert isinstance(binding, SkillCapabilityBinding)
        assert isinstance(binding.catalog, SkillCatalogSnapshot)
        values = await binding.materialize(CapabilityRuntimeContext(Principal("p", "t"), ResourceRef(ResourceKind.EXECUTION, "e", "t")))
        assert len(values) == 1 and isinstance(values[0], SkillCapability)

    asyncio.run(run())


def test_mcp_binding_defers_execution_to_runtime_provider() -> None:
    class Runtime:
        fingerprint = canonical_sha256("mcp-runtime")
        calls = 0

        async def connect(self, server_id: str) -> object:
            del server_id
            raise AssertionError("connect belongs to runtime materialization")

        async def toolsets(self, servers, *, principal, execution):
            del principal, execution
            self.calls += 1
            return (FunctionToolset(id=servers[0].id),)

    async def run() -> None:
        runtime = Runtime()
        binding = bind_mcp_capability(
            (AgentCapabilityRef("mcp", "server"),),
            (MCPServerSpec("server", 1, "command"),),
            runtime,
        )
        assert isinstance(binding, MCPServerCapabilityBinding)
        assert runtime.calls == 0
        await binding.materialize(CapabilityRuntimeContext(Principal("p", "t"), ResourceRef(ResourceKind.EXECUTION, "e", "t")))
        assert runtime.calls == 1

    asyncio.run(run())


@pytest.mark.asyncio
async def test_composer_loads_agent_prompt_and_skill_from_asset_repository() -> None:
    store = await _asset_store()
    await store.put(AssetKey("agent", "coding"), AgentSpecCodec().encode(AgentSpec("coding", 1, "route", (AgentCapabilityRef("skill", "python"),), "output", 1)))
    await store.put(AssetKey("prompt", "review"), PromptSpecCodec().encode(PromptSpec("review", 1, "system", ("prompt",))))
    await store.put(AssetKey("skill", "python"), SkillSpecCodec().encode(SkillSpec("python", 2, "skill body")))
    repository = build_asset_repository(store)
    output_types = OutputTypeRegistry()
    output_types.register("output", 1, _Output)
    output_types.freeze()
    routes = ModelRegistry()
    snapshot = routes.prime({"route": ModelRoute("route", "test", "test")})
    composer = build_agent_binding_composer(
        repository,
        model_resolver=SnapshotModelResolver(snapshot),
        model_connections=ModelConnectionRegistry(),
        output_types=output_types,
        execution_profile_fingerprint=BOUND_RUNTIME_PROFILE_FINGERPRINT,
    )
    binding = await composer.compose(agent_id="coding", prompt_id="review")
    assert binding.spec.id == "coding"
    assert binding.prompt.id == "review"
    assert isinstance(binding.capability_bindings[0].catalog, SkillCatalogSnapshot)


@pytest.mark.asyncio
async def test_composer_preserves_required_and_optional_missing_capability_contracts() -> None:
    store = await _asset_store()
    await store.put(AssetKey("agent", "optional"), AgentSpecCodec().encode(AgentSpec("optional", 1, "route", (AgentCapabilityRef("skill", "missing", required=False),), "output", 1)))
    await store.put(AssetKey("prompt", "default"), PromptSpecCodec().encode(PromptSpec("default", 1, "system", ())))
    output_types = OutputTypeRegistry()
    output_types.register("output", 1, _Output)
    output_types.freeze()
    routes = ModelRegistry()
    snapshot = routes.prime({"route": ModelRoute("route", "test", "test")})
    composer = build_agent_binding_composer(
        build_asset_repository(store),
        model_resolver=SnapshotModelResolver(snapshot),
        model_connections=ModelConnectionRegistry(),
        output_types=output_types,
        execution_profile_fingerprint=BOUND_RUNTIME_PROFILE_FINGERPRINT,
    )
    optional = await composer.compose(agent_id="optional", prompt_id="default")
    assert optional.capability_bindings[0].resolutions[0].status == "unresolved"
    await store.put(AssetKey("agent", "required"), AgentSpecCodec().encode(AgentSpec("required", 1, "route", (AgentCapabilityRef("skill", "missing"),), "output", 1)))
    with pytest.raises(AIError) as error:
        await composer.compose(agent_id="required", prompt_id="default")
    assert error.value.code is ErrorCode.CAPABILITY_REQUIRED_MISSING


@pytest.mark.asyncio
async def test_downstream_preparer_is_the_only_extension_point() -> None:
    class FooPreparer:
        provider = "foo"

        async def prepare(self, refs):
            return _Binding(
                "foo",
                tuple(CapabilityRefResolution(ref.id, ref.revision, 1, ref.required, "resolved", canonical_sha256(ref.id)) for ref in refs),
                canonical_sha256("foo-binding"),
            )

    store = await _asset_store()
    await store.put(AssetKey("agent", "foo-agent"), AgentSpecCodec().encode(AgentSpec("foo-agent", 1, "route", (AgentCapabilityRef("foo", "tool"),), "output", 1)))
    await store.put(AssetKey("prompt", "default"), PromptSpecCodec().encode(PromptSpec("default", 1, "system", ())))
    output_types = OutputTypeRegistry()
    output_types.register("output", 1, _Output)
    output_types.freeze()
    routes = ModelRegistry()
    snapshot = routes.prime({"route": ModelRoute("route", "test", "test")})
    composer = build_agent_binding_composer(
        build_asset_repository(store),
        model_resolver=SnapshotModelResolver(snapshot),
        model_connections=ModelConnectionRegistry(),
        output_types=output_types,
        execution_profile_fingerprint=BOUND_RUNTIME_PROFILE_FINGERPRINT,
        extra_preparers=(FooPreparer(),),
    )
    binding = await composer.compose(agent_id="foo-agent", prompt_id="default")
    assert binding.capability_bindings[0].provider == "foo"


@pytest.mark.asyncio
async def test_existing_skill_binding_is_stable_after_asset_update() -> None:
    store = await _asset_store()
    await store.put(AssetKey("agent", "stable"), AgentSpecCodec().encode(AgentSpec("stable", 1, "route", (AgentCapabilityRef("skill", "python"),), "output", 1)))
    await store.put(AssetKey("prompt", "default"), PromptSpecCodec().encode(PromptSpec("default", 1, "system", ())))
    await store.put(AssetKey("skill", "python"), SkillSpecCodec().encode(SkillSpec("python", 1, "old")))
    output_types = OutputTypeRegistry()
    output_types.register("output", 1, _Output)
    output_types.freeze()
    routes = ModelRegistry()
    snapshot = routes.prime({"route": ModelRoute("route", "test", "test")})
    composer = build_agent_binding_composer(
        build_asset_repository(store),
        model_resolver=SnapshotModelResolver(snapshot),
        model_connections=ModelConnectionRegistry(),
        output_types=output_types,
        execution_profile_fingerprint=BOUND_RUNTIME_PROFILE_FINGERPRINT,
    )
    first = await composer.compose(agent_id="stable", prompt_id="default")
    await store.put(AssetKey("skill", "python"), SkillSpecCodec().encode(SkillSpec("python", 2, "new")))
    second = await composer.compose(agent_id="stable", prompt_id="default")
    first_skill = first.capability_bindings[0].catalog.specifications[0]
    second_skill = second.capability_bindings[0].catalog.specifications[0]
    assert first_skill.content == "old"
    assert second_skill.content == "new"
    assert first.manifest.digest != second.manifest.digest


@pytest.mark.asyncio
async def test_optional_skill_layout_conflict_is_not_downgraded() -> None:
    store = await _asset_store()
    await store.put(AssetKey("agent", "conflict"), AgentSpecCodec().encode(AgentSpec("conflict", 1, "route", (AgentCapabilityRef("skill", "python", required=False),), "output", 1)))
    await store.put(AssetKey("prompt", "default"), PromptSpecCodec().encode(PromptSpec("default", 1, "system", ())))
    await store.put(AssetKey("skill", "python"), SkillSpecCodec().encode(SkillSpec("python", 1, "old")))
    await store.put(AssetKey("skill", "python/SKILL.md"), b"---\nname: python\ndescription: Python\n---\nnew\n")
    output_types = OutputTypeRegistry()
    output_types.register("output", 1, _Output)
    output_types.freeze()
    routes = ModelRegistry()
    snapshot = routes.prime({"route": ModelRoute("route", "test", "test")})
    composer = build_agent_binding_composer(
        build_asset_repository(store),
        model_resolver=SnapshotModelResolver(snapshot),
        model_connections=ModelConnectionRegistry(),
        output_types=output_types,
        execution_profile_fingerprint=BOUND_RUNTIME_PROFILE_FINGERPRINT,
    )
    with pytest.raises(AIError) as error:
        await composer.compose(agent_id="conflict", prompt_id="default")
    assert error.value.code is ErrorCode.ASSET_LAYOUT_CONFLICT


def test_binding_registry_reuses_the_same_manifest() -> None:
    registry = AgentBindingRegistry()
    binding = _binder().bind(AgentSpec("agent", 1, "route", (), "output", 1), PromptSpec("prompt", 1, "system", ()))
    registry.register(binding)
    registry.register(binding)
    assert registry.resolve(binding.manifest.digest) is binding
