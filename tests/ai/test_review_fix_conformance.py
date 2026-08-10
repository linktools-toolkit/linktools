#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end evidence for the review-fix composition boundaries."""

from dataclasses import dataclass

import pytest
from linktools.ai.agent import AgentBinder, AgentCatalogItem, OutputTypeRegistry
from linktools.ai.app import (
    RuntimeDependencies,
    RuntimePersistenceConfig,
    build_local_runtime_services,
    build_runtime,
    open_runtime_resources,
)
from linktools.ai.capability import CapabilityResolverRegistry
from linktools.ai.core import Principal, TenantAuthorizationPolicy, canonical_sha256
from linktools.ai.model import (
    ModelConnectionConfig,
    ModelConnectionRegistry,
    ModelRegistry,
    ModelRoute,
    OpenAIModelMaterializer,
    SnapshotModelResolver,
    StaticModelCredentialProvider,
)
from linktools.ai.runtime import ExecutionRequest
from linktools.ai.spec import AgentSpec, PromptSpec
from pydantic import BaseModel
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel


class _Output(BaseModel):
    value: str


class _Materializer:
    def materialize(self, route: ModelRoute, connection: "ModelConnectionConfig | None") -> Model:
        del route, connection
        return TestModel(call_tools=[], custom_output_args={"value": "ok"})


class _Catalog:
    def __init__(self) -> None:
        self.calls = 0

    async def list_agents(self) -> "tuple[AgentCatalogItem, ...]":
        self.calls += 1
        return (AgentCatalogItem("child", "child", "Child", "Execute the child task.", None),)


@dataclass(frozen=True, slots=True)
class _Fingerprint:
    fingerprint: str


def _runtime_dependencies(local, *, connection: "ModelConnectionConfig | None" = None) -> RuntimeDependencies:
    routes = ModelRegistry()
    route = ModelRoute("route", "test", "test", None if connection is None else connection.connection_id)
    snapshot = routes.prime({"route": route})
    output_types = OutputTypeRegistry()
    output_types.register("output", 1, _Output)
    return RuntimeDependencies(
        model_resolver=SnapshotModelResolver(snapshot),
        capability_resolvers=(),
        model_connections=ModelConnectionRegistry(() if connection is None else (connection,)),
        middleware=_Fingerprint("middleware"),
        sandbox=_Fingerprint("sandbox"),
        tool_policy=_Fingerprint("tool-policy"),
        output_types=output_types,
        local=local,
    )


@pytest.mark.asyncio
async def test_local_runtime_uses_launcher_registry_and_catalog() -> None:
    catalog = _Catalog()
    async with open_runtime_resources(RuntimePersistenceConfig.in_memory(namespace="review-fix-local")) as resources:
        local = build_local_runtime_services(
            resources,
            TenantAuthorizationPolicy(),
            grant_key=b"review-fix-key",
            materializer=_Materializer(),
            agent_catalog=catalog,
        )
        runtime = build_runtime(
            AgentSpec("parent", 1, "route", (), "output", 1),
            PromptSpec("prompt", 1, "system", ()),
            dependencies=_runtime_dependencies(local),
        )
        result = await runtime.run(
            ExecutionRequest(
                "run without durable memory",
                Principal("owner", "tenant"),
                "review-fix-runtime",
            )
        )

    assert result.output == {"value": "ok"}
    assert catalog.calls == 1


def test_model_connection_identity_excludes_secret_and_tracks_configuration() -> None:
    def bind(connection: ModelConnectionConfig) -> str:
        routes = ModelRegistry()
        snapshot = routes.prime({"route": ModelRoute("route", "openai", "openai:gpt-test", "connection")})
        output_types = OutputTypeRegistry()
        output_types.register("output", 1, _Output)
        binder = AgentBinder(
            model_resolver=SnapshotModelResolver(snapshot),
            model_connections=ModelConnectionRegistry((connection,)),
            output_types=output_types,
            capability_resolvers=CapabilityResolverRegistry(()),
            execution_profile_fingerprint=canonical_sha256("review-fix-profile"),
        )
        return binder.bind(AgentSpec("agent", 1, "route", (), "output", 1), PromptSpec("prompt", 1, "system", ())).manifest.digest

    base = ModelConnectionConfig("connection", "https://example.test/v1", 10.0, "credential-a")
    digests = []
    for secret in ("secret-a", "secret-b"):
        model = OpenAIModelMaterializer(StaticModelCredentialProvider({"credential-a": secret})).materialize(
            ModelRoute("route", "openai", "openai:gpt-test"),
            base,
        )
        assert secret not in repr(model)
        digests.append(bind(base))
    assert digests[0] == digests[1]
    assert bind(base) != bind(ModelConnectionConfig("connection", "https://other.test/v1", 10.0, "credential-a"))
    assert bind(base) != bind(ModelConnectionConfig("connection", "https://example.test/v1", 20.0, "credential-a"))
    assert bind(base) != bind(ModelConnectionConfig("connection", "https://example.test/v1", 10.0, "credential-b"))
    assert "secret-a" not in repr(base)
