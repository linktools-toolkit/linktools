#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evidence for the closed local runtime composition boundary."""

from pathlib import Path

import pytest
from linktools.ai.agent import AgentBinder, OutputTypeRegistry
from linktools.ai.app import (
    RuntimePersistenceConfig,
    build_local_runtime_services,
    build_runtime,
    open_runtime_resources,
)
from linktools.ai.core import TenantAuthorizationPolicy, canonical_sha256
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
from linktools.ai.spec import AgentSpec, PromptSpec
from pydantic import BaseModel


class _Output(BaseModel):
    value: str


def _binder(connection: "ModelConnectionConfig | None" = None) -> AgentBinder:
    routes = ModelRegistry()
    route = ModelRoute("route", "openai", "openai:gpt-test", None if connection is None else connection.connection_id)
    snapshot = routes.prime({"route": route})
    output_types = OutputTypeRegistry()
    output_types.register("output", 1, _Output)
    return AgentBinder(
        model_resolver=SnapshotModelResolver(snapshot),
        model_connections=ModelConnectionRegistry(() if connection is None else (connection,)),
        output_types=output_types,
        execution_profile_fingerprint=canonical_sha256("review-fix-profile"),
    )


@pytest.mark.asyncio
async def test_local_runtime_uses_factory_bundle_and_explicit_execution_root(tmp_path: Path) -> None:
    binding = _binder().bind(AgentSpec("parent", 1, "route", (), "output", 1), PromptSpec("prompt", 1, "system", ()))
    async with open_runtime_resources(RuntimePersistenceConfig.in_memory(namespace="review-fix-local")) as resources:
        local = build_local_runtime_services(
            resources,
            TenantAuthorizationPolicy(),
            grant_key=b"review-fix-key",
            materializer=OpenAIModelMaterializer(StaticModelCredentialProvider({})),
            execution_root=tmp_path,
        )
        runtime = build_runtime(binding, local=local)
    assert runtime.binding is binding


def test_local_runtime_rejects_missing_execution_root(tmp_path: Path) -> None:
    async def run() -> None:
        async with open_runtime_resources(RuntimePersistenceConfig.in_memory(namespace="review-fix-root")) as resources:
            with pytest.raises(AIError) as error:
                build_local_runtime_services(
                    resources,
                    TenantAuthorizationPolicy(),
                    grant_key=b"review-fix-key",
                    materializer=OpenAIModelMaterializer(StaticModelCredentialProvider({})),
                    execution_root=tmp_path / "missing",
                )
            assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID

    import asyncio

    asyncio.run(run())


def test_model_connection_identity_excludes_secret_and_tracks_configuration() -> None:
    base = ModelConnectionConfig("connection", "https://example.test/v1", 10.0, "credential-a")
    digests = []
    for secret in ("secret-a", "secret-b"):
        model = OpenAIModelMaterializer(StaticModelCredentialProvider({"credential-a": secret})).materialize(
            ModelRoute("route", "openai", "openai:gpt-test"),
            base,
        )
        assert secret not in repr(model)
        digests.append(_binder(base).bind(AgentSpec("agent", 1, "route", (), "output", 1), PromptSpec("prompt", 1, "system", ())).manifest.digest)
    assert digests[0] == digests[1]
    assert _binder(base).bind(AgentSpec("agent", 1, "route", (), "output", 1), PromptSpec("prompt", 1, "system", ())).manifest.digest != _binder(ModelConnectionConfig("connection", "https://other.test/v1", 10.0, "credential-a")).bind(AgentSpec("agent", 1, "route", (), "output", 1), PromptSpec("prompt", 1, "system", ())).manifest.digest
    assert "secret-a" not in repr(base)
