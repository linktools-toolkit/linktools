#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conformance tests for bound runtime attachment and local composition."""

import inspect
import json
import sys
from pathlib import Path

import pytest
from linktools.ai.agent import AgentBinder, OutputTypeRegistry
from linktools.ai.app import (
    RuntimePersistenceConfig,
    build_local_runtime_services,
    build_runtime,
    open_runtime_resources,
)
from linktools.ai.core import (
    AuthorizationAction,
    Principal,
    PrincipalKind,
    ResourceKind,
    ResourceRef,
    TenantAuthorizationPolicy,
    service_principal,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import (
    ModelConnectionRegistry,
    ModelMaterializer,
    ModelRegistry,
    ModelRoute,
    SnapshotModelResolver,
)
from linktools.ai.runtime import (
    AllowAllToolPolicy,
    ExecutionRequest,
    ToolAuthorization,
    ToolDescriptor,
)
from linktools.ai.spec import AgentSpec, PromptSpec, PromptSpecCodec
from linktools.ai.workspace import DisabledSandbox
from pydantic import BaseModel
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel


class _Answer(BaseModel):
    value: str


class _Materializer(ModelMaterializer):
    def materialize(self, route: ModelRoute, connection: "object | None") -> Model:
        del route, connection
        return TestModel(call_tools=[], custom_output_args={"value": "ok"})


def _binder() -> AgentBinder:
    registry = ModelRegistry()
    snapshot = registry.prime({"route": ModelRoute("route", "test", "test")})
    output_types = OutputTypeRegistry()
    output_types.register("answer", 1, _Answer)
    return AgentBinder(
        model_resolver=SnapshotModelResolver(snapshot),
        model_connections=ModelConnectionRegistry(),
        output_types=output_types,
        execution_profile_fingerprint="0" * 64,
    )


@pytest.mark.asyncio
async def test_disabled_sandbox_is_fail_closed_without_side_effects(tmp_path: Path) -> None:
    sandbox = DisabledSandbox()
    target = tmp_path / "should-not-exist"
    command = f'{sys.executable} -c "from pathlib import Path; Path({str(target)!r}).write_text(\'created\')"'
    for operation in (sandbox.read_file(str(target)), sandbox.write_file(str(target), "content"), sandbox.run(command)):
        with pytest.raises(AIError) as error:
            await operation
        assert error.value.code is ErrorCode.SANDBOX_UNAVAILABLE
    assert not target.exists()


@pytest.mark.asyncio
async def test_allow_all_policy_is_explicit_and_deterministic() -> None:
    policy = AllowAllToolPolicy()
    principal = service_principal("tenant-a", "worker-a")
    other_principal = Principal("user-a", "tenant-a")
    execution = ResourceRef(ResourceKind.EXECUTION, "execution-a", "tenant-a")
    tool = ToolDescriptor("temporary-tool", replay_safe=True)
    for current_principal, digest in ((principal, "digest-a"), (other_principal, "digest-b")):
        assert await policy.authorize_tool(current_principal, execution, tool, digest) is ToolAuthorization.ALLOW
    assert policy.fingerprint == AllowAllToolPolicy().fingerprint
    assert policy.fingerprint != DisabledSandbox().fingerprint


@pytest.mark.asyncio
async def test_service_principal_keeps_tenant_authorization() -> None:
    principal = service_principal("tenant-a", "worker-a")
    assert principal.kind == PrincipalKind.SERVICE.value
    assert principal.tenant_id == "tenant-a"
    owned = ResourceRef(ResourceKind.EXECUTION, "execution-a", "tenant-a", owner_principal_id="worker-a")
    await TenantAuthorizationPolicy().authorize(principal, AuthorizationAction.EXECUTION_RUN, owned)
    with pytest.raises(AIError) as error:
        await TenantAuthorizationPolicy().authorize(principal, AuthorizationAction.EXECUTION_RUN, ResourceRef(ResourceKind.EXECUTION, "execution-b", "tenant-b"))
    assert error.value.code is ErrorCode.AUTHORIZATION_DENIED


def test_prompt_spec_variables_are_removed_and_legacy_payloads_fail() -> None:
    prompt = PromptSpec("prompt", 1, "system", ("instruction",))
    codec = PromptSpecCodec()
    assert "variables" not in json.loads(codec.encode(prompt))
    with pytest.raises(TypeError):
        PromptSpec("prompt", 1, "system", (), variables=("name",))


@pytest.mark.asyncio
async def test_bound_runtime_reuses_registered_binding_and_runs(tmp_path: Path) -> None:
    authorization = TenantAuthorizationPolicy()
    principal = service_principal("tenant-a", "worker-a")
    spec = AgentSpec("pipeline-agent", 1, "route", (), "answer", 1)
    prompt = PromptSpec("pipeline-prompt", 1, "system", ())
    binding = _binder().bind(spec, prompt)
    async with open_runtime_resources(RuntimePersistenceConfig.in_memory(namespace="bound-runtime")) as resources:
        local = build_local_runtime_services(
            resources,
            authorization,
            grant_key=b"bound-runtime-key",
            materializer=_Materializer(),
            execution_root=tmp_path,
        )
        runtime = build_runtime(binding, local=local)
        same_runtime = build_runtime(binding, local=local)
        first = await runtime.run(ExecutionRequest("first", principal, idempotency_key="bound-job-1"))
        second = await same_runtime.run(ExecutionRequest("second", principal, idempotency_key="bound-job-2"))
    assert runtime.binding.manifest.digest == same_runtime.binding.manifest.digest
    assert first.output == second.output == {"value": "ok"}


def test_runtime_resource_entrypoint_accepts_only_the_external_session_factory() -> None:
    from linktools.ai.app import open_runtime_resources

    parameters = inspect.signature(open_runtime_resources).parameters
    assert "session_factory" in parameters
    assert "engine" not in parameters
