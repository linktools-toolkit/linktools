#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Current public workspace and authorization contracts."""

import json
from pathlib import Path

import pytest
from linktools.ai.core import (
    AuthorizationAction,
    PrincipalKind,
    ResourceKind,
    ResourceRef,
    TenantAuthorizationPolicy,
    service_principal,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.spec import AgentSpec, AgentSpecCodec
from linktools.ai.workspace import (
    ToolPermissionRule,
    ToolPermissionPolicy,
)


@pytest.mark.asyncio
async def test_disabled_workspace_factory_is_fail_closed(tmp_path: Path) -> None:
    from linktools.ai.runtime import Runtime, RuntimeFeatures
    from linktools.ai.persistence import PersistenceStorage
    from linktools.ai.capability import FileSystem
    from linktools.ai.workspace import Workspace
    from ._runtime_test_helpers import RuntimeUsageModels, model_features

    features = RuntimeFeatures("disabled", workspace_factory=None)
    features.capability(FileSystem(Workspace.load(tmp_path)))
    (tmp_path / "evidence.txt").write_text("private evidence")
    async with Runtime.open("disabled", storage=PersistenceStorage.in_memory(), features=model_features(RuntimeUsageModels(), features)) as runtime:
        with pytest.raises(AIError) as error:
            await runtime.agents.get().run("inspect", files=("evidence.txt",))
    assert error.value.code is ErrorCode.SANDBOX_UNAVAILABLE


@pytest.mark.asyncio
async def test_workspace_permission_policy_is_explicit_and_deterministic() -> None:
    policy = ToolPermissionPolicy(
        (ToolPermissionRule("deny", tool_class="shell"),),
        default="allow",
    )
    assert policy.decide(tool_name="read_file", tool_class="filesystem.read") == "allow"
    assert policy.decide(tool_name="run_command", tool_class="shell") == "deny"
    assert policy == ToolPermissionPolicy(
        (ToolPermissionRule("deny", tool_class="shell"),),
        default="allow",
    )


@pytest.mark.asyncio
async def test_service_principal_keeps_tenant_authorization() -> None:
    principal = service_principal("tenant-a", "worker-a")
    assert principal.kind == PrincipalKind.SERVICE.value
    owned = ResourceRef(
        ResourceKind.EXECUTION,
        "execution-a",
        "tenant-a",
        owner_principal_id="worker-a",
    )
    await TenantAuthorizationPolicy().authorize(
        principal,
        AuthorizationAction.EXECUTION_RUN,
        owned,
    )
    with pytest.raises(AIError) as error:
        await TenantAuthorizationPolicy().authorize(
            principal,
            AuthorizationAction.EXECUTION_RUN,
            ResourceRef(ResourceKind.EXECUTION, "execution-b", "tenant-b"),
        )
    assert error.value.code is ErrorCode.AUTHORIZATION_DENIED


def test_agent_spec_codec_preserves_system_prompt_and_instructions() -> None:
    spec = AgentSpec(
        "agent",
        model="model",
        system_prompt="system",
        instructions=("instruction",),
    )
    payload = json.loads(AgentSpecCodec().encode(spec))
    assert payload["system_prompt"] == "system"
    assert payload["instructions"] == ["instruction"]
