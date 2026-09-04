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
from linktools.ai.runtime._tool import (
    AllowAllToolPolicy,
    ToolAuthorization,
    ToolDescriptor,
)
from linktools.ai.spec import AgentSpec, AgentSpecCodec
from linktools.ai.workspace import DisabledSandbox


@pytest.mark.asyncio
async def test_disabled_sandbox_is_fail_closed() -> None:
    with pytest.raises(AIError) as error:
        await DisabledSandbox().open()
    assert error.value.code is ErrorCode.SANDBOX_UNAVAILABLE


@pytest.mark.asyncio
async def test_allow_all_policy_is_explicit_and_deterministic() -> None:
    policy = AllowAllToolPolicy()
    principal = service_principal("tenant-a", "worker-a")
    execution = ResourceRef(
        ResourceKind.EXECUTION,
        "execution-a",
        "tenant-a",
    )
    assert (
        await policy.authorize_tool(
            principal,
            execution,
            ToolDescriptor("tool", replay_safe=True),
            "digest",
        )
        is ToolAuthorization.ALLOW
    )
    assert policy.fingerprint == AllowAllToolPolicy().fingerprint


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
