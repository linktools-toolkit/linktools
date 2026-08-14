#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Current public workspace and authorization contracts."""

import json
import sys
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
from linktools.ai.runtime._tool import AllowAllToolPolicy, ToolAuthorization, ToolDescriptor
from linktools.ai.spec import PromptSpec, PromptSpecCodec
from linktools.ai.workspace import DisabledSandbox
from linktools.ai.workspace._tools import build_workspace_tool_map


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
    execution = ResourceRef(ResourceKind.EXECUTION, "execution-a", "tenant-a")
    assert await policy.authorize_tool(principal, execution, ToolDescriptor("tool", replay_safe=True), "digest") is ToolAuthorization.ALLOW
    assert policy.fingerprint == AllowAllToolPolicy().fingerprint


@pytest.mark.asyncio
async def test_service_principal_keeps_tenant_authorization() -> None:
    principal = service_principal("tenant-a", "worker-a")
    assert principal.kind == PrincipalKind.SERVICE.value
    owned = ResourceRef(ResourceKind.EXECUTION, "execution-a", "tenant-a", owner_principal_id="worker-a")
    await TenantAuthorizationPolicy().authorize(principal, AuthorizationAction.EXECUTION_RUN, owned)
    with pytest.raises(AIError) as error:
        await TenantAuthorizationPolicy().authorize(
            principal,
            AuthorizationAction.EXECUTION_RUN,
            ResourceRef(ResourceKind.EXECUTION, "execution-b", "tenant-b"),
        )
    assert error.value.code is ErrorCode.AUTHORIZATION_DENIED


def test_prompt_spec_codec_rejects_legacy_variables() -> None:
    prompt = PromptSpec("prompt", 1, "system", ("instruction",))
    assert "variables" not in json.loads(PromptSpecCodec().encode(prompt))
    with pytest.raises(TypeError):
        PromptSpec("prompt", 1, "system", (), variables=("name",))


@pytest.mark.asyncio
async def test_workspace_tools_reject_paths_outside_the_project(tmp_path: Path) -> None:
    result = await build_workspace_tool_map(tmp_path)["read_file"](path="../outside")
    assert result == {"error": "PATH_OUTSIDE_ROOT"}
