#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bubblewrap policy contracts that do not require an installed rootfs."""

from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.workspace import (
    BubblewrapSandbox,
    ReadOnlySandboxPolicy,
    SandboxResource,
)
from linktools.ai.workspace import _bubblewrap


@pytest.mark.parametrize(
    ("read_policy", "workspace_access"),
    (
        (None, "read_write"),
        (ReadOnlySandboxPolicy(readable_paths=("**",)), "read"),
        (ReadOnlySandboxPolicy(readable_paths=()), "none"),
    ),
)
def test_stdio_policy_reports_supported_workspace_access(
    read_policy: ReadOnlySandboxPolicy | None,
    workspace_access: str,
    tmp_path: Path,
) -> None:
    sandbox = BubblewrapSandbox(
        runtime_root=tmp_path / "runtime",
        bwrap_executable=tmp_path / "bwrap",
        read_policy=read_policy,
    )
    policy = sandbox.stdio_execution_policy()
    assert policy["workspace_access"] == workspace_access
    assert policy["boundary"] == "workspace-stdio"
    assert policy["network"] == "isolated"


def test_stdio_policy_rejects_narrow_read_globs(tmp_path: Path) -> None:
    sandbox = BubblewrapSandbox(
        runtime_root=tmp_path / "runtime",
        bwrap_executable=tmp_path / "bwrap",
        read_policy=ReadOnlySandboxPolicy(readable_paths=("src/**",)),
    )
    with pytest.raises(AIError) as error:
        sandbox.stdio_execution_policy()
    assert error.value.code is ErrorCode.SANDBOX_UNAVAILABLE
    assert error.value.safe_details == {"reason": "stdio_read_policy_unsupported"}


def test_stdio_policy_is_read_only_and_returns_detached_values(
    tmp_path: Path,
) -> None:
    sandbox = BubblewrapSandbox(
        runtime_root=tmp_path / "runtime",
        bwrap_executable=tmp_path / "bwrap",
        hidden_paths=("private", "tmp/cache"),
    )
    policy = sandbox.stdio_execution_policy()
    hidden_paths = policy["hidden_paths"]
    assert isinstance(hidden_paths, list)
    hidden_paths.append("outside")
    assert "outside" not in policy["hidden_paths"]
    with pytest.raises(TypeError):
        policy["network"] = "public"  # type: ignore[index]


def test_readonly_hidden_path_can_be_absent_without_host_mutation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    hidden = (".linktools",)

    assert _bubblewrap._prepare_hidden_paths(
        workspace,
        hidden,
        create_missing=False,
    ) == hidden
    assert not (workspace / ".linktools").exists()


def test_stdio_resources_are_not_mounted_into_worker_session(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    runtime = tmp_path / "runtime"
    locks = tmp_path / "locks"
    resource_root = tmp_path / "mcp"
    for path in (workspace, runtime, locks, resource_root):
        path.mkdir()
    resource = SandboxResource("mcp", resource_root)

    worker_args = _bubblewrap._build_bwrap_args(
        root=workspace,
        runtime_root=runtime,
        bwrap=tmp_path / "bwrap",
        lock_root=locks,
        resources=(),
        hidden_paths=(),
        worker_resources=[],
    )
    stdio_args = _bubblewrap._build_bwrap_args(
        root=workspace,
        runtime_root=runtime,
        bwrap=tmp_path / "bwrap",
        lock_root=locks,
        resources=(resource,),
        hidden_paths=(),
        worker_resources=[],
        mode="stdio",
        command="/usr/bin/python3",
    )

    guest_path = _bubblewrap._resource_guest_path(resource.id)
    assert str(resource_root) not in worker_args
    assert guest_path not in worker_args
    assert str(resource_root) in stdio_args
    assert guest_path in stdio_args

