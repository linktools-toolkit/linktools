#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bubblewrap policy contracts that do not require an installed rootfs."""

from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.workspace import BubblewrapSandbox, ReadOnlySandboxPolicy


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
