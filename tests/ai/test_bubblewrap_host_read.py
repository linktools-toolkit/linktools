#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Host byte reads stay anchored despite concurrent workspace mutations."""

import os
import sys
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.workspace import WorkspaceReadPolicy, _bubblewrap

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Bubblewrap requires Linux")


@pytest.mark.parametrize("swap", ("file", "ancestor", "fifo"))
def test_host_read_rejects_swapped_symlinks_and_special_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap: str,
) -> None:
    root = tmp_path / "workspace"
    directory = root / "directory"
    directory.mkdir(parents=True)
    target = directory / "value.bin"
    target.write_bytes(b"allowed")
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "value.bin"
    secret.write_bytes(b"host secret")
    original_open = os.open
    swapped = False

    def swap_before_open(path: str | Path, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        component = "directory" if swap == "ancestor" else "value.bin"
        if path == component and not swapped:
            swapped = True
            if swap == "ancestor":
                directory.rename(root / "old-directory")
                directory.symlink_to(outside, target_is_directory=True)
            else:
                target.unlink()
                if swap == "file":
                    target.symlink_to(secret)
                else:
                    os.mkfifo(target)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(_bubblewrap.os, "open", swap_before_open)
    with pytest.raises(AIError) as error:
        _bubblewrap._read_workspace_bytes(
            root, "directory/value.bin", read_policy=None, hidden_paths=(), max_bytes=None
        )
    assert swapped
    assert error.value.code in {
        ErrorCode.AUTHORIZATION_DENIED, ErrorCode.REQUEST_FIELD_INVALID
    }


def test_host_read_preserves_contained_file_symlinks_and_byte_limit(tmp_path: Path) -> None:
    (tmp_path / "data.bin").write_bytes(b"allowed")
    (tmp_path / "link.bin").symlink_to("data.bin")
    assert _bubblewrap._read_workspace_bytes(
        tmp_path, "link.bin", read_policy=None, hidden_paths=(), max_bytes=7
    ) == b"allowed"
    with pytest.raises(AIError) as error:
        _bubblewrap._read_workspace_bytes(
            tmp_path, "link.bin", read_policy=None, hidden_paths=(), max_bytes=6
        )
    assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID


@pytest.mark.parametrize("path", ("directory", "root-link"))
def test_host_read_rejects_directories_with_typed_error(tmp_path: Path, path: str) -> None:
    (tmp_path / "directory").mkdir()
    (tmp_path / "root-link").symlink_to(".")
    with pytest.raises(AIError) as error:
        _bubblewrap._read_workspace_bytes(
            tmp_path, path, read_policy=None, hidden_paths=(), max_bytes=None
        )
    assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID


def test_restricted_host_read_never_follows_new_file_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = tmp_path / "allowed.txt"
    allowed.write_bytes(b"allowed")
    (tmp_path / "secret.txt").write_bytes(b"secret")
    is_symlink = Path.is_symlink

    def swap_after_check(path: Path) -> bool:
        result = is_symlink(path)
        if path == allowed and not result:
            allowed.unlink()
            allowed.symlink_to("secret.txt")
        return result

    monkeypatch.setattr(Path, "is_symlink", swap_after_check)
    try:
        value = _bubblewrap._read_workspace_bytes(
            tmp_path,
            "allowed.txt",
            read_policy=WorkspaceReadPolicy(("allowed.txt",)),
            hidden_paths=(),
            max_bytes=None,
        )
    except AIError as error:
        assert error.code is ErrorCode.AUTHORIZATION_DENIED
    else:
        assert value == b"allowed"


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_kind", ("directory", "members", "source_less"))
@pytest.mark.parametrize("restricted", (False, True))
async def test_host_resource_reads_share_canonical_mount_grants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resource_kind: str, restricted: bool,
) -> None:
    import asyncio
    from typing import Any
    from unittest.mock import AsyncMock, MagicMock
    from linktools.ai.workspace import Workspace, WorkspaceResource

    root, runtime = tmp_path / "workspace", tmp_path / "runtime"
    real, outside = tmp_path / "real" / "assets", tmp_path / "outside" / "assets"
    for directory in (root, runtime, real, outside):
        directory.mkdir(parents=True)
    (root / Workspace.STORAGE_DIR_NAME).mkdir()
    for directory, value in ((real, b"granted"), (outside, b"not granted")):
        for name in ("allowed", "policy-denied", "nonmember"):
            (directory / name).write_bytes(value)
    alias = tmp_path / "alias"
    alias.symlink_to(real.parent, target_is_directory=True)
    source = None if resource_kind == "source_less" else alias / "assets"
    files = None if resource_kind == "directory" else {
        name: alias / "assets" / name for name in ("allowed", "policy-denied")
    }
    resource = WorkspaceResource("docs", source, files)
    policy = WorkspaceReadPolicy((), resource_paths={"docs": ("allowed",)}) if restricted else None
    backend = _bubblewrap.BubblewrapWorkspaceBackend(
        root, runtime_root=runtime, bwrap_executable=Path(sys.executable).resolve(),
        resources=(resource,), read_policy=policy,
    )
    configs: list[dict[str, Any]] = []
    process = MagicMock(spec=asyncio.subprocess.Process)
    process.stdin = None
    process.returncode = 0
    process.wait = AsyncMock(return_value=0)

    def validate_rootfs(runtime_root: Path, workspace_root: Path) -> None:
        return None

    def validate_bwrap(executable: Path) -> None:
        return None

    async def spawn_guardian(config: dict[str, Any], runtime_pidfd: int) -> tuple[asyncio.subprocess.Process, int]:
        configs.append(config)
        return process, -1

    async def ready() -> None:
        backend._state = "OPEN"

    # Only the host preparation/read owner is exercised, not a namespace launch.
    monkeypatch.setattr(_bubblewrap.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(_bubblewrap, "_validate_rootfs", validate_rootfs)
    monkeypatch.setattr(_bubblewrap, "_validate_bwrap", validate_bwrap)
    monkeypatch.setattr(_bubblewrap, "_spawn_guardian", spawn_guardian)
    monkeypatch.setattr(backend, "_open_handshake", ready)
    try:
        await backend.working_dir()
        guest = backend.resource_path("docs")
        assert guest is not None
        mounted = real if resource_kind == "directory" else real / "allowed"
        assert str(mounted) in configs[0]["bwrap_args"]
        alias.unlink()
        alias.symlink_to(outside.parent, target_is_directory=True)
        assert await backend.read_bytes(guest + "/allowed") == b"granted"
        if resource_kind != "directory":
            with pytest.raises(PermissionError):
                await backend.read_bytes(guest + "/nonmember")
        if restricted:
            with pytest.raises(PermissionError):
                await backend.read_bytes(guest + "/policy-denied")
        else:
            assert await backend.read_bytes(guest + "/policy-denied") == b"granted"
    finally:
        await backend.close()
