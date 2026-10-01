#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-process resource working directories preserve sandbox boundaries."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from linktools.ai.asset import (
    AssetKey,
    AssetMaterializer,
    AssetStore,
    DirectoryAssetBackend,
    InMemoryAssetBackend,
    PrefixAssetPathAdapter,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import StorageOverlay
from linktools.ai.workspace import (
    BubblewrapSandbox,
    LocalSandbox,
    SandboxResource,
    StdioSandboxSession,
)
from linktools.ai.workspace import _bubblewrap, sandbox_guardian


def _guardian_config(
    root: Path,
    resources: tuple[SandboxResource, ...],
    cwd_resource_id: str | None,
    *,
    mode: str = "stdio",
) -> dict[str, Any]:
    return _bubblewrap._guardian_config(
        root=root / "workspace",
        runtime_root=root / "runtime",
        bwrap=root / "bwrap",
        lock_root=root / "locks",
        resources=resources,
        hidden_paths=(".linktools",),
        read_policy=None,
        mode=mode,
        command="/usr/bin/python3",
        command_args=("server.py",),
        environment={"PWD": "/caller-value"} if cwd_resource_id else None,
        cwd_resource_id=cwd_resource_id,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_resource", (False, True))
async def test_local_stdio_starts_relative_script_in_selected_directory(
    tmp_path: Path,
    use_resource: bool,
) -> None:
    workspace = tmp_path / "workspace"
    package = tmp_path / "package"
    for root in (workspace, package):
        root.mkdir()
        (root / "server.py").write_text(
            "import json, os\n"
            "from pathlib import Path\n"
            "print(json.dumps([os.getcwd(), os.getenv('PWD'), "
            "Path('value.txt').read_text()]), flush=True)\n",
            encoding="utf-8",
        )
        (root / "value.txt").write_text(root.name, encoding="utf-8")
    resource = SandboxResource("package", package)
    session = await LocalSandbox().open(root=workspace)
    assert isinstance(session, StdioSandboxSession)
    try:
        process = await session.open_stdio_process(
            sys.executable,
            ("server.py",),
            resources=(resource,),
            environment={"PWD": "/caller-value"},
            cwd_resource_id="package" if use_resource else None,
        )
        try:
            chunks: list[bytes] = []
            while chunk := await asyncio.wait_for(process.read_stdout(), 5):
                chunks.append(chunk)
            cwd, pwd, value = json.loads(b"".join(chunks))
            selected = package if use_resource else workspace
            assert Path(cwd) == selected.resolve()
            assert value == selected.name
            if use_resource:
                assert Path(pwd) == package.resolve()
        finally:
            await process.close()
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_session_resource_does_not_grant_process_working_directory(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    package = tmp_path / "package"
    workspace.mkdir()
    package.mkdir()
    resource = SandboxResource("package", package)
    session = await LocalSandbox().open(root=workspace, resources=(resource,))
    assert isinstance(session, StdioSandboxSession)
    try:
        with pytest.raises(AIError) as error:
            await session.open_stdio_process(
                sys.executable,
                ("-c", "raise AssertionError('must not start')"),
                cwd_resource_id="package",
            )
        assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("local", "bubblewrap"))
@pytest.mark.parametrize(
    "invalid_resource", ("ungranted", "files_only", "missing", "changed")
)
async def test_stdio_resource_working_directory_fails_closed(
    tmp_path: Path,
    backend: str,
    invalid_resource: str,
) -> None:
    workspace = tmp_path / "workspace"
    package = tmp_path / "package"
    workspace.mkdir()
    package.mkdir()
    script = package / "server.py"
    script.write_text("raise AssertionError('must not start')\n", encoding="utf-8")
    resource = SandboxResource(
        "package",
        None if invalid_resource == "files_only" else package,
        {"server.py": script},
    )
    resources = () if invalid_resource == "ungranted" else (resource,)
    if invalid_resource == "missing":
        script.unlink()
        package.rmdir()
    elif invalid_resource == "changed":
        script.unlink()
        script.mkdir()

    if backend == "bubblewrap":
        with pytest.raises(AIError):
            _guardian_config(tmp_path, resources, "package")
        return

    session = await LocalSandbox().open(root=workspace)
    assert isinstance(session, StdioSandboxSession)
    try:
        with pytest.raises(AIError):
            await session.open_stdio_process(
                sys.executable,
                ("server.py",),
                resources=resources,
                cwd_resource_id="package",
            )
    finally:
        await session.close()


def test_bubblewrap_resource_cwd_preserves_readonly_and_process_isolation(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    script = package / "server.py"
    script.write_text("print('ready')\n", encoding="utf-8")
    sibling = package / "secret.txt"
    sibling.write_text("not granted", encoding="utf-8")
    resource = SandboxResource("package", package, {"server.py": script})

    config = _guardian_config(tmp_path, (resource,), "package")
    sandbox_guardian._validate_config(config)
    args = config["bwrap_args"]
    guest = _bubblewrap._resource_guest_path("package")
    assert args[args.index("--chdir") + 1] == guest
    assert args[args.index("PWD") + 1] == guest
    assert "/caller-value" not in args
    assert args[-3:] == ["--", "/usr/bin/python3", "server.py"]
    index = args.index(str(script))
    assert args[index - 1 : index + 2] == [
        "--ro-bind", str(script), f"{guest}/server.py"
    ]
    assert str(package) not in args
    assert str(sibling) not in args
    assert ["--remount-ro", "/resources"] in [
        args[index:index + 2] for index in range(len(args))
    ]
    assert {
        "--unshare-net", "--unshare-pid", "--die-with-parent", "--as-pid-1"
    } <= set(args)


@pytest.mark.parametrize("mode", ("worker", "stdio"))
def test_bubblewrap_default_cwd_remains_workspace(tmp_path: Path, mode: str) -> None:
    config = _guardian_config(tmp_path, (), None, mode=mode)
    sandbox_guardian._validate_config(config)
    args = config["bwrap_args"]
    assert args[args.index("--chdir") + 1] == "/workspace"
    assert args[args.index("PWD") + 1] == "/workspace"


def test_worker_cannot_select_stdio_resource_cwd(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    with pytest.raises(AIError) as error:
        _guardian_config(
            tmp_path,
            (SandboxResource("package", package),),
            "package",
            mode="worker",
        )
    assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID


def test_guardian_wire_preserves_empty_arguments_and_environment(tmp_path: Path) -> None:
    config = _bubblewrap._guardian_config(
        root=tmp_path,
        runtime_root=tmp_path / "runtime",
        bwrap=tmp_path / "bwrap",
        lock_root=tmp_path / "locks",
        resources=(),
        hidden_paths=(),
        read_policy=None,
        mode="stdio",
        command="/usr/bin/python3",
        command_args=("server.py", ""),
        environment={"EMPTY": ""},
    )
    sandbox_guardian._validate_config(config)
    args = config["bwrap_args"]
    assert args[-3:] == ["/usr/bin/python3", "server.py", ""]
    assert args[args.index("EMPTY") + 1] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_kind", ("memory", "native", "remapped_native"))
async def test_asset_resource_materialization_preserves_native_paths_when_usable(
    tmp_path: Path,
    backend_kind: str,
) -> None:
    data = b"#!/bin/sh\necho pinned\n"
    native = tmp_path / "assets" / "packages" / "example" / "actual.sh"
    key = AssetKey("mcp", "example/actual.sh")
    if backend_kind == "memory":
        backend = InMemoryAssetBackend()
        store = AssetStore(StorageOverlay(backend, writer=backend))
    else:
        native.parent.mkdir(parents=True)
        native.write_bytes(data)
        native.chmod(0o700)
        store = AssetStore(StorageOverlay(DirectoryAssetBackend(
            str(tmp_path / "assets"),
            path_adapter=PrefixAssetPathAdapter({"mcp": "packages"}),
            kinds=("mcp",),
        )))
    await store.initialize()
    materialized_root: Path | None = None
    try:
        if backend_kind == "memory":
            await store.put(key, data)
        ref = (await store.resolve_versions((key,)))[0]
        relative = "actual.sh" if backend_kind == "native" else "server.sh"
        async with AssetMaterializer() as materializer:
            resource = await SandboxResource.from_asset_versions(
                "package",
                store,
                {relative: ref},
                executable_bits={relative: 0o100},
                materializer=materializer,
            )
            assert resource is not None
            assert resource.source is not None
            assert resource.files is not None
            assert resource.files[relative].read_bytes() == data
            assert resource.files[relative].stat().st_mode & 0o111 == 0o100
            assert resource.source / relative == resource.files[relative]
            if backend_kind == "native":
                assert resource.source == native.parent.resolve()
            else:
                materialized_root = resource.source
                assert resource.source != native.parent
        if materialized_root is not None:
            assert not materialized_root.exists()
        if backend_kind != "memory":
            assert native.read_bytes() == data
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_empty_asset_resource_remains_absent_with_materializer() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        async with AssetMaterializer() as materializer:
            assert await SandboxResource.from_asset_versions(
                "package", store, {}, materializer=materializer
            ) is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_bubblewrap_runs_relative_script_from_readonly_resource_cwd(
    tmp_path: Path,
) -> None:
    runtime_root = os.environ.get("LINKTOOLS_BWRAP_RUNTIME_ROOT")
    executable = os.environ.get("LINKTOOLS_BWRAP_EXECUTABLE")
    if not runtime_root or not executable:
        if os.environ.get("LINKTOOLS_REQUIRE_BUBBLEWRAP") == "1":
            pytest.fail("the required Bubblewrap acceptance environment is missing")
        pytest.skip("Bubblewrap acceptance rootfs is not configured")
    workspace = tmp_path / "workspace"
    package = tmp_path / "package"
    workspace.mkdir()
    package.mkdir()
    (package / "value.txt").write_text("resource", encoding="utf-8")
    (package / "server.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "try:\n"
        "    Path('value.txt').write_text('changed')\n"
        "    writable = True\n"
        "except OSError:\n"
        "    writable = False\n"
        "print(json.dumps([os.getcwd(), os.getenv('PWD'), "
        "Path('value.txt').read_text(), writable]), flush=True)\n",
        encoding="utf-8",
    )
    session = await BubblewrapSandbox(
        runtime_root=Path(runtime_root),
        bwrap_executable=Path(executable),
    ).open(root=workspace)
    assert isinstance(session, StdioSandboxSession)
    try:
        process = await session.open_stdio_process(
            "/usr/bin/python3",
            ("server.py",),
            resources=(SandboxResource("package", package),),
            cwd_resource_id="package",
        )
        try:
            chunks: list[bytes] = []
            while chunk := await asyncio.wait_for(process.read_stdout(), 10):
                chunks.append(chunk)
            guest = _bubblewrap._resource_guest_path("package")
            assert json.loads(b"".join(chunks)) == [guest, guest, "resource", False]
        finally:
            await process.close()
    finally:
        await session.close()
