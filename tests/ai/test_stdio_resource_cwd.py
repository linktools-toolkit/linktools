#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-process resource working directories preserve sandbox boundaries."""

import asyncio
import json
import sys
from pathlib import Path

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
    LocalWorkspaceBackend,
    WorkspaceResource,
    SupportsStdio,
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
    resource = WorkspaceResource("package", package)
    session = LocalWorkspaceBackend(workspace)
    await session.working_dir()
    assert isinstance(session, SupportsStdio)
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
    resource = WorkspaceResource("package", package)
    session = LocalWorkspaceBackend(workspace, resources=(resource,))
    await session.working_dir()
    assert isinstance(session, SupportsStdio)
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
@pytest.mark.parametrize(
    "invalid_resource", ("ungranted", "files_only", "missing", "changed")
)
async def test_stdio_resource_working_directory_fails_closed(
    tmp_path: Path,
    invalid_resource: str,
) -> None:
    workspace = tmp_path / "workspace"
    package = tmp_path / "package"
    workspace.mkdir()
    package.mkdir()
    script = package / "server.py"
    script.write_text("raise AssertionError('must not start')\n", encoding="utf-8")
    resource = WorkspaceResource(
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

    session = LocalWorkspaceBackend(workspace)
    await session.working_dir()
    assert isinstance(session, SupportsStdio)
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
            resource = await WorkspaceResource.from_asset_versions(
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
            assert await WorkspaceResource.from_asset_versions(
                "package", store, {}, materializer=materializer
            ) is None
    finally:
        await store.close()
