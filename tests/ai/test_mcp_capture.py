#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCP registration, declaration capture, and Asset binding contracts."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pytest

from linktools.ai.asset import (
    AssetKey,
    AssetStore,
    AssetVersionRef,
    DirectoryAssetBackend,
    InMemoryAssetBackend,
)
from linktools.ai.capability import (
    CapabilityContribution,
    CapabilityGroup,
    CapabilityLoadContext,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.spec import MCPServerSpec, MCPServerSpecAdapter, MCPServerSpecCodec
from linktools.ai.storage import StorageOverlay


class _MCPLoader:
    def __init__(
        self,
        *values: MCPServerSpec | CapabilityContribution[object],
    ) -> None:
        self.values = values

    async def load(
        self,
        context: CapabilityLoadContext,
    ) -> Sequence[MCPServerSpec | CapabilityContribution[object]]:
        del context
        return self.values


async def _memory_store() -> AssetStore:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    return store


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("stdio", "streamable-http", "sse"))
async def test_resource_free_registration_needs_no_asset_store(
    transport: Literal["stdio", "streamable-http", "sse"],
) -> None:
    group = CapabilityGroup("application")
    server = (
        MCPServerSpec("server", "python", ("-m", "server"), env={"MODE": "safe"})
        if transport == "stdio"
        else MCPServerSpec(
            "server",
            transport=transport,
            url="https://example.test/service",
            headers={"Authorization": "Bearer test-token"},
        )
    )

    assert group.mcp(server) is server
    capture = await group.capture()

    assert capture.asset_reader is None
    assert capture.source_revision is None
    assert len(capture.contributions) == 1
    contribution = capture.contributions[0]
    assert contribution.kind == "mcp"
    assert contribution.value is server
    assert contribution.contract == MCPServerSpecCodec().to_contract_payload(server)
    assert "resource_versions" not in contribution.contract
    assert "asset_source_id" not in contribution.contract


@pytest.mark.asyncio
async def test_resource_registration_requires_an_explicit_asset_store() -> None:
    group = CapabilityGroup("application")
    group.mcp(
        MCPServerSpec(
            "server", "python", ("resource:server.py",), AssetKey("mcp", "server")
        )
    )

    with pytest.raises(AIError) as error:
        await group.capture()

    assert error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ("registration", "declaration", "contribution"))
async def test_mcp_entries_share_the_captured_resource_binding(entry: str) -> None:
    store = await _memory_store()
    resource = AssetKey("mcp", "server/server.py")
    declaration = AssetKey("mcp", "server/mcp.json")
    server = MCPServerSpec(
        "server",
        "python",
        ("resource:server.py",),
        AssetKey("mcp", "server"),
        env={"MODE": "readonly"},
        revision=3,
    )
    codec = MCPServerSpecCodec()
    try:
        await store.put(resource, b"original resource")
        await store.put(declaration, codec.encode(server))
        loaded = await CapabilityGroup("application", assets=store).capture()
        group = CapabilityGroup("application", assets=store)
        if entry == "registration":
            group.loader("mcp", _MCPLoader())
            group.mcp(server)
        else:
            group.loader(
                "mcp",
                _MCPLoader(
                    server
                    if entry == "declaration"
                    else CapabilityContribution.from_declaration(server)
                ),
            )

        capture = await group.capture()
        contribution = capture.contributions[0]
        loaded_contribution = loaded.contributions[0]
        assert contribution.contract == loaded_contribution.contract
        assert isinstance(loaded_contribution.value, MCPServerSpec)
        assert codec.to_wire_payload(server) == codec.to_wire_payload(
            loaded_contribution.value
        )
        versions = codec.decode_binding_payload(contribution.contract, declaration=server)
        assert versions is not None
        assert tuple(ref.key for ref in versions) == (resource,)
        assert contribution.contract["asset_source_id"] == "application"

        await store.put(resource, b"updated resource")
        assert capture.asset_reader is not None
        assert await capture.asset_reader.read_versions(versions) == (b"original resource",)
        refreshed = await group.capture()
        assert refreshed.contributions[0].contract != contribution.contract
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_registered_resource_arguments_are_validated_during_capture() -> None:
    store = await _memory_store()
    try:
        group = CapabilityGroup("application", assets=store)
        group.mcp(
            MCPServerSpec(
                "server", "python", ("resource:missing.py",), AssetKey("mcp", "server")
            )
        )

        with pytest.raises(AIError) as error:
            await group.capture()

        assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("duplicate_source", ("registration", "loader"))
async def test_duplicate_mcp_registration_preserves_conflict_semantics(
    duplicate_source: str,
) -> None:
    store = await _memory_store()
    try:
        server = MCPServerSpec(
            "server", transport="streamable-http", url="https://example.test/service"
        )
        group = CapabilityGroup("application", assets=store)
        group.mcp(server)
        if duplicate_source == "registration":
            group.mcp(server)
        else:
            group.loader("mcp", _MCPLoader(server))

        with pytest.raises(AIError) as error:
            await group.capture()

        assert error.value.code is ErrorCode.CAPABILITY_CONFLICT
    finally:
        await store.close()


class _RemoteDeclarationStore(AssetStore):
    def __init__(
        self,
        backend: InMemoryAssetBackend | DirectoryAssetBackend,
        declaration: AssetKey,
    ) -> None:
        super().__init__(StorageOverlay(backend, writer=backend if backend.writable else None))
        self.declaration = declaration

    async def read_versions(
        self,
        refs: Sequence[AssetVersionRef],
    ) -> tuple[bytes, ...]:
        assert all(ref.key == self.declaration for ref in refs)
        return await super().read_versions(refs)

    async def local_paths(self, keys: Sequence[AssetKey]) -> tuple[Path | None, ...]:
        assert not keys, "Remote declarations must not resolve local resource paths"
        return ()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_name", ("memory", "directory"))
@pytest.mark.parametrize("format", ("json", "yaml"))
async def test_remote_packages_do_not_bind_or_read_sibling_resources(
    backend_name: str,
    format: str,
    tmp_path: Path,
) -> None:
    declaration = AssetKey("mcp", f"server/mcp.{format}")
    sibling = AssetKey("mcp", "server/private.bin")
    raw = (
        b'{"url":"https://example.test/mcp","headers":{"X-Service":"test"}}'
        if format == "json"
        else b'transport: sse\nurl: https://example.test/events\nheaders:\n  X-Service: test\n'
    )
    backend = (
        InMemoryAssetBackend()
        if backend_name == "memory"
        else DirectoryAssetBackend(str(tmp_path), kinds=("mcp",))
    )
    if backend_name == "directory":
        package = tmp_path / "mcp" / "server"
        package.mkdir(parents=True)
        (package / f"mcp.{format}").write_bytes(raw)
        (package / "private.bin").write_bytes(b"private sibling contents")
    store = _RemoteDeclarationStore(backend, declaration)
    await store.initialize()
    try:
        if backend_name == "memory":
            await store.put(declaration, raw)
            await store.put(sibling, b"private sibling contents")

        capture = await CapabilityGroup("application", assets=store).capture()

        assert len(capture.contributions) == 1
        contribution = capture.contributions[0]
        server = contribution.value
        assert isinstance(server, MCPServerSpec)
        assert server.resource is None
        assert server.transport == ("streamable-http" if format == "json" else "sse")
        assert dict(server.headers) == {"X-Service": "test"}
        assert "resource_versions" not in contribution.contract
        assert "asset_source_id" not in contribution.contract
        assert "private" not in json.dumps(contribution.contract)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_shared_config_registration_freezes_expanded_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LINKTOOLS_MCP_CAPTURE_TOKEN", "first-token")
    config = json.dumps(
        {
            "mcpServers": {
                "server": {
                    "url": "https://example.test/mcp",
                    "headers": {"Authorization": "Bearer ${LINKTOOLS_MCP_CAPTURE_TOKEN}"},
                }
            }
        }
    ).encode()
    group = CapabilityGroup("application")
    for server in MCPServerSpecAdapter().decode_config(config):
        group.mcp(server)

    capture = await group.capture()
    monkeypatch.setenv("LINKTOOLS_MCP_CAPTURE_TOKEN", "rotated-token")
    captured = capture.contributions[0].value
    assert isinstance(captured, MCPServerSpec)
    assert captured.url == "https://example.test/mcp"
    assert dict(captured.headers) == {"Authorization": "Bearer first-token"}
    assert "first-token" not in json.dumps(capture.contributions[0].contract)

    refreshed = CapabilityGroup("application")
    for server in MCPServerSpecAdapter().decode_config(config):
        refreshed.mcp(server)
    refreshed_capture = await refreshed.capture()
    rotated = refreshed_capture.contributions[0].value
    assert isinstance(rotated, MCPServerSpec)
    assert dict(rotated.headers) == {"Authorization": "Bearer rotated-token"}
    assert refreshed_capture.contributions[0].contract == capture.contributions[0].contract
