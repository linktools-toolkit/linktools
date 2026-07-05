#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for registry/mcp.py: MCPRegistry resolves MCPServerSpec from {name}.yaml
files via SpecLoader, revision-cached."""

import asyncio

import pytest

from linktools.ai.errors import (
    InvalidSpecError,
    RegistryNotFoundError,
    RegistryParseError,
)
from linktools.ai.registry.mcp import MCPRegistry, MCPServerSpec, parse_mcp_spec
from linktools.ai.registry.parser import SpecLoader


def _write_mcps(tmp_path) -> None:
    """Write mcp/search.yaml (stdio) + mcp/remote.yaml (sse) under tmp_path."""
    mcp = tmp_path / "mcp"
    mcp.mkdir()
    (mcp / "search.yaml").write_text(
        "name: search\n"
        "transport: stdio\n"
        "command: [\"python\", \"-m\", \"search_server\"]\n"
        "env:\n"
        "  API_KEY: xxx\n",
        encoding="utf-8",
    )
    (mcp / "remote.yaml").write_text(
        "name: remote\n"
        "transport: sse\n"
        "url: https://example.com/sse\n",
        encoding="utf-8",
    )


# 1. get() parses a stdio YAML into an MCPServerSpec with command joined from list.
def test_get_returns_stdio_mcp_spec(tmp_path):
    _write_mcps(tmp_path)
    registry = MCPRegistry(SpecLoader.from_filesystem(tmp_path / "mcp"))

    async def _run():
        return await registry.get("search")

    spec = asyncio.run(_run())
    assert isinstance(spec, MCPServerSpec)
    assert spec.id == "search"
    assert spec.name == "search"
    assert spec.transport == "stdio"
    assert spec.command_or_url == "python -m search_server"
    assert dict(spec.env) == {"API_KEY": "xxx"}


# 2. get() parses an sse YAML into an MCPServerSpec using url as command_or_url.
def test_get_returns_sse_mcp_spec(tmp_path):
    _write_mcps(tmp_path)
    registry = MCPRegistry(SpecLoader.from_filesystem(tmp_path / "mcp"))

    async def _run():
        return await registry.get("remote")

    spec = asyncio.run(_run())
    assert spec.transport == "sse"
    assert spec.command_or_url == "https://example.com/sse"
    assert dict(spec.env) == {}


# 3. list_ids() returns every mcp id.
def test_list_ids_returns_all_mcp_ids(tmp_path):
    _write_mcps(tmp_path)
    registry = MCPRegistry(SpecLoader.from_filesystem(tmp_path / "mcp"))

    async def _run():
        return await registry.list_ids()

    ids = asyncio.run(_run())
    assert ids == ("remote", "search")


# 4. Invalid transport -> InvalidSpecError.
def test_get_invalid_transport_raises_invalid_spec(tmp_path):
    mcp = tmp_path / "mcp"
    mcp.mkdir()
    (mcp / "pigeon.yaml").write_text(
        "name: pigeon\ntransport: carrier_pigeon\ncommand: bird\n",
        encoding="utf-8",
    )
    registry = MCPRegistry(SpecLoader.from_filesystem(mcp))

    async def _run():
        await registry.get("pigeon")

    with pytest.raises(InvalidSpecError):
        asyncio.run(_run())


# 5. Missing command + url -> InvalidSpecError.
def test_get_missing_command_and_url_raises_invalid_spec(tmp_path):
    mcp = tmp_path / "mcp"
    mcp.mkdir()
    (mcp / "empty.yaml").write_text(
        "name: empty\ntransport: stdio\n",
        encoding="utf-8",
    )
    registry = MCPRegistry(SpecLoader.from_filesystem(mcp))

    async def _run():
        await registry.get("empty")

    with pytest.raises(InvalidSpecError):
        asyncio.run(_run())


# 6. Missing mcp -> RegistryNotFoundError.
def test_get_missing_mcp_raises_not_found(tmp_path):
    _write_mcps(tmp_path)
    registry = MCPRegistry(SpecLoader.from_filesystem(tmp_path / "mcp"))

    async def _run():
        await registry.get("nope")

    with pytest.raises(RegistryNotFoundError):
        asyncio.run(_run())


# 7. get() caches the parsed spec per revision: second read does not hit the loader.
def test_get_caches_spec_per_revision():
    files = {
        "search.yaml": "name: search\ntransport: stdio\ncommand: python\n",
    }
    read_count = [0]

    async def read(path):
        read_count[0] += 1
        if path not in files:
            raise RegistryNotFoundError(path)
        return files[path]

    async def list_ids(suffix):
        return tuple(sorted(k[: -len(suffix)] for k in files if k.endswith(suffix)))

    async def revision():
        return 1

    loader = SpecLoader(read=read, list_ids=list_ids, revision=revision)
    registry = MCPRegistry(loader)

    async def _run():
        a = await registry.get("search")
        b = await registry.get("search")
        return a, b

    a, b = asyncio.run(_run())
    assert a is b
    assert read_count[0] == 1


# 8. parse_mcp_spec: defaults transport to stdio; name falls back to mcp_id.
def test_parse_mcp_spec_defaults_transport_and_name():
    spec = parse_mcp_spec("fallback", {"command": "run"})
    assert spec.name == "fallback"
    assert spec.transport == "stdio"
    assert spec.command_or_url == "run"


# 9. parse_mcp_spec: type alias works as transport.
def test_parse_mcp_spec_accepts_type_alias_for_transport():
    spec = parse_mcp_spec("r", {"type": "http", "url": "https://x"})
    assert spec.transport == "http"
    assert spec.command_or_url == "https://x"


# 10. Malformed YAML -> RegistryParseError.
def test_get_malformed_yaml_raises_parse_error(tmp_path):
    mcp = tmp_path / "mcp"
    mcp.mkdir()
    (mcp / "broken.yaml").write_text("command: [unterminated\n", encoding="utf-8")
    registry = MCPRegistry(SpecLoader.from_filesystem(mcp))

    async def _run():
        await registry.get("broken")

    with pytest.raises(RegistryParseError):
        asyncio.run(_run())
