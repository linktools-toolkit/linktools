#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for registry/tool.py: ToolSpec declaration + ToolRegistry loading."""

import asyncio

import pytest

from linktools.ai.errors import (
    InvalidSpecError,
    RegistryNotFoundError,
    RegistryParseError,
)
from linktools.ai.policy.rule import (
    ApprovalMode,
    Permission,
    RiskLevel,
    SideEffectKind,
    ToolPolicyMetadata,
)
from linktools.ai.registry.parser import SpecLoader
from linktools.ai.registry.tool import ToolRegistry, ToolSpec


def _write_tools(tmp_path) -> None:
    """Write terminal.yaml + reader.yaml under tmp_path/tools."""
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "terminal.yaml").write_text(
        "description: shell\n"
        "permissions: [execute, write]\n"
        "risk: HIGH\n"
        "side_effect: destructive\n"
        "approval: on_risk\n",
        encoding="utf-8",
    )
    (tools / "reader.yaml").write_text(
        "description: read files\n"
        "permissions: [read]\n"
        "risk: LOW\n"
        "side_effect: read_only\n"
        "approval: never\n",
        encoding="utf-8",
    )


# 1. get() parses a full YAML into a ToolSpec with the declared policy slice.
def test_get_returns_tool_spec_from_yaml(tmp_path):
    _write_tools(tmp_path)
    registry = ToolRegistry(SpecLoader.from_filesystem(tmp_path / "tools"))

    async def run():
        return await registry.get("terminal")

    spec = asyncio.run(run())
    assert isinstance(spec, ToolSpec)
    assert spec.name == "terminal"
    assert spec.description == "shell"
    assert spec.permissions == frozenset({Permission.EXECUTE, Permission.WRITE})
    assert spec.risk is RiskLevel.HIGH
    assert spec.side_effect is SideEffectKind.DESTRUCTIVE
    assert spec.approval is ApprovalMode.ON_RISK


# 2. list_ids() returns every tool id (sorted by filename via SpecLoader).
def test_list_ids_returns_all_tool_ids(tmp_path):
    _write_tools(tmp_path)
    registry = ToolRegistry(SpecLoader.from_filesystem(tmp_path / "tools"))

    async def run():
        return await registry.list_ids()

    ids = asyncio.run(run())
    assert ids == ("reader", "terminal")


# 3. get_metadata_map() exposes the ToolPolicyMetadata slice the rule modules consume.
def test_get_metadata_map_returns_policy_metadata(tmp_path):
    _write_tools(tmp_path)
    registry = ToolRegistry(SpecLoader.from_filesystem(tmp_path / "tools"))

    async def run():
        return await registry.get_metadata_map()

    mapping = asyncio.run(run())
    assert set(mapping.keys()) == {"reader", "terminal"}
    terminal = mapping["terminal"]
    assert isinstance(terminal, ToolPolicyMetadata)
    assert terminal.permissions == frozenset({Permission.EXECUTE, Permission.WRITE})
    assert terminal.risk is RiskLevel.HIGH
    assert terminal.side_effect is SideEffectKind.DESTRUCTIVE
    assert terminal.approval is ApprovalMode.ON_RISK
    # reader uses the read_only/LOW/never profile
    reader = mapping["reader"]
    assert reader.permissions == frozenset({Permission.READ})
    assert reader.risk is RiskLevel.LOW
    assert reader.approval is ApprovalMode.NEVER


# 4. get() caches a parsed spec per revision: second read does not hit the loader.
def test_get_caches_spec_per_revision():
    files = {"terminal.yaml": "description: shell\npermissions: [execute]\n"}
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
    registry = ToolRegistry(loader)

    async def run():
        a = await registry.get("terminal")
        b = await registry.get("terminal")
        return a, b

    a, b = asyncio.run(run())
    assert a is b
    assert read_count[0] == 1


# 5. Missing tool -> RegistryNotFoundError (propagated from the loader).
def test_get_missing_tool_raises_not_found(tmp_path):
    _write_tools(tmp_path)
    registry = ToolRegistry(SpecLoader.from_filesystem(tmp_path / "tools"))

    async def run():
        await registry.get("nope")

    with pytest.raises(RegistryNotFoundError):
        asyncio.run(run())


# 6a. Malformed YAML -> RegistryParseError.
def test_get_malformed_yaml_raises_parse_error(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "broken.yaml").write_text("permissions: [unterminated\n", encoding="utf-8")
    registry = ToolRegistry(SpecLoader.from_filesystem(tools))

    async def run():
        await registry.get("broken")

    with pytest.raises(RegistryParseError):
        asyncio.run(run())


# 6b. Unknown permission -> InvalidSpecError.
def test_get_unknown_permission_raises_invalid_spec(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "weird.yaml").write_text("permissions: [fly]\n", encoding="utf-8")
    registry = ToolRegistry(SpecLoader.from_filesystem(tools))

    async def run():
        await registry.get("weird")

    with pytest.raises(InvalidSpecError):
        asyncio.run(run())


# 6c. Unknown risk -> InvalidSpecError.
def test_get_unknown_risk_raises_invalid_spec(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "risky.yaml").write_text("risk: EXTREME\n", encoding="utf-8")
    registry = ToolRegistry(SpecLoader.from_filesystem(tools))

    async def run():
        await registry.get("risky")

    with pytest.raises(InvalidSpecError):
        asyncio.run(run())


# 7. A YAML declaring only a description falls back to the safe defaults.
def test_get_applies_defaults_when_only_description(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "minimal.yaml").write_text(
        "description: just a description\n", encoding="utf-8"
    )
    registry = ToolRegistry(SpecLoader.from_filesystem(tools))

    async def run():
        return await registry.get("minimal")

    spec = asyncio.run(run())
    assert spec.description == "just a description"
    assert spec.permissions == frozenset({Permission.READ})
    assert spec.risk is RiskLevel.LOW
    assert spec.side_effect is SideEffectKind.READ_ONLY
    assert spec.approval is ApprovalMode.NEVER
    assert spec.idempotent is None
    assert spec.timeout_seconds is None
