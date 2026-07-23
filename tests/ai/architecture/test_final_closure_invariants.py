#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Architecture invariants that lock the final closure: static contract
checks (signatures, field shapes, source-level revision properties) plus
behavioral locks that exercise each closed gain end-to-end through a real
registry/parser, so a future change -- or a deleted per-area test file --
cannot silently re-introduce the gap each WP closed."""

import inspect
import math
import re
from pathlib import Path

import pytest

from linktools.ai.errors import InvalidSpecError
from linktools.ai.catalog.parsing import StrictConfigReader, parse_model_policy


def test_tool_executor_requires_final_descriptor_and_policy():
    """GovernedToolInvoker.execute() must require the finalized descriptor and
    effective_policy (no default) -- a default Descriptor/Policy could
    mis-classify a mutating tool as non-mutating and retry a write."""
    from linktools.ai.tool.executor import GovernedToolInvoker

    signature = inspect.signature(GovernedToolInvoker.execute)
    assert signature.parameters["descriptor"].default is inspect.Parameter.empty, (
        "descriptor must be a required argument (no default)"
    )
    assert (
        signature.parameters["effective_policy"].default is inspect.Parameter.empty
    ), "effective_policy must be a required argument (no default)"


def test_mcp_provider_requires_connection_manager():
    """MCPProvider must require an MCPConnectionPool (no default) -- without
    one it cannot enumerate live tools, so governance would be silently
    skipped. A missing manager is a configuration error, fail-closed."""
    import dataclasses

    from linktools.ai.mcp.provider import MCPProvider

    fields = {f.name: f for f in dataclasses.fields(MCPProvider)}
    assert "connection_manager" in fields, "MCPProvider must have connection_manager"
    assert fields["connection_manager"].default is dataclasses.MISSING, (
        "connection_manager must have no default (required at construction)"
    )


# ---------------------------------------------------------------------------
# Registry null semantics: missing -> default, explicit null -> reject.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("optional_str", {}),
        ("bool", {"default": True}),
        ("non_negative_int", {"default": 0}),
        ("positive_int", {"default": 1}),
        ("positive_number", {"default": 1.0}),
        ("string_tuple", {"default": ()}),
        ("mapping", {}),
        ("string_mapping", {}),
    ],
)
def test_reader_rejects_explicit_null(method, kwargs):
    """Every accessor rejects an explicit null rather than treating it as
    missing (a typo'd ``field: null`` must not silently take the default)."""
    reader = StrictConfigReader({"f": None}, allowed={"f"}, context="lock")
    with pytest.raises(InvalidSpecError, match="must not be null"):
        getattr(reader, method)("f", **kwargs)


def test_model_policy_timeout_rejects_non_finite():
    """ModelPolicy timeout_seconds rejects NaN/+Inf/-Inf (the hand-written
    ``timeout <= 0`` check used to let them through because NaN compares
    false)."""
    base = {"primary": "gpt"}
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(InvalidSpecError, match="positive number"):
            parse_model_policy({**base, "timeout_seconds": bad})


# ---------------------------------------------------------------------------
# MCPServerSpec domain invariants: programmatic construction is strict.
# ---------------------------------------------------------------------------


def test_mcp_server_spec_has_post_init():
    """MCPServerSpec must enforce invariants at construction (not just via the
    registry parser), so a custom provider cannot build an ungovernable server."""
    from linktools.ai.mcp.spec import MCPServerSpec

    assert hasattr(MCPServerSpec, "__post_init__"), (
        "MCPServerSpec must define __post_init__"
    )
    # And it rejects an invalid direct construction.
    with pytest.raises(ValueError):
        MCPServerSpec(id="", name="x", transport="stdio", command=("a",))


# ---------------------------------------------------------------------------
# Registry revision closure: source-level locks so the revisions cannot
# regress to constant-0 or second-level mtime.
# ---------------------------------------------------------------------------

_PARSER = (
    Path(__file__).resolve().parents[3]
    / "linktools-ai"
    / "src"
    / "linktools"
    / "ai"
    / "catalog"
    / "parsing.py"
)


def test_resource_backed_revision_is_not_constant_zero():
    """SpecLoader.from_assets must compute a real revision (not ``return 0``)
    or the registry cache would pin the first read forever."""
    source = _PARSER.read_text(encoding="utf-8")
    assert "return 0" not in source, (
        "registry parser must not pin revision to a constant 0"
    )


def test_filesystem_revision_uses_nanosecond_mtime():
    """SpecLoader.from_filesystem must hash st_mtime_ns (nanosecond), not a
    second-level int(st_mtime), or same-second edits would miss the cache."""
    source = _PARSER.read_text(encoding="utf-8")
    assert "st_mtime_ns" in source, (
        "filesystem revision must use st_mtime_ns for sub-second precision"
    )
    assert not re.search(r"int\(\s*[^)]*st_mtime", source), (
        "filesystem revision must not floor mtime to seconds"
    )


# ---------------------------------------------------------------------------
# Behavioral locks: the central fence exercises each closed gain end-to-end,
# not just its static shape, so a deleted per-WP test file cannot silently
# remove a lock.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resource_backed_registry_refreshes_after_change():
    """A asset-backed AgentCatalog re-reads after its underlying asset
    is modified (the cache invalidates on revision change) -- the
    asset-refresh gain must hold through a real registry."""
    from linktools.ai.agent.catalog import AgentCatalog
    from linktools.ai.catalog.parsing import SpecLoader
    from linktools.ai.asset.memory import MemoryAssetBackend
    from linktools.ai.asset.models import WriteOptions
    from linktools.ai.asset.path import AssetPath
    from linktools.ai.asset.store import AssetStore

    store = AssetStore(primary=MemoryAssetBackend())
    await store.put(
        AssetPath("/specs/agents/a.md"),
        b"---\nname: a\nmodel:\n  primary: gpt\n---\nv1",
        options=WriteOptions(content_type="text/markdown"),
    )
    registry = AgentCatalog.from_specloader(SpecLoader.from_assets(store, prefix="specs/agents"))
    assert "v1" in (await registry.get("a")).instructions.instructions

    await store.put(
        AssetPath("/specs/agents/a.md"),
        b"---\nname: a\nmodel:\n  primary: gpt\n---\nv2",
        options=WriteOptions(content_type="text/markdown"),
    )
    assert "v2" in (await registry.get("a")).instructions.instructions, (
        "registry must refresh after the asset is modified"
    )


def test_nested_declaration_unknown_field_rejected():
    """A nested declaration with an unknown field is rejected (ToolRef and
    AgentRef) -- the strict-nesting gain must hold in the central fence."""
    from linktools.ai.tool.codec import parse_tool_refs
    from linktools.ai.swarm.codec import _parse_agent_ref

    with pytest.raises(InvalidSpecError, match="unknown fields"):
        parse_tool_refs([{"kind": "k", "name": "n", "extra": 1}])
    with pytest.raises(InvalidSpecError, match="unknown fields"):
        _parse_agent_ref({"agentd_id": "x"}, swarm_id="sw", kind="agent")


def test_mcp_command_blank_part_rejected():
    """An MCP command with a whitespace-only part is rejected."""
    from linktools.ai.mcp.codec import parse_mcp_spec

    with pytest.raises(InvalidSpecError, match="must not be blank"):
        parse_mcp_spec("s", {"transport": "stdio", "command": ["python", "   "]})


@pytest.mark.asyncio
async def test_filesystem_registry_refreshes_within_same_second(tmp_path):
    """A filesystem-backed AgentCatalog sees a same-second modify and a same
    -second add -- the high-resolution revision gain must hold end-to-end."""
    from linktools.ai.agent.catalog import AgentCatalog
    from linktools.ai.catalog.parsing import SpecLoader

    root = tmp_path / "agents"
    root.mkdir()
    (root / "a.md").write_text(
        "---\nname: a\nmodel:\n  primary: gpt\n---\nv1\n", encoding="utf-8"
    )
    registry = AgentCatalog.from_specloader(SpecLoader.from_filesystem(root), suffix=".md")
    assert "v1" in (await registry.get("a")).instructions.instructions

    (root / "a.md").write_text(
        "---\nname: a\nmodel:\n  primary: gpt\n---\nv2\n", encoding="utf-8"
    )
    assert "v2" in (await registry.get("a")).instructions.instructions, (
        "same-second modify must invalidate the registry cache"
    )
    (root / "b.md").write_text(
        "---\nname: b\nmodel:\n  primary: gpt\n---\nx\n", encoding="utf-8"
    )
    assert await registry.list_ids() == ("a", "b"), (
        "same-second add must be visible in list_ids"
    )
