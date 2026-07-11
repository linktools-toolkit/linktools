#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable Provider/scenario assertions (contract). Both the default
file-backed registries and any business Provider/Store implementation must
satisfy these. Downstream systems import these helpers to validate their own
custom providers."""

from collections.abc import Mapping

import pytest

from linktools.ai.agent.spec import AgentSpec
from linktools.ai.errors import RegistryNotFoundError
from linktools.ai.package.entrypoint import EntrypointRef
from linktools.ai.package.resource import ResourceRef
from linktools.ai.package.scope import PackageScope


async def assert_spec_provider_contract(provider, *, sample_id, expected_type, missing_exc=(KeyError, LookupError, RegistryNotFoundError)):
    """list_ids returns a tuple of str; get returns the standard Spec type for a
    known id; a missing id raises."""
    ids = await provider.list_ids()
    assert isinstance(ids, tuple), f"list_ids must return tuple, got {type(ids)}"
    assert all(isinstance(i, str) for i in ids), "list_ids must contain only strings"
    assert sample_id in ids, f"{sample_id!r} not in {ids}"
    spec = await provider.get(sample_id)
    assert isinstance(spec, expected_type), f"get must return {expected_type}, got {type(spec)}"
    with pytest.raises(missing_exc):
        await provider.get("__contract_missing_id__")


async def assert_tool_policy_provider_contract(provider, *, sample_name):
    """get_metadata_map returns a Mapping of tool name -> ToolPolicyMetadata with
    the required fields populated."""
    from linktools.ai.policy.rule import ToolPolicyMetadata

    meta_map = await provider.get_metadata_map()
    assert isinstance(meta_map, Mapping)
    assert sample_name in meta_map, f"{sample_name!r} not in metadata map"
    meta = meta_map[sample_name]
    assert isinstance(meta, ToolPolicyMetadata)
    for field in ("permissions", "risk", "side_effect", "approval"):
        assert hasattr(meta, field), f"ToolPolicyMetadata missing {field}"


async def assert_package_resource_provider_contract(provider, *, package_id, sample_path):
    """list_resources paginates (limit/cursor); read_resource honors max_bytes;
    parent-traversal is rejected."""
    scope = PackageScope(package_id)
    page = await provider.list_resources(scope, "", limit=1)
    assert hasattr(page, "next_cursor")
    content = await provider.read_resource(ResourceRef(scope, sample_path), max_bytes=1 << 20)
    assert hasattr(content, "content") and hasattr(content, "size_bytes")
    with pytest.raises(Exception):
        await provider.read_resource(ResourceRef(scope, "../escape"))


async def assert_entrypoint_resolver_contract(resolver, *, package_id, agent_name):
    """list_entrypoints honors kind filter + pagination; resolve_agent returns a
    scoped AgentSpec; same-named entrypoints in two packages stay distinct."""
    scope = PackageScope(package_id)
    listed = await resolver.list_entrypoints(scope, kind="agent")
    assert any(i.name == agent_name and i.package_id == package_id for i in listed.items)
    paged = await resolver.list_entrypoints(scope, kind="agent", limit=1)
    assert len(paged.items) <= 1
    agent = await resolver.resolve_agent(EntrypointRef("agent", agent_name, scope))
    assert isinstance(agent, AgentSpec)
    assert agent.id == f"package:{package_id}:agent:{agent_name}"
