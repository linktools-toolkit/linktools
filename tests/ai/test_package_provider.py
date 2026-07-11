#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PackageProvider (capability integration) + PackageRegistry (contract/contract):
catalog-only for `package:`, read tools for `package-resource:`, list tool for
`package-entrypoint:`; call stays opt-in; PackageRegistry lists packages."""

import pytest

from linktools.ai.capability import CapabilityContext, CapabilityToolExposurePolicy
from linktools.ai.capability.ref import CapabilityRef
from linktools.ai.errors import PackageResourceAccessDeniedError, PackageNotFoundError
from linktools.ai.package.capability_provider import PackageProvider
from linktools.ai.package.provider import DirectoryPackageResourceProvider
from linktools.ai.package.resolver import (
    DirectoryEntrypointResolver, DirectoryPackageRegistry, PackageRegistry,
)
from linktools.ai.package.scope import PackageScope


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "skill-creator"
    (root / "agents").mkdir(parents=True)
    (root / "references").mkdir()
    (root / "SKILL.md").write_text("# s", encoding="utf-8")
    (root / "references" / "r.md").write_text("ref", encoding="utf-8")
    (root / "agents" / "grader.md").write_text(
        "---\nname: grader\nmodel:\n  primary: gpt-4o\n---\ngrade.\n", encoding="utf-8")
    rp = DirectoryPackageResourceProvider({"skill-creator": root})
    er = DirectoryEntrypointResolver({"skill-creator": root})
    return PackageProvider(resource_provider=rp, entrypoint_resolver=er), root


def _ctx():
    return CapabilityContext(agent_id="a1", exposure_policy=CapabilityToolExposurePolicy())


@pytest.mark.asyncio
async def test_package_kind_is_prompt_catalog_only(env):
    provider, _ = env
    bundle = await provider.resolve(CapabilityRef("package", "skill-creator"), _ctx())
    assert bundle.toolsets == ()
    assert "packages" in bundle.prompt_sections


@pytest.mark.asyncio
async def test_package_resource_exposes_read_tools(env):
    provider, _ = env
    bundle = await provider.resolve(CapabilityRef("package-resource", "skill-creator"), _ctx())
    names = {md.descriptor.name for c in bundle.tool_contributions for md in c.tools}
    assert names == {"list_package_resources", "read_package_resource"}


@pytest.mark.asyncio
async def test_package_resource_read_tool_authorized_and_sandboxed(env):
    provider, _ = env
    bundle = await provider.resolve(CapabilityRef("package-resource", "skill-creator"), _ctx())
    tools = {md.descriptor.name: md.handler
             for c in bundle.tool_contributions for md in c.tools}
    list_fn = tools["list_package_resources"]
    read_fn = tools["read_package_resource"]
    # declared package -> allowed
    listing = await list_fn("skill-creator", "")
    assert any("SKILL.md" in i["path"] for i in listing["items"])
    content = await read_fn("skill-creator", "SKILL.md")
    assert content["size_bytes"] > 0
    # undeclared package -> denied
    with pytest.raises(PackageResourceAccessDeniedError):
        await read_fn("other-pkg", "SKILL.md")


@pytest.mark.asyncio
async def test_package_entrypoint_lists_only_by_default(env):
    provider, _ = env
    bundle = await provider.resolve(CapabilityRef("package-entrypoint", "skill-creator"), _ctx())
    names = {md.descriptor.name for c in bundle.tool_contributions for md in c.tools}
    # Default: only list, no call tool.
    assert names == {"list_package_entrypoints"}
    list_fn = next(md.handler for c in bundle.tool_contributions for md in c.tools
                   if md.descriptor.name == "list_package_entrypoints")
    result = await list_fn("skill-creator", kind="agent")
    assert any(i["name"] == "grader" for i in result["items"])


@pytest.mark.asyncio
async def test_package_entrypoint_call_is_opt_in(env):
    provider, _ = env
    ctx = CapabilityContext(
        agent_id="a1",
        exposure_policy=CapabilityToolExposurePolicy(expose_execution_tools=True),
    )
    ref = CapabilityRef("package-entrypoint", "skill-creator", config={
        "allowed_kinds": ["agent"], "allowed_names": ["grader"], "expose_call_tool": True,
    })
    bundle = await provider.resolve(ref, ctx)
    names = {md.descriptor.name for c in bundle.tool_contributions for md in c.tools}
    assert "call_package_entrypoint" in names


@pytest.mark.asyncio
async def test_package_registry_lists_packages(tmp_path):
    (tmp_path / "skill-creator").mkdir()
    (tmp_path / "skill-creator" / "package.yaml").write_text("kind: skill\nname: Skill Creator\n", encoding="utf-8")
    (tmp_path / "agentpack-x").mkdir()
    reg = PackageRegistry(tmp_path)
    assert set(await reg.list_ids()) == {"skill-creator", "agentpack-x"}
    spec = await reg.get("skill-creator")
    assert spec.kind == "skill" and spec.name == "Skill Creator"
    assert spec.scope == PackageScope("skill-creator", "skill")
    with pytest.raises(PackageNotFoundError):
        await reg.get("nope")


def test_directory_package_registry_alias():
    assert DirectoryPackageRegistry is PackageRegistry


@pytest.mark.asyncio
async def test_package_registry_implements_resource_provider(tmp_path):
    # contract: PackageRegistry satisfies BOTH PackageSpecProvider and
    # PackageResourceProvider.
    from linktools.ai.providers import PackageResourceProvider, PackageSpecProvider
    from linktools.ai.package.resolver import PackageRegistry
    root = tmp_path / "skill-creator"
    root.mkdir()
    (root / "SKILL.md").write_text("# s", encoding="utf-8")
    reg = PackageRegistry(tmp_path)
    assert isinstance(reg, PackageSpecProvider)
    assert isinstance(reg, PackageResourceProvider)
    from linktools.ai.package.scope import PackageScope
    from linktools.ai.package.resource import ResourceRef
    content = await reg.read_resource(ResourceRef(scope=PackageScope("skill-creator"), path="SKILL.md"))
    assert content.size_bytes > 0
