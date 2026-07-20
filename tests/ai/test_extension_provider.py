#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ExtensionProvider (capability integration) + ExtensionRegistry (contract/contract):
catalog-only for `extension:`, read tools for `extension-resource:`, list tool for
`extension-entrypoint:`; call stays opt-in; ExtensionRegistry lists extensions."""

import pytest

from linktools.ai.capability.exposure import CapabilityToolExposurePolicy
from linktools.ai.capability.provider import CapabilityContext
from linktools.ai.capability.models import CapabilityRef
from linktools.ai.errors import ExtensionResourceAccessDeniedError, ExtensionNotFoundError
from linktools.ai.extension.capability_provider import ExtensionProvider
from linktools.ai.extension.provider import DirectoryExtensionResourceProvider
from linktools.ai.extension.resolver import (
    DirectoryEntrypointResolver,
    DirectoryExtensionRegistry,
    ExtensionRegistry,
)
from linktools.ai.extension.scope import ExtensionScope


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "skill-creator"
    (root / "agents").mkdir(parents=True)
    (root / "references").mkdir()
    (root / "SKILL.md").write_text("# s", encoding="utf-8")
    (root / "references" / "r.md").write_text("ref", encoding="utf-8")
    (root / "agents" / "grader.md").write_text(
        "---\nname: grader\nmodel:\n  primary: gpt-4o\n---\ngrade.\n", encoding="utf-8"
    )
    rp = DirectoryExtensionResourceProvider({"skill-creator": root})
    er = DirectoryEntrypointResolver({"skill-creator": root})
    return ExtensionProvider(resource_provider=rp, entrypoint_resolver=er), root


def _ctx():
    return CapabilityContext(
        agent_id="a1", exposure_policy=CapabilityToolExposurePolicy()
    )


@pytest.mark.asyncio
async def test_extension_kind_is_prompt_catalog_only(env):
    provider, _ = env
    bundle = await provider.resolve(CapabilityRef("extension", "skill-creator"), _ctx())
    assert "extensions" in bundle.prompt_sections


@pytest.mark.asyncio
async def test_extension_resource_exposes_read_tools(env):
    provider, _ = env
    bundle = await provider.resolve(
        CapabilityRef("extension-resource", "skill-creator"), _ctx()
    )
    names = {md.descriptor.name for c in bundle.tool_contributions for md in c.tools}
    assert names == {"list_extension_resources", "read_extension_resource"}


@pytest.mark.asyncio
async def test_extension_resource_read_tool_authorized_and_sandboxed(env):
    provider, _ = env
    bundle = await provider.resolve(
        CapabilityRef("extension-resource", "skill-creator"), _ctx()
    )
    tools = {
        md.descriptor.name: md.handler
        for c in bundle.tool_contributions
        for md in c.tools
    }
    list_fn = tools["list_extension_resources"]
    read_fn = tools["read_extension_resource"]
    # declared extension -> allowed
    listing = await list_fn("skill-creator", "")
    assert any("SKILL.md" in i["path"] for i in listing["items"])
    content = await read_fn("skill-creator", "SKILL.md")
    assert content["size_bytes"] > 0
    # undeclared extension -> denied
    with pytest.raises(ExtensionResourceAccessDeniedError):
        await read_fn("other-pkg", "SKILL.md")


@pytest.mark.asyncio
async def test_extension_entrypoint_lists_only_by_default(env):
    provider, _ = env
    bundle = await provider.resolve(
        CapabilityRef("extension-entrypoint", "skill-creator"), _ctx()
    )
    names = {md.descriptor.name for c in bundle.tool_contributions for md in c.tools}
    # Default: only list, no call tool.
    assert names == {"list_extension_entrypoints"}
    list_fn = next(
        md.handler
        for c in bundle.tool_contributions
        for md in c.tools
        if md.descriptor.name == "list_extension_entrypoints"
    )
    result = await list_fn("skill-creator", kind="agent")
    assert any(i["name"] == "grader" for i in result["items"])


@pytest.mark.asyncio
async def test_extension_entrypoint_call_is_opt_in(env):
    provider, _ = env
    ctx = CapabilityContext(
        agent_id="a1",
        exposure_policy=CapabilityToolExposurePolicy(expose_execution_tools=True),
    )
    ref = CapabilityRef(
        "extension-entrypoint",
        "skill-creator",
        config={
            "allowed_kinds": ["agent"],
            "allowed_names": ["grader"],
            "expose_call_tool": True,
        },
    )
    bundle = await provider.resolve(ref, ctx)
    names = {md.descriptor.name for c in bundle.tool_contributions for md in c.tools}
    assert "call_extension_entrypoint" in names


@pytest.mark.asyncio
async def test_extension_registry_lists_extensions(tmp_path):
    (tmp_path / "skill-creator").mkdir()
    (tmp_path / "skill-creator" / "extension.yaml").write_text(
        "kind: skill\nname: Skill Creator\n", encoding="utf-8"
    )
    (tmp_path / "agentpack-x").mkdir()
    reg = ExtensionRegistry(tmp_path)
    assert set(await reg.list_ids()) == {"skill-creator", "agentpack-x"}
    spec = await reg.get("skill-creator")
    assert spec.kind == "skill" and spec.name == "Skill Creator"
    assert spec.scope == ExtensionScope("skill-creator", "skill")
    with pytest.raises(ExtensionNotFoundError):
        await reg.get("nope")


def test_directory_extension_registry_alias():
    assert DirectoryExtensionRegistry is ExtensionRegistry


@pytest.mark.asyncio
async def test_extension_registry_implements_resource_provider(tmp_path):
    # contract: ExtensionRegistry satisfies BOTH ExtensionSpecProvider and
    # ExtensionResourceProvider.
    from linktools.ai.extension.spec import (
        ExtensionResourceProvider,
        ExtensionSpecProvider,
    )
    from linktools.ai.extension.resolver import ExtensionRegistry

    root = tmp_path / "skill-creator"
    root.mkdir()
    (root / "SKILL.md").write_text("# s", encoding="utf-8")
    reg = ExtensionRegistry(tmp_path)
    assert isinstance(reg, ExtensionSpecProvider)
    assert isinstance(reg, ExtensionResourceProvider)
    from linktools.ai.extension.scope import ExtensionScope
    from linktools.ai.extension.resource import ResourceRef

    content = await reg.read_resource(
        ResourceRef(scope=ExtensionScope("skill-creator"), path="SKILL.md")
    )
    assert content.size_bytes > 0
