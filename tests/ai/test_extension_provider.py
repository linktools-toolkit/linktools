#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ExtensionProvider (capability integration) + ExtensionRegistry (contract/contract):
catalog-only for `extension:`, read tools for `extension-asset:`, list tool for
`extension-entrypoint:`; call stays opt-in; ExtensionRegistry lists extensions."""

import pytest

from linktools.ai.agent.tool.exposure import ToolExposurePolicy
from linktools.ai.agent.assembly.provider import AgentFeatureContext
from linktools.ai.agent.assembly.models import AgentFeatureRef
from linktools.ai.errors import ExtensionContentAccessDeniedError, ExtensionNotFoundError
from linktools.ai.agent.extension.provider import ExtensionProvider
from linktools.ai.agent.extension.content_source import DirectoryExtensionContentSource
from linktools.ai.agent.extension.resolver import (
    DirectoryEntrypointResolver,
    DirectoryExtensionRegistry,
    ExtensionRegistry,
)
from linktools.ai.agent.extension.scope import ExtensionScope


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
    rp = DirectoryExtensionContentSource({"skill-creator": root})
    er = DirectoryEntrypointResolver({"skill-creator": root})
    return ExtensionProvider(content_source=rp, entrypoint_resolver=er), root


def _ctx():
    return AgentFeatureContext(
        agent_id="a1",
        execution_id="e1",
        root_execution_id="e1",
        parent_execution_id=None,
        session_id="s1",
        tenant_id="t1",
        user_id="u1",
        workspace=None,
        sandbox=None,
    )


@pytest.mark.asyncio
async def test_extension_kind_is_prompt_catalog_only(env):
    provider, _ = env
    bundle = await provider.resolve(AgentFeatureRef("extension", "skill-creator"), _ctx())
    assert "extensions" in bundle.prompt_sections


@pytest.mark.asyncio
async def test_extension_resource_exposes_read_tools(env):
    provider, _ = env
    bundle = await provider.resolve(
        AgentFeatureRef("extension-asset", "skill-creator"), _ctx()
    )
    names = {md.descriptor.name for md in bundle.tools}
    assert names == {"list_extension_content", "read_extension_content"}


@pytest.mark.asyncio
async def test_extension_resource_read_tool_authorized_and_sandboxed(env):
    provider, _ = env
    bundle = await provider.resolve(
        AgentFeatureRef("extension-asset", "skill-creator"), _ctx()
    )
    tools = {
        md.descriptor.name: md.handler
        for md in bundle.tools
    }
    list_fn = tools["list_extension_content"]
    read_fn = tools["read_extension_content"]
    # declared extension -> allowed
    listing = await list_fn("skill-creator", "")
    assert any("SKILL.md" in i["path"] for i in listing["items"])
    content = await read_fn("skill-creator", "SKILL.md")
    assert content["size_bytes"] > 0
    # undeclared extension -> denied
    with pytest.raises(ExtensionContentAccessDeniedError):
        await read_fn("other-pkg", "SKILL.md")


@pytest.mark.asyncio
async def test_extension_entrypoint_lists_only_by_default(env):
    provider, _ = env
    bundle = await provider.resolve(
        AgentFeatureRef("extension-entrypoint", "skill-creator"), _ctx()
    )
    names = {md.descriptor.name for md in bundle.tools}
    # Default: only list, no call tool.
    assert names == {"list_extension_entrypoints"}
    list_fn = next(
        md.handler
        for md in bundle.tools
        if md.descriptor.name == "list_extension_entrypoints"
    )
    result = await list_fn("skill-creator", kind="agent")
    assert any(i["name"] == "grader" for i in result["items"])


@pytest.mark.asyncio
async def test_extension_entrypoint_call_is_opt_in(env):
    provider, _ = env
    ctx = _ctx()
    ref = AgentFeatureRef(
        "extension-entrypoint",
        "skill-creator",
        config={
            "allowed_kinds": ["agent"],
            "allowed_names": ["grader"],
            "expose_call_tool": True,
        },
    )
    bundle = await provider.resolve(ref, ctx)
    names = {md.descriptor.name for md in bundle.tools}
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
async def test_extension_registry_implements_content_source(tmp_path):
    # contract: ExtensionRegistry satisfies BOTH ExtensionSpecProvider and
    # ExtensionContentSource.
    from linktools.ai.agent.extension.spec import (
        ExtensionContentSource,
        ExtensionSpecProvider,
    )
    from linktools.ai.agent.extension.resolver import ExtensionRegistry

    root = tmp_path / "skill-creator"
    root.mkdir()
    (root / "SKILL.md").write_text("# s", encoding="utf-8")
    reg = ExtensionRegistry(tmp_path)
    assert isinstance(reg, ExtensionSpecProvider)
    assert isinstance(reg, ExtensionContentSource)
    from linktools.ai.agent.extension.scope import ExtensionScope
    from linktools.ai.agent.extension.content import ExtensionContentRef

    content = await reg.read_content(
        ExtensionContentRef(scope=ExtensionScope("skill-creator"), path="SKILL.md")
    )
    assert content.size_bytes > 0
