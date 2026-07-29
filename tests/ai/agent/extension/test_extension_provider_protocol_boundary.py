#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ExtensionProvider works against any ExtensionContentSource / EntrypointResolver
implementation, not just the Directory defaults (contract)."""

import pytest

from linktools.ai.agent.tool.exposure import ToolExposurePolicy
from linktools.ai.agent.assembly.provider import AgentFeatureContext
from linktools.ai.agent.assembly.models import AgentFeatureRef
from linktools.ai.agent.extension.provider import ExtensionProvider


class _FakeResourceProvider:
    async def list_entries(self, scope, path="", *, limit=50, cursor=None):
        from linktools.ai.agent.extension.content import AssetInfo, ExtensionContentPage

        return ExtensionContentPage(
            items=[AssetInfo(path="SKILL.md", kind="file", size_bytes=3)]
        )

    async def read_content(self, ref, *, max_bytes=None):
        from linktools.ai.agent.extension.content import ExtensionContent

        return ExtensionContent(path="SKILL.md", content=b"abc", size_bytes=3)


class _FakeEntrypointResolver:
    async def list_entrypoints(self, scope, *, kind=None, limit=50, cursor=None):
        from linktools.ai.agent.extension.entrypoint import EntrypointInfo, EntrypointListResult

        return EntrypointListResult(items=[EntrypointInfo(kind="agent", name="grader")])

    async def resolve_agent(self, ref):
        raise NotImplementedError("not needed for this test")


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
async def test_fake_content_source_works():
    provider = ExtensionProvider(content_source=_FakeResourceProvider())
    bundle = await provider.resolve(AgentFeatureRef("extension-asset", "pkg"), _ctx())
    read_fn = next(
        md.handler
        for md in bundle.tools
        if md.descriptor.name == "read_extension_content"
    )
    out = await read_fn("pkg", "SKILL.md")
    assert out["size_bytes"] == 3


@pytest.mark.asyncio
async def test_fake_entrypoint_resolver_works():
    provider = ExtensionProvider(entrypoint_resolver=_FakeEntrypointResolver())
    bundle = await provider.resolve(AgentFeatureRef("extension-entrypoint", "pkg"), _ctx())
    list_fn = next(
        md.handler
        for md in bundle.tools
        if md.descriptor.name == "list_extension_entrypoints"
    )
    out = await list_fn("pkg", kind="agent")
    assert any(i["name"] == "grader" for i in out["items"])
