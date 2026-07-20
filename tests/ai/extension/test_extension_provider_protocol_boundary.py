#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ExtensionProvider works against any ExtensionResourceProvider / EntrypointResolver
implementation, not just the Directory defaults (contract)."""

import pytest

from linktools.ai.capability.exposure import CapabilityToolExposurePolicy
from linktools.ai.capability.provider import CapabilityContext
from linktools.ai.capability.models import CapabilityRef
from linktools.ai.extension.capability_provider import ExtensionProvider


class _FakeResourceProvider:
    async def list_resources(self, scope, path="", *, limit=50, cursor=None):
        from linktools.ai.extension.resource import AssetInfo, ResourceListResult

        return ResourceListResult(
            items=[AssetInfo(path="SKILL.md", kind="file", size_bytes=3)]
        )

    async def read_resource(self, ref, *, max_bytes=None):
        from linktools.ai.extension.resource import ResourceContent

        return ResourceContent(path="SKILL.md", content=b"abc", size_bytes=3)


class _FakeEntrypointResolver:
    async def list_entrypoints(self, scope, *, kind=None, limit=50, cursor=None):
        from linktools.ai.extension.entrypoint import EntrypointInfo, EntrypointListResult

        return EntrypointListResult(items=[EntrypointInfo(kind="agent", name="grader")])

    async def resolve_agent(self, ref):
        raise NotImplementedError("not needed for this test")


def _ctx():
    return CapabilityContext(
        agent_id="a1", exposure_policy=CapabilityToolExposurePolicy()
    )


@pytest.mark.asyncio
async def test_fake_resource_provider_works():
    provider = ExtensionProvider(resource_provider=_FakeResourceProvider())
    bundle = await provider.resolve(CapabilityRef("extension-resource", "pkg"), _ctx())
    read_fn = next(
        md.handler
        for c in bundle.tool_contributions
        for md in c.tools
        if md.descriptor.name == "read_extension_resource"
    )
    out = await read_fn("pkg", "SKILL.md")
    assert out["size_bytes"] == 3


@pytest.mark.asyncio
async def test_fake_entrypoint_resolver_works():
    provider = ExtensionProvider(entrypoint_resolver=_FakeEntrypointResolver())
    bundle = await provider.resolve(CapabilityRef("extension-entrypoint", "pkg"), _ctx())
    list_fn = next(
        md.handler
        for c in bundle.tool_contributions
        for md in c.tools
        if md.descriptor.name == "list_extension_entrypoints"
    )
    out = await list_fn("pkg", kind="agent")
    assert any(i["name"] == "grader" for i in out["items"])
