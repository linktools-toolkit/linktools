#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PackageProvider works against any PackageResourceProvider / EntrypointResolver
implementation, not just the Directory defaults (spec §7)."""

import pytest

from linktools.ai.capability import CapabilityContext, CapabilityToolExposurePolicy
from linktools.ai.capability.ref import CapabilityRef
from linktools.ai.package.capability_provider import PackageProvider
from linktools.ai.package.scope import PackageScope


class _FakeResourceProvider:
    async def list_resources(self, scope, path="", *, limit=50, cursor=None):
        from linktools.ai.package.resource import ResourceInfo, ResourceListResult
        return ResourceListResult(items=[ResourceInfo(path="SKILL.md", kind="file", size_bytes=3)])

    async def read_resource(self, ref, *, max_bytes=None):
        from linktools.ai.package.resource import ResourceContent
        return ResourceContent(path="SKILL.md", content=b"abc", size_bytes=3)


class _FakeEntrypointResolver:
    async def list_entrypoints(self, scope, *, kind=None, limit=50, cursor=None):
        from linktools.ai.package.entrypoint import EntrypointInfo, EntrypointListResult
        return EntrypointListResult(items=[EntrypointInfo(kind="agent", name="grader")])

    async def resolve_agent(self, ref):
        raise NotImplementedError("not needed for this test")


def _ctx():
    return CapabilityContext(agent_id="a1", exposure_policy=CapabilityToolExposurePolicy())


@pytest.mark.asyncio
async def test_fake_resource_provider_works():
    provider = PackageProvider(resource_provider=_FakeResourceProvider())
    bundle = await provider.resolve(CapabilityRef("package-resource", "pkg"), _ctx())
    read_fn = bundle.toolsets[0].tools["read_package_resource"].function
    out = await read_fn("pkg", "SKILL.md")
    assert out["size_bytes"] == 3


@pytest.mark.asyncio
async def test_fake_entrypoint_resolver_works():
    provider = PackageProvider(entrypoint_resolver=_FakeEntrypointResolver())
    bundle = await provider.resolve(CapabilityRef("package-entrypoint", "pkg"), _ctx())
    list_fn = bundle.toolsets[0].tools["list_package_entrypoints"].function
    out = await list_fn("pkg", kind="agent")
    assert any(i["name"] == "grader" for i in out["items"])
