#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""call_package_entrypoint must execute or raise -- never return a fake-success
'reserved' marker (spec §9)."""

import pytest

from linktools.ai.capability import CapabilityContext, CapabilityToolExposurePolicy
from linktools.ai.capability.ref import CapabilityRef
from linktools.ai.errors import PackageEntrypointDeniedError
from linktools.ai.package.capability_provider import PackageProvider


def _ctx(execution_tools=True):
    return CapabilityContext(
        agent_id="a1",
        exposure_policy=CapabilityToolExposurePolicy(expose_execution_tools=execution_tools),
    )


@pytest.mark.asyncio
async def test_call_without_executor_raises_not_reserved():
    # expose_call_tool=True but no entrypoint_executor wired -> denial, not
    # a {"status": "reserved"} fake-success.
    provider = PackageProvider(entrypoint_resolver=object())  # resolver set, executor None
    ref = CapabilityRef("package-entrypoint", "pkg", config={
        "allowed_kinds": ["agent"], "allowed_names": ["grader"], "expose_call_tool": True,
    })
    bundle = await provider.resolve(ref, _ctx())
    call = bundle.toolsets[0].tools["call_package_entrypoint"].function
    with pytest.raises(Exception):  # PackageEntrypointDeniedError (no executor)
        await call("pkg", "agent", "grader", "do it")


def test_no_reserved_marker_in_source():
    import pathlib
    src = pathlib.Path("linktools-ai/src/linktools/ai/package/toolset.py").read_text()
    assert '"status": "reserved"' not in src
    assert "status=\"reserved\"" not in src


@pytest.mark.asyncio
async def test_call_without_resolver_raises_not_nameerror():
    # resolver=None must raise PackageEntrypointNotFoundError, not NameError
    # from a missing import (the bug the review caught). Exercise the toolset
    # directly since PackageProvider guards resolver=None -> empty bundle.
    from linktools.ai.errors import PackageEntrypointNotFoundError
    from linktools.ai.package.scope import PackageScope
    from linktools.ai.package.toolset import build_package_entrypoint_toolset
    ts = build_package_entrypoint_toolset(
        resolver=None, allowed={"pkg": PackageScope("pkg")},
        allowed_kinds=("agent",), allowed_names=("grader",),
        expose_call_tool=True, executor=object(),
    )
    call = ts.tools["call_package_entrypoint"].function
    with pytest.raises(PackageEntrypointNotFoundError):
        await call("pkg", "agent", "grader", "do it")
