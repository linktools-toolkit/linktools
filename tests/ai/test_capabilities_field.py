#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RuntimeDependencies.capabilities lets a caller pass pre-built
CapabilityProviders that override the bundle-constructed ones."""

from linktools.ai.capability.exposure import CapabilityToolExposurePolicy
from linktools.ai.capability.models import CapabilityBundle
from linktools.ai.runtime import RuntimeDependencies


class _CustomProvider:
    """A custom provider for kind 'custom' that records it was consulted."""

    kind = "custom"
    supported_kinds = frozenset({"custom"})

    def __init__(self):
        self.called = False

    async def resolve(self, ref, context):
        self.called = True
        return CapabilityBundle(prompt_sections={"custom": "injected"})


def test_capabilities_field_registers_custom_provider():
    provider = _CustomProvider()
    bundle = RuntimeDependencies(capabilities=(provider,))
    assert not bundle.is_empty()

    from linktools.ai.runtime.builder import _build_capability_registry

    registry = _build_capability_registry(
        bundle,
        execution=None,
        options=CapabilityToolExposurePolicy().__class__(),
        mcp_manager=None,
    )
    assert registry is not None
    assert "custom" in registry.providers
    assert registry.providers["custom"] is provider
