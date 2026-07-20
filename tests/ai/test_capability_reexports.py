#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""linktools.ai.capability already exports CapabilityRef/CapabilityRuntimeOptions/
CapabilityInspection/CapabilityProvider; this test proves
CapabilityToolExposurePolicy is now re-exported the same way, resolving to
the exact same object as the deep submodule import."""


def test_capability_tool_exposure_policy_reexport_identity():
    from linktools.ai.capability import CapabilityToolExposurePolicy as Shallow
    from linktools.ai.capability.exposure import CapabilityToolExposurePolicy as Deep
    assert Shallow is Deep


def test_existing_capability_exports_still_work():
    """Regression guard: this task must not remove or break existing exports."""
    from linktools.ai.capability import (
        CapabilityInspection,
        CapabilityProvider,
        CapabilityProviderRegistry,
        CapabilityRef,
        CapabilityRuntimeOptions,
    )
    assert all(
        symbol is not None
        for symbol in (
            CapabilityRef,
            CapabilityRuntimeOptions,
            CapabilityInspection,
            CapabilityProvider,
            CapabilityProviderRegistry,
        )
    )


def test_capability_provider_registry_reexport_identity():
    """CapabilityProviderRegistry is the runtime registry surface; it must
    re-export to the exact same object as the deep submodule import."""
    from linktools.ai.capability import CapabilityProviderRegistry as Shallow
    from linktools.ai.capability.registry import CapabilityProviderRegistry as Deep
    assert Shallow is Deep
