#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CapabilityToolExposurePolicy (spec §11.4): conservative defaults + immutability."""

from linktools.ai.capability import CapabilityToolExposurePolicy


def test_defaults_are_conservative():
    p = CapabilityToolExposurePolicy()
    assert p.expose_prompt_catalog is True
    assert p.expose_discovery_tools is True
    # Execution tools must NOT be on by default.
    assert p.expose_execution_tools is False
    assert p.max_tools_total == 64
    assert p.max_tools_per_capability == 16
    assert p.max_resources_per_list == 50
    assert p.max_read_bytes == 65536
    assert p.max_entrypoints_per_package == 20
    assert p.allowed_entrypoint_kinds == ("agent",)
    assert p.require_explicit_entrypoint_allowlist is True


def test_policy_is_frozen():
    import pytest
    p = CapabilityToolExposurePolicy()
    with pytest.raises(Exception):
        p.expose_execution_tools = True  # type: ignore[misc]


def test_policy_overridable_via_constructor():
    p = CapabilityToolExposurePolicy(expose_execution_tools=True, max_tools_total=8)
    assert p.expose_execution_tools is True
    assert p.max_tools_total == 8
