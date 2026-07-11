#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MetadataBackedPolicyProvider: the only real ToolPolicyProvider Runtime.build
wires. Verifies it produces tri-state ResolvedToolPolicy layers (None for
fields the source ToolSpec has no opinion on) so merge_policies can never be
clamped by a phantom concrete default (contract/contract)."""

import pytest

from linktools.ai.policy.rule import (
    ApprovalMode, Permission, RiskLevel, SideEffectKind, ToolPolicyMetadata,
)
from linktools.ai.security.descriptor import ToolDescriptor
from linktools.ai.tool.policy_adapter import MetadataBackedPolicyProvider


def _descriptor(name="my_tool"):
    return ToolDescriptor(
        name=name, source="builtin", category="file-read",
        risk="low", mutating=False,
    )


class _MetadataSource:
    def __init__(self, mapping):
        self._mapping = mapping

    async def get_metadata_map(self):
        return self._mapping


@pytest.mark.asyncio
async def test_known_tool_maps_risk_approval_idempotent_timeout():
    meta = ToolPolicyMetadata(
        permissions=frozenset({Permission.WRITE}),
        risk=RiskLevel.HIGH,
        side_effect=SideEffectKind.NAMESPACE_MUTATING,
        approval=ApprovalMode.ON_RISK,
    )
    provider = MetadataBackedPolicyProvider(_MetadataSource({"my_tool": meta}))
    policy = await provider.resolve(_descriptor(), context=None)
    assert policy.risk == "high"
    assert policy.require_approval is True
    assert policy.idempotent is False


@pytest.mark.asyncio
async def test_known_tool_leaves_max_retries_undeclared():
    """Regression: ToolSpec has no max_retries concept. The provider MUST
    declare None (not 0) so merge_policies' min-of-declared rule does not
    clamp every tool's retries to 0 regardless of what a baseline/descriptor
    layer wants. A concrete 0 here would silently defeat contract's tri-state fix
    end-to-end on the one production path."""
    meta = ToolPolicyMetadata(
        permissions=frozenset({Permission.READ}),
        risk=RiskLevel.LOW,
        side_effect=SideEffectKind.READ_ONLY,
        approval=ApprovalMode.NEVER,
    )
    provider = MetadataBackedPolicyProvider(_MetadataSource({"my_tool": meta}))
    policy = await provider.resolve(_descriptor(), context=None)
    assert policy.max_retries is None


@pytest.mark.asyncio
async def test_schema_version_is_a_policy_field_not_metadata():
    """The version is carried as the policy authority used by idempotency."""
    meta = ToolPolicyMetadata(
        permissions=frozenset({Permission.READ}),
        risk=RiskLevel.LOW,
        side_effect=SideEffectKind.READ_ONLY,
        approval=ApprovalMode.NEVER,
        schema_version="7",
        metadata={"team": "platform"},
    )
    provider = MetadataBackedPolicyProvider(_MetadataSource({"my_tool": meta}))
    policy = await provider.resolve(_descriptor(), context=None)
    assert policy.schema_version == "7"
    assert "schema_version" not in policy.metadata
    assert policy.metadata["source_metadata"] == {"team": "platform"}
    assert "permissions" in policy.metadata and "side_effect" in policy.metadata


@pytest.mark.asyncio
async def test_unknown_tool_returns_all_undeclared_layer():
    """A tool absent from the metadata map contributes no opinion -- every
    field None, so it can never override a more specific layer."""
    provider = MetadataBackedPolicyProvider(_MetadataSource({}))
    policy = await provider.resolve(_descriptor("not_listed"), context=None)
    assert policy.max_retries is None
    assert policy.idempotent is None
    assert policy.require_approval is None
    assert policy.enabled is None


@pytest.mark.asyncio
async def test_provider_failure_raises_policy_resolution_error():
    """A metadata-source failure fails closed: the provider raises
    ToolPolicyResolutionError (the contract default) so the ManagedToolAdapter can
    emit a SecurityDegraded event and deny the call -- it never returns a
    silently-degraded policy."""
    from linktools.ai.errors import ToolPolicyResolutionError

    class _Boom:
        async def get_metadata_map(self):
            raise RuntimeError("provider down")

    provider = MetadataBackedPolicyProvider(_Boom())
    with pytest.raises(ToolPolicyResolutionError):
        await provider.resolve(_descriptor(), context=None)
