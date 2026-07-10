#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2A security execution: data model, policy merge, and ManagedToolAdapter
governance tests."""

import asyncio
import pytest

from linktools.ai.security.descriptor import ToolDescriptor, default_risk_for_category
from linktools.ai.security.pipeline import (
    PipelineAction, PipelineDecision, SecurityPipeline,
    ToolInvocationEvent, ToolResultEvent,
)
from linktools.ai.tool.contribution import ToolContribution
from linktools.ai.tool.managed import ManagedToolAdapter
from linktools.ai.tool.policy import (
    ResolvedToolPolicy, ToolInvocationContext, ToolPolicyProvider,
    merge_policies,
)
from linktools.ai.errors import ToolDeniedError, ToolTimeoutError


# --- ToolDescriptor ---

def test_descriptor_known_category_risk():
    assert default_risk_for_category("terminal") == "high"
    assert default_risk_for_category("file-read") == "low"
    assert default_risk_for_category("discovery") == "low"


def test_descriptor_unknown_category_conservative():
    assert default_risk_for_category("bogus") == "high"


def test_descriptor_with_capability_fields():
    d = ToolDescriptor(
        name="read_file", source="builtin", category="file-read",
        risk="low", mutating=False, capability_kind="builtin", capability_name="file-read",
    )
    assert d.capability_kind == "builtin"
    assert d.capability_name == "file-read"


# --- ResolvedToolPolicy validation ---

def test_policy_validates_timeout():
    with pytest.raises(ValueError, match="timeout"):
        ResolvedToolPolicy(timeout_seconds=-1)


def test_policy_validates_retries():
    with pytest.raises(ValueError, match="max_retries"):
        ResolvedToolPolicy(max_retries=-1)


def test_policy_default_allows():
    p = ResolvedToolPolicy()
    assert p.enabled and not p.require_approval


# --- merge_policies ---

def test_merge_enabled_any_false():
    result = merge_policies(
        ResolvedToolPolicy(enabled=True),
        ResolvedToolPolicy(enabled=False),
        None,
    )
    assert result.enabled is False


def test_merge_timeout_smallest():
    result = merge_policies(
        ResolvedToolPolicy(timeout_seconds=30),
        ResolvedToolPolicy(timeout_seconds=10),
        ResolvedToolPolicy(timeout_seconds=60),
    )
    assert result.timeout_seconds == 10


def test_merge_approval_any_true():
    result = merge_policies(
        ResolvedToolPolicy(),
        ResolvedToolPolicy(require_approval=True),
        None,
    )
    assert result.require_approval is True


def test_merge_risk_highest():
    result = merge_policies(
        ResolvedToolPolicy(risk="low"),
        ResolvedToolPolicy(risk="high"),
        ResolvedToolPolicy(risk="medium"),
    )
    assert result.risk == "high"


def test_merge_idempotent_all_true():
    result = merge_policies(
        ResolvedToolPolicy(idempotent=True),
        ResolvedToolPolicy(idempotent=True),
    )
    assert result.idempotent is True


def test_merge_idempotent_one_false():
    result = merge_policies(
        ResolvedToolPolicy(idempotent=True),
        ResolvedToolPolicy(idempotent=False),
    )
    assert result.idempotent is False


# --- ManagedToolAdapter ---

def _descriptor(name="test_tool", category="discovery", risk="low", mutating=False):
    return ToolDescriptor(name=name, source="builtin", category=category,
                          risk=risk, mutating=mutating)


@pytest.mark.asyncio
async def test_adapter_success():
    async def handler(x: str = "default") -> str:
        return f"done:{x}"

    adapter = ManagedToolAdapter(descriptor=_descriptor(), handler=handler)
    result = await adapter.invoke(x="hello")
    assert result == "done:hello"


@pytest.mark.asyncio
async def test_adapter_pipeline_deny():
    class _DenyPipeline:
        async def before_model(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
        async def after_model(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
        async def before_tool(self, e):
            return PipelineDecision(action=PipelineAction.DENY, reason="blocked")
        async def after_tool(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
        async def on_security_event(self, e): return PipelineDecision(action=PipelineAction.AUDIT_ONLY)

    async def handler(): return "should not reach"

    adapter = ManagedToolAdapter(
        descriptor=_descriptor(), handler=handler,
        security_pipeline=_DenyPipeline(),
    )
    with pytest.raises(ToolDeniedError, match="blocked"):
        await adapter.invoke()


@pytest.mark.asyncio
async def test_adapter_pipeline_require_approval():
    class _ApprovalPipeline:
        async def before_model(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
        async def after_model(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
        async def before_tool(self, e):
            return PipelineDecision(action=PipelineAction.REQUIRE_APPROVAL)
        async def after_tool(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
        async def on_security_event(self, e): return PipelineDecision(action=PipelineAction.AUDIT_ONLY)

    async def handler(): return "should not reach"

    adapter = ManagedToolAdapter(
        descriptor=_descriptor(), handler=handler,
        security_pipeline=_ApprovalPipeline(),
    )
    with pytest.raises(ToolDeniedError, match="approval"):
        await adapter.invoke()


@pytest.mark.asyncio
async def test_adapter_pipeline_modify_args():
    class _ModifyPipeline:
        async def before_model(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
        async def after_model(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
        async def before_tool(self, e):
            return PipelineDecision(action=PipelineAction.MODIFY, modified_payload={"x": "modified"})
        async def after_tool(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
        async def on_security_event(self, e): return PipelineDecision(action=PipelineAction.AUDIT_ONLY)

    async def handler(x: str = "") -> str:
        return f"got:{x}"

    adapter = ManagedToolAdapter(
        descriptor=_descriptor(), handler=handler,
        security_pipeline=_ModifyPipeline(),
    )
    result = await adapter.invoke(x="original")
    assert result == "got:modified"


@pytest.mark.asyncio
async def test_adapter_timeout():
    async def slow_handler():
        await asyncio.sleep(10)
        return "done"

    adapter = ManagedToolAdapter(
        descriptor=_descriptor(), handler=handler if False else slow_handler,
        baseline_policy=ResolvedToolPolicy(timeout_seconds=0.01),
    )
    with pytest.raises(ToolTimeoutError):
        await adapter.invoke()


@pytest.mark.asyncio
async def test_adapter_policy_disabled():
    async def handler(): return "done"

    adapter = ManagedToolAdapter(
        descriptor=_descriptor(), handler=handler,
        baseline_policy=ResolvedToolPolicy(enabled=False),
    )
    with pytest.raises(ToolDeniedError, match="disabled"):
        await adapter.invoke()


@pytest.mark.asyncio
async def test_adapter_after_tool_deny():
    class _AfterDenyPipeline:
        async def before_model(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
        async def after_model(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
        async def before_tool(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
        async def after_tool(self, e):
            return PipelineDecision(action=PipelineAction.DENY, reason="result blocked")
        async def on_security_event(self, e): return PipelineDecision(action=PipelineAction.AUDIT_ONLY)

    async def handler(): return "executed"

    adapter = ManagedToolAdapter(
        descriptor=_descriptor(), handler=handler,
        security_pipeline=_AfterDenyPipeline(),
    )
    with pytest.raises(ToolDeniedError, match="result denied"):
        await adapter.invoke()


# --- ToolContribution ---

def test_tool_contribution():
    d = _descriptor()
    tc = ToolContribution(toolset=object(), descriptors=(d,))
    assert len(tc.descriptors) == 1
    assert tc.descriptors[0].name == "test_tool"
