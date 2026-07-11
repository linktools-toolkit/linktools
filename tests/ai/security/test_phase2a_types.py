#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""security execution security execution: data model, policy merge, and ManagedToolAdapter
governance tests."""

import asyncio
import pytest

from linktools.ai.security.descriptor import ToolDescriptor, default_risk_for_category
from linktools.ai.security.pipeline import (
    PipelineAction, PipelineDecision, SecurityPipeline,
    ToolInvocationEvent, ToolResultEvent,
)
from linktools.ai.errors import RunPaused, ToolDeniedError, ToolTimeoutError
from linktools.ai.tool.contribution import ToolContribution
from linktools.ai.tool.managed import ManagedToolAdapter
from linktools.ai.tool.policy import (
    EffectiveToolPolicy, ResolvedToolPolicy, ToolInvocationContext,
    ToolPolicyProvider, finalize_policy, merge_policies,
)


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


def test_policy_default_is_undeclared_tri_state():
    """A bare ResolvedToolPolicy layer declares nothing -- every field stays
    None ("not declared"), distinguishable from an explicit 0/False."""
    p = ResolvedToolPolicy()
    assert p.enabled is None
    assert p.max_retries is None
    assert p.idempotent is None
    assert p.require_approval is None


def test_finalize_undeclared_policy_defaults_open_but_not_unsafe():
    """finalize_policy() collapses "nothing declared" to the safe concrete
    defaults: enabled (tools aren't disabled by omission), but idempotent/
    require_approval stay closed (an absent layer must never be read as an
    implicit retry/approval-skip opt-in)."""
    effective = finalize_policy(merge_policies(None, None, None))
    assert effective == EffectiveToolPolicy()
    assert effective.enabled is True
    assert effective.idempotent is False
    assert effective.require_approval is False
    assert effective.max_retries == 0


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


def test_merge_max_retries_survives_undeclared_layers():
    """Regression: prior non-tri-state defaults (max_retries=0) made an
    undeclared layer indistinguishable from an explicit 0, so merge (min())
    could never raise max_retries above 0. A provider-declared positive
    max_retries must now survive an undeclared descriptor/baseline layer."""
    result = merge_policies(None, None, ResolvedToolPolicy(max_retries=3))
    assert result.max_retries == 3
    assert finalize_policy(result).max_retries == 3


def test_merge_idempotent_survives_undeclared_layers():
    """Same regression for idempotent: an undeclared baseline layer must not
    force idempotent back to False when the provider explicitly declares True."""
    result = merge_policies(None, ResolvedToolPolicy(), ResolvedToolPolicy(idempotent=True))
    assert result.idempotent is True
    assert finalize_policy(result).idempotent is True


def test_merge_metadata_not_silently_dropped():
    result = merge_policies(
        ResolvedToolPolicy(metadata={"a": 1}),
        ResolvedToolPolicy(metadata={"b": 2}),
        ResolvedToolPolicy(metadata={"c": 3}),
    )
    assert result.metadata == {"a": 1, "b": 2, "c": 3}


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
    with pytest.raises(RunPaused, match="approval"):
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
async def test_adapter_modify_revalidates_against_schema_and_denies_invalid():
    """A pipeline MODIFY that produces schema-invalid arguments is rejected
    (fail closed) -- a misbehaving pipeline cannot inject an unsafe payload."""
    class _BadModify:
        async def before_model(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
        async def after_model(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
        async def before_tool(self, e):
            # `count` must be an integer per the schema; inject a string.
            return PipelineDecision(action=PipelineAction.MODIFY, modified_payload={"count": "not-an-int"})
        async def after_tool(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
        async def on_security_event(self, e): return PipelineDecision(action=PipelineAction.AUDIT_ONLY)

    async def handler(count: int = 0) -> int:
        return count

    # Schema: count must be an integer.
    schema = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": []}
    adapter = ManagedToolAdapter(
        descriptor=_descriptor(), handler=handler, security_pipeline=_BadModify())
    from linktools.ai.errors import ToolSchemaValidationError
    with pytest.raises(ToolSchemaValidationError, match="schema validation"):
        await adapter.invoke(parameter_schema=schema, count=1)


@pytest.mark.asyncio
async def test_adapter_modify_allows_schema_valid_payload():
    class _GoodModify:
        async def before_model(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
        async def after_model(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
        async def before_tool(self, e):
            return PipelineDecision(action=PipelineAction.MODIFY, modified_payload={"count": 42})
        async def after_tool(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
        async def on_security_event(self, e): return PipelineDecision(action=PipelineAction.AUDIT_ONLY)

    async def handler(count: int = 0) -> int:
        return count

    schema = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": []}
    adapter = ManagedToolAdapter(
        descriptor=_descriptor(), handler=handler, security_pipeline=_GoodModify())
    assert await adapter.invoke(parameter_schema=schema, count=1) == 42


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
            return PipelineDecision(action=PipelineAction.DENY_RESULT, reason="result blocked")
        async def on_security_event(self, e): return PipelineDecision(action=PipelineAction.AUDIT_ONLY)

    async def handler(): return "executed"

    adapter = ManagedToolAdapter(
        descriptor=_descriptor(), handler=handler,
        security_pipeline=_AfterDenyPipeline(),
    )
    with pytest.raises(ToolDeniedError, match="result denied"):
        await adapter.invoke()


@pytest.mark.asyncio
async def test_adapter_after_tool_modify_result_replaces_result():
    """after_tool MODIFY_RESULT replaces the tool's result (previously the
    after_tool MODIFY payload was silently ignored)."""
    class _AfterModifyPipeline:
        async def before_model(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
        async def after_model(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
        async def before_tool(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
        async def after_tool(self, e):
            return PipelineDecision(action=PipelineAction.MODIFY_RESULT, modified_payload="redacted")
        async def on_security_event(self, e): return PipelineDecision(action=PipelineAction.AUDIT_ONLY)

    async def handler(): return "secret-output"

    adapter = ManagedToolAdapter(
        descriptor=_descriptor(), handler=handler,
        security_pipeline=_AfterModifyPipeline(),
    )
    assert await adapter.invoke() == "redacted"


@pytest.mark.asyncio
async def test_adapter_provider_failure_emits_degraded_event_and_denies():
    """A ToolPolicyProvider failure fails closed (deny) AND emits a
    SecurityDegraded event so the silent fallback is observable."""
    from linktools.ai.tool.policy import ToolPolicyProvider as _Proto

    class _BoomProvider:
        async def resolve(self, descriptor, context):
            raise RuntimeError("provider down")

    class _RecordingStore:
        def __init__(self): self.events = []
        async def append(self, **kw): self.events.append(kw)

    async def handler(): return "should not reach"

    store = _RecordingStore()
    from linktools.ai.run.context import RunContext
    from linktools.ai.run.models import RunnableType
    ctx = RunContext(
        run_id="r1", root_run_id="r1", parent_run_id=None, session_id="s1",
        runnable_id="a1", runnable_type=RunnableType.AGENT,
        user_id=None, tenant_id=None, workspace=None,
    )
    adapter = ManagedToolAdapter(
        descriptor=_descriptor(), handler=handler,
        policy_provider=_BoomProvider(), event_store=store, run_context=ctx,
    )
    with pytest.raises(ToolDeniedError, match="policy resolution failed"):
        await adapter.invoke()
    assert any(e["payload"].__class__.__name__ == "SecurityDegraded" for e in store.events)


# --- ToolContribution ---

def test_tool_contribution():
    d = _descriptor()
    tc = ToolContribution(toolset=object(), descriptors=(d,))
    assert len(tc.descriptors) == 1
    assert tc.descriptors[0].name == "test_tool"
