#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CompositeSecurityPipeline: decision precedence and sequential MODIFY
threading (spec §12)."""

import pytest

from linktools.ai.security.pipeline import (
    CompositeSecurityPipeline,
    PipelineAction,
    PipelineDecision,
    ToolInvocationEvent,
    ToolResultEvent,
)


class _FixedPipeline:
    """Returns the same decision for every hook, for tests that only exercise
    one hook at a time."""

    def __init__(self, before_tool=None, after_tool=None):
        self._before_tool = before_tool or (lambda e: PipelineDecision(action=PipelineAction.ALLOW))
        self._after_tool = after_tool or (lambda e: PipelineDecision(action=PipelineAction.ALLOW))

    async def before_model(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
    async def after_model(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
    async def before_tool(self, e): return self._before_tool(e)
    async def after_tool(self, e): return self._after_tool(e)
    async def on_security_event(self, e): return PipelineDecision(action=PipelineAction.AUDIT_ONLY)


@pytest.mark.asyncio
async def test_later_deny_overrides_earlier_require_approval():
    approval = _FixedPipeline(
        before_tool=lambda e: PipelineDecision(action=PipelineAction.REQUIRE_APPROVAL))
    deny = _FixedPipeline(
        before_tool=lambda e: PipelineDecision(action=PipelineAction.DENY, reason="blocked"))
    composite = CompositeSecurityPipeline([approval, deny])
    decision = await composite.before_tool(
        ToolInvocationEvent(tool_name="t", arguments={}))
    assert decision.action == PipelineAction.DENY
    assert decision.reason == "blocked"


@pytest.mark.asyncio
async def test_require_approval_survives_when_no_later_deny():
    approval = _FixedPipeline(
        before_tool=lambda e: PipelineDecision(action=PipelineAction.REQUIRE_APPROVAL, reason="need ok"))
    allow = _FixedPipeline()
    composite = CompositeSecurityPipeline([approval, allow])
    decision = await composite.before_tool(
        ToolInvocationEvent(tool_name="t", arguments={}))
    assert decision.action == PipelineAction.REQUIRE_APPROVAL
    assert decision.reason == "need ok"


@pytest.mark.asyncio
async def test_sequential_modify_second_pipeline_sees_first_pipelines_edit():
    """The core §12.2 requirement: pipeline B's before_tool must observe
    pipeline A's modified arguments, not the original call."""
    seen_by_b = {}

    def _a_modify(e):
        return PipelineDecision(action=PipelineAction.MODIFY, modified_payload={**e.arguments, "a": 1})

    def _b_modify(e):
        seen_by_b.update(e.arguments)
        return PipelineDecision(action=PipelineAction.MODIFY, modified_payload={**e.arguments, "b": 2})

    pipeline_a = _FixedPipeline(before_tool=_a_modify)
    pipeline_b = _FixedPipeline(before_tool=_b_modify)
    composite = CompositeSecurityPipeline([pipeline_a, pipeline_b])

    decision = await composite.before_tool(
        ToolInvocationEvent(tool_name="t", arguments={"x": "orig"}))

    assert seen_by_b == {"x": "orig", "a": 1}, "pipeline B must see pipeline A's edit"
    assert decision.action == PipelineAction.ALLOW
    assert decision.modified_payload == {"x": "orig", "a": 1, "b": 2}


@pytest.mark.asyncio
async def test_sequential_modify_applies_to_result_field_for_after_tool():
    def _a_modify(e):
        return PipelineDecision(action=PipelineAction.MODIFY, modified_payload=f"{e.result}-a")

    def _b_modify(e):
        return PipelineDecision(action=PipelineAction.MODIFY, modified_payload=f"{e.result}-b")

    composite = CompositeSecurityPipeline([
        _FixedPipeline(after_tool=_a_modify),
        _FixedPipeline(after_tool=_b_modify),
    ])
    decision = await composite.after_tool(
        ToolResultEvent(tool_name="t", result="orig"))
    assert decision.modified_payload == "orig-a-b"


@pytest.mark.asyncio
async def test_after_tool_deny_result_short_circuits_like_deny():
    composite = CompositeSecurityPipeline([
        _FixedPipeline(after_tool=lambda e: PipelineDecision(action=PipelineAction.DENY_RESULT)),
        _FixedPipeline(after_tool=lambda e: PipelineDecision(action=PipelineAction.MODIFY, modified_payload="never")),
    ])
    decision = await composite.after_tool(ToolResultEvent(tool_name="t", result="orig"))
    assert decision.action == PipelineAction.DENY_RESULT


@pytest.mark.asyncio
async def test_after_tool_modify_result_treated_as_modify():
    composite = CompositeSecurityPipeline([
        _FixedPipeline(after_tool=lambda e: PipelineDecision(
            action=PipelineAction.MODIFY_RESULT, modified_payload="replaced")),
    ])
    decision = await composite.after_tool(ToolResultEvent(tool_name="t", result="orig"))
    assert decision.action == PipelineAction.ALLOW
    assert decision.modified_payload == "replaced"


@pytest.mark.asyncio
async def test_per_step_schema_validation_denies_bad_intermediate_modify():
    """A pipeline whose MODIFY produces schema-invalid arguments is DENYed at
    that step -- the bad payload never reaches the next pipeline. Validates
    between MODIFYs, not only at the final adapter check."""
    schema = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": []}

    def _bad_modify(e):
        # count must be int; inject a string.
        return PipelineDecision(action=PipelineAction.MODIFY, modified_payload={"count": "bad"})

    def _downstream(e):
        # This pipeline must never observe the invalid payload.
        raise AssertionError("downstream pipeline observed an invalid MODIFY payload")

    composite = CompositeSecurityPipeline([
        _FixedPipeline(before_tool=_bad_modify),
        _FixedPipeline(before_tool=_downstream),
    ])
    event = ToolInvocationEvent(tool_name="t", arguments={"count": 1}, parameter_schema=schema)
    decision = await composite.before_tool(event)
    assert decision.action == PipelineAction.DENY
    assert "schema validation" in (decision.reason or "")
