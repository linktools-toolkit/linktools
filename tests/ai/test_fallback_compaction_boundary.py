#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summary sub-agents account for fallback before terminal error projection."""

from collections.abc import Sequence
from types import SimpleNamespace

import pytest
from pydantic_ai import RunContext, messages as sdk_messages
from pydantic_ai.exceptions import FallbackExceptionGroup
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

from linktools.ai.core import PromptLimits
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ReferencedContent
from linktools.ai.runtime._compaction import CompactionCapability
from linktools.ai.runtime._journal import ModelRequestFact, ModelRequestJournal

from .test_referenced_input_failure import _REF, _SOURCE


class _RejectedSummary(TestModel):
    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart("rejected summary")],
            usage=RequestUsage(input_tokens=17, output_tokens=4),
        )


class _FailedSummary(TestModel):
    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        raise ValueError("provider repeated private referenced document")


def _reject_summary(response: ModelResponse) -> bool:
    return True


def _history(referenced: bool) -> list[ModelMessage]:
    messages: list[ModelMessage] = []
    for index in range(15):
        messages.append(ModelRequest(parts=[UserPromptPart(f"user {index} " + "x" * 200)]))
        messages.append(ModelResponse(parts=[TextPart(f"assistant {index} " + "y" * 200)]))
    if referenced:
        messages[0] = ModelRequest(parts=[UserPromptPart([
            "inspect", ReferencedContent(_REF, media_type="application/pdf", source=_SOURCE),
        ])])
    return messages


@pytest.mark.asyncio
@pytest.mark.parametrize("referenced", [False, True])
@pytest.mark.parametrize("limit", [False, True])
async def test_failed_summary_keeps_sdk_usage_before_projecting_without_an_outer_request_scope(
    referenced: bool, limit: bool,
) -> None:
    if getattr(sdk_messages, "ModelRequestAttempt", None) is None:
        pytest.skip("installed SDK does not expose fallback attempt observations")
    messages = _history(referenced)
    model = FallbackModel(_RejectedSummary(), _FailedSummary(), fallback_on=[_reject_summary, ValueError])
    usage = RunUsage()
    ctx = RunContext(
        deps=SimpleNamespace(correlation={}), model=model, usage=usage, run_step=1,
        usage_limits=UsageLimits(input_tokens_limit=10) if limit else None,
    )
    ctx.messages = list(messages)
    journal = ModelRequestJournal(
        source_namespace="workspace", tenant_id="tenant", execution_id="execution", agent_run_id="run",
    )
    observed: list[tuple[str, ModelRequestFact, BaseException | None]] = []

    async def observer(
        context: RunContext[object], fact: ModelRequestFact, phase: str,
        selected: Model, response: ModelResponse | None, error: BaseException | None,
    ) -> None:
        observed.append((phase, fact, error))

    async def handler(request: ModelRequestContext) -> ModelResponse:
        pytest.fail("the main model must not run after failed summary compaction")

    capability = CompactionCapability(1, limits=PromptLimits(), journal=journal, observer=observer)
    request = ModelRequestContext(
        model=model, messages=list(messages), model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    with pytest.raises(AIError) as caught:
        await capability.wrap_model_request(ctx, request_context=request, handler=handler)

    assert usage.input_tokens == 17
    assert usage.output_tokens == 4
    assert usage.requests == 0
    assert caught.value.code is (
        ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED if limit else ErrorCode.INTERNAL_ERROR
    )
    assert [phase for phase, _, _ in observed] == ["started", "failed"]
    assert isinstance(observed[-1][2], FallbackExceptionGroup)
    attempts = caught.value.safe_details["failed_attempts"]
    assert isinstance(attempts, Sequence)
    assert len(attempts) == 2
    assert attempts[0]["usage"]["input_tokens"] == 17
    assert attempts[1]["usage"] is None
    assert caught.value.safe_details["purpose"] == "compaction"
    assert caught.value.safe_details["model_request_seq"] == observed[-1][1].model_request_seq
    if referenced:
        assert all(attempt["error"] is None for attempt in attempts)
        assert caught.value.diagnostics.content_omission_reason == "referenced_content_lifecycle"
        assert "private referenced document" not in str(caught.value.diagnostics.to_payload())
    else:
        assert "private referenced document" in attempts[1]["error"]


@pytest.mark.asyncio
async def test_successful_summary_omits_attempt_content_using_original_referenced_input_policy() -> None:
    if getattr(sdk_messages, "ModelRequestAttempt", None) is None:
        pytest.skip("installed SDK does not expose fallback attempt observations")
    messages = _history(True)
    model = FallbackModel(_FailedSummary(), _RejectedSummary(), fallback_on=ValueError)
    usage = RunUsage()
    ctx = RunContext(deps=SimpleNamespace(correlation={}), model=model, usage=usage, run_step=1)
    ctx.messages = list(messages)
    journal = ModelRequestJournal(
        source_namespace="workspace", tenant_id="tenant", execution_id="execution", agent_run_id="run",
    )
    responses: list[ModelResponse] = []

    async def observer(
        context: RunContext[object], fact: ModelRequestFact, phase: str,
        selected: Model, response: ModelResponse | None, error: BaseException | None,
    ) -> None:
        if phase == "completed":
            assert response is not None
            assert response.failed_attempts[0].error is None
            responses.append(response)

    async def handler(request: ModelRequestContext) -> ModelResponse:
        return ModelResponse(parts=[TextPart("main response")])

    capability = CompactionCapability(1, limits=PromptLimits(), journal=journal, observer=observer)
    request = ModelRequestContext(
        model=model, messages=list(messages), model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    response = await capability.wrap_model_request(ctx, request_context=request, handler=handler)

    assert response.parts == [TextPart("main response")]
    assert len(responses) == 1
    assert responses[0].failed_attempts[0].usage is None
    assert responses[0].usage.input_tokens == usage.input_tokens == 17
    assert usage.requests == 1
