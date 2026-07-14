#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SEC-02 (v5 guide §6): every model security-pipeline decision must be honored
-- MODIFY rewrites the prompt actually sent, MODIFY_RESULT rewrites the output
the caller sees, DENY/DENY_RESULT stops the call, an invalid action fails
closed, and a configured pipeline never lets an un-audited token leak through
request_stream.

Each test drives a SecuredModel directly with a recording delegate Model so the
prompt the delegate received and the output the caller got are both observable.
The MODIFY / MODIFY_RESULT / streaming tests fail before the fix (the decision
is ignored and the raw SECRET passes through)."""

from contextlib import asynccontextmanager
from typing import Any

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import Model

from linktools.ai.errors import (
    ModelInvocationDeniedError,
    ModelResultDeniedError,
    PipelineExecutionError,
)
from linktools.ai.security.pipeline import PipelineAction, PipelineDecision
from linktools.ai.security.secured_model import SecuredModel

_ALLOW = PipelineDecision(action=PipelineAction.ALLOW)


class _Script:
    """Pipeline whose before_model/after_model return canned decisions."""

    def __init__(
        self,
        *,
        before: PipelineDecision | None = None,
        after: PipelineDecision | None = None,
    ) -> None:
        self.before = before
        self.after = after

    async def before_model(self, e):  # noqa: ARG002
        return self.before or _ALLOW

    async def after_model(self, e):  # noqa: ARG002
        return self.after or _ALLOW

    async def before_tool(self, e):  # noqa: ARG002
        return _ALLOW

    async def after_tool(self, e):  # noqa: ARG002
        return _ALLOW

    async def on_security_event(self, e):  # noqa: ARG002
        return PipelineDecision(action=PipelineAction.AUDIT_ONLY)


class _RecordingDelegate(Model):
    """Minimal concrete Model: records the messages it received and returns a
    canned ModelResponse. ``request_stream`` is unused by the buffered path."""

    def __init__(self, output_text: str = "DELEGATE-OUT") -> None:
        object.__setattr__(self, "_output_text", output_text)
        object.__setattr__(self, "sent_messages", None)

    @property
    def model_name(self) -> str:
        return "fake"

    @property
    def system(self) -> str:
        return ""

    async def request(self, messages, model_settings, model_request_parameters):  # noqa: ARG002
        object.__setattr__(self, "sent_messages", list(messages))
        return ModelResponse(parts=[TextPart(content=self._output_text)])

    @asynccontextmanager
    async def request_stream(
        self, messages, model_settings, model_request_parameters, run_context=None
    ):  # noqa: ARG002
        yield self  # never used by the buffered (pipeline-enabled) path

    async def get(self) -> ModelResponse:  # pragma: no cover - stand-in only
        return ModelResponse(parts=[TextPart(content=self._output_text)])


def _messages(prompt: str) -> list:
    return [ModelRequest(parts=[UserPromptPart(content=prompt)])]


def _last_user_text(messages: Any) -> str:
    for msg in reversed(list(messages or [])):
        for part in reversed(getattr(msg, "parts", None) or []):
            if isinstance(part, UserPromptPart):
                return str(part.content)
    return ""


def _response_text(response: ModelResponse) -> str:
    return "\n".join(
        str(p.content)
        for p in getattr(response, "parts", None) or []
        if isinstance(p, TextPart)
    )


@pytest.mark.asyncio
async def test_before_modify_rewrites_prompt_sent_to_delegate():
    delegate = _RecordingDelegate(output_text="ok")
    pipeline = _Script(
        before=PipelineDecision(
            action=PipelineAction.MODIFY, modified_payload="[REDACTED]"
        )
    )
    secured = SecuredModel(delegate, pipeline=pipeline, run_id="r", agent_id="a")

    await secured.request(_messages("SECRET-PROMPT"), None, None)

    assert _last_user_text(delegate.sent_messages) == "[REDACTED]"
    assert "SECRET-PROMPT" not in _last_user_text(delegate.sent_messages)


@pytest.mark.asyncio
async def test_after_modify_result_rewrites_output_seen_by_caller():
    delegate = _RecordingDelegate(output_text="SECRET-OUTPUT")
    pipeline = _Script(
        after=PipelineDecision(
            action=PipelineAction.MODIFY_RESULT, modified_payload="[REDACTED]"
        )
    )
    secured = SecuredModel(delegate, pipeline=pipeline, run_id="r", agent_id="a")

    response = await secured.request(_messages("hi"), None, None)

    assert _response_text(response) == "[REDACTED]"
    assert "SECRET" not in _response_text(response)


@pytest.mark.asyncio
async def test_before_deny_skips_delegate_call():
    delegate = _RecordingDelegate(output_text="ok")
    pipeline = _Script(before=PipelineDecision(action=PipelineAction.DENY))
    secured = SecuredModel(delegate, pipeline=pipeline, run_id="r", agent_id="a")

    with pytest.raises(ModelInvocationDeniedError):
        await secured.request(_messages("hi"), None, None)

    assert delegate.sent_messages is None, "delegate must not be called on DENY"


@pytest.mark.asyncio
async def test_after_deny_result_raises_and_hides_output():
    delegate = _RecordingDelegate(output_text="SECRET")
    pipeline = _Script(after=PipelineDecision(action=PipelineAction.DENY_RESULT))
    secured = SecuredModel(delegate, pipeline=pipeline, run_id="r", agent_id="a")

    with pytest.raises(ModelResultDeniedError):
        await secured.request(_messages("hi"), None, None)


@pytest.mark.asyncio
async def test_before_require_approval_fails_closed():
    delegate = _RecordingDelegate(output_text="ok")
    pipeline = _Script(before=PipelineDecision(action=PipelineAction.REQUIRE_APPROVAL))
    secured = SecuredModel(delegate, pipeline=pipeline, run_id="r", agent_id="a")

    with pytest.raises(PipelineExecutionError):
        await secured.request(_messages("hi"), None, None)

    assert delegate.sent_messages is None


@pytest.mark.asyncio
async def test_request_stream_buffers_until_after_model_sanitizes():
    delegate = _RecordingDelegate(output_text="SECRET-STREAM")
    pipeline = _Script(
        after=PipelineDecision(
            action=PipelineAction.MODIFY_RESULT, modified_payload="[REDACTED]"
        )
    )
    secured = SecuredModel(delegate, pipeline=pipeline, run_id="r", agent_id="a")

    async with secured.request_stream(_messages("hi"), None, None) as sr:
        async for _event in sr:  # consume the buffered stream
            pass
        response = sr.get()  # StreamedResponse.get() is synchronous

    assert _response_text(response) == "[REDACTED]"
    assert "SECRET" not in _response_text(response)


@pytest.mark.asyncio
async def test_request_stream_deny_result_raises_before_any_token():
    delegate = _RecordingDelegate(output_text="SECRET-STREAM")
    pipeline = _Script(after=PipelineDecision(action=PipelineAction.DENY_RESULT))
    secured = SecuredModel(delegate, pipeline=pipeline, run_id="r", agent_id="a")

    with pytest.raises(ModelResultDeniedError):
        async with secured.request_stream(_messages("hi"), None, None) as sr:
            async for _event in sr:
                pass


@pytest.mark.asyncio
async def test_no_pipeline_passthrough_unchanged():
    delegate = _RecordingDelegate(output_text="plain")
    secured = SecuredModel(delegate, pipeline=None, run_id="r", agent_id="a")

    response = await secured.request(_messages("hi"), None, None)

    assert _response_text(response) == "plain"


@pytest.mark.asyncio
async def test_before_modify_without_payload_fails_closed():
    """A MODIFY that carries no payload must fail closed -- the original
    (possibly sensitive) prompt must not pass through unchanged."""
    delegate = _RecordingDelegate(output_text="ok")
    pipeline = _Script(
        before=PipelineDecision(action=PipelineAction.MODIFY, modified_payload=None)
    )
    secured = SecuredModel(delegate, pipeline=pipeline, run_id="r", agent_id="a")

    with pytest.raises(PipelineExecutionError):
        await secured.request(_messages("SECRET"), None, None)

    assert delegate.sent_messages is None, "delegate must not be called on a bad MODIFY"


@pytest.mark.asyncio
async def test_after_modify_result_without_payload_fails_closed():
    """A MODIFY_RESULT that carries no payload must fail closed -- the original
    (possibly sensitive) result must not be returned unchanged."""
    delegate = _RecordingDelegate(output_text="SECRET")
    pipeline = _Script(
        after=PipelineDecision(
            action=PipelineAction.MODIFY_RESULT, modified_payload=None
        )
    )
    secured = SecuredModel(delegate, pipeline=pipeline, run_id="r", agent_id="a")

    with pytest.raises(PipelineExecutionError):
        await secured.request(_messages("hi"), None, None)
