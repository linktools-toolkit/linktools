#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SecuredModel: a pydantic-ai Model subclass that fires the security pipeline
around EVERY model request (before_model / after_model), not just once around
the whole run. A tool loop drives several model requests; each must pass the
pipeline.

Every pipeline decision is honored:
- before_model DENY -> the model is never called.
- before_model MODIFY -> the prompt actually sent is the redacted/rewritten one.
- after_model DENY_RESULT -> the caller never sees the raw output.
- after_model MODIFY_RESULT -> the caller sees the redacted/rewritten output.
- an action invalid for the stage (e.g. before_model REQUIRE_APPROVAL, for
  which there is no model-approval recovery) fails closed.
- when a pipeline is configured, request_stream never exposes a token before
  after_model has run -- the stream is buffered through the non-stream path.

Subclassing Model (not duck-typing) is required: pydantic-ai's Agent.iter()
recognizes a real Model and routes through request()/request_stream(). Runtime
passes model=SecuredModel(...) per call (never mutating the shared compiled
agent)."""

import dataclasses
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic_ai.messages import ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse

from ..errors import (
    ModelInvocationDeniedError,
    ModelResultDeniedError,
    PipelineExecutionError,
)
from .pipeline import PipelineAction, validate_model_decision


def _last_user_text(messages: Any) -> str:
    """Best-effort latest user prompt from the message list, for before_model.
    Empty string when no user part is present (e.g. a resume re-drive)."""
    for msg in reversed(list(messages or [])):
        for part in reversed(getattr(msg, "parts", None) or []):
            if isinstance(part, UserPromptPart):
                return str(part.content)
    return ""


def _response_text(response: Any) -> Any:
    """Text output of a ModelResponse, for after_model."""
    texts = []
    for part in getattr(response, "parts", None) or []:
        content = getattr(part, "content", None)
        if isinstance(content, str):
            texts.append(content)
    return "\n".join(texts) if texts else response


def _replace_last_user_text(messages: Any, replacement: str) -> Any:
    """Return a copy of ``messages`` with the last UserPromptPart's content set
    to ``replacement``. The caller's list is not mutated. Raises if there is no
    user prompt to replace (a MODIFY must target a real prompt)."""
    copied = list(messages)
    for i in range(len(copied) - 1, -1, -1):
        msg = copied[i]
        parts = list(getattr(msg, "parts", None) or [])
        for j in range(len(parts) - 1, -1, -1):
            if isinstance(parts[j], UserPromptPart):
                parts[j] = dataclasses.replace(parts[j], content=replacement)
                copied[i] = dataclasses.replace(msg, parts=tuple(parts))
                return copied
    raise PipelineExecutionError(
        "before_model MODIFY could not find a user prompt to replace"
    )


def _replace_model_response_output(
    response: ModelResponse, payload: Any
) -> ModelResponse:
    """Apply an after_model MODIFY_RESULT payload to a ModelResponse. A
    ModelResponse payload replaces wholesale; a str payload replaces the text;
    anything else fails closed (never returns the un-redacted original)."""
    if isinstance(payload, ModelResponse):
        return payload
    if isinstance(payload, str):
        return dataclasses.replace(response, parts=(TextPart(content=payload),))
    raise PipelineExecutionError(
        "after_model MODIFY_RESULT payload must be a string or ModelResponse"
    )


@dataclass
class _BufferedStreamedResponse(StreamedResponse):
    """A StreamedResponse that exposes ONLY an after_model-sanitized response.
    The underlying request completed (and was audited) before this object is
    constructed, so no un-audited token can reach the consumer. Used when a
    security pipeline is configured: streaming the delegate raw would leak
    content before after_model runs.

    The full sanitized ModelResponse is re-streamed (every part, including
    ToolCallParts) so pydantic-ai's agent graph still drives the tool loop --
    buffering the request must not drop a tool call."""

    _model_name: str
    _response: ModelResponse
    _timestamp: "datetime" = field(default_factory=lambda: datetime.now(timezone.utc))

    async def _get_event_iterator(self):  # type: ignore[override]
        for index, part in enumerate(getattr(self._response, "parts", None) or ()):
            yield self._parts_manager.handle_part(vendor_part_id=index, part=part)

    async def close_stream(self) -> None:
        return None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> "str | None":
        return None

    @property
    def provider_url(self) -> "str | None":
        return None

    @property
    def timestamp(self) -> "datetime":
        return self._timestamp


class SecuredModel(Model):
    """Wraps a pydantic-ai Model so the security pipeline fires per request."""

    def __init__(
        self,
        delegate: Model,
        *,
        pipeline: Any,
        run_id: str,
        agent_id: "str | None",
    ) -> None:
        # NOTE: intentionally no super().__init__() -- Model's abstract members
        # are satisfied by the properties/methods below, and we must NOT copy
        # the delegate's state (we delegate to it live).
        object.__setattr__(self, "_delegate", delegate)
        object.__setattr__(self, "_pipeline", pipeline)
        object.__setattr__(self, "_run_id", run_id)
        object.__setattr__(self, "_agent_id", agent_id)

    def __getattr__(self, name: str) -> Any:
        # Delegate non-intercepted attributes (customize_request_parameters,
        # count_tokens, prepare_messages, ...) to the wrapped model.
        return getattr(self._delegate, name)

    @property
    def model_name(self) -> str:
        return self._delegate.model_name

    @property
    def system(self) -> str:
        return self._delegate.system

    async def _before(self, messages: Any) -> Any:
        """Run before_model and return the messages to actually send. DENY
        raises; MODIFY rewrites the last user prompt; an invalid action fails
        closed. The payload is applied whenever it is non-None (a composite
        pipeline may return action=ALLOW with a modified_payload)."""
        if self._pipeline is None or not hasattr(self._pipeline, "before_model"):
            return messages
        from .pipeline import ModelInvocationEvent

        decision = await self._pipeline.before_model(
            ModelInvocationEvent(
                prompt=_last_user_text(messages),
                run_id=self._run_id,
                agent_id=self._agent_id,
            )
        )
        validate_model_decision(decision, stage="before")
        if decision.action is PipelineAction.DENY:
            raise ModelInvocationDeniedError(
                decision.reason or "model call denied by pipeline"
            )
        if decision.modified_payload is None:
            return messages
        if not isinstance(decision.modified_payload, str):
            raise PipelineExecutionError("before_model MODIFY payload must be a string")
        return _replace_last_user_text(messages, decision.modified_payload)

    async def _after(self, response: Any) -> Any:
        """Run after_model and return the response to return to the caller.
        DENY_RESULT raises; MODIFY_RESULT rewrites the output; an invalid action
        fails closed. The payload is applied whenever it is non-None."""
        if self._pipeline is None or not hasattr(self._pipeline, "after_model"):
            return response
        from .pipeline import ModelResultEvent

        decision = await self._pipeline.after_model(
            ModelResultEvent(output=_response_text(response), run_id=self._run_id)
        )
        validate_model_decision(decision, stage="after")
        if decision.action is PipelineAction.DENY_RESULT:
            raise ModelResultDeniedError(
                decision.reason or "model result denied by pipeline"
            )
        if decision.modified_payload is None:
            return response
        return _replace_model_response_output(response, decision.modified_payload)

    async def request(self, messages, model_settings, model_request_parameters):  # type: ignore[override]
        effective_messages = await self._before(messages)
        response = await self._delegate.request(
            effective_messages, model_settings, model_request_parameters
        )
        return await self._after(response)

    @asynccontextmanager
    async def request_stream(
        self, messages, model_settings, model_request_parameters, run_context=None
    ):  # type: ignore[override]
        if self._pipeline is None:
            # No pipeline: stream the delegate directly (before_model is a no-op
            # when there is no pipeline).
            effective_messages = await self._before(messages)
            async with self._delegate.request_stream(
                effective_messages,
                model_settings,
                model_request_parameters,
                run_context,
            ) as sr:
                yield sr
            return
        # A pipeline is configured: never expose an un-audited token. Run the
        # full non-stream request (before -> delegate.request -> after) and
        # yield a buffered adapter over the sanitized response. DENY at either
        # stage raises before anything is yielded.
        sanitized = await self.request(
            messages, model_settings, model_request_parameters
        )
        mrp = model_request_parameters or ModelRequestParameters()
        yield _BufferedStreamedResponse(
            model_request_parameters=mrp,
            _model_name=self._delegate.model_name,
            _response=sanitized,
        )
