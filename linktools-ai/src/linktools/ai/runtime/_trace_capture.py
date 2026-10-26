#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-run model interaction payload capture and trace projection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol

from linktools.core import environ

from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapperCapability
from pydantic_ai.usage import UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings

from ..capability import TraceCapability
from ..core import JsonValue, UsageMetrics
from ..errors import AIError, ErrorCode
from ..model import ModelClientRequest, ModelStreamAttempt, model_binding_error
from linktools.ai.model.messages import request_attachment_facts
from linktools.ai.observe.journal import ModelRequestFact
from linktools.ai.model.messages import freeze_model_messages
from linktools.ai.persistence.runs import (
    StagedContextProjection,
    StagedContextSpan,
    StagedModelInteraction,
    StagedStreamAttemptRecord,
    build_context_projection,
    build_inline_context_projection,
    model_identity,
    request_envelope,
)
from ._run_context import RunContextProjection


def record_http_bodies_enabled(capabilities: Iterable[AbstractCapability[object]]) -> bool:
    """Read the active run's Trace policy through public native capability trees."""
    for capability in capabilities:
        if isinstance(capability, WrapperCapability):
            if record_http_bodies_enabled((capability.wrapped,)):
                return True
        elif isinstance(capability, CombinedCapability):
            if record_http_bodies_enabled(capability.capabilities):
                return True
        elif isinstance(capability, TraceCapability) and capability.record_http_bodies:
            return True
    return False


class TraceStagingPort(Protocol):
    def intern_payload(self, agent_run_id: str, payload: bytes) -> tuple[str, int]: ...
    def stage_model_interaction(self, interaction: object) -> None: ...
    def prepare_model_interaction(self, interaction: object) -> None: ...


class ModelInteractionCapture:
    """Own prepared requests, attachment facts and completed trace payloads."""

    def __init__(
        self, staging: TraceStagingPort, agent_run_id: str,
        context: RunContextProjection, initial_attachments: Sequence[Mapping[str, JsonValue]],
    ) -> None:
        self._staging_store = staging
        self._agent_run_id = agent_run_id
        self._context = context
        self._initial_attachments = tuple(dict(value) for value in initial_attachments)
        self._accepted_attachment_ids = {
            attachment_id
            for value in self._initial_attachments
            if isinstance((attachment_id := value.get("attachment_id")), str)
            and attachment_id
        }

        self._interaction_projections: dict[int, StagedContextProjection | None] = {}
        self._interaction_payloads: dict[int, str | None] = {}
        self._interaction_models: dict[int, dict[str, str]] = {}
        self._interaction_attachments: dict[
            int,
            tuple[Mapping[str, JsonValue], ...],
        ] = {}

    def begin_model_interaction(
        self,
        fact: ModelRequestFact,
        model: Model,
        messages: Sequence[ModelMessage],
        model_settings: ModelSettings | None,
        parameters: ModelRequestParameters,
        streaming: bool,
        model_id: str | None = None,
        source_messages: Sequence[ModelMessage] | None = None,
    ) -> None:
        self._stage_model_interaction(
            fact, model, messages, model_settings, parameters, streaming,
            model_id, source_messages, prepared=False,
        )

    def prepare_model_interaction(
        self,
        fact: ModelRequestFact,
        model: Model,
        messages: Sequence[ModelMessage],
        model_settings: ModelSettings | None,
        parameters: ModelRequestParameters,
        streaming: bool,
        model_id: str | None = None,
        source_messages: Sequence[ModelMessage] | None = None,
    ) -> None:
        self._stage_model_interaction(
            fact, model, messages, model_settings, parameters, streaming,
            model_id, source_messages, prepared=True,
        )

    def _stage_model_interaction(
        self,
        fact: ModelRequestFact,
        model: Model,
        messages: Sequence[ModelMessage],
        model_settings: ModelSettings | None,
        parameters: ModelRequestParameters,
        streaming: bool,
        model_id: str | None,
        source_messages: Sequence[ModelMessage] | None,
        *,
        prepared: bool,
    ) -> None:
        frozen = freeze_model_messages(messages)
        projection = None
        digest = None
        if prepared:
            source = frozen if source_messages is None else freeze_model_messages(source_messages)
            source_refs, source_keys = self._context.source_refs_for(source)
            projection = build_context_projection(
                source,
                frozen,
                lambda payload: self._staging_store.intern_payload(
                    self._agent_run_id,
                    payload,
                ),
                source_refs=source_refs,
                source_keys=source_keys,
            )
            _envelope, envelope_bytes = request_envelope(
                model_settings=model_settings,
                parameters=parameters,
                streaming=streaming,
            )
            digest, _size = self._staging_store.intern_payload(
                self._agent_run_id,
                envelope_bytes,
            )
        model_value = model_identity(
            model,
            route_id=model_id,
        )
        attachments = request_attachment_facts(
            frozen,
            self._initial_attachments,
            accepted_attachment_ids=self._accepted_attachment_ids,
        )
        # Acceptance survives provider preparation even when the final input
        # drops an attachment. Inclusion follows the current prepared request.
        accepted = tuple(
            value
            for value in self._interaction_attachments.get(fact.model_request_seq, ())
            if value["fact"] == "accepted"
        )
        attachments = (*accepted, *attachments)
        if not prepared:
            attachments = tuple(value for value in attachments if value["fact"] == "accepted")
        stage = (
            self._staging_store.prepare_model_interaction
            if prepared
            else self._staging_store.stage_model_interaction
        )
        stage(
            StagedModelInteraction(
                agent_run_id=self._agent_run_id,
                step_index=fact.step_index,
                model_request_seq=fact.model_request_seq,
                purpose=fact.purpose,
                output_retry_index=fact.output_retry_index,
                model=model_value,
                request_context=projection,
                request_envelope_digest=digest,
                response_context=None,
                status="RUNNING",
                error_code=None,
                duration_ns=None,
                usage=None,
                attachments=attachments,
                started_at=fact.started_at,
                finished_at=None,
            )
        )
        self._interaction_projections[fact.model_request_seq] = projection
        self._interaction_payloads[fact.model_request_seq] = digest
        self._interaction_models[fact.model_request_seq] = model_value
        self._interaction_attachments[fact.model_request_seq] = attachments

    def finish_model_interaction(
        self,
        fact: ModelRequestFact,
        *,
        model: Model,
        response: ModelResponse | None,
        status: str,
        error_code: str | None,
        duration_ns: int,
        usage: object | None,
        client_requests: tuple[ModelClientRequest, ...] | None = None,
        response_message_index: int | None = None,
        stream_attempts: tuple[ModelStreamAttempt, ...] | None = None,
    ) -> None:
        del model
        model_request_seq = fact.model_request_seq
        try:
            projection = self._interaction_projections.pop(model_request_seq)
            envelope_digest = self._interaction_payloads.pop(model_request_seq)
            model_value = self._interaction_models.pop(model_request_seq)
            attachments = self._interaction_attachments.pop(model_request_seq)
        except KeyError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

        response_projection = None
        if response is not None:
            if response_message_index is not None:
                response_projection = StagedContextProjection((
                    StagedContextSpan(response_message_index, response_message_index + 1),
                ))
            else:
                frozen_response = freeze_model_messages((response,))
                response_projection = build_inline_context_projection(
                    frozen_response,
                    lambda payload: self._staging_store.intern_payload(
                        self._agent_run_id,
                        payload,
                    ),
                )
        self._staging_store.stage_model_interaction(
            StagedModelInteraction(
                agent_run_id=self._agent_run_id,
                step_index=fact.step_index,
                model_request_seq=model_request_seq,
                purpose=fact.purpose,
                output_retry_index=fact.output_retry_index,
                model=model_value,
                request_context=projection,
                request_envelope_digest=envelope_digest,
                response_context=response_projection,
                status=status,
                error_code=error_code,
                duration_ns=duration_ns,
                usage=_usage_metrics(usage),
                attachments=attachments,
                started_at=fact.started_at,
                finished_at=fact.finished_at,
                client_requests=client_requests,
                stream_attempts=None if stream_attempts is None else tuple(
                    self._stream_attempt(value) for value in stream_attempts
                ),
            )
        )

    def _stream_attempt(self, attempt: ModelStreamAttempt) -> StagedStreamAttemptRecord:
        error = attempt.error
        projected = (model_binding_error(error) if isinstance(error, Exception) else None)
        error_code = (error.code.value if isinstance(error, AIError) else
                      ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED.value if isinstance(error, UsageLimitExceeded) else
                      projected.code.value if projected is not None else
                      ErrorCode.INTERNAL_ERROR.value if attempt.status == "FAILED" else None)
        context = None
        if attempt.status != "SUCCEEDED" and attempt.response is not None:
            context = build_inline_context_projection(
                freeze_model_messages((attempt.response,)),
                lambda payload: self._staging_store.intern_payload(self._agent_run_id, payload),
            )
        return StagedStreamAttemptRecord(
            stream_attempt_seq=attempt.stream_attempt_seq, status=attempt.status,
            started_at=attempt.started_at, finished_at=attempt.finished_at,
            duration_ns=attempt.duration_ns,
            client_request_start=attempt.client_request_start, client_request_end=attempt.client_request_end,
            response_context=context, usage=_usage_metrics(attempt.usage),
            error_code=error_code, error_type=(None if error is None else
                environ.logging.redact(f"{type(error).__module__}.{type(error).__qualname__}")),
            retry_scheduled=attempt.retry_scheduled,
        )


def _usage_metrics(value: object | None) -> UsageMetrics | None:
    if value is None:
        return None
    return UsageMetrics(
        model_requests=1,
        input_tokens=int(getattr(value, "input_tokens", 0)),
        output_tokens=int(getattr(value, "output_tokens", 0)),
        cache_read_tokens=int(getattr(value, "cache_read_tokens", 0)),
        cache_write_tokens=int(getattr(value, "cache_write_tokens", 0)),
    )


__all__ = ["ModelInteractionCapture", "TraceStagingPort", "record_http_bodies_enabled"]
