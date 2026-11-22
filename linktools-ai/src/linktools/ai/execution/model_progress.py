#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metadata-only model progress on the optional live execution transport."""

from dataclasses import dataclass
from datetime import datetime
from typing import cast

from pydantic import TypeAdapter

from ..core import JsonValue, UsageMetrics, canonical_json_bytes
from ._service_api import ModelInteractionItem, ModelStreamAttemptItem


_MODEL_IDENTITY_FIELDS = {"route_id", "system", "model_name"}


@dataclass(frozen=True, slots=True)
class _Attempt:
    stream_attempt_seq: int
    status: str
    started_at: datetime
    finished_at: datetime
    duration_ns: int
    client_request_start: int
    client_request_end: int
    usage: UsageMetrics | None
    error_code: str | None
    error_type: str | None
    retry_scheduled: bool

    @classmethod
    def capture(cls, value: ModelStreamAttemptItem) -> "_Attempt":
        return cls(
            value.stream_attempt_seq, value.status, value.started_at, value.finished_at,
            value.duration_ns, value.client_request_start, value.client_request_end,
            value.usage, value.error_code, value.error_type, value.retry_scheduled,
        )

    def project(self) -> ModelStreamAttemptItem:
        return ModelStreamAttemptItem(
            self.stream_attempt_seq, self.status, self.started_at, self.finished_at,
            self.duration_ns, self.client_request_start, self.client_request_end,
            None, self.usage, self.error_code, self.error_type, self.retry_scheduled,
        )


@dataclass(frozen=True, slots=True)
class _Metadata:
    agent_run_seq: int
    model_request_seq: int
    purpose: str
    step_index: int
    output_retry_index: int | None
    model: dict[str, str]
    status: str
    error_code: str | None
    duration_ns: int | None
    usage: UsageMetrics | None
    started_at: datetime | None
    finished_at: datetime | None
    stream_attempts: tuple[_Attempt, ...] | None

    def project(self, execution_id: str) -> ModelInteractionItem:
        if self.model.keys() - _MODEL_IDENTITY_FIELDS:
            raise ValueError("live model identity is invalid")
        return ModelInteractionItem(
            execution_id=execution_id, agent_run_seq=self.agent_run_seq, depth=0,
            model_request_seq=self.model_request_seq, purpose=self.purpose,
            step_index=self.step_index, output_retry_index=self.output_retry_index,
            model=self.model, request={}, response=None, status=self.status,
            error_code=self.error_code, duration_ns=self.duration_ns, usage=self.usage,
            content_included=False, started_at=self.started_at, finished_at=self.finished_at,
            stream_attempts=None if self.stream_attempts is None else tuple(
                value.project() for value in self.stream_attempts
            ),
        )


_ADAPTER = TypeAdapter(_Metadata)


def encode_model_progress(value: ModelInteractionItem) -> JsonValue:
    if value.content_included or value.client_requests_included:
        raise ValueError("live model progress must omit model content")
    model = {
        key: raw for key, raw in value.model.items()
        if key in _MODEL_IDENTITY_FIELDS and isinstance(raw, str)
    }
    if len(model) != len(value.model):
        raise ValueError("live model identity is invalid")
    return cast(JsonValue, _ADAPTER.dump_python(_Metadata(
        value.agent_run_seq, value.model_request_seq, value.purpose, value.step_index,
        value.output_retry_index, model, value.status, value.error_code,
        value.duration_ns, value.usage, value.started_at, value.finished_at,
        None if value.stream_attempts is None else tuple(
            _Attempt.capture(attempt) for attempt in value.stream_attempts
        ),
    ), mode="json"))


def decode_model_progress(execution_id: str, value: JsonValue) -> ModelInteractionItem:
    return _ADAPTER.validate_json(canonical_json_bytes(value), strict=True).project(execution_id)


__all__ = ["encode_model_progress", "decode_model_progress"]
