#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One request-fact authority shared by Runtime metrics and history."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from time import monotonic_ns
from typing import Literal

from ._metric_id import _model_observation_id

ModelRequestPurpose = Literal["agent", "compaction"]
REQUEST_PURPOSE_METADATA_KEY = "linktools.ai.request_purpose"
MESSAGE_SEQ_METADATA_KEY = "linktools.ai.message_seq"
REPLAYED_TOOL_CALL_INDICES_METADATA_KEY = "linktools.ai.replayed_tool_call_indices"
MODEL_REQUEST_SEQ_METADATA_KEY = "linktools.ai.model_request_seq"
OUTPUT_RETRY_INDEX_METADATA_KEY = "linktools.ai.output_retry_index"
OBSERVATION_ID_METADATA_KEY = "linktools.ai.observation_id"
DURATION_NS_METADATA_KEY = "linktools.ai.duration_ns"
MODEL_USAGE_INPUT_METADATA_KEY = "linktools.ai.model_usage.input_tokens"
MODEL_USAGE_OUTPUT_METADATA_KEY = "linktools.ai.model_usage.output_tokens"
MODEL_USAGE_CACHE_READ_METADATA_KEY = "linktools.ai.model_usage.cache_read_tokens"
MODEL_USAGE_CACHE_WRITE_METADATA_KEY = "linktools.ai.model_usage.cache_write_tokens"


@dataclass(frozen=True, slots=True)
class ModelRequestFact:
    """Identity and timing facts for one public SDK model request."""

    step_index: int
    model_request_seq: int
    purpose: ModelRequestPurpose
    observation_id: str
    started_ns: int
    started_at: datetime
    output_retry_index: int | None = None
    duration_ns: int | None = None
    status: str | None = None
    finished_at: datetime | None = None

    def metadata(self, *, include_observation: bool) -> dict[str, str]:
        values = {
            MODEL_REQUEST_SEQ_METADATA_KEY: str(self.model_request_seq),
            REQUEST_PURPOSE_METADATA_KEY: self.purpose,
        }
        if include_observation:
            values[OBSERVATION_ID_METADATA_KEY] = self.observation_id
        if self.output_retry_index is not None:
            values[OUTPUT_RETRY_INDEX_METADATA_KEY] = str(self.output_retry_index)
        if self.duration_ns is not None:
            values[DURATION_NS_METADATA_KEY] = str(self.duration_ns)
        return values


class ModelRequestJournal:
    """Allocate request facts once and hand the same fact to all producers."""

    def __init__(
        self,
        *,
        source_namespace: str,
        tenant_id: str,
        execution_id: str,
        agent_run_id: str,
        next_model_request_seq: int = 1,
    ) -> None:
        self._source_namespace = source_namespace
        self._tenant_id = tenant_id
        self._execution_id = execution_id
        if (
            isinstance(next_model_request_seq, bool)
            or not isinstance(next_model_request_seq, int)
            or next_model_request_seq < 1
        ):
            raise ValueError("next_model_request_seq must be a positive integer")
        self._agent_run_id = agent_run_id
        self._next_model_request_seq = next_model_request_seq
        self._facts: dict[int, ModelRequestFact] = {}

    def begin(
        self,
        step_index: int,
        *,
        purpose: ModelRequestPurpose = "agent",
        output_retry_index: int | None = None,
    ) -> ModelRequestFact:
        if output_retry_index is not None and (
            isinstance(output_retry_index, bool)
            or not isinstance(output_retry_index, int)
            or output_retry_index < 1
        ):
            raise ValueError("output_retry_index must be a positive integer or None")
        sequence = self._next_model_request_seq
        self._next_model_request_seq += 1
        fact = ModelRequestFact(
            step_index=step_index,
            model_request_seq=sequence,
            purpose=purpose,
            observation_id=_model_observation_id(
                self._source_namespace,
                self._tenant_id,
                self._execution_id,
                self._agent_run_id,
                sequence,
                purpose,
            ),
            started_ns=monotonic_ns(),
            started_at=datetime.now(timezone.utc),
            output_retry_index=output_retry_index,
        )
        self._facts[sequence] = fact
        return fact

    def finish(
        self,
        model_request_seq: int,
        *,
        status: str,
        duration_ns: int | None = None,
    ) -> ModelRequestFact:
        fact = self._facts.get(model_request_seq)
        if fact is None:
            raise RuntimeError("model request fact is missing")
        if fact.duration_ns is not None:
            raise RuntimeError("model request fact is already finished")
        elapsed = monotonic_ns() - fact.started_ns
        return_value = replace(
            fact,
            duration_ns=max(0, elapsed if duration_ns is None else duration_ns),
            status=status,
            finished_at=datetime.now(timezone.utc),
        )
        self._facts[model_request_seq] = return_value
        return return_value

    def current(self, model_request_seq: int) -> ModelRequestFact:
        try:
            return self._facts[model_request_seq]
        except KeyError as error:
            raise RuntimeError("model request fact is missing") from error

    def latest_for_step(
        self,
        step_index: int,
        *,
        purpose: ModelRequestPurpose = "agent",
    ) -> ModelRequestFact | None:
        values = tuple(
            fact
            for fact in self._facts.values()
            if fact.step_index == step_index and fact.purpose == purpose
        )
        return max(values, key=lambda fact: fact.model_request_seq, default=None)

    def consume(self, model_request_seq: int) -> ModelRequestFact:
        try:
            return self._facts.pop(model_request_seq)
        except KeyError as error:
            raise RuntimeError("model request fact is missing") from error


async def _await_request_handoff(awaitable: Awaitable[None]) -> bool:
    task = asyncio.create_task(awaitable)
    interrupted = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            interrupted = True
    if task.cancelled():
        raise asyncio.CancelledError
    task.result()
    return interrupted


__all__ = [
    "DURATION_NS_METADATA_KEY",
    "MODEL_USAGE_CACHE_READ_METADATA_KEY",
    "MODEL_USAGE_CACHE_WRITE_METADATA_KEY",
    "MODEL_USAGE_INPUT_METADATA_KEY",
    "MODEL_USAGE_OUTPUT_METADATA_KEY",
    "ModelRequestFact",
    "ModelRequestJournal",
    "ModelRequestPurpose",
    "OBSERVATION_ID_METADATA_KEY",
    "OUTPUT_RETRY_INDEX_METADATA_KEY",
    "REQUEST_PURPOSE_METADATA_KEY",
    "MODEL_REQUEST_SEQ_METADATA_KEY",
    "MESSAGE_SEQ_METADATA_KEY",
    "REPLAYED_TOOL_CALL_INDICES_METADATA_KEY",
]
