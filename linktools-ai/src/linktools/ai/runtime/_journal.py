#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One request-fact authority shared by Runtime metrics and history."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import monotonic_ns
from typing import Literal

from ._metric_id import _model_observation_id

ModelRequestPurpose = Literal["agent", "compaction"]
REQUEST_PURPOSE_METADATA_KEY = "linktools.ai.request_purpose"
REQUEST_SEQUENCE_METADATA_KEY = "linktools.ai.request_sequence"


@dataclass(frozen=True, slots=True)
class ModelRequestFact:
    """Identity and timing facts for one public SDK model request."""

    step_index: int
    request_sequence: int
    purpose: ModelRequestPurpose
    observation_id: str
    started_ns: int
    duration_ns: int | None = None
    status: str | None = None

    def metadata(self, *, include_observation: bool) -> dict[str, str]:
        values = {
            REQUEST_SEQUENCE_METADATA_KEY: str(self.request_sequence),
            REQUEST_PURPOSE_METADATA_KEY: self.purpose,
        }
        if include_observation:
            values["linktools.ai.observation_id"] = self.observation_id
        if self.duration_ns is not None:
            values["linktools.ai.duration_ns"] = str(self.duration_ns)
        return values


class ModelRequestJournal:
    """Allocate request facts once and hand the same fact to all producers."""

    def __init__(
        self,
        *,
        source_namespace: str,
        tenant_id: str,
        execution_id: str,
        step_run_id: str,
    ) -> None:
        self._source_namespace = source_namespace
        self._tenant_id = tenant_id
        self._execution_id = execution_id
        self._step_run_id = step_run_id
        self._next_sequence = 1
        self._facts: dict[int, ModelRequestFact] = {}

    def begin(
        self,
        step_index: int,
        *,
        purpose: ModelRequestPurpose = "agent",
    ) -> ModelRequestFact:
        sequence = self._next_sequence
        self._next_sequence += 1
        fact = ModelRequestFact(
            step_index=step_index,
            request_sequence=sequence,
            purpose=purpose,
            observation_id=_model_observation_id(
                self._source_namespace,
                self._tenant_id,
                self._execution_id,
                self._step_run_id,
                sequence,
                purpose,
            ),
            started_ns=monotonic_ns(),
        )
        self._facts[sequence] = fact
        return fact

    def finish(
        self,
        request_sequence: int,
        *,
        status: str,
        duration_ns: int | None = None,
    ) -> ModelRequestFact:
        fact = self._facts.get(request_sequence)
        if fact is None:
            raise RuntimeError("model request fact is missing")
        if fact.duration_ns is not None:
            return fact
        elapsed = monotonic_ns() - fact.started_ns
        return_value = replace(
            fact,
            duration_ns=max(0, elapsed if duration_ns is None else duration_ns),
            status=status,
        )
        self._facts[request_sequence] = return_value
        return return_value

    def current(self, request_sequence: int) -> ModelRequestFact:
        try:
            return self._facts[request_sequence]
        except KeyError as error:
            raise RuntimeError("model request fact is missing") from error

    def consume(self, request_sequence: int) -> ModelRequestFact:
        try:
            return self._facts.pop(request_sequence)
        except KeyError as error:
            raise RuntimeError("model request fact is missing") from error


__all__ = [
    "ModelRequestFact",
    "ModelRequestJournal",
    "ModelRequestPurpose",
    "REQUEST_PURPOSE_METADATA_KEY",
    "REQUEST_SEQUENCE_METADATA_KEY",
]
