#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation query and execution API."""

from typing import Protocol

from ..core import Principal
from ..core.errors import ErrorCode, LinktoolsAIError
from ..observe.snapshot import RunSnapshot
from .services import (
    CompareEvaluationRequest,
    EvaluationComparison,
    EvaluationHandle,
    EvaluationView,
    ReplayEvaluationRequest,
    RunEvaluationRequest,
    ExecutionHandle,
)


def validate_compare_request(request: CompareEvaluationRequest) -> None:
    values = (
        request.baseline_id,
        request.candidate_id,
        request.dataset_id,
        request.evaluator_contract_id,
        request.target_kind,
        request.snapshot_digest,
        request.artifact_digest,
        request.output_schema_fingerprint,
    )
    revisions = (
        request.dataset_revision,
        request.evaluator_contract_revision,
        request.metric_contract_revision,
    )
    if any(value is None or not value.strip() for value in values) or any(value is None or value < 1 for value in revisions):
        raise LinktoolsAIError(ErrorCode.EVALUATION_INCOMPATIBLE)


class EvaluationQueryApi(Protocol):
    async def inspect(self, evaluation_id: str, *, principal: Principal) -> EvaluationView: ...
    async def compare(self, request: CompareEvaluationRequest) -> EvaluationComparison: ...
    async def snapshot(self, evaluation_id: str, *, principal: Principal) -> RunSnapshot: ...


class EvaluationApi(EvaluationQueryApi, Protocol):
    async def run(self, request: RunEvaluationRequest) -> EvaluationHandle: ...
    async def replay(self, snapshot_id: str, request: ReplayEvaluationRequest) -> ExecutionHandle: ...


__all__ = ["EvaluationApi", "EvaluationQueryApi", "validate_compare_request"]
