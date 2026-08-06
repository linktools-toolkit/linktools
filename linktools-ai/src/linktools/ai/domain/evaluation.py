#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluation product values independent of Pydantic Evals."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ..foundation.errors import ErrorCode, LinktoolsAIError


class EvaluationStatus(StrEnum):
    """Evaluation lifecycle."""

    ACCEPTED = "ACCEPTED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class EvaluationRun(BaseModel):
    """Fixed-release evaluation identity."""

    model_config = ConfigDict(frozen=True)

    evaluation_id: str
    tenant_id: str
    agent_id: str
    agent_revision: int
    dataset_digest: str
    evaluator_digest: "str | None" = None
    target_digest: "str | None" = None
    prompt_digest: "str | None" = None
    status: EvaluationStatus = EvaluationStatus.ACCEPTED

    def transition_to(self, target: EvaluationStatus) -> "EvaluationRun":
        """Return a new evaluation projection in a valid state."""
        allowed = {
            EvaluationStatus.ACCEPTED: {EvaluationStatus.RUNNING, EvaluationStatus.FAILED},
            EvaluationStatus.RUNNING: {EvaluationStatus.SUCCEEDED, EvaluationStatus.FAILED},
            EvaluationStatus.SUCCEEDED: set(),
            EvaluationStatus.FAILED: set(),
        }
        if target not in allowed[self.status]:
            raise ValueError(f"invalid evaluation transition: {self.status} -> {target}")
        return self.model_copy(update={"status": target})


class EvaluationCase(BaseModel):
    """A product evaluation case linked to a Runtime execution."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    evaluation_id: str
    execution_id: str
    trace_id: "str | None" = None
    score: "float | None" = None


class EvaluationAggregate(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int = Field(ge=0)
    metrics: "dict[str, float]"
    pass_rate: float = Field(default=0, ge=0, le=1)
    error_rate: float = Field(default=0, ge=0, le=1)
    refusal_rate: float = Field(default=0, ge=0, le=1)
    retry_count: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    model_tokens: int = Field(default=0, ge=0)
    cost_microusd: int = Field(default=0, ge=0)
    latency_p50_ms: "float | None" = Field(default=None, ge=0)
    latency_p95_ms: "float | None" = Field(default=None, ge=0)
    p50: "float | None" = None
    p95: "float | None" = None
    dataset_revision: "str | None" = None
    evaluator_contract: "str | None" = None


class EvaluationComparison(BaseModel):
    """Baseline/candidate aggregate and metric differences."""

    model_config = ConfigDict(frozen=True)

    baseline: EvaluationAggregate
    candidate: EvaluationAggregate
    deltas: "dict[str, float]"

    @classmethod
    def compare(cls, baseline: EvaluationAggregate, candidate: EvaluationAggregate) -> "EvaluationComparison":
        if baseline.count == 0 or candidate.count == 0:
            raise LinktoolsAIError(ErrorCode.EVALUATION_INCOMPATIBLE, "evaluation aggregates must contain cases")
        if baseline.dataset_revision != candidate.dataset_revision or baseline.evaluator_contract != candidate.evaluator_contract:
            raise LinktoolsAIError(ErrorCode.EVALUATION_INCOMPATIBLE, "evaluation inputs are not comparable")
        keys = sorted(set(baseline.metrics) | set(candidate.metrics))
        return cls(
            baseline=baseline,
            candidate=candidate,
            deltas={key: candidate.metrics.get(key, 0.0) - baseline.metrics.get(key, 0.0) for key in keys},
        )
