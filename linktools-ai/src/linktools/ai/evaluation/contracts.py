#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Product Evaluation DTOs independent of execution implementation."""

from pydantic import BaseModel, ConfigDict

from ..domain.evaluation import EvaluationAggregate, EvaluationComparison


class EvaluationTarget(BaseModel):
    model_config = ConfigDict(frozen=True)
    target_id: str
    target_kind: str
    release_digest: str


class EvaluationCaseResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    case_id: str
    execution_id: str
    score: float
    artifact_digests: "tuple[str, ...]" = ()
    snapshot_id: "str | None" = None


class ReplayRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    snapshot_id: str
    verify_artifacts: bool = True
    artifact_digests: "tuple[str, ...]" = ()


__all__ = ["EvaluationAggregate", "EvaluationCaseResult", "EvaluationComparison", "EvaluationTarget", "ReplayRequest"]
