#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation domain models

The evaluation plane is an optional quality layer. It runs an Agent / Skill /
SubAgent target against a set of cases, captures the full run snapshot, and
scores it with pluggable evaluators. It does NOT block dynamic resource usage
or gate production -- it answers "is this target actually better?".
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class EvalRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class EvalTarget:
    kind: str  # "agent", "skill", "subagent"
    id: str
    revision: "str | None" = None
    config: "Mapping[str, object]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    input_artifact_id: str
    expected_artifact_id: "str | None" = None
    tags: "tuple[str, ...]" = ()
    metadata: "Mapping[str, object]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalSuite:
    id: str
    name: str
    version: str
    target: EvalTarget
    case_ids: "tuple[str, ...]"
    evaluator_names: "tuple[str, ...]"
    baseline_target: "EvalTarget | None" = None
    max_concurrency: int = 1
    metadata: "Mapping[str, object]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalRun:
    id: str
    suite_id: str
    target: EvalTarget
    status: EvalRunStatus
    baseline_target: "EvalTarget | None"
    created_at: datetime
    started_at: "datetime | None" = None
    finished_at: "datetime | None" = None
    metadata: "Mapping[str, object]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalResult:
    id: str
    eval_run_id: str
    case_id: str
    run_id: "str | None" = None
    job_id: "str | None" = None
    task_id: "str | None" = None
    output_artifact_id: "str | None" = None
    snapshot_artifact_id: "str | None" = None
    scores: "Mapping[str, float]" = field(default_factory=dict)
    metrics: "Mapping[str, object]" = field(default_factory=dict)
    error_type: "str | None" = None
    error_message: "str | None" = None


@dataclass(frozen=True, slots=True)
class EvalScore:
    evaluator_name: str
    score: float
    details: "Mapping[str, object]" = field(default_factory=dict)


def normalize_usage(usage: "Mapping[str, object] | None") -> "dict[str, object]":
    """Normalize a RunResult.token_usage mapping for the eval metrics.

    The Runtime reports ``input_tokens`` + ``output_tokens``; the eval plane's
    aggregation reads ``total_tokens``. Derive it when absent so
    avg tokens populate from real data. (``total_cost`` is left as-is -- the
    Runtime does not currently produce a cost figure.) Returns a new dict; does
    not mutate the input."""
    if not usage:
        return {}
    out: "dict[str, object]" = dict(usage)
    if "total_tokens" not in out:
        inp = out.get("input_tokens")
        gen = out.get("output_tokens")
        if (
            isinstance(inp, (int, float))
            and not isinstance(inp, bool)
            and isinstance(gen, (int, float))
            and not isinstance(gen, bool)
        ):
            out["total_tokens"] = inp + gen
    return out


@dataclass(frozen=True, slots=True)
class EvalExecution:
    """Output of running one case against the target."""

    case_id: str
    run_id: "str | None"
    output: object  # the target's result (RunResult.output or raw value)
    error: "str | None" = None
    # When the executor can seal the run output to an artifact, its id lands
    # here so the EvalResult can carry provenance for downstream comparison.
    output_artifact_id: "str | None" = None
    # Per-model usage (tokens / cost) aggregated from the RunResult, fed into
    # EvalResult.metrics so aggregate() can compute avg tokens / avg cost.
    model_usage: "Mapping[str, object]" = field(default_factory=dict)
    # When the executor captured a full RunSnapshot (run record / definition /
    # events sealed to artifacts), its artifact id + the snapshot itself land
    # here so the result carries replay provenance and evaluators can see the
    # run trajectory.
    snapshot_artifact_id: "str | None" = None
    snapshot: "object | None" = None


__all__: "list[str]" = [
    "EvalRunStatus",
    "EvalTarget",
    "EvalCase",
    "EvalSuite",
    "EvalRun",
    "EvalResult",
    "EvalScore",
    "EvalExecution",
    "normalize_usage",
]
