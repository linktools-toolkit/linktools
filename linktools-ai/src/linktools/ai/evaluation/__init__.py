#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation plane."""

from .comparison import aggregate, compare
from .executors import DirectEvalExecutor, TaskEvalExecutor
from .models import (
    EvalCase,
    EvalExecution,
    EvalResult,
    EvalRun,
    EvalRunStatus,
    EvalScore,
    EvalSuite,
    EvalTarget,
    RunEvaluation,
)
from .protocols import EvalExecutor, Evaluator
from .replay import SnapshotValidationError, replay, validate_snapshot
from .runner import EvalRunner
from .snapshot import EvalSnapshot
from .targets import MappingTargetResolver

__all__: "list[str]" = [
    "EvalTarget",
    "EvalCase",
    "EvalSuite",
    "EvalRun",
    "EvalResult",
    "EvalScore",
    "EvalExecution",
    "EvalRunStatus",
    "EvalSnapshot",
    "RunEvaluation",
    "EvalExecutor",
    "Evaluator",
    "EvalRunner",
    "DirectEvalExecutor",
    "TaskEvalExecutor",
    "MappingTargetResolver",
    "aggregate",
    "compare",
    "validate_snapshot",
    "replay",
    "SnapshotValidationError",
]
