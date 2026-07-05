#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AggregationPolicy + aggregate(): reduces a tuple of completed SwarmTasks into
one RunResult (the value written back to the shared Session)."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..run.models import RunResult
from .models import SwarmTask


class AggregationMode(str, Enum):
    CONCAT = "concat"
    FIRST = "first"
    LAST = "last"
    MERGE = "merge"


@dataclass(frozen=True, slots=True)
class AggregationPolicy:
    mode: AggregationMode = AggregationMode.CONCAT


def aggregate(policy: AggregationPolicy, tasks: "tuple[SwarmTask, ...]") -> RunResult:
    """Reduce the SUCCEEDED tasks' results per policy.mode. Returns RunResult with
    empty token_usage (token accumulation is SwarmRunner's job) and a metadata
    dict carrying task_count."""
    succeeded = tuple(t for t in tasks if t.result is not None)
    outputs = [t.result.output for t in succeeded]
    if policy.mode == AggregationMode.CONCAT:
        out: Any = "\n".join(str(o) for o in outputs)
    elif policy.mode == AggregationMode.FIRST:
        out = outputs[0] if outputs else ""
    elif policy.mode == AggregationMode.LAST:
        out = outputs[-1] if outputs else ""
    elif policy.mode == AggregationMode.MERGE:
        merged: "dict[str, Any]" = {}
        for o in outputs:
            if isinstance(o, dict):
                merged.update(o)
        out = merged
    else:
        raise ValueError(f"unknown aggregation mode: {policy.mode}")
    return RunResult(output=out, token_usage={}, metadata={"task_count": len(succeeded)})
