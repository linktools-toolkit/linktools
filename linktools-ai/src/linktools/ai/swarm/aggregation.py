#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AggregationPolicy + aggregate(): reduces a tuple of completed SwarmSteps into
one RunResult (the value written back to the shared Session)."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..run.models import RunResult
from .models import SwarmStep


class AggregationMode(str, Enum):
    CONCAT = "concat"
    FIRST = "first"
    LAST = "last"
    MERGE = "merge"


@dataclass(frozen=True, slots=True)
class AggregationPolicy:
    mode: AggregationMode = AggregationMode.CONCAT


def aggregate(policy: AggregationPolicy, tasks: "tuple[SwarmStep, ...]") -> RunResult:
    """Reduce the SUCCEEDED tasks' results per policy.mode. Returns RunResult
    whose ``token_usage`` carries the SUM of per-task input/output tokens (each
    worker RunResult.token_usage is populated by AgentEngine from the model's
    usage) so SwarmEngine can enforce ``max_total_tokens`` , and a
    metadata dict carrying task_count."""
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
    total_input = sum(
        int(t.result.token_usage.get("input_tokens", 0)) for t in succeeded
    )
    total_output = sum(
        int(t.result.token_usage.get("output_tokens", 0)) for t in succeeded
    )
    return RunResult(
        output=out,
        token_usage={"input_tokens": total_input, "output_tokens": total_output},
        metadata={"task_count": len(succeeded)},
    )
