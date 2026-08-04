#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AggregationPolicy + the collect/aggregate reductions.

``collect`` is the task_graph projection: it walks EVERY terminal node (not
only successful ones) and preserves FAILED/SKIPPED/CANCELLED. The legacy
CONCAT/FIRST/LAST/MERGE modes reduce only succeeded SwarmStep results and are
retained for the coordinator_delegation strategy."""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping

from ...json import JsonValue
from ...agent.models import RunResult


class AggregationMode(str, Enum):
    CONCAT = "concat"
    FIRST = "first"
    LAST = "last"
    MERGE = "merge"
    COLLECT = "collect"


@dataclass(frozen=True, slots=True)
class AggregationPolicy:
    mode: AggregationMode = AggregationMode.CONCAT


def aggregate(policy: AggregationPolicy, tasks: "tuple[Any, ...]") -> RunResult:
    """Reduce the SUCCEEDED tasks' results per policy.mode (CONCAT/FIRST/LAST/
    MERGE). Returns a RunResult whose token_usage sums per-task tokens."""
    succeeded = tuple(t for t in tasks if getattr(t, "result", None) is not None)
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
        raise ValueError(f"aggregate() does not handle mode {policy.mode!r}")
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


def collect(
    plan_id: str,
    nodes_in_plan_order: "tuple[Any, ...]",
) -> JsonValue:
    """Build the task_graph collect projection over every node.

    ``nodes_in_plan_order`` is the per-node view, already in TaskPlan
    declaration order: each carries ``agent_id``, ``status``, ``output``,
    ``error`` (redacted), ``blocked_by``, ``reason``, ``attempts``,
    ``child_run_id`` and ``usage``. FAILED/SKIPPED/CANCELLED are preserved.
    ``status_counts`` tally every terminal status; the projection is NOT an
    authority for limit accounting (usage is)."""
    counts: "dict[str, int]" = {
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "cancelled": 0,
    }
    encoded_nodes: "dict[str, JsonValue]" = {}
    for node in nodes_in_plan_order:
        status = str(node["status"])
        if status in counts:
            counts[status] += 1
        encoded_nodes[node["node_id"]] = _encode_node_view(node)
    return {
        "plan_id": plan_id,
        "status_counts": counts,
        "nodes": encoded_nodes,
    }


def _encode_node_view(node: "Mapping[str, Any]") -> JsonValue:
    usage = node.get("usage")
    total_cost: "str | None" = None
    if usage is not None and usage.get("total_cost") is not None:
        total_cost = format(Decimal(str(usage["total_cost"])), "f")
    error = node.get("error")
    error_message: "str | None" = None
    if error is not None:
        error_message = getattr(error, "message", None) or str(error)
    return {
        "agent_id": node["agent_id"],
        "status": str(node["status"]),
        "output": node.get("output"),
        "error": error_message,
        "blocked_by": list(node.get("blocked_by", ())),
        "reason": node.get("reason"),
        "attempts": int(node.get("attempts", 0)),
        "child_run_id": node.get("child_run_id"),
        "usage": {
            "input_tokens": int(usage.get("input_tokens", 0)) if usage else 0,
            "output_tokens": int(usage.get("output_tokens", 0)) if usage else 0,
            "total_cost": total_cost,
        },
    }


__all__ = ["AggregationMode", "AggregationPolicy", "aggregate", "collect"]
