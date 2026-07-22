#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comparison: aggregate and compare Baseline vs Candidate eval results.

aggregate() reduces a sequence of EvalResults into the metric set
(per-evaluator avg + pass rate, error rate, P50/P95 latency, average tokens
and cost, safety-refusal rate, average retry count). The per-case signals are
read from each result's ``metrics`` mapping under conventional keys
(``latency_seconds``, ``total_tokens``, ``total_cost``, ``safety_refusal``,
``retry_count``); a signal that no result captures aggregates to 0.0 rather than
erroring, so the formulas are ready as capture lands."""

from collections.abc import Mapping, Sequence

from .models import EvalResult

# Exact metric keys for which LOWER is better; compare() inverts their delta so
# a positive delta always means "candidate better". Matched by EXACT key (not
# substring) so a business evaluator whose name happens to contain "cost" /
# "error" / "latency" is NOT silently inverted -- only the known aggregate
# metrics with lower-is-better semantics flip.
_LOWER_IS_BETTER_KEYS = frozenset(
    {
        "error_rate",
        "p50_latency_seconds",
        "p95_latency_seconds",
        "avg_cost",
        "avg_tokens",
        "avg_retry_count",
        "retry_rate",
    }
)
# safety_refusal_rate is intentionally NOT lower-is-better: whether more
# refusals are "better" (safer) or "worse" (over-blocking) is context-dependent,
# so compare() returns its raw delta and the caller interprets the sign.

_DEFAULT_PASS_THRESHOLD = 0.5


def _avg(values: "Sequence[float]") -> float:
    return sum(values) / len(values) if values else 0.0


def _metric_series(results: "Sequence[EvalResult]", key: str) -> "list[float]":
    out: "list[float]" = []
    for r in results:
        v = r.metrics.get(key)
        if isinstance(v, bool):
            out.append(1.0 if v else 0.0)
        elif isinstance(v, (int, float)):
            out.append(float(v))
    return out


def _percentile(sorted_vals: "Sequence[float]", p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    floor = int(k)
    ceil = min(floor + 1, len(sorted_vals) - 1)
    if floor == ceil:
        return sorted_vals[floor]
    return sorted_vals[floor] + (sorted_vals[ceil] - sorted_vals[floor]) * (k - floor)


def aggregate(
    results: "Sequence[EvalResult]",
    *,
    pass_threshold: float = _DEFAULT_PASS_THRESHOLD,
) -> "Mapping[str, float]":
    """Compute aggregate metrics over a set of EvalResults."""
    if not results:
        return {}
    evaluator_names: "set[str]" = set()
    for r in results:
        evaluator_names.update(r.scores.keys())
    agg: "dict[str, float]" = {}
    for name in sorted(evaluator_names):
        scores = [r.scores[name] for r in results if name in r.scores]
        agg[f"{name}_avg"] = _avg(scores)
        agg[f"{name}_pass_rate"] = _avg(
            [1.0 if s >= pass_threshold else 0.0 for s in scores]
        )
    agg["error_rate"] = sum(1 for r in results if r.error_type is not None) / len(
        results
    )
    latency = sorted(_metric_series(results, "latency_seconds"))
    agg["p50_latency_seconds"] = _percentile(latency, 0.5)
    agg["p95_latency_seconds"] = _percentile(latency, 0.95)
    agg["avg_tokens"] = _avg(_metric_series(results, "total_tokens"))
    agg["avg_cost"] = _avg(_metric_series(results, "total_cost"))
    agg["safety_refusal_rate"] = _avg(_metric_series(results, "safety_refusal"))
    agg["avg_retry_count"] = _avg(_metric_series(results, "retry_count"))
    agg["retry_rate"] = _avg(
        [1.0 if v > 0 else 0.0 for v in _metric_series(results, "retry_count")]
    )
    return agg


def compare(
    baseline: "Sequence[EvalResult]",
    candidate: "Sequence[EvalResult]",
    *,
    pass_threshold: float = _DEFAULT_PASS_THRESHOLD,
) -> "Mapping[str, float]":
    """Compare baseline vs candidate; a positive delta means the candidate is
    better. Lower-is-better metrics are inverted (by exact key) so the sign is
    uniform."""
    b = aggregate(baseline, pass_threshold=pass_threshold)
    c = aggregate(candidate, pass_threshold=pass_threshold)
    delta: "dict[str, float]" = {}
    for key in sorted(set(b) | set(c)):
        diff = c.get(key, 0.0) - b.get(key, 0.0)
        if key in _LOWER_IS_BETTER_KEYS:
            diff = -diff
        delta[key] = diff
    return delta


__all__: "list[str]" = ["aggregate", "compare"]
