#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`lt ai-metrics`: inspect local Runtime metrics."""

import asyncio
from argparse import Namespace
from typing import TYPE_CHECKING

from linktools.ai.errors import AIError
from linktools.ai.observe import (
    MetricPoint,
    MetricQuery,
    MetricQueryResult,
    Metrics,
    MetricWindow,
)
from linktools.cli import BaseCommand, CommandError

from ._ai_common import _load_workspace, _local_metrics

if TYPE_CHECKING:
    from linktools.cli import CommandParser

_SUMMARY_METRICS = (
    ("Executions", "linktools.execution.count"),
    ("Execution failures", "linktools.execution.failure_ratio"),
    ("Execution latency", "linktools.execution.latency"),
    ("Model requests", "linktools.model.request.count"),
    ("Model failures", "linktools.model.request.failure_ratio"),
    ("Model latency", "linktools.model.request.latency"),
    ("Tool executions", "linktools.tool.execution.count"),
    ("Tool failures", "linktools.tool.execution.failure_ratio"),
    ("Input tokens", "linktools.model.input_tokens"),
    ("Output tokens", "linktools.model.output_tokens"),
    ("Cache read tokens", "linktools.model.cache_read_tokens"),
    ("Cache write tokens", "linktools.model.cache_write_tokens"),
)


class Command(BaseCommand):
    """Inspect local AI runtime metrics."""

    @property
    def name(self) -> str:
        return "ai-metrics"

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("metric", nargs="?", help="metric name")

    def run(self, args: Namespace) -> int:
        workspace = _load_workspace()

        async def execute() -> int:
            metrics = await _local_metrics(workspace)
            if args.metric is None:
                _emit_summary(await _query_summary(metrics))
                return 0
            _emit_metric(await _query_metric(metrics, args.metric))
            return 0

        try:
            return asyncio.run(execute())
        except (AIError, TypeError, ValueError) as error:
            raise CommandError(str(error)) from error


async def _query_metric(metrics: Metrics, metric: str) -> MetricQueryResult:
    return await metrics.query(
        MetricQuery(
            metric,
            MetricWindow.recent(days=1),
        )
    )


async def _query_summary(
    metrics: Metrics,
) -> tuple[tuple[str, str, MetricQueryResult], ...]:
    values: list[tuple[str, str, MetricQueryResult]] = []
    for label, metric in _SUMMARY_METRICS:
        values.append((label, metric, await _query_metric(metrics, metric)))
    return tuple(values)


def _emit_summary(values: tuple[tuple[str, str, MetricQueryResult], ...]) -> None:
    print("Last 24 hours")
    width = max(len(label) for label, _, _ in values)
    for label, metric, result in values:
        point = _single_point(result)
        print(f"{label:<{width}}  {_format_value(metric, result.unit, point.value)}")


def _emit_metric(result: MetricQueryResult) -> None:
    point = _single_point(result)
    print(f"Metric:      {result.metric}")
    print(f"Aggregation: {result.aggregation.value}")
    print(f"Unit:        {result.unit}")
    print(
        "Window:      "
        f"{result.window_start.isoformat()} .. {result.window_end.isoformat()}"
    )
    print(f"Value:       {_format_value(result.metric, result.unit, point.value)}")
    print(f"Samples:     {point.sample_count}")


def _single_point(result: MetricQueryResult) -> MetricPoint:
    if len(result.points) != 1:
        raise RuntimeError("ungrouped metric query must return one point")
    return result.points[0]


def _format_value(metric: str, unit: str, value: "int | float | None") -> str:
    if value is None:
        return "-"
    if metric.endswith("_ratio"):
        return f"{float(value) * 100:.2f}%"
    if unit == "ns":
        return _format_duration_ns(float(value))
    if isinstance(value, int):
        return f"{value:,}"
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.3f}"


def _format_duration_ns(value: float) -> str:
    seconds = value / 1_000_000_000
    if seconds >= 1:
        return f"{seconds:.3f}s"
    milliseconds = value / 1_000_000
    if milliseconds >= 1:
        return f"{milliseconds:.3f}ms"
    microseconds = value / 1_000
    if microseconds >= 1:
        return f"{microseconds:.3f}us"
    return f"{value:.0f}ns"


command = Command()
