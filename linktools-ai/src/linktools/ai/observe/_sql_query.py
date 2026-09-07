#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL execution for Metrics query pushdown."""

from __future__ import annotations

import calendar
import json
import math
from datetime import timezone
from decimal import Decimal
from typing import TYPE_CHECKING

from ..errors import AIError, ErrorCode
from ._model import MetricAggregation, MetricSourceKind, MetricType
from ._store import (
    _MetricQueryPushdownPlan,
    _MetricQueryPushdownResult,
    _MetricQueryPushdownRow,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy.ext.asyncio import AsyncSession


_COUNT_AGGREGATIONS = frozenset(
    {
        MetricAggregation.COUNT,
        MetricAggregation.SUM,
        MetricAggregation.RATE,
    }
)
_INDICATOR_AGGREGATIONS = frozenset(
    {
        MetricAggregation.COUNT,
        MetricAggregation.SUM,
        MetricAggregation.MEAN,
        MetricAggregation.RATE,
    }
)
_MEASUREMENT_AGGREGATIONS = frozenset(MetricAggregation)
_SELECTED_MEASUREMENT_AGGREGATIONS = frozenset(
    {
        MetricAggregation.MIN,
        MetricAggregation.MAX,
        MetricAggregation.LATEST,
        MetricAggregation.PERCENTILE,
    }
)
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def sql_query_pushdown_supported(plan: _MetricQueryPushdownPlan) -> bool:
    if plan.source_kind is MetricSourceKind.OBSERVATION_COUNT:
        return plan.aggregation in _COUNT_AGGREGATIONS
    if plan.source_kind is MetricSourceKind.INDICATOR:
        return plan.aggregation in _INDICATOR_AGGREGATIONS
    if plan.source_kind is not MetricSourceKind.MEASUREMENT:
        return False
    if plan.aggregation not in _MEASUREMENT_AGGREGATIONS:
        return False
    if plan.measurement_name is None or plan.measurement_revision is None:
        return False
    if plan.aggregation is MetricAggregation.PERCENTILE:
        return plan.percentile is not None
    return plan.percentile is None


async def execute_sql_metric_query(
    session: "AsyncSession",
    *,
    namespace_key: str,
    dialect_name: str,
    plan: _MetricQueryPushdownPlan,
) -> _MetricQueryPushdownResult | None:
    if dialect_name not in {"sqlite", "mysql", "postgresql"}:
        return None
    if not sql_query_pushdown_supported(plan):
        return None

    from sqlalchemy import text

    params = _base_params(namespace_key, dialect_name, plan)
    scanned = await session.scalar(text(_scan_count_sql()), params)
    if scanned is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    scanned_count = int(scanned)
    if scanned_count > plan.max_scanned_observations:
        raise AIError(ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED)

    if (
        plan.source_kind is MetricSourceKind.OBSERVATION_COUNT
        and not plan.filters
        and not plan.correlation_filters
        and not plan.group_by
        and plan.bucket_microseconds is None
    ):
        return _MetricQueryPushdownResult(
            rows=(
                _MetricQueryPushdownRow(
                    group=(),
                    bucket_index=None,
                    sample_count=scanned_count,
                    sample_sum=scanned_count,
                ),
            )
        )

    if plan.source_kind is MetricSourceKind.MEASUREMENT:
        statement, statement_params = _measurement_sql(dialect_name, plan, params)
        rows = (await session.execute(text(statement), statement_params)).mappings().all()
        return _decode_measurement_rows(rows, plan)

    statement, statement_params = _aggregate_sql(dialect_name, plan, params)
    rows = (await session.execute(text(statement), statement_params)).mappings().all()
    return _decode_aggregate_rows(rows, plan)


def _scan_count_sql() -> str:
    return """
SELECT COUNT(*)
FROM (
    SELECT 1
    FROM ai_metric_observations
    WHERE namespace_digest = :namespace_key
      AND kind = :kind
      AND occurred_at >= :window_start
      AND occurred_at < :window_end
    LIMIT :scan_cap
) AS bounded_metric_scan
"""


def _base_params(
    namespace_key: str,
    dialect_name: str,
    plan: _MetricQueryPushdownPlan,
) -> dict[str, object]:
    start_utc = plan.start.astimezone(timezone.utc)
    end_utc = plan.end.astimezone(timezone.utc)
    if dialect_name == "sqlite":
        window_start: object = start_utc.replace(tzinfo=None).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )
        window_end: object = end_utc.replace(tzinfo=None).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )
    elif dialect_name == "mysql":
        window_start = start_utc.replace(tzinfo=None)
        window_end = end_utc.replace(tzinfo=None)
    else:
        window_start = start_utc
        window_end = end_utc

    partition_limit = (
        (plan.max_groups if plan.group_by else 1) * plan.bucket_count
    )
    params: dict[str, object] = {
        "namespace_key": namespace_key,
        "kind": plan.observation_kind,
        "window_start": window_start,
        "window_end": window_end,
        "scan_cap": plan.max_scanned_observations + 1,
        "result_cap": min(plan.max_result_points, partition_limit) + 1,
        "max_extracted": plan.max_extracted_samples,
        "int64_min": _INT64_MIN,
        "int64_max": _INT64_MAX,
    }
    if plan.bucket_microseconds is not None:
        params["bucket_us"] = plan.bucket_microseconds
        if dialect_name == "sqlite":
            params["start_epoch_seconds"] = calendar.timegm(start_utc.utctimetuple())
            params["start_microsecond"] = start_utc.microsecond
    return params


def _aggregate_sql(
    dialect_name: str,
    plan: _MetricQueryPushdownPlan,
    base_params: dict[str, object],
) -> tuple[str, dict[str, object]]:
    params = dict(base_params)
    select_parts, group_aliases = _projection_parts(dialect_name, plan)
    conditions = _filter_conditions(dialect_name, plan, params)
    group_columns = [*group_aliases]
    if plan.bucket_microseconds is not None:
        group_columns.append("bucket_index")

    source_select = ",\n        ".join(
        [*select_parts, "b.payload_json AS payload_json"]
    )
    if not select_parts:
        source_select = "b.payload_json AS payload_json"

    if plan.source_kind is MetricSourceKind.INDICATOR:
        indicator = _indicator_condition(dialect_name, plan, params, alias="f")
        sample_sum = f"SUM(CASE WHEN {indicator} THEN 1 ELSE 0 END)"
    else:
        sample_sum = "COUNT(*)"

    aggregate_select = [
        *group_columns,
        "COUNT(*) AS sample_count",
        f"{sample_sum} AS sample_sum",
    ]
    aggregate_sql = ",\n        ".join(aggregate_select)
    group_sql = f"\n    GROUP BY {', '.join(group_columns)}" if group_columns else ""
    order_sql = f"\nORDER BY {', '.join(group_columns)}" if group_columns else ""
    where_sql = "\n      AND " + "\n      AND ".join(conditions) if conditions else ""

    sql = f"""
WITH filtered AS (
    SELECT
        {source_select}
    FROM ai_metric_observations AS b
    WHERE b.namespace_digest = :namespace_key
      AND b.kind = :kind
      AND b.occurred_at >= :window_start
      AND b.occurred_at < :window_end{where_sql}
), aggregated AS (
    SELECT
        {aggregate_sql}
    FROM filtered AS f{group_sql}
    LIMIT :result_cap
)
SELECT *
FROM aggregated{order_sql}
"""
    return sql, params


def _measurement_sql(
    dialect_name: str,
    plan: _MetricQueryPushdownPlan,
    base_params: dict[str, object],
) -> tuple[str, dict[str, object]]:
    params = dict(base_params)
    params["measurement_name"] = plan.measurement_name
    params["measurement_revision"] = plan.measurement_revision
    if plan.percentile is not None:
        params["percentile"] = plan.percentile

    select_parts, group_aliases = _projection_parts(dialect_name, plan)
    conditions = _filter_conditions(dialect_name, plan, params)
    partition_columns = [*group_aliases]
    if plan.bucket_microseconds is not None:
        partition_columns.append("bucket_index")

    filtered_select = ",\n        ".join(
        [
            *select_parts,
            "b.occurred_at AS occurred_at",
            "b.observation_digest AS observation_digest",
            "b.payload_json AS payload_json",
        ]
    )
    where_sql = "\n      AND " + "\n      AND ".join(conditions) if conditions else ""
    measurement_sql = _measurement_source_sql(dialect_name, partition_columns)
    valid_predicate = _measurement_valid_predicate(dialect_name, plan)
    reduced_sql = _measurement_reduce_sql(dialect_name, plan, partition_columns)

    result_columns = [*partition_columns, "sample_count"]
    if plan.aggregation in _SELECTED_MEASUREMENT_AGGREGATIONS:
        result_columns.append("selected_value")
    elif plan.aggregation is not MetricAggregation.COUNT:
        result_columns.append("sample_sum")
    selected_columns = ", ".join(f"p.{column}" for column in result_columns)
    if selected_columns:
        selected_columns = ",\n    " + selected_columns
    order_sql = (
        ", ".join(f"p.{column}" for column in partition_columns)
        if partition_columns
        else "p.sample_count"
    )

    sql = f"""
WITH filtered AS (
    SELECT
        {filtered_select}
    FROM ai_metric_observations AS b
    WHERE b.namespace_digest = :namespace_key
      AND b.kind = :kind
      AND b.occurred_at >= :window_start
      AND b.occurred_at < :window_end{where_sql}
), samples_raw AS (
    {measurement_sql}
), sample_stats AS (
    SELECT
        COUNT(*) AS extracted_count,
        COALESCE(SUM(CASE WHEN {valid_predicate} THEN 0 ELSE 1 END), 0) AS invalid_count
    FROM samples_raw AS r
), valid_samples AS (
    SELECT r.*
    FROM samples_raw AS r
    CROSS JOIN sample_stats AS s
    WHERE s.extracted_count <= :max_extracted
      AND s.invalid_count = 0
      AND {valid_predicate}
), reduced AS (
    {reduced_sql}
    LIMIT :result_cap
)
SELECT
    s.extracted_count,
    s.invalid_count{selected_columns}
FROM sample_stats AS s
LEFT JOIN reduced AS p ON 1 = 1
ORDER BY {order_sql}
"""
    return sql, params


def _projection_parts(
    dialect_name: str,
    plan: _MetricQueryPushdownPlan,
) -> tuple[list[str], list[str]]:
    parts: list[str] = []
    aliases: list[str] = []
    for index, field in enumerate(plan.group_by):
        alias = f"g{index}"
        parts.append(f"{_facet_text_expression(dialect_name, 'b', field)} AS {alias}")
        aliases.append(alias)
    if plan.bucket_microseconds is not None:
        parts.append(f"{_bucket_expression(dialect_name)} AS bucket_index")
    return parts, aliases


def _bucket_expression(dialect_name: str) -> str:
    if dialect_name == "sqlite":
        return (
            "CAST((((CAST(strftime('%s', substr(CAST(b.occurred_at AS TEXT), 1, 19)) AS INTEGER) "
            "- :start_epoch_seconds) * 1000000 + "
            "CAST(COALESCE(NULLIF(substr(CAST(b.occurred_at AS TEXT), 21, 6), ''), '0') AS INTEGER) "
            "- :start_microsecond) / :bucket_us) AS INTEGER)"
        )
    if dialect_name == "mysql":
        return "(TIMESTAMPDIFF(MICROSECOND, :window_start, b.occurred_at) DIV :bucket_us)"
    return (
        "CAST(FLOOR(EXTRACT(EPOCH FROM (b.occurred_at - :window_start)) "
        "* 1000000 / :bucket_us) AS BIGINT)"
    )


def _filter_conditions(
    dialect_name: str,
    plan: _MetricQueryPushdownPlan,
    params: dict[str, object],
) -> list[str]:
    conditions: list[str] = []
    for index, (field, value) in enumerate(plan.filters):
        parameter = f"filter_{index}"
        params[parameter] = value
        conditions.append(
            _text_equality_condition(
                dialect_name,
                alias="b",
                parts=_facet_parts(field),
                parameter=parameter,
            )
        )
    for index, (field, value) in enumerate(plan.correlation_filters):
        parameter = f"correlation_{index}"
        params[parameter] = value
        parts = ("observation", "correlation", field)
        if isinstance(value, str):
            conditions.append(
                _text_equality_condition(
                    dialect_name,
                    alias="b",
                    parts=parts,
                    parameter=parameter,
                )
            )
        else:
            conditions.append(
                _integer_equality_condition(
                    dialect_name,
                    alias="b",
                    parts=parts,
                    parameter=parameter,
                )
            )
    return conditions


def _indicator_condition(
    dialect_name: str,
    plan: _MetricQueryPushdownPlan,
    params: dict[str, object],
    *,
    alias: str,
) -> str:
    field = plan.indicator_field
    if field not in {"status", "error_code"} or not plan.indicator_values:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    values = []
    for index, value in enumerate(plan.indicator_values):
        parameter = f"indicator_{index}"
        params[parameter] = value
        values.append(f":{parameter}")
    parts = ("observation", field)
    text_value = _json_text_expression(dialect_name, alias, parts)
    return (
        f"{_json_type_condition(dialect_name, alias, parts, string=True)} "
        f"AND {text_value} IN ({', '.join(values)})"
    )


def _measurement_source_sql(
    dialect_name: str,
    partition_columns: list[str],
) -> str:
    prefix = ", ".join(f"f.{column}" for column in partition_columns)
    if prefix:
        prefix += ",\n        "
    common = (
        f"{prefix}f.occurred_at AS occurred_at,\n"
        "        f.observation_digest AS observation_digest,\n        "
    )
    if dialect_name == "sqlite":
        raw_value = "json_extract(m.value, '$.value')"
        value_type = "json_type(m.value, '$.value')"
        numeric_value = (
            f"CASE WHEN {value_type} IN ('integer', 'real') "
            f"THEN {raw_value} ELSE NULL END"
        )
        return f"""SELECT
        {common}{numeric_value} AS numeric_value,
        {raw_value} AS raw_value,
        {value_type} AS value_type
    FROM filtered AS f
    JOIN json_each(f.payload_json, '$.observation.measurements') AS m
      ON 1 = 1
    WHERE json_type(m.value, '$.name') = 'text'
      AND json_extract(m.value, '$.name') = :measurement_name
      AND json_type(m.value, '$.revision') = 'integer'
      AND json_extract(m.value, '$.revision') = :measurement_revision"""
    if dialect_name == "mysql":
        raw_value = (
            "JSON_EXTRACT(f.payload_json, CONCAT('$.observation.measurements[', "
            "m.ordinality - 1, '].value'))"
        )
        value_type = f"JSON_TYPE({raw_value})"
        numeric_value = (
            "CASE "
            f"WHEN {value_type} = 'INTEGER' THEN "
            f"CAST(JSON_UNQUOTE({raw_value}) AS DECIMAL(65, 0)) "
            f"WHEN {value_type} = 'DOUBLE' THEN "
            f"CAST(JSON_UNQUOTE({raw_value}) AS DOUBLE) "
            "ELSE NULL END"
        )
        return f"""SELECT
        {common}{numeric_value} AS numeric_value,
        {raw_value} AS raw_value,
        {value_type} AS value_type
    FROM filtered AS f
    JOIN JSON_TABLE(
        JSON_EXTRACT(f.payload_json, '$.observation.measurements'),
        '$[*]' COLUMNS(
            ordinality FOR ORDINALITY,
            measurement_name VARCHAR(128) PATH '$.name' NULL ON EMPTY NULL ON ERROR,
            measurement_revision BIGINT PATH '$.revision' NULL ON EMPTY NULL ON ERROR
        )
    ) AS m
      ON 1 = 1
    WHERE m.measurement_name = :measurement_name
      AND m.measurement_revision = :measurement_revision"""
    raw_value = "m.value ->> 'value'"
    value_type = "json_typeof(m.value -> 'value')"
    numeric_value = (
        f"CASE WHEN {value_type} = 'number' "
        f"THEN CAST({raw_value} AS NUMERIC) ELSE NULL END"
    )
    return f"""SELECT
        {common}{numeric_value} AS numeric_value,
        {raw_value} AS raw_value,
        {value_type} AS value_type
    FROM filtered AS f
    CROSS JOIN LATERAL json_array_elements(
        f.payload_json -> 'observation' -> 'measurements'
    ) AS m(value)
    WHERE json_typeof(m.value -> 'name') = 'string'
      AND m.value ->> 'name' = :measurement_name
      AND json_typeof(m.value -> 'revision') = 'number'
      AND m.value ->> 'revision' ~ '^[0-9]+$'
      AND CAST(m.value ->> 'revision' AS BIGINT) = :measurement_revision"""


def _measurement_valid_predicate(
    dialect_name: str,
    plan: _MetricQueryPushdownPlan,
) -> str:
    if dialect_name == "sqlite":
        base = (
            "r.value_type IN ('integer', 'real') AND "
            "(r.value_type != 'integer' OR "
            "r.numeric_value BETWEEN :int64_min AND :int64_max)"
        )
    elif dialect_name == "mysql":
        base = (
            "r.value_type IN ('INTEGER', 'DOUBLE') AND "
            "(r.value_type != 'INTEGER' OR "
            "CAST(JSON_UNQUOTE(r.raw_value) AS DECIMAL(65, 0)) "
            "BETWEEN :int64_min AND :int64_max)"
        )
    else:
        base = (
            "r.value_type = 'number' AND "
            "(r.raw_value !~ '^-?[0-9]+$' OR "
            "CAST(r.raw_value AS NUMERIC) BETWEEN :int64_min AND :int64_max)"
        )
    if plan.metric_type is MetricType.COUNTER:
        return f"({base}) AND r.numeric_value >= 0"
    if plan.metric_type is MetricType.RATIO:
        return f"({base}) AND r.numeric_value >= 0 AND r.numeric_value <= 1"
    return base


def _measurement_reduce_sql(
    dialect_name: str,
    plan: _MetricQueryPushdownPlan,
    partition_columns: list[str],
) -> str:
    aggregation = plan.aggregation
    if aggregation is MetricAggregation.PERCENTILE:
        return _percentile_pick_sql(dialect_name, partition_columns)
    if aggregation in {
        MetricAggregation.MIN,
        MetricAggregation.MAX,
        MetricAggregation.LATEST,
    }:
        return _measurement_pick_sql(dialect_name, aggregation, partition_columns)
    if aggregation in {
        MetricAggregation.COUNT,
        MetricAggregation.SUM,
        MetricAggregation.MEAN,
        MetricAggregation.RATE,
    }:
        return _measurement_sum_sql(aggregation, partition_columns)
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _measurement_sum_sql(
    aggregation: MetricAggregation,
    partition_columns: list[str],
) -> str:
    select_prefix = ", ".join(partition_columns)
    if select_prefix:
        select_prefix += ",\n        "
    group_sql = f"\n    GROUP BY {', '.join(partition_columns)}" if partition_columns else ""
    sample_sum = (
        "NULL AS sample_sum"
        if aggregation is MetricAggregation.COUNT
        else "SUM(numeric_value) AS sample_sum"
    )
    return f"""SELECT
        {select_prefix}COUNT(*) AS sample_count,
        {sample_sum}
    FROM valid_samples{group_sql}"""


def _measurement_pick_sql(
    dialect_name: str,
    aggregation: MetricAggregation,
    partition_columns: list[str],
) -> str:
    select_prefix = ", ".join(partition_columns)
    if select_prefix:
        select_prefix += ",\n        "
    partition_sql = (
        f"PARTITION BY {', '.join(partition_columns)} "
        if partition_columns
        else ""
    )
    if aggregation is MetricAggregation.LATEST:
        order_sql = "occurred_at DESC, observation_digest DESC"
    else:
        value = "raw_value" if dialect_name == "mysql" else "numeric_value"
        direction = "ASC" if aggregation is MetricAggregation.MIN else "DESC"
        order_sql = (
            f"{value} {direction}, occurred_at ASC, observation_digest ASC"
        )
    return f"""SELECT
        {select_prefix}sample_count,
        raw_value AS selected_value
    FROM (
        SELECT
            {select_prefix}raw_value,
            COUNT(*) OVER ({partition_sql}) AS sample_count,
            ROW_NUMBER() OVER ({partition_sql}ORDER BY {order_sql}) AS sample_rank
        FROM valid_samples
    ) AS ranked
    WHERE sample_rank = 1"""


def _percentile_pick_sql(
    dialect_name: str,
    partition_columns: list[str],
) -> str:
    select_prefix = ", ".join(partition_columns)
    if select_prefix:
        select_prefix += ",\n        "
    group_sql = f"\n    GROUP BY {', '.join(partition_columns)}" if partition_columns else ""
    if dialect_name == "postgresql":
        return f"""SELECT
        {select_prefix}COUNT(*) AS sample_count,
        percentile_disc(:percentile) WITHIN GROUP (ORDER BY numeric_value) AS selected_value
    FROM valid_samples{group_sql}"""

    partition_sql = (
        f"PARTITION BY {', '.join(partition_columns)} "
        if partition_columns
        else ""
    )
    order_value = "raw_value" if dialect_name == "mysql" else "numeric_value"
    if dialect_name == "sqlite":
        target_rank = (
            "CAST(:percentile * sample_count AS INTEGER) + "
            "CASE WHEN :percentile * sample_count > "
            "CAST(:percentile * sample_count AS INTEGER) THEN 1 ELSE 0 END"
        )
    else:
        target_rank = "CAST(CEIL(:percentile * sample_count) AS UNSIGNED)"

    return f"""SELECT
        {select_prefix}sample_count,
        raw_value AS selected_value
    FROM (
        SELECT
            {select_prefix}raw_value,
            COUNT(*) OVER ({partition_sql}) AS sample_count,
            ROW_NUMBER() OVER (
                {partition_sql}ORDER BY {order_value} ASC, occurred_at ASC, observation_digest ASC
            ) AS sample_rank
        FROM valid_samples
    ) AS ranked
    WHERE sample_rank = {target_rank}"""


def _facet_parts(field: str) -> tuple[str, ...]:
    if field in {"source_namespace", "tenant_id", "status", "error_code"}:
        return ("observation", field)
    return ("observation", "dimensions", field)


def _facet_text_expression(dialect_name: str, alias: str, field: str) -> str:
    return _json_text_expression(dialect_name, alias, _facet_parts(field))


def _text_equality_condition(
    dialect_name: str,
    *,
    alias: str,
    parts: tuple[str, ...],
    parameter: str,
) -> str:
    text_value = _json_text_expression(dialect_name, alias, parts)
    return (
        f"{_json_type_condition(dialect_name, alias, parts, string=True)} "
        f"AND {text_value} = :{parameter}"
    )


def _integer_equality_condition(
    dialect_name: str,
    *,
    alias: str,
    parts: tuple[str, ...],
    parameter: str,
) -> str:
    raw = _json_raw_expression(dialect_name, alias, parts)
    text_value = _json_text_expression(dialect_name, alias, parts)
    if dialect_name == "sqlite":
        return (
            f"json_type({alias}.payload_json, '{_json_path(parts)}') = 'integer' "
            f"AND {raw} = :{parameter}"
        )
    if dialect_name == "mysql":
        return (
            f"JSON_TYPE({raw}) = 'INTEGER' "
            f"AND CAST({text_value} AS SIGNED) = :{parameter}"
        )
    return (
        f"json_typeof({raw}) = 'number' AND {text_value} ~ '^-?[0-9]+$' "
        f"AND CAST({text_value} AS BIGINT) = :{parameter}"
    )


def _json_type_condition(
    dialect_name: str,
    alias: str,
    parts: tuple[str, ...],
    *,
    string: bool,
) -> str:
    raw = _json_raw_expression(dialect_name, alias, parts)
    if dialect_name == "sqlite":
        expected = "text" if string else "integer"
        return f"json_type({alias}.payload_json, '{_json_path(parts)}') = '{expected}'"
    if dialect_name == "mysql":
        expected = "STRING" if string else "INTEGER"
        return f"JSON_TYPE({raw}) = '{expected}'"
    expected = "string" if string else "number"
    return f"json_typeof({raw}) = '{expected}'"


def _json_raw_expression(
    dialect_name: str,
    alias: str,
    parts: tuple[str, ...],
) -> str:
    if dialect_name == "sqlite":
        return f"json_extract({alias}.payload_json, '{_json_path(parts)}')"
    if dialect_name == "mysql":
        return f"JSON_EXTRACT({alias}.payload_json, '{_json_path(parts)}')"
    arguments = ", ".join(f"'{part}'" for part in parts)
    return f"json_extract_path({alias}.payload_json, {arguments})"


def _json_text_expression(
    dialect_name: str,
    alias: str,
    parts: tuple[str, ...],
) -> str:
    if dialect_name == "sqlite":
        return _json_raw_expression(dialect_name, alias, parts)
    if dialect_name == "mysql":
        raw = _json_raw_expression(dialect_name, alias, parts)
        return (
            f"CASE WHEN JSON_TYPE({raw}) = 'NULL' THEN NULL "
            f"ELSE JSON_UNQUOTE({raw}) END"
        )
    arguments = ", ".join(f"'{part}'" for part in parts)
    return f"json_extract_path_text({alias}.payload_json, {arguments})"


def _json_path(parts: tuple[str, ...]) -> str:
    return "$" + "".join(f'."{part}"' for part in parts)


def _decode_aggregate_rows(
    rows: list["Mapping[str, object]"],
    plan: _MetricQueryPushdownPlan,
) -> _MetricQueryPushdownResult:
    if len(rows) > plan.max_result_points:
        raise AIError(ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED)
    decoded = tuple(
        _decode_row(row, plan, selected=False)
        for row in rows
    )
    _validate_groups(decoded, plan)
    return _MetricQueryPushdownResult(rows=decoded)


def _decode_measurement_rows(
    rows: list["Mapping[str, object]"],
    plan: _MetricQueryPushdownPlan,
) -> _MetricQueryPushdownResult:
    if not rows:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    extracted = int(rows[0]["extracted_count"])
    invalid = int(rows[0]["invalid_count"])
    if invalid:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if extracted > plan.max_extracted_samples:
        raise AIError(ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED)

    materialized = [row for row in rows if row.get("sample_count") is not None]
    if len(materialized) > plan.max_result_points:
        raise AIError(ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED)
    selected = plan.aggregation in _SELECTED_MEASUREMENT_AGGREGATIONS
    decoded = tuple(
        _decode_row(row, plan, selected=selected)
        for row in materialized
    )
    if sum(row.sample_count for row in decoded) != extracted:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _validate_groups(decoded, plan)
    return _MetricQueryPushdownResult(rows=decoded)


def _decode_row(
    row: "Mapping[str, object]",
    plan: _MetricQueryPushdownPlan,
    *,
    selected: bool,
) -> _MetricQueryPushdownRow:
    group_values = []
    for index in range(len(plan.group_by)):
        value = row[f"g{index}"]
        if value is not None and not isinstance(value, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        group_values.append(value)

    bucket_index = row.get("bucket_index")
    if bucket_index is not None:
        bucket_index = int(bucket_index)
        if not 0 <= bucket_index < plan.bucket_count:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    sample_count = int(row["sample_count"])
    if sample_count < 0:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    selected_value = (
        _coerce_sample_number(row.get("selected_value")) if selected else None
    )
    sample_sum = (
        None if selected else _coerce_aggregate_number(row.get("sample_sum"))
    )
    return _MetricQueryPushdownRow(
        group=tuple(group_values),
        bucket_index=bucket_index,
        sample_count=sample_count,
        sample_sum=sample_sum,
        selected_value=selected_value,
    )


def _coerce_sample_number(value: object) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if isinstance(value, int):
        if not _INT64_MIN <= value <= _INT64_MAX:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return 0.0 if value == 0.0 else value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if value == value.to_integral_value() and _INT64_MIN <= value <= _INT64_MAX:
            return int(value)
        floating = float(value)
        if not math.isfinite(floating):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return 0.0 if floating == 0.0 else floating
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        return _coerce_sample_number(decoded)
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _coerce_aggregate_number(value: object) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return 0.0 if value == 0.0 else value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if value == value.to_integral_value():
            return int(value)
        floating = float(value)
        if not math.isfinite(floating):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return 0.0 if floating == 0.0 else floating
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            try:
                decoded_decimal = Decimal(value)
            except Exception as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            return _coerce_aggregate_number(decoded_decimal)
        return _coerce_aggregate_number(decoded)
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _validate_groups(
    rows: tuple[_MetricQueryPushdownRow, ...],
    plan: _MetricQueryPushdownPlan,
) -> None:
    if plan.group_by:
        groups = {row.group for row in rows}
        if len(groups) > plan.max_groups:
            raise AIError(ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED)
        if len(groups) * plan.bucket_count > plan.max_result_points:
            raise AIError(ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED)
    elif plan.bucket_count > plan.max_result_points:
        raise AIError(ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED)


__all__: list[str] = []
