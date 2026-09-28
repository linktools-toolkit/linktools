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
_SUM_LIMB_BASE = 2**32


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
    namespace: str = "",
    verify: bool = False,
) -> _MetricQueryPushdownResult | None:
    if dialect_name not in {"sqlite", "mysql", "postgresql"}:
        return None
    if not sql_query_pushdown_supported(plan):
        return None

    from sqlalchemy import text

    params = _base_params(namespace_key, dialect_name, plan)
    if verify:
        params["verified_namespace"] = namespace
    if (
        not verify
        and plan.source_kind is MetricSourceKind.OBSERVATION_COUNT
        and not plan.filters
        and not plan.correlation_filters
        and not plan.group_by
        and plan.bucket_microseconds is None
    ):
        scanned = await session.scalar(text(_scan_count_sql()), params)
        if scanned is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        scanned_count = int(scanned)
        if scanned_count > plan.max_scanned_observations:
            raise AIError(ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED)
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

    if not await _sql_features_available(session, dialect_name, plan):
        return None

    if plan.source_kind is MetricSourceKind.MEASUREMENT:
        statement, statement_params = _measurement_sql(dialect_name, plan, params, verify=verify)
        rows = (await session.execute(text(statement), statement_params)).mappings().all()
        return _decode_measurement_rows(rows, plan)

    statement, statement_params = _aggregate_sql(dialect_name, plan, params, verify=verify)
    rows = (await session.execute(text(statement), statement_params)).mappings().all()
    return _decode_aggregate_rows(rows, plan)


async def _sql_features_available(
    session: "AsyncSession",
    dialect_name: str,
    plan: _MetricQueryPushdownPlan,
) -> bool:
    connection = await session.connection()
    version = connection.dialect.server_version_info
    if version is None:
        return False
    if dialect_name == "mysql":
        minimum = (
            (8, 0, 17)
            if plan.source_kind is MetricSourceKind.MEASUREMENT
            else (8, 0, 1)
        )
        return version >= minimum
    if dialect_name == "postgresql":
        minimum = (9, 4) if plan.aggregation is MetricAggregation.PERCENTILE else (9, 3)
        return version >= minimum

    window = plan.aggregation in {
        MetricAggregation.LATEST,
        MetricAggregation.PERCENTILE,
        MetricAggregation.SUM,
        MetricAggregation.MEAN,
        MetricAggregation.RATE,
    }
    if version < ((3, 25, 0) if window else (3, 9, 0)):
        return False

    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError

    window_column = ", ROW_NUMBER() OVER ()" if window else ""
    probe = (
        "SELECT json_type(value), json_extract(value, '$')"
        f"{window_column} FROM json_each('[1]')"
    )
    try:
        await session.execute(text(probe))
    except OperationalError as error:
        if str(error.orig) in {
            "no such function: json_type",
            "no such function: json_extract",
            "no such table: json_each",
        }:
            return False
        raise
    return True


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


def _bounded_observations_sql(
    columns: tuple[str, ...], *, verify: bool = False
) -> str:
    if verify:
        columns = tuple(dict.fromkeys((
            *columns, "namespace_digest", "kind", "occurred_at",
            "observation_digest", "payload_digest", "payload_json",
        )))
    projection = ", ".join(columns)
    bounded = f"""bounded_observations AS (
    SELECT {projection}
    FROM ai_metric_observations
    WHERE namespace_digest = :namespace_key
      AND kind = :kind
      AND occurred_at >= :window_start
      AND occurred_at < :window_end
    LIMIT :scan_cap
), scan_stats AS (
    SELECT COUNT(*) AS scanned_count
    FROM bounded_observations
)"""
    if not verify:
        return bounded
    # SQLite evaluates this scalar only after the capped admission succeeds.
    # Both validation and aggregation consume the same statement-local rows.
    verified = bounded + """, verified_scan AS (
    SELECT scanned_count,
        CASE WHEN scanned_count < :scan_cap THEN (
            SELECT MAX(linktools_metric_verify(
                :verified_namespace, :namespace_key, namespace_digest, observation_digest, payload_digest,
                kind, occurred_at, payload_json
            )) FROM bounded_observations
        ) ELSE NULL END AS record_error
    FROM scan_stats
    LIMIT -1 OFFSET 0
)"""
    verified_columns = ", ".join(
        "CASE WHEN t.record_error IS NULL THEN b.payload_json END AS payload_json"
        if column == "payload_json" else f"b.{column}"
        for column in columns
    )
    # SQL may reorder filters. Never feed unverified JSON into their functions.
    return verified + f""", verified_observations AS (
    SELECT {verified_columns}
    FROM bounded_observations AS b CROSS JOIN verified_scan AS t
    WHERE t.scanned_count < :scan_cap AND t.record_error IS NULL
)"""


def _result_row_limit(plan: _MetricQueryPushdownPlan) -> int:
    partition_limit = (plan.max_groups if plan.group_by else 1) * plan.bucket_count
    return min(plan.max_result_points, partition_limit)


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

    params: dict[str, object] = {
        "namespace_key": namespace_key,
        "kind": plan.observation_kind,
        "window_start": window_start,
        "window_end": window_end,
        "scan_cap": plan.max_scanned_observations + 1,
        "result_cap": _result_row_limit(plan) + 1,
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
    *,
    verify: bool = False,
) -> tuple[str, dict[str, object]]:
    params = dict(base_params)
    select_parts, group_columns = _projection_parts(dialect_name, plan)
    conditions = _filter_conditions(dialect_name, plan, params)
    if plan.bucket_microseconds is not None:
        group_columns.append("bucket_index")

    if plan.source_kind is MetricSourceKind.INDICATOR:
        indicator = _indicator_condition(dialect_name, plan, params, alias="b")
        sample = f"CASE WHEN {indicator} THEN 1 ELSE 0 END"
    else:
        sample = "1"
    source_select = ",\n        ".join([*select_parts, f"{sample} AS sample"])
    aggregate_select = ",\n        ".join(
        [*group_columns, "COUNT(*) AS sample_count", "SUM(sample) AS sample_sum"]
    )
    result_select = ", ".join(
        f"a.{column}" for column in [*group_columns, "sample_count", "sample_sum"]
    )
    group_sql = f"\n    GROUP BY {', '.join(group_columns)}" if group_columns else ""
    where_sql = "\n      AND " + "\n      AND ".join(conditions) if conditions else ""
    columns = (
        ("payload_json", "occurred_at")
        if plan.bucket_microseconds is not None
        else ("payload_json",)
    )
    bounded_sql = _bounded_observations_sql(columns, verify=verify)

    scan_relation = "verified_scan" if verify else "scan_stats"
    input_relation = "verified_observations" if verify else "bounded_observations"
    if verify:
        where_sql = "\n      AND t.record_error IS NULL" + where_sql
    record_error_column = "t.record_error, " if verify else ""
    sql = f"""
WITH {bounded_sql}, filtered AS (
    SELECT
        {source_select}
    FROM {input_relation} AS b
    CROSS JOIN {scan_relation} AS t
    WHERE t.scanned_count < :scan_cap{where_sql}
), aggregated AS (
    SELECT
        {aggregate_select}
    FROM filtered{group_sql}
    LIMIT :result_cap
)
SELECT t.scanned_count, {record_error_column}{result_select}
FROM {scan_relation} AS t
LEFT JOIN aggregated AS a ON 1 = 1
"""
    return sql, params


def _measurement_sql(
    dialect_name: str,
    plan: _MetricQueryPushdownPlan,
    base_params: dict[str, object],
    *,
    verify: bool = False,
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

    ordered = plan.aggregation in {
        MetricAggregation.LATEST,
        MetricAggregation.PERCENTILE,
        MetricAggregation.SUM,
        MetricAggregation.MEAN,
        MetricAggregation.RATE,
    } or (
        dialect_name != "sqlite"
        and plan.aggregation in {MetricAggregation.MIN, MetricAggregation.MAX}
    )
    columns = ["payload_json"]
    if ordered:
        columns.extend(("occurred_at", "observation_digest"))
    elif plan.bucket_microseconds is not None:
        columns.append("occurred_at")
    filtered_select = ",\n        ".join(
        [*select_parts, *(f"b.{column} AS {column}" for column in columns)]
    )
    where_sql = "\n      AND " + "\n      AND ".join(conditions) if conditions else ""
    measurement_sql = _measurement_source_sql(
        dialect_name, partition_columns, ordered=ordered
    )
    valid_predicate = _measurement_valid_predicate(dialect_name, plan)
    reduced_sql = _measurement_reduce_sql(dialect_name, plan, partition_columns)

    result_columns = [*partition_columns, "sample_count"]
    if plan.aggregation in _SELECTED_MEASUREMENT_AGGREGATIONS:
        result_columns.append("selected_value")
    elif plan.aggregation is not MetricAggregation.COUNT:
        result_columns.extend(
            ("sample_sum", "integer_sum_high", "integer_sum_low", "floating_count")
        )
    selected_columns = ", ".join(f"p.{column}" for column in result_columns)
    if selected_columns:
        selected_columns = ",\n    " + selected_columns
    bounded_sql = _bounded_observations_sql(tuple(columns), verify=verify)
    # Keep statistics and reduction from expanding the same JSON array twice.
    if dialect_name == "sqlite":
        measurement_sql += "\n    LIMIT -1 OFFSET 0"
    elif dialect_name == "mysql":
        measurement_sql += "\n    LIMIT 18446744073709551615"

    scan_relation = "verified_scan" if verify else "scan_stats"
    input_relation = "verified_observations" if verify else "bounded_observations"
    if verify:
        where_sql = "\n      AND t.record_error IS NULL" + where_sql
    record_error_column = "t.record_error, " if verify else ""
    sql = f"""
WITH {bounded_sql}, filtered AS (
    SELECT
        {filtered_select}
    FROM {input_relation} AS b
    CROSS JOIN {scan_relation} AS t
    WHERE t.scanned_count < :scan_cap{where_sql}
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
    t.scanned_count, {record_error_column}
    s.extracted_count,
    s.invalid_count{selected_columns}
FROM {scan_relation} AS t
CROSS JOIN sample_stats AS s
LEFT JOIN reduced AS p ON 1 = 1
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
    *,
    ordered: bool,
) -> str:
    prefix = ", ".join(f"f.{column}" for column in partition_columns)
    if prefix:
        prefix += ",\n        "
    common = prefix
    if ordered:
        common += (
            "f.occurred_at AS occurred_at,\n"
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
            f"WHEN {value_type} IN ('INTEGER', 'UNSIGNED INTEGER') THEN "
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
    WHERE CAST(m.measurement_name AS BINARY) = CAST(:measurement_name AS BINARY)
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
            "r.numeric_value BETWEEN :int64_min AND :int64_max) AND "
            "(r.value_type != 'real' OR ABS(r.numeric_value) <= 1.7976931348623157e308)"
        )
    elif dialect_name == "mysql":
        base = (
            "r.value_type IN ('INTEGER', 'UNSIGNED INTEGER', 'DOUBLE') AND "
            "(r.value_type NOT IN ('INTEGER', 'UNSIGNED INTEGER') OR "
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
    if dialect_name == "sqlite" and aggregation in {
        MetricAggregation.MIN,
        MetricAggregation.MAX,
    }:
        columns = ", ".join(partition_columns)
        prefix = f"{columns}, " if columns else ""
        group_sql = f" GROUP BY {columns}" if columns else ""
        function = "MIN" if aggregation is MetricAggregation.MIN else "MAX"
        return (
            f"SELECT {prefix}COUNT(*) AS sample_count, "
            f"{function}(numeric_value) AS selected_value "
            f"FROM valid_samples{group_sql}"
        )
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
        return _measurement_sum_sql(dialect_name, aggregation, partition_columns)
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _measurement_sum_sql(
    dialect_name: str,
    aggregation: MetricAggregation,
    partition_columns: list[str],
) -> str:
    select_prefix = ", ".join(partition_columns)
    if select_prefix:
        select_prefix += ",\n        "
    group_sql = f"\n    GROUP BY {', '.join(partition_columns)}" if partition_columns else ""
    if aggregation is MetricAggregation.COUNT:
        return f"""SELECT
        {select_prefix}COUNT(*) AS sample_count
    FROM valid_samples{group_sql}"""

    if dialect_name == "sqlite":
        return _sqlite_sum_sql(partition_columns)

    return _server_sum_sql(dialect_name, partition_columns)


def _server_sum_sql(dialect_name: str, partition_columns: list[str]) -> str:
    columns = ", ".join(partition_columns)
    prefix = f"{columns}, " if columns else ""
    source_prefix = "".join(f"v.{column}, " for column in partition_columns)
    group_sql = f" GROUP BY {columns}" if columns else ""
    partition_sql = f"PARTITION BY {columns} " if columns else ""
    source_partition = (
        "PARTITION BY " + ", ".join(f"v.{column}" for column in partition_columns) + " "
        if columns else ""
    )
    if dialect_name == "mysql":
        integer_test = "value_type IN ('INTEGER', 'UNSIGNED INTEGER')"
        integer_value = "CAST(JSON_UNQUOTE(raw_value) AS DECIMAL(65, 0))"
        floating_value = "CAST(JSON_UNQUOTE(v.raw_value) AS DOUBLE)"
        prefix_value = "CAST(integer_prefix AS DOUBLE)"
        equal = "<=>"
    else:
        integer_test = "raw_value ~ '^-?[0-9]+$'"
        integer_value = "numeric_value"
        floating_value = "CAST(v.raw_value AS DOUBLE PRECISION)"
        prefix_value = "CAST(integer_prefix AS DOUBLE PRECISION)"
        equal = "IS NOT DISTINCT FROM"
    join_sql = " AND ".join(
        f"v.{column} {equal} t.{column}" for column in partition_columns
    ) or "1 = 1"
    # Native ordered sums consume only the floating suffix. The integer prefix
    # is converted exactly once, at the same transition as the scan executor.
    common = f"""WITH totals AS (
        SELECT {prefix}COUNT(*) AS sample_count,
            SUM(CASE WHEN {integer_test} THEN {integer_value} ELSE NULL END) AS integer_sum_low,
            SUM(CASE WHEN {integer_test} THEN 0 ELSE 1 END) AS floating_count
        FROM valid_samples{group_sql}
    ), floating_rows AS (
        SELECT {source_prefix}v.occurred_at, v.observation_digest,
            t.sample_count, t.floating_count, {floating_value} AS floating_value,
            CASE WHEN {integer_test} THEN 0 ELSE 1 END AS is_floating,
            SUM(CASE WHEN {integer_test} THEN {integer_value} ELSE 0 END) OVER (
                {source_partition}ORDER BY v.occurred_at, v.observation_digest
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS integer_prefix,
            SUM(CASE WHEN {integer_test} THEN 0 ELSE 1 END) OVER (
                {source_partition}ORDER BY v.occurred_at, v.observation_digest
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS floating_seen
        FROM valid_samples AS v JOIN totals AS t ON {join_sql}
        WHERE t.floating_count > 0
    ), floating_values AS (
        SELECT {prefix}occurred_at, observation_digest, sample_count, floating_count,
            CASE WHEN floating_seen = 0 THEN NULL
                WHEN floating_seen = 1 AND is_floating = 1
                    THEN {prefix_value} + floating_value
                ELSE floating_value END AS value
        FROM floating_rows
    )"""
    if dialect_name == "postgresql":
        reduction = f"""SELECT {prefix}MAX(sample_count) AS sample_count,
            MAX(floating_count) AS floating_count,
            SUM(value ORDER BY occurred_at, observation_digest) AS sample_sum
        FROM floating_values{group_sql}"""
    else:
        reduction = f"""SELECT {prefix}sample_count, floating_count, sample_sum
        FROM (
            SELECT {prefix}sample_count, floating_count,
                ROW_NUMBER() OVER (
                    {partition_sql}ORDER BY occurred_at, observation_digest
                ) AS ordinal,
                SUM(value) OVER (
                    {partition_sql}ORDER BY occurred_at, observation_digest
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS sample_sum
            FROM floating_values
        ) AS accumulated WHERE ordinal = sample_count"""
    return f"""{common}, floating_totals AS (
        {reduction}
    )
    SELECT {prefix}sample_count, 0 AS integer_sum_high, integer_sum_low,
        floating_count, NULL AS sample_sum
    FROM totals WHERE COALESCE(floating_count, 0) = 0
    UNION ALL
    SELECT {prefix}sample_count, 0 AS integer_sum_high, NULL AS integer_sum_low,
        floating_count, sample_sum
    FROM floating_totals"""


def _sqlite_sum_sql(partition_columns: list[str]) -> str:
    columns = ", ".join(partition_columns)
    prefix = f"{columns}, " if columns else ""
    group_sql = f" GROUP BY {columns}" if columns else ""
    partition_sql = (
        "PARTITION BY " + ", ".join(f"v.{column}" for column in partition_columns) + " "
        if partition_columns else ""
    )
    source_prefix = "".join(f"v.{column}, " for column in partition_columns)
    next_prefix = "".join(f"n.{column}, " for column in partition_columns)
    totals_join = " AND ".join(
        f"v.{column} IS t.{column}" for column in partition_columns
    ) or "1 = 1"
    next_join = " AND ".join(
        f"n.{column} IS s.{column}" for column in partition_columns
    ) or "1 = 1"
    base = _SUM_LIMB_BASE
    # SQLite SUM uses compensated floating addition. Scalar addition in scan
    # order preserves the query contract; all-integer partitions need no fold.
    return f"""WITH RECURSIVE totals AS (
        SELECT {prefix}COUNT(*) AS sample_count,
            SUM(CASE WHEN value_type = 'integer'
                THEN numeric_value / {base} ELSE 0 END) AS integer_sum_high,
            SUM(CASE WHEN value_type = 'integer'
                THEN numeric_value % {base} ELSE 0 END) AS integer_sum_low,
            SUM(CASE WHEN value_type = 'real' THEN 1 ELSE 0 END) AS floating_count
        FROM valid_samples{group_sql}
    ), ordered AS (
        SELECT {source_prefix}v.numeric_value, v.value_type, t.sample_count,
            ROW_NUMBER() OVER (
                {partition_sql}ORDER BY occurred_at, observation_digest
            ) AS ordinal
        FROM valid_samples AS v
        JOIN totals AS t ON {totals_join}
        WHERE t.floating_count > 0
    ), folded AS (
        SELECT {prefix}ordinal, sample_count,
            CASE WHEN value_type = 'integer'
                THEN numeric_value / {base} ELSE 0 END AS integer_sum_high,
            CASE WHEN value_type = 'integer'
                THEN numeric_value % {base} ELSE 0 END AS integer_sum_low,
            CASE WHEN value_type = 'real' THEN 1 ELSE 0 END AS floating_count,
            CASE WHEN value_type = 'real' THEN numeric_value ELSE NULL END AS sample_sum
        FROM ordered WHERE ordinal = 1
        UNION ALL
        SELECT {next_prefix}n.ordinal, n.sample_count,
            s.integer_sum_high + CASE WHEN n.value_type = 'integer'
                THEN n.numeric_value / {base} ELSE 0 END,
            s.integer_sum_low + CASE WHEN n.value_type = 'integer'
                THEN n.numeric_value % {base} ELSE 0 END,
            s.floating_count + CASE WHEN n.value_type = 'real' THEN 1 ELSE 0 END,
            CASE WHEN s.floating_count > 0 THEN s.sample_sum + n.numeric_value
                WHEN n.value_type = 'real' THEN
                    (s.integer_sum_high * 1.0 * {base} + s.integer_sum_low * 1.0)
                    + n.numeric_value
                ELSE NULL END
        FROM folded AS s
        JOIN ordered AS n ON n.ordinal = s.ordinal + 1 AND {next_join}
    )
    SELECT {prefix}sample_count, integer_sum_high, integer_sum_low,
        floating_count, NULL AS sample_sum
    FROM totals WHERE COALESCE(floating_count, 0) = 0
    UNION ALL
    SELECT {prefix}sample_count, integer_sum_high, integer_sum_low,
        floating_count, sample_sum
    FROM folded WHERE ordinal = sample_count"""


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
        direction = "ASC" if aggregation is MetricAggregation.MIN else "DESC"
        if dialect_name == "postgresql":
            coarse, residual = _postgresql_numeric_order()
            value_order = f"{coarse} {direction}, {residual} {direction}"
        elif dialect_name == "mysql":
            coarse, residual = _mysql_numeric_order()
            value_order = f"{coarse} {direction}, {residual} {direction}"
        else:
            value_order = f"numeric_value {direction}"
        order_sql = f"{value_order}, occurred_at ASC, observation_digest ASC"
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
        coarse, residual = _postgresql_numeric_order()
        return f"""SELECT
        {select_prefix}COUNT(*) AS sample_count,
        row_to_json(percentile_disc(CAST(:percentile AS DOUBLE PRECISION)) WITHIN GROUP (
            ORDER BY ROW({coarse}, {residual}, occurred_at, observation_digest, raw_value)
        )) ->> 'f5' AS selected_value
    FROM valid_samples{group_sql}"""

    partition_sql = (
        f"PARTITION BY {', '.join(partition_columns)} "
        if partition_columns
        else ""
    )
    if dialect_name == "mysql":
        coarse, residual = _mysql_numeric_order()
        order_sql = f"{coarse} ASC, {residual} ASC"
    else:
        order_sql = "numeric_value ASC"
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
                {partition_sql}ORDER BY {order_sql}, occurred_at ASC, observation_digest ASC
            ) AS sample_rank
        FROM valid_samples
    ) AS ranked
    WHERE sample_rank = {target_rank}"""



def _mysql_numeric_order() -> tuple[str, str]:
    coarse = "CAST(JSON_UNQUOTE(raw_value) AS DOUBLE)"
    integer = "CAST(JSON_UNQUOTE(raw_value) AS DECIMAL(65, 0))"
    # JSON numeric comparison uses a decimal approximation for doubles.
    # An exact integer residual preserves mixed binary-float ordering.
    residual = (
        "CASE WHEN value_type IN ('INTEGER', 'UNSIGNED INTEGER') THEN "
        f"{integer} - CASE WHEN {coarse} >= 9223372036854775808e0 "
        "THEN 9223372036854775808 "
        f"WHEN {coarse} <= -9223372036854775808e0 THEN -9223372036854775808 "
        f"ELSE CAST({coarse} AS SIGNED) END "
        "ELSE 0 END"
    )
    return coarse, residual


def _postgresql_numeric_order() -> tuple[str, str]:
    coarse = "CAST(raw_value AS DOUBLE PRECISION)"
    integer = "CAST(raw_value AS BIGINT)"
    # Rounding an int64 to float is monotone. A small exact residual orders
    # integers sharing that float, without treating a float's short JSON
    # representation as its exact decimal value.
    residual = (
        "CASE WHEN raw_value ~ '^-?[0-9]+$' THEN CASE "
        f"WHEN {coarse} >= 9223372036854775808.0 THEN "
        f"{integer} - 9223372036854775807 - 1 "
        f"ELSE {integer} - CAST({coarse} AS BIGINT) END ELSE 0 END"
    )
    return coarse, residual


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
            f"JSON_TYPE({raw}) IN ('INTEGER', 'UNSIGNED INTEGER') "
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
        if string:
            return f"JSON_TYPE({raw}) = 'STRING'"
        return f"JSON_TYPE({raw}) IN ('INTEGER', 'UNSIGNED INTEGER')"
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


def _validate_scan_count(
    rows: list["Mapping[str, object]"],
    plan: _MetricQueryPushdownPlan,
) -> None:
    if not rows:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if int(rows[0]["scanned_count"]) > plan.max_scanned_observations:
        raise AIError(ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED)
    record_error = rows[0].get("record_error")
    if record_error is not None:
        raise AIError(ErrorCode(record_error))


def _decode_aggregate_rows(
    rows: list["Mapping[str, object]"],
    plan: _MetricQueryPushdownPlan,
) -> _MetricQueryPushdownResult:
    _validate_scan_count(rows, plan)
    materialized = [row for row in rows if row["sample_count"] is not None]
    if len(materialized) > _result_row_limit(plan):
        raise AIError(ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED)
    decoded = tuple(
        _decode_row(row, plan, selected=False)
        for row in materialized
    )
    _validate_groups(decoded, plan)
    return _MetricQueryPushdownResult(rows=decoded)


def _decode_measurement_rows(
    rows: list["Mapping[str, object]"],
    plan: _MetricQueryPushdownPlan,
) -> _MetricQueryPushdownResult:
    _validate_scan_count(rows, plan)
    extracted = int(rows[0]["extracted_count"])
    invalid = int(rows[0]["invalid_count"])
    if invalid:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if extracted > plan.max_extracted_samples:
        raise AIError(ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED)

    materialized = [row for row in rows if row.get("sample_count") is not None]
    if len(materialized) > _result_row_limit(plan):
        raise AIError(ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED)
    selected = plan.aggregation in _SELECTED_MEASUREMENT_AGGREGATIONS
    decoded = tuple(
        _decode_row(row, plan, selected=selected)
        for row in materialized
    )
    _validate_groups(decoded, plan)
    if sum(row.sample_count for row in decoded) != extracted:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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
    sample_sum = None
    if not selected and plan.source_kind is MetricSourceKind.MEASUREMENT:
        if plan.aggregation is not MetricAggregation.COUNT and sample_count:
            if int(row["floating_count"]) == 0:
                high = _coerce_aggregate_number(row["integer_sum_high"])
                low = _coerce_aggregate_number(row["integer_sum_low"])
                if not isinstance(high, int) or not isinstance(low, int):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                sample_sum = high * _SUM_LIMB_BASE + low
            else:
                sample_sum = _coerce_aggregate_number(row["sample_sum"])
    elif not selected:
        sample_sum = _coerce_aggregate_number(row["sample_sum"])
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
        if math.isnan(value):
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
