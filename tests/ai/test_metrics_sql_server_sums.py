#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exercise database-owned chronological sums against the scan contract."""

import json
import os
import shlex
import shutil
import subprocess
import time
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from linktools.ai.observe import MetricAggregation, MetricSourceKind, MetricType
from linktools.ai.observe._sql_query import (
    _base_params, _decode_measurement_rows, _measurement_sql,
)
from linktools.ai.observe._store import _MetricQueryPushdownPlan
from sqlalchemy import text
from sqlalchemy.dialects import mysql, postgresql


@pytest.fixture(scope="module", params=("postgresql", "mysql"))
def _server(request: pytest.FixtureRequest) -> Iterator[tuple[str, list[str]]]:
    name = request.param
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("Server fixtures require an unprivileged local user")
    with TemporaryDirectory(prefix="metrics-sum-") as directory:
        root = Path(directory)
        data = root / "data"
        if name == "postgresql":
            candidates = sorted(Path("/usr/lib/postgresql").glob("*/bin"), reverse=True)
            initdb = shutil.which("initdb")
            if initdb:
                candidates.insert(0, Path(initdb).parent)
            candidates = [p for p in candidates if all((p / n).is_file() for n in ("initdb", "pg_ctl", "psql"))]
            if not candidates:
                pytest.skip("PostgreSQL server tools are not installed")
            binaries = candidates[0]
            subprocess.run(
                [str(binaries / "initdb"), "-D", str(data), "--auth=trust", "--no-locale", "--encoding=UTF8"],
                check=True, capture_output=True, text=True, timeout=30,
            )
            subprocess.run(
                [str(binaries / "pg_ctl"), "-D", str(data), "-l", str(root / "server.log"),
                 "-w", "-t", "20", "-o", shlex.join(["-F", "-h", "", "-k", str(root)]), "start"],
                check=True, capture_output=True, text=True, timeout=30,
            )
            try:
                yield name, [str(binaries / "psql"), "-X", "-A", "-t", "-q", "-v", "ON_ERROR_STOP=1", "-h", str(root), "-d", "postgres"]
            finally:
                subprocess.run(
                    [str(binaries / "pg_ctl"), "-D", str(data), "-w", "-t", "20", "-m", "fast", "stop"],
                    check=True, capture_output=True, text=True, timeout=30,
                )
        else:
            server = shutil.which("mysqld")
            client = shutil.which("mysql")
            if not server or not client:
                pytest.skip("MySQL server tools are not installed")
            version = subprocess.run([server, "--version"], check=True, capture_output=True, text=True).stdout
            if "MariaDB" in version:
                pytest.skip("The MySQL query contract does not target MariaDB")
            subprocess.run(
                [server, "--no-defaults", "--initialize-insecure", f"--datadir={data}", f"--log-error={root / 'init.log'}"],
                check=True, capture_output=True, text=True, timeout=60,
            )
            command = [client, "--no-defaults", f"--socket={root / 'mysql.sock'}", "-uroot", "--batch", "--raw", "--skip-column-names"]
            with (root / "server-output.log").open("w") as log:
                process = subprocess.Popen(
                    [server, "--no-defaults", f"--datadir={data}", f"--socket={root / 'mysql.sock'}",
                     f"--pid-file={root / 'server.pid'}", f"--log-error={root / 'server.log'}", "--skip-networking",
                     "--mysqlx=OFF", "--innodb-buffer-pool-size=32M"],
                    stdout=log, stderr=subprocess.STDOUT,
                )
                try:
                    deadline = time.monotonic() + 30
                    while True:
                        ready = subprocess.run(command, input="SELECT 1;", capture_output=True, text=True, timeout=3)
                        if ready.returncode == 0:
                            break
                        if process.poll() is not None or time.monotonic() >= deadline:
                            pytest.fail((root / "server.log").read_text())
                        time.sleep(0.1)
                    subprocess.run(command, input="CREATE DATABASE metrics_test;", check=True, capture_output=True, text=True, timeout=5)
                    yield name, [*command, "metrics_test"]
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)


def _plan() -> _MetricQueryPushdownPlan:
    start = datetime(2026, 9, 7, tzinfo=timezone.utc)
    return _MetricQueryPushdownPlan(
        observation_kind="business.numeric", source_kind=MetricSourceKind.MEASUREMENT,
        metric_type=MetricType.GAUGE, measurement_name="value", measurement_revision=1,
        indicator_field=None, indicator_values=(), aggregation=MetricAggregation.MEAN, percentile=None,
        start=start, end=start + timedelta(seconds=2), filters=(), correlation_filters=(),
        group_by=(), bucket_microseconds=None, bucket_count=1,
        max_scanned_observations=100000, max_extracted_samples=100000,
        max_groups=256, max_result_points=16384,
    )


def _run(
    server: tuple[str, list[str]], records: tuple[tuple[int, str | None, int | float], ...],
    plan: _MetricQueryPushdownPlan,
) -> dict[tuple[tuple[str | None, ...], int | None], tuple[int | float | None, int]]:
    name, command = server
    dialect = postgresql.dialect(paramstyle="named") if name == "postgresql" else mysql.dialect(paramstyle="named")
    statement, params = _measurement_sql(name, plan, _base_params("ns", name, plan))
    sql = str(text(statement).bindparams(**params).compile(dialect=dialect, compile_kwargs={"literal_binds": True}))
    inserts = []
    for index, (microseconds, group, value) in enumerate(records):
        payload = {"observation": {"dimensions": {} if group is None else {"group": group},
                   "measurements": [{"name": "value", "revision": 1, "value": value}]}}
        occurred_at = plan.start + timedelta(microseconds=microseconds)
        if name == "mysql":
            occurred_at = occurred_at.replace(tzinfo=None)
        row = text(
            "INSERT INTO ai_metric_observations VALUES ('ns', 'business.numeric', :occurred, :identity, CAST(:payload AS JSON));"
        ).bindparams(occurred=occurred_at, identity=f"{index:064x}", payload=json.dumps(payload))
        insert = str(row.compile(dialect=dialect, compile_kwargs={"literal_binds": True}))
        inserts.append(insert)
    timestamp = "TIMESTAMPTZ" if name == "postgresql" else "DATETIME(6)"
    temporary = "TEMPORARY " if name == "postgresql" else ""
    script = "DROP TABLE IF EXISTS ai_metric_observations;\n" + f"""CREATE {temporary}TABLE ai_metric_observations (
        namespace_digest VARCHAR(64), kind VARCHAR(128), occurred_at {timestamp},
        observation_digest VARCHAR(64), payload_json JSON
    );\n""" + "\n".join(inserts)
    if name == "postgresql":
        script += (
            "\nSELECT COALESCE(json_agg(typed), '[]') FROM ("
            "SELECT q.*, pg_typeof(q.sample_sum)::text AS sum_type FROM ("
            + sql + ") AS q) AS typed;"
        )
    else:
        columns = ["scanned_count", "extracted_count", "invalid_count", "sample_count", "sample_sum", "integer_sum_high", "integer_sum_low", "floating_count"]
        columns += [f"g{n}" for n in range(len(plan.group_by))]
        if plan.bucket_microseconds is not None:
            columns.append("bucket_index")
        fields = ", ".join(f"'{column}', q.{column}" for column in columns)
        script += f"\nSELECT JSON_OBJECT({fields}) FROM ({sql}) AS q;"
    result = subprocess.run(command, input=script, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    rows = json.loads(result.stdout) if name == "postgresql" else [json.loads(line) for line in result.stdout.splitlines()]
    if name == "postgresql":
        for row in rows:
            assert row["sum_type"] == "double precision"
            if row["sample_sum"] is not None:
                # json_agg renders integral float8 values without a decimal point.
                row["sample_sum"] = float(row["sample_sum"])
    decoded = _decode_measurement_rows(rows, plan)
    return {(row.group, row.bucket_index): (row.sample_sum, row.sample_count) for row in decoded.rows}


@pytest.mark.parametrize("values", (
    (1e16, 1.0, -1e16),
    (2**53 + 1, 1, 0.0),
    (2**63 - 1, 2**63 - 1, 1, 0.0, 1),
    (-2**63, -1, 1.0, 2**63 - 1),
    (1.0, 2**53 + 1, -float(2**53)),
    (5e-324, 5e-324, -5e-324),
    (0, 0.0, -0.0),
    (2**63 - 1, 2**63 - 1),
    (1.0,),
    (),
))
def test_server_chronological_sum_matches_python(
    _server: tuple[str, list[str]], values: tuple[int | float, ...],
) -> None:
    result = _run(_server, tuple((i, None, value) for i, value in enumerate(values)), _plan())
    expected = None
    for value in values:
        expected = value if expected is None else expected + value
    actual, count = result.get(((), None), (None, 0))
    assert actual == expected
    assert type(actual) is type(expected)
    assert count == len(values)


def test_server_sum_separates_missing_groups_and_buckets(_server: tuple[str, list[str]]) -> None:
    plan = replace(_plan(), group_by=("group",), bucket_microseconds=1000000, bucket_count=2)
    values = (1e16, 1.0, -1e16)
    records = tuple((bucket * 1000000 + index, group, value)
                    for bucket in (0, 1) for group in (None, "x", "X") for index, value in enumerate(values))
    result = _run(_server, records, plan)
    assert result == {((group,), bucket): (0.0, 3) for group in (None, "x", "X") for bucket in (0, 1)}


def test_mysql_integer_scalar_representation(_server: tuple[str, list[str]]) -> None:
    name, command = _server
    if name != "mysql":
        return
    values = (1, 2**53 + 1, 2**63 - 1, -(2**63))
    fields = []
    for value in values:
        raw = f"CAST('{value}' AS JSON)"
        decimal = f"CAST(JSON_UNQUOTE({raw}) AS DECIMAL(65, 0))"
        fields.append(
            f"SELECT JSON_OBJECT('type', JSON_TYPE({raw}), 'text', JSON_UNQUOTE({raw}), "
            f"'decimal', CAST({decimal} AS CHAR), "
            f"'valid', {decimal} BETWEEN -9223372036854775808 AND 9223372036854775807);"
        )
    result = subprocess.run(command, input="\n".join(fields), capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    actual = [json.loads(line) for line in result.stdout.splitlines()]
    expected = [{"type": "INTEGER", "text": str(value), "decimal": str(value), "valid": 1} for value in values]
    assert actual == expected
