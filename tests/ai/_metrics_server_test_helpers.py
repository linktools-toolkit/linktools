#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Share server lifetimes while isolating each metrics test module's database."""

import getpass
import os
import shlex
import shutil
import subprocess
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import pytest_asyncio
from linktools.ai.migrate import provision_metrics_database
from linktools.ai.observe import (
    MetricAggregation, MetricDefinition, MetricSource, MetricType, Metrics, build_metrics_sql_metadata,
)
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


_NAMESPACE = "server-integrity"
_COUNT = "business.integrity.count"
_RATIO = "business.integrity.ratio"
_VALUE = "business.integrity.value"
_ServerMetrics = tuple[Metrics, AsyncEngine, AsyncEngine]


def _server_process(name: str) -> Iterator[tuple[str, list[str]]]:
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
            started = False
            try:
                subprocess.run(
                    [str(binaries / "pg_ctl"), "-D", str(data), "-l", str(root / "server.log"),
                     "-w", "-t", "20", "-o", shlex.join(["-F", "-h", "", "-k", str(root)]), "start"],
                    check=True, capture_output=True, text=True, timeout=30,
                )
                started = True
                yield name, [str(binaries / "psql"), "-X", "-A", "-t", "-q", "-v", "ON_ERROR_STOP=1", "-h", str(root), "-d", "postgres"]
            finally:
                subprocess.run(
                    [str(binaries / "pg_ctl"), "-D", str(data), "-w", "-t", "20", "-m", "fast", "stop"],
                    check=started, capture_output=True, text=True, timeout=30,
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
                    yield name, [*command, "mysql"]
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)


@pytest.fixture(scope="session")
def _postgresql_server() -> Iterator[tuple[str, list[str]]]:
    yield from _server_process("postgresql")


@pytest.fixture(scope="session")
def _mysql_server() -> Iterator[tuple[str, list[str]]]:
    yield from _server_process("mysql")


@contextmanager
def _server_database(server: tuple[str, list[str]]) -> Iterator[tuple[str, list[str]]]:
    name, command = server
    database = "metrics_" + uuid.uuid4().hex
    try:
        subprocess.run(command, input=f"CREATE DATABASE {database};", check=True,
                       capture_output=True, text=True, timeout=5)
        yield name, [*command[:-1], database]
    finally:
        subprocess.run(command, input=f"DROP DATABASE IF EXISTS {database};", check=True,
                       capture_output=True, text=True, timeout=5)


@pytest.fixture(scope="module", params=("postgresql", "mysql"))
def _server(request: pytest.FixtureRequest) -> Iterator[tuple[str, list[str]]]:
    server = request.getfixturevalue(f"_{request.param}_server")
    with _server_database(server) as database:
        yield database


@pytest_asyncio.fixture
async def server_metrics(_server: tuple[str, list[str]]) -> AsyncIterator[_ServerMetrics]:
    name, command = _server
    if name == "postgresql":
        url = URL.create(
            "postgresql+asyncpg", username=getpass.getuser(), database=command[-1],
            query={"host": command[command.index("-h") + 1]},
        )
    else:
        socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
        url = URL.create(
            "mysql+asyncmy", username="root", database=command[-1],
            query={"unix_socket": socket, "charset": "utf8mb4"},
        )
    engine = create_async_engine(url, isolation_level="READ COMMITTED", pool_size=1, max_overflow=0)
    writer = create_async_engine(url, isolation_level="READ COMMITTED", pool_size=1, max_overflow=0)
    metadata = build_metrics_sql_metadata()
    try:
        async with engine.begin() as connection:
            await connection.run_sync(metadata.drop_all)
        await provision_metrics_database(engine)
        metrics = Metrics.sql(engine, namespace=_NAMESPACE)
        for name, source, metric_type in (
            (_COUNT, MetricSource.observation_count(), MetricType.COUNTER),
            (_RATIO, MetricSource.indicator("status", ("FAILED",)), MetricType.RATIO),
            (_VALUE, MetricSource.measurement("value"), MetricType.GAUGE),
        ):
            await metrics.define(MetricDefinition(
                name=name, revision=1, observation_kind="business.integrity", source=source,
                metric_type=metric_type, unit="1", default_aggregation=MetricAggregation.COUNT,
                query_fields=("group", "status"),
            ))
        yield metrics, engine, writer
    finally:
        await writer.dispose()
        await engine.dispose()
