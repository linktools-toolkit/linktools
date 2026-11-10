#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit matched SQL-adapter message-body cache measurement."""

import hashlib
import json
import os
import statistics
import time
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.persistence import SqlStepStore
from linktools.ai.persistence import _step_store as module

from .test_step_message_cache import _SqlTraffic

pytestmark = pytest.mark.skipif(not os.getenv("STEP_CACHE_MEASUREMENT"), reason="Explicit SQL cache measurement only")


@pytest.mark.asyncio
async def test_long_history_message_cache_traffic_and_cpu(tmp_path: Path) -> None:
    directory = Path(os.environ["STEP_CACHE_MEASUREMENT"])
    directory.mkdir(parents=True, exist_ok=True)
    history = [ModelRequest(parts=[UserPromptPart(f"{index}:" + "x" * 1024)]) for index in range(1024)]
    snapshot = ContinuableSnapshot(run_id="long-history", step_index=0, messages=history, idempotency_key="0")
    samples = []
    cases = [("disabled", SqlStepStore, 0), ("cached", SqlStepStore, 4096)]
    for round_number in range(3):
        offset = round_number % len(cases)
        ordered = cases[offset:] + cases[:offset]
        for name, constructor, capacity in ordered:
            engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / f'{name}-{round_number}.db'}")
            store = constructor(engine, namespace="measurement", tenant_id="one", max_cached_messages=capacity)
            await store.initialize(local=True)
            traffic = _SqlTraffic()
            commits = 0

            def commit(connection):
                nonlocal commits
                commits += 1

            event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
            event.listen(engine.sync_engine, "commit", commit)
            cpu_start, wall_start = time.process_time(), time.perf_counter()
            for index in range(24):
                await store.save_snapshot(replace(snapshot, step_index=index, idempotency_key=str(index)))
            cpu = time.process_time() - cpu_start
            wall = time.perf_counter() - wall_start
            event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
            event.remove(engine.sync_engine, "commit", commit)
            assert await store.latest_snapshot(run_id="long-history") == replace(snapshot, step_index=23, idempotency_key="23")
            async with engine.connect() as connection:
                rows = {table: await connection.scalar(text(f"SELECT count(*) FROM ai_step_{table}"))
                        for table in ("snapshots", "messages", "sequences")}
            assert rows == {"snapshots": 24, "messages": 1024, "sequences": 1024}
            assert commits == 24 and traffic.statements["SELECT"] == 0
            assert traffic.message_rows == (1024 if name == "cached" else 24576)
            samples.append({"mode": name, "round": round_number, "cpu_seconds": cpu, "wall_seconds": wall,
                            "commits": commits, "statements": dict(traffic.statements), "rows": rows,
                            "body_insert_rows": traffic.message_rows, "body_bind_bytes": traffic.message_bytes,
                            "all_bind_bytes": traffic.bound_bytes})
            await store.close()
            await engine.dispose()
    summary = {}
    for name, _, _ in cases:
        selected = [sample for sample in samples if sample["mode"] == name]
        summary[name] = {key: statistics.median(sample[key] for sample in selected)
                         for key in ("cpu_seconds", "wall_seconds", "commits", "body_insert_rows", "body_bind_bytes", "all_bind_bytes")}
        summary[name]["statements"] = selected[0]["statements"]
        assert all(sample["statements"] == selected[0]["statements"] for sample in selected)
    output = {"workload": {"messages": 1024, "characters_per_message": 1024, "snapshots": 24, "repetitions": 3},
              "samples": samples, "median": summary,
              "source_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
    (directory / "measurement.json").write_text(json.dumps(output, indent=2))
