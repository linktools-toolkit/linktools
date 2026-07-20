#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 9 AC-16: the §6.4 absolute performance/usability thresholds, as
measurable checks. Each assertion pins a fixed acceptance threshold from the
plan; the reference env is 4 vCPU / 8 GiB RAM / Linux / Python 3.10 / SSD
(§6.4). These run in the default suite so a regression past a threshold is
caught immediately -- the thresholds carry generous headroom (an in-process
lock op at p95 <= 5ms; a Runtime.build at p95 <= 1s), so they do not flake on
a normally-loaded machine.

Coverage of the §6.4 table: RuntimeBuilder.build p95 <= 1s; Catalog cache get
p95 <= 5ms / cold-load 100 specs <= 500ms; Event single append p95 <= 20ms;
Asset list 1000 items p95 <= 200ms; Artifact integrity 10k r/w 0 mismatch; Job
defaults (heartbeat 5s / lease 30s / poll 1s); process-local coordination p95
<= 5ms; stability stress 10k ops < 0.1% non-injected error. The rows not
asserted inline are: the 1 GiB streaming-RSS cap (needs a ~1 GiB fixture +
tracemalloc; verified manually in the benchmark env, not per-suite); the
SQLite-specific Event batch rate (>=500/s) and SQLite Asset-list number (p95
<= 300ms), which need the optional SQLAlchemy extra (the Filesystem baseline
numbers are asserted here as the always-installed baseline); and the
relative-regression half (see below).

The plan's relative-regression half ("<= 20% vs Phase 0 hot path") is not
asserted here: no Phase-0 baseline was captured. Per §6.4 ("若阶段 0 已优于
表中目标 ... 取 ... 绝对门槛"), when no Phase-0 baseline exists the absolute
thresholds govern, which is what this module checks."""

import asyncio
import json
import statistics
import tempfile
import time
from datetime import timedelta

import pytest

from linktools.ai.jobs.runtime import JobRuntimeOptions
from linktools.ai.runtime import Runtime
from linktools.ai.storage.coordination.process_local import (
    ProcessLocalLeaseCoordinator,
)
from linktools.ai.storage.facade import FilesystemStorage


def _p95_ms(samples_s: "list[float]") -> float:
    """p95 of a list of per-sample durations (seconds) -> milliseconds."""
    if not samples_s:
        return 0.0
    s = sorted(samples_s)
    idx = max(0, int(len(s) * 0.95) - 1)
    return s[idx] * 1000.0


# --- §6.4 Job defaults (heartbeat 5s, lease TTL 30s, worker claim poll 1s) ---


def test_job_defaults_match_section_6_4():
    opts = JobRuntimeOptions()
    assert opts.heartbeat_seconds == 5.0, "§6.4: Job heartbeat default 5s"
    assert opts.lease_seconds == 30.0, "§6.4: Job lease TTL default 30s"
    assert opts.poll_interval_seconds == 1.0, "§6.4: worker claim poll default 1s"


# --- §6.4 process-local coordination: acquire/renew/release p95 <= 5ms ---


@pytest.mark.asyncio
async def test_coordination_acquire_renew_release_p95_under_5ms():
    coord = ProcessLocalLeaseCoordinator()
    # Warm up (first op may pay import/alloc costs).
    t = await coord.acquire(key="warmup", owner_id="o", ttl=timedelta(seconds=30))
    await coord.renew(token=t, ttl=timedelta(seconds=30))
    await coord.release(token=t)

    acquire_s: "list[float]" = []
    renew_s: "list[float]" = []
    release_s: "list[float]" = []
    for i in range(500):
        k = f"k{i}"
        a0 = time.perf_counter()
        tok = await coord.acquire(key=k, owner_id="o", ttl=timedelta(seconds=30))
        acquire_s.append(time.perf_counter() - a0)
        assert tok is not None
        r0 = time.perf_counter()
        tok = await coord.renew(token=tok, ttl=timedelta(seconds=30))
        renew_s.append(time.perf_counter() - r0)
        x0 = time.perf_counter()
        await coord.release(token=tok)
        release_s.append(time.perf_counter() - x0)
    assert _p95_ms(acquire_s) <= 5.0, f"acquire p95 {_p95_ms(acquire_s):.3f}ms > 5ms"
    assert _p95_ms(renew_s) <= 5.0, f"renew p95 {_p95_ms(renew_s):.3f}ms > 5ms"
    assert _p95_ms(release_s) <= 5.0, f"release p95 {_p95_ms(release_s):.3f}ms > 5ms"


# --- §6.4 RuntimeBuilder.build: p95 <= 1s (no MCP network discovery) ---


def test_runtime_build_p95_under_1s(tmp_path):
    storage = FilesystemStorage(root=tmp_path)

    def _build() -> None:
        Runtime.build(storage=FilesystemStorage(root=tmp_path / "x"))

    # Warm up.
    for _ in range(3):
        _build()
    samples: "list[float]" = []
    for _ in range(20):
        t0 = time.perf_counter()
        _build()
        samples.append(time.perf_counter() - t0)
    p95 = _p95_ms(samples)
    assert p95 <= 1000.0, f"Runtime.build p95 {p95:.1f}ms > 1000ms"


# --- §6.4 Catalog: cache get p95 <= 5ms / 10k, cold-load 100 specs <= 500ms ---


class _StaticCatalogSource:
    """Tiny in-memory CatalogSource: stable revision + a fixed item map. Drives
    the real RevisionCache code path (revision check + cache key + single-flight)
    without a filesystem dependency, so the cache-threshold test is deterministic."""

    def __init__(self, items: "dict[str, str]") -> None:
        self._items = items

    async def revision(self) -> str:
        return "rev-1"

    async def list_ids(self, suffix: str) -> "tuple[str, ...]":
        return tuple(
            sorted(
                k[: -len(suffix)] if suffix and k.endswith(suffix) else k
                for k in self._items
            )
        )

    async def read(self, path: str) -> str:
        if path not in self._items:
            raise FileNotFoundError(path)
        return self._items[path]


class _JsonCodec:
    """Minimal CatalogCodec: JSON decode (a stand-in for strict spec parsing on
    the perf path; the parse strictness itself is covered by catalog unit tests)."""

    def decode(self, item_id: str, raw: str):
        return json.loads(raw)


@pytest.mark.asyncio
async def test_catalog_cache_get_p95_under_5ms():
    from linktools.ai.catalog.contracts import RevisionCache

    cache = RevisionCache(
        _StaticCatalogSource({"agent.md": json.dumps({"name": "agent"})}),
        _JsonCodec(),
        suffix=".md",
    )
    await cache.get("agent")  # prime the cache (first get is a cold read+decode)
    samples: "list[float]" = []
    for _ in range(10000):
        t0 = time.perf_counter()
        await cache.get("agent")
        samples.append(time.perf_counter() - t0)
    p95 = _p95_ms(samples)
    assert p95 <= 5.0, f"Catalog cache get p95 {p95:.3f}ms > 5ms"


@pytest.mark.asyncio
async def test_catalog_cold_load_100_specs_under_500ms():
    from linktools.ai.catalog.contracts import RevisionCache

    items = {
        f"agent-{i:03d}.md": json.dumps({"name": f"agent-{i}"})
        for i in range(100)
    }
    cache = RevisionCache(_StaticCatalogSource(items), _JsonCodec(), suffix=".md")
    ids = await cache.list_ids()
    assert len(ids) == 100
    t0 = time.perf_counter()
    for item_id in ids:
        await cache.get(item_id)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms <= 500.0, f"cold load 100 specs {elapsed_ms:.1f}ms > 500ms"


# --- §6.4 Event: single append p95 <= 20ms, batch >= 500 events/s (Filesystem) ---


async def _append_run_started(store, *, stream_id: str = "r") -> None:
    from linktools.ai.events.payloads import RunStarted

    await store.append(
        stream_id=stream_id,
        run_id="r",
        root_run_id="r",
        parent_run_id=None,
        session_id="s",
        runnable_id="a",
        payload=RunStarted(run_id="r", runnable_id="a"),
    )


@pytest.mark.asyncio
async def test_event_single_append_p95_under_20ms(tmp_path):
    from linktools.ai.storage.filesystem.event import FilesystemEventStore

    store = FilesystemEventStore(root=tmp_path)
    for _ in range(10):  # warm
        await _append_run_started(store)
    samples: "list[float]" = []
    for _ in range(200):
        t0 = time.perf_counter()
        await _append_run_started(store)
        samples.append(time.perf_counter() - t0)
    assert _p95_ms(samples) <= 20.0, (
        f"Event single append p95 {_p95_ms(samples):.3f}ms > 20ms"
    )


# NOTE: §6.4's Event BATCH row ("1,000 条事件，SQLite ≥ 500 events/s") pins the
# >=500/s bar to SQLite specifically; no Filesystem batch rate is specified. The
# SQLite extra is optional and not installed in the default env, so the batch
# threshold is not asserted here -- see the docstring's deferred list.


# --- §6.4 Asset list: 1000-item single-level list, Filesystem p95 <= 200ms ---


@pytest.mark.asyncio
async def test_asset_list_1000_filesystem_p95_under_200ms(tmp_path):
    from linktools.ai.asset.file import FileAssetBackend
    from linktools.ai.asset.models import Depth
    from linktools.ai.asset.path import AssetPath
    from linktools.ai.asset.store import AssetStore

    store = AssetStore(primary=FileAssetBackend(root=tmp_path))
    for i in range(1000):
        await store.put(AssetPath(f"/ns/item-{i:04d}"), f"body-{i}".encode())

    async def _list_all() -> list:
        seen: list = []
        page = await store.propfind(AssetPath("/ns"), depth=Depth.ONE, limit=100)
        seen.extend(page.items)
        while page.cursor is not None:
            page = await store.propfind(
                AssetPath("/ns"), depth=Depth.ONE, limit=100, cursor=page.cursor
            )
            seen.extend(page.items)
        return seen

    items = await _list_all()
    assert len(items) == 1000, f"expected 1000 listed, got {len(items)}"
    await _list_all()  # warm
    samples: "list[float]" = []
    for _ in range(10):
        t0 = time.perf_counter()
        await _list_all()
        samples.append(time.perf_counter() - t0)
    assert _p95_ms(samples) <= 200.0, (
        f"Asset list 1000 p95 {_p95_ms(samples):.1f}ms > 200ms"
    )


# --- §6.4 Artifact integrity: 10,000 r/w with 0 digest mismatch ---


@pytest.mark.asyncio
async def test_artifact_integrity_10000_rw_zero_mismatch():
    from linktools.ai.artifact.store import ArtifactStore

    from tests.ai.storage.example_external_adapter import (
        InMemoryArtifactBlobStore,
        InMemoryArtifactRecordStore,
    )

    store = ArtifactStore(InMemoryArtifactBlobStore(), InMemoryArtifactRecordStore())
    for i in range(10000):
        content = f"payload-{i}".encode()
        record = await store.put(content, media_type="text/plain", tenant_id="t1")
        blob = await store.get(record.ref.id, tenant_id="t1")
        assert blob == content, f"digest mismatch at iteration {i}"


# --- §6.4 stability stress: 10,000 core store ops, < 0.1% non-injected error ---


@pytest.mark.asyncio
async def test_stability_stress_10000_ops_under_0_1_percent_error():
    # 10,000 acquire/renew/release cycles on the process-local LeaseCoordinator --
    # an in-memory core storage operation, so the 10k-op count + <0.1% error-rate
    # assertion run fast in the default suite (the threshold is about stability
    # under load, not filesystem throughput).
    coord = ProcessLocalLeaseCoordinator()
    errors = 0
    for i in range(10000):
        try:
            token = await coord.acquire(
                key=f"k{i}", owner_id="o", ttl=timedelta(seconds=30)
            )
            if token is None:
                errors += 1
                continue
            await coord.renew(token=token, ttl=timedelta(seconds=30))
            await coord.release(token=token)
        except Exception:
            errors += 1
    assert errors / 10000 < 0.001, (
        f"stability stress error rate {errors}/10000 >= 0.1%"
    )
