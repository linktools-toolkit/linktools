#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from linktools.ai.artifact import ANONYMOUS_PROVENANCE
"""op 7: benchmark summary. Runs every threshold in a single
process, captures each row's measured value + pass/fail state, and writes a
JSON + Markdown summary to ``PERF_RESULTS_DIR`` (default ``.docs/review-fix``)
so a reviewer can see the actual numbers behind rather than just a
pytest pass bit.

This is a separate test from ``test_thresholds.py`` (which has the load-bearing
assertions) and from ``test_artifact_streaming_rss.py`` (the 1 GiB RSS
benchmark). It COMPOSES those -- running each measurement inline, collecting
the result, and writing the summary -- so the summary reflects what the
assertion-gated tests actually measured on this run, not a hand-typed table.

The summary is written even when a measurement fails (the JSON records the
actual number and the pass/fail bit) so a regression is visible in the artifact
rather than hidden behind a pytest failure. The load-bearing assertions stay
in their own tests; this test only fails if the summary could not be written.

The 1 GiB RSS benchmark itself runs unconditionally in the default suite (see
``test_artifact_streaming_rss.py`` -- a skipped acceptance test is not
evidence). This summary's OWN inline re-measurement of it is still
opt-in via ``RUN_PERF_RSS=1`` (set ``skipped: true`` otherwise) purely to
avoid duplicating the ~8s I/O cost in a report generator that is not itself
an acceptance gate; the load-bearing assertion lives in the dedicated test."""

import asyncio
import json
import os
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

# Re-use the helpers from test_thresholds so the summary measures the SAME
# thing the assertion-gated tests measure (no separate code path that could
# diverge).
from tests.ai.perf.test_thresholds import (  # noqa: E402
    _append_run_started,
    _p95_ms,
)


def _record_result(
    results: "list[dict[str, Any]]",
    *,
    name: str,
    category: str,
    threshold: str,
    measured: "str | float | None",
    passed: "bool | None",
    note: "str | None" = None,
) -> None:
    results.append(
        {
            "name": name,
            "category": category,
            "threshold": threshold,
            "measured": measured,
            "passed": passed,
            "note": note,
        }
    )


def _write_summary(
    results: "list[dict[str, Any]]", *, out_dir: Path, name: str
) -> Path:
    """Write JSON + Markdown summary; returns the JSON path. The JSON is
    machine-readable for CI; the Markdown is for reviewer-facing reports."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_env": "4 vCPU / 8 GiB RAM / Linux / Python 3.10 / SSD",
        "rows": results,
        "overall_passed": all(r["passed"] for r in results if r["passed"] is not None),
    }
    json_path = out_dir / "thresholds_summary.json"
    md_path = out_dir / "thresholds_summary.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Markdown table -- reviewer-facing. The 'measured' column is the actual
    # number; the 'result' column is PASS/FAIL/SKIP.
    rows = ["# Plan §6.4 / §7.5 threshold summary", ""]
    rows.append(f"- generated: {payload['generated_at']}")
    rows.append(
        f"- overall: {'PASS' if payload['overall_passed'] else 'FAIL or SKIP'}"
    )
    rows.append("")
    rows.append("| category | check | threshold | measured | result |")
    rows.append("|---|---|---|---|---|")
    for r in results:
        result = (
            "PASS" if r["passed"] else "FAIL" if r["passed"] is False else "SKIP"
        )
        measured = r["measured"] if r["measured"] is not None else "-"
        rows.append(
            f"| {r['category']} | {r['name']} | {r['threshold']} | "
            f"{measured} | {result} |"
        )
    rows.append("")
    md_path.write_text("\n".join(rows), encoding="utf-8")
    return json_path


# --- per-row measurement helpers --------------------------------------------


def _measure_job_defaults() -> "dict[str, Any]":
    from linktools.ai.jobs.runtime import JobRuntimeOptions

    opts = JobRuntimeOptions()
    return {
        "heartbeat_s": opts.heartbeat_seconds,
        "lease_s": opts.lease_seconds,
        "poll_s": opts.poll_interval_seconds,
        "passed": (
            opts.heartbeat_seconds == 5.0
            and opts.lease_seconds == 30.0
            and opts.poll_interval_seconds == 1.0
        ),
    }


def _measure_coordination_p95() -> "dict[str, Any]":
    from datetime import timedelta

    from linktools.ai.storage.coordination.process_local import (
        ProcessLocalLeaseCoordinator,
    )

    async def _run() -> "dict[str, Any]":
        coord = ProcessLocalLeaseCoordinator()
        t = await coord.acquire(key="warmup", owner_id="o", ttl=timedelta(seconds=30))
        await coord.renew(token=t, ttl=timedelta(seconds=30))
        await coord.release(token=t)
        acquire_s: "list[float]" = []
        renew_s: "list[float]" = []
        release_s: "list[float]" = []
        for i in range(200):
            k = f"k{i}"
            a0 = time.perf_counter()
            tok = await coord.acquire(key=k, owner_id="o", ttl=timedelta(seconds=30))
            acquire_s.append(time.perf_counter() - a0)
            r0 = time.perf_counter()
            tok = await coord.renew(token=tok, ttl=timedelta(seconds=30))
            renew_s.append(time.perf_counter() - r0)
            x0 = time.perf_counter()
            await coord.release(token=tok)
            release_s.append(time.perf_counter() - x0)
        return {
            "acquire_p95_ms": _p95_ms(acquire_s),
            "renew_p95_ms": _p95_ms(renew_s),
            "release_p95_ms": _p95_ms(release_s),
        }

    out = asyncio.run(_run())
    worst = max(out["acquire_p95_ms"], out["renew_p95_ms"], out["release_p95_ms"])
    out["worst_p95_ms"] = worst
    out["passed"] = worst <= 5.0
    return out


def _measure_runtime_build_p95(tmp_path: Path) -> "dict[str, Any]":
    from linktools.ai.runtime import Runtime, build_runtime
    from linktools.ai.storage.facade import FilesystemStorage
    from linktools.ai.storage.filesystem.commit import (
        FilesystemRunCommitCoordinator,
    )

    def _build() -> None:
        fresh = FilesystemStorage(root=tmp_path / "x")
        build_runtime(
            storage=fresh,
            commit_coordinator=FilesystemRunCommitCoordinator.from_storage(fresh),
        )

    for _ in range(3):
        _build()
    samples: "list[float]" = []
    for _ in range(20):
        t0 = time.perf_counter()
        _build()
        samples.append(time.perf_counter() - t0)
    p95 = _p95_ms(samples)
    return {"p95_ms": p95, "passed": p95 <= 1000.0}


def _measure_event_append_p95(tmp_path: Path) -> "dict[str, Any]":
    from linktools.ai.storage.filesystem.event import FilesystemEventStore

    async def _run() -> "dict[str, Any]":
        store = FilesystemEventStore(root=tmp_path)
        for _ in range(10):
            await _append_run_started(store)
        samples: "list[float]" = []
        for _ in range(200):
            t0 = time.perf_counter()
            await _append_run_started(store)
            samples.append(time.perf_counter() - t0)
        return {"p95_ms": _p95_ms(samples)}

    out = asyncio.run(_run())
    out["passed"] = out["p95_ms"] <= 20.0
    return out


def _measure_artifact_integrity_10k() -> "dict[str, Any]":
    from linktools.ai.artifact.coordination import InProcessArtifactDigestCoordinator
    from linktools.ai.artifact.store import ArtifactStore

    from external_adapter import (
        InMemoryArtifactBlobStore,
        InMemoryArtifactRecordStore,
    )

    async def _run() -> "dict[str, Any]":
        store = ArtifactStore(
            InMemoryArtifactBlobStore(),
            InMemoryArtifactRecordStore(),
            InProcessArtifactDigestCoordinator(),
        )
        mismatches = 0
        for i in range(10000):
            content = f"payload-{i}".encode()
            record = await store.put(content=content, media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,)
            blob = await store.get(artifact_id=record.ref.id, tenant_id="t1")
            if blob != content:
                mismatches += 1
        return {"iterations": 10000, "mismatches": mismatches}

    out = asyncio.run(_run())
    out["passed"] = out["mismatches"] == 0
    return out


def _measure_stability_stress() -> "dict[str, Any]":
    from datetime import timedelta

    from linktools.ai.storage.coordination.process_local import (
        ProcessLocalLeaseCoordinator,
    )

    async def _run() -> "dict[str, Any]":
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
            except Exception:  # noqa: BLE001
                errors += 1
        return {
            "ops": 10000,
            "errors": errors,
            "error_rate": errors / 10000,
        }

    out = asyncio.run(_run())
    out["passed"] = out["error_rate"] < 0.001
    return out


def _measure_job_recovery_threshold(tmp_path: Path) -> "dict[str, Any]":
    """Repeats the test_thresholds.test_job_recovery_worker_reclaim_under_35_seconds
    measurement and reports the elapsed clock time. The bound is 35s; the
    default-lease case completes at ~lease_seconds (30s) when recovery runs
    immediately after expiry."""
    from datetime import datetime, timedelta, timezone

    from linktools.ai.jobs.models import (
        ActorChain,
        ActorRef,
        JobRecord,
        JobStatus,
        RetryPolicy,
        SideEffectPolicy,
        TaskBudget,
        TaskPrincipal,
        TaskRecord,
        TaskStatus,
    )
    from linktools.ai.storage.filesystem.job import FilesystemJobStore

    class _Clock:
        def __init__(self) -> None:
            self._t = datetime(2026, 1, 1, tzinfo=timezone.utc)

        def now(self) -> "datetime":
            return self._t

        def advance(self, seconds: float) -> None:
            self._t = self._t + timedelta(seconds=seconds)

    clock = _Clock()
    store = FilesystemJobStore(tmp_path / "jobs", clock=clock)

    async def _run() -> "dict[str, Any]":
        now = clock.now()
        await store.create_job(
            JobRecord(
                id="j-recv",
                status=JobStatus.PENDING,
                principal=TaskPrincipal(tenant_id="t1", user_id="alice"),
                actor_chain=ActorChain(actors=(ActorRef("user", "alice"),)),
                budget=TaskBudget(),
                root_task_id="t-recv",
                input_artifact_id=None,
                output_artifact_id=None,
                version=1,
                created_at=now,
                started_at=None,
                finished_at=None,
            ),
            TaskRecord(
                id="t-recv",
                job_id="j-recv",
                parent_task_id=None,
                key="root",
                handler="runtime",
                status=TaskStatus.PENDING,
                input_artifact_id=None,
                output_artifact_id=None,
                dependencies=(),
                retry_policy=RetryPolicy(max_attempts=2),
                side_effect_policy=SideEffectPolicy(),
                attempt_count=0,
                available_at=now,
                lease_owner=None,
                lease_expires_at=None,
                fencing_token=0,
                active_attempt_id=None,
                timeout_seconds=None,
                asset_snapshots=(),
                version=1,
                created_at=now,
                updated_at=now,
            ),
        )
        await store.claim(worker_id="A", now=clock.now(), lease_seconds=30.0)
        clock.advance(31.0)  # past the 30s lease
        recovered = await store.recover_expired(now=clock.now(), limit=10)
        reclaimed = await store.claim(worker_id="B", now=clock.now(), lease_seconds=30.0)
        elapsed_s = (clock.now() - now).total_seconds()
        return {
            "elapsed_s": elapsed_s,
            "recovered_count": len(recovered),
            "reclaimed_worker": reclaimed.claim.worker_id if reclaimed else None,
        }

    out = asyncio.run(_run())
    out["passed"] = (
        out["elapsed_s"] <= 35.0
        and out["recovered_count"] == 1
        and out["reclaimed_worker"] == "B"
    )
    return out


def _measure_streaming_rss_if_enabled(tmp_path: Path) -> "dict[str, Any]":
    """Re-run the 1 GiB RSS benchmark inline for this summary report ONLY when
    RUN_PERF_RSS is set -- this is a duplicate measurement for the JSON/MD
    report, not the acceptance gate itself (that lives in the always-on
    test_artifact_streaming_rss.py). Without the env var the row reports SKIP
    so a default-suite summary run still emits without paying the ~8s cost
    twice."""
    if not os.environ.get("RUN_PERF_RSS"):
        return {
            "skipped": True,
            "reason": "set RUN_PERF_RSS=1 to measure (perf slice)",
            "passed": None,
        }
    # Import inline so a missing /proc or skip inside the test does not break
    # the rest of the summary.
    from tests.ai.perf.test_artifact_streaming_rss import (
        _RssSampler,
        _gen_non_repeating_gib,
        _GIB,
        _KIB,
        _MIB,
        _RSS_LIMIT_MIB,
        _CHUNK_KIB,
        _read_vm_rss_kib,
    )
    from linktools.ai.artifact.coordination import InProcessArtifactDigestCoordinator
    from linktools.ai.artifact.store import ArtifactStore
    from linktools.ai.storage.filesystem.artifact import (
        FilesystemArtifactBlobStore,
        FilesystemArtifactRecordStore,
    )
    import gc
    import hashlib

    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")
    records = FilesystemArtifactRecordStore(records_root=tmp_path / "records")
    store = ArtifactStore(blob, records, InProcessArtifactDigestCoordinator())
    pre_baseline = _read_vm_rss_kib()
    if pre_baseline is None:
        return {"skipped": True, "reason": "/proc unavailable", "passed": None}

    sampler_up = _RssSampler()
    sampler_up.start()
    try:
        async def _up():
            return await store.put_stream(
                source=_gen_non_repeating_gib(),
                media_type="application/octet-stream",
                tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,
    )

        record = asyncio.run(_up())
    finally:
        sampler_up.stop()
    gc.collect()
    sampler_down = _RssSampler()
    sampler_down.start()
    hasher = hashlib.sha256()
    try:
        async def _down():
            async with store.open_stream(artifact_id=record.ref.id, tenant_id="t1") as chunks:
                async for chunk in chunks:
                    hasher.update(chunk)

        asyncio.run(_down())
    finally:
        sampler_down.stop()

    delta_up = (sampler_up.peak_kib - sampler_up.baseline_kib) / _KIB
    delta_down = (sampler_down.peak_kib - sampler_down.baseline_kib) / _KIB
    return {
        "skipped": False,
        "scale_bytes": _GIB,
        "upload_extra_rss_mib": delta_up,
        "download_extra_rss_mib": delta_down,
        "limit_mib": _RSS_LIMIT_MIB,
        "sha256_match": hasher.hexdigest() == record.ref.sha256,
        "passed": (
            delta_up <= _RSS_LIMIT_MIB
            and delta_down <= _RSS_LIMIT_MIB
            and hasher.hexdigest() == record.ref.sha256
        ),
    }


# --- the summary-emitting test -------------------------------------------------


def test_thresholds_summary(tmp_path: Path) -> None:
    """Run every threshold inline, collect the measured numbers + pass/fail
    state, and write a JSON + Markdown summary. The summary is the artifact a
    reviewer reads to confirm ; the load-bearing assertions stay in their
    own tests (this test only fails if the summary could not be written).

    The summary is written to ``PERF_RESULTS_DIR`` (default
    ``.docs/review-fix``) so it lands alongside the other review-fix
    artifacts."""

    storage_tmp = tmp_path / "storage"
    storage_tmp.mkdir(parents=True, exist_ok=True)
    results: "list[dict[str, Any]]" = []

    # --- Job defaults ---
    try:
        d = _measure_job_defaults()
        _record_result(
            results,
            name="job_defaults",
            category="job",
            threshold="heartbeat 5s / lease 30s / poll 1s",
            measured=(
                f"heartbeat={d['heartbeat_s']}s, lease={d['lease_s']}s, "
                f"poll={d['poll_s']}s"
            ),
            passed=d["passed"],
        )
    except Exception as exc:  # noqa: BLE001
        _record_result(
            results,
            name="job_defaults",
            category="job",
            threshold="heartbeat 5s / lease 30s / poll 1s",
            measured=None,
            passed=False,
            note=f"measurement error: {exc!r}",
        )

    # --- Coordination p95 ---
    try:
        c = _measure_coordination_p95()
        _record_result(
            results,
            name="coordination_acquire_renew_release_p95",
            category="coordination",
            threshold="p95 <= 5 ms",
            measured=(
                f"acquire={c['acquire_p95_ms']:.3f}ms, "
                f"renew={c['renew_p95_ms']:.3f}ms, "
                f"release={c['release_p95_ms']:.3f}ms"
            ),
            passed=c["passed"],
        )
    except Exception as exc:  # noqa: BLE001
        _record_result(
            results,
            name="coordination_acquire_renew_release_p95",
            category="coordination",
            threshold="p95 <= 5 ms",
            measured=None,
            passed=False,
            note=f"measurement error: {exc!r}",
        )

    # --- RuntimeBuilder.build p95 ---
    try:
        r = _measure_runtime_build_p95(storage_tmp / "rt")
        _record_result(
            results,
            name="runtime_build_p95",
            category="runtime",
            threshold="p95 <= 1 s",
            measured=f"{r['p95_ms']:.1f} ms",
            passed=r["passed"],
        )
    except Exception as exc:  # noqa: BLE001
        _record_result(
            results,
            name="runtime_build_p95",
            category="runtime",
            threshold="p95 <= 1 s",
            measured=None,
            passed=False,
            note=f"measurement error: {exc!r}",
        )

    # --- Event append p95 ---
    try:
        e = _measure_event_append_p95(storage_tmp / "events")
        _record_result(
            results,
            name="event_single_append_p95",
            category="event",
            threshold="p95 <= 20 ms",
            measured=f"{e['p95_ms']:.3f} ms",
            passed=e["passed"],
        )
    except Exception as exc:  # noqa: BLE001
        _record_result(
            results,
            name="event_single_append_p95",
            category="event",
            threshold="p95 <= 20 ms",
            measured=None,
            passed=False,
            note=f"measurement error: {exc!r}",
        )

    # --- Artifact integrity 10k ---
    try:
        a = _measure_artifact_integrity_10k()
        _record_result(
            results,
            name="artifact_integrity_10000_rw",
            category="artifact",
            threshold="0 mismatch in 10,000 r/w",
            measured=(
                f"{a['iterations']} iterations, {a['mismatches']} mismatches"
            ),
            passed=a["passed"],
        )
    except Exception as exc:  # noqa: BLE001
        _record_result(
            results,
            name="artifact_integrity_10000_rw",
            category="artifact",
            threshold="0 mismatch in 10,000 r/w",
            measured=None,
            passed=False,
            note=f"measurement error: {exc!r}",
        )

    # --- Stability stress ---
    try:
        s = _measure_stability_stress()
        _record_result(
            results,
            name="stability_stress_10000_ops",
            category="stability",
            threshold="< 0.1% non-injected error",
            measured=(
                f"{s['ops']} ops, {s['errors']} errors "
                f"({s['error_rate'] * 100:.4f}%)"
            ),
            passed=s["passed"],
        )
    except Exception as exc:  # noqa: BLE001
        _record_result(
            results,
            name="stability_stress_10000_ops",
            category="stability",
            threshold="< 0.1% non-injected error",
            measured=None,
            passed=False,
            note=f"measurement error: {exc!r}",
        )

    # --- Job recovery ---
    try:
        v = _measure_job_recovery_threshold(storage_tmp / "jobs")
        _record_result(
            results,
            name="job_recovery_worker_reclaim",
            category="job",
            threshold="<= 35 s after worker exit",
            measured=(
                f"elapsed={v['elapsed_s']}s, recovered={v['recovered_count']}, "
                f"reclaimed_by={v['reclaimed_worker']}"
            ),
            passed=v["passed"],
        )
    except Exception as exc:  # noqa: BLE001
        _record_result(
            results,
            name="job_recovery_worker_reclaim",
            category="job",
            threshold="<= 35 s after worker exit",
            measured=None,
            passed=False,
            note=f"measurement error: {exc!r}",
        )

    # --- Streaming RSS (perf slice only) ---
    try:
        rss = _measure_streaming_rss_if_enabled(storage_tmp / "rss")
        if rss.get("skipped"):
            _record_result(
                results,
                name="streaming_rss_1gib",
                category="artifact",
                threshold="upload + download extra RSS <= 64 MiB",
                measured=None,
                passed=None,
                note=rss.get("reason"),
            )
        else:
            _record_result(
                results,
                name="streaming_rss_1gib",
                category="artifact",
                threshold="upload + download extra RSS <= 64 MiB",
                measured=(
                    f"upload={rss['upload_extra_rss_mib']:.1f} MiB, "
                    f"download={rss['download_extra_rss_mib']:.1f} MiB"
                ),
                passed=rss["passed"],
            )
    except Exception as exc:  # noqa: BLE001
        _record_result(
            results,
            name="streaming_rss_1gib",
            category="artifact",
            threshold="upload + download extra RSS <= 64 MiB",
            measured=None,
            passed=False,
            note=f"measurement error: {exc!r}",
        )

    out_dir = Path(os.environ.get("PERF_RESULTS_DIR", ".docs/review-fix"))
    json_path = _write_summary(results, out_dir=out_dir, name="ac16_thresholds")

    # The summary test passes if the file was written; per-row pass/fail is
    # visible inside the file. A failure here means we could not write the
    # summary (e.g. PERF_RESULTS_DIR pointed at an unwritable path).
    assert json_path.exists(), f"summary was not written to {json_path}"
