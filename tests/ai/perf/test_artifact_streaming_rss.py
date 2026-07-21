#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from linktools.ai.artifact import ANONYMOUS_PROVENANCE
"""RF-04 / plan §6.4 streaming-RSS cap: the headline 1 GiB benchmark.

The plan §5 R7 op 1-3 requires a REAL RSS measurement on a REAL 1 GiB stream of
NON-REPEATING bytes through ``ArtifactStore.put_stream`` and ``open_stream``
over a ``FilesystemArtifactBlobStore``, asserting the EXTRA RSS (peak during
the op minus baseline) is ≤ 64 MiB in both directions. The failure-handling
section (line 1255) forbids lowering the scale, restricting to the bytes API,
or removing the RSS assertion to make the test green -- the proof has to be
honest.

Why non-repeating bytes: §6.2 calls out a constant buffer (e.g. all-zero) as
forbidden because the OS page cache could compress/dedup it, masking RSS. The
generator below yields 1 MiB chunks whose contents are derived from a
per-chunk-index PRNG seed: no two chunks share bytes, and within each chunk the
bytes are pseudo-random.

Why /proc/self/status VmRSS: it is the resident-set size the kernel reports
for THIS process, sampled by a background thread at ~5 ms granularity (a thread
rather than an asyncio task so the sampler is not gated by the I/O-bound async
work). The peak across the op minus the pre-op baseline is the EXTRA RSS
attributable to the streaming operation.

This test runs in the default suite (it moves ~1 GiB of I/O but completes in a
few seconds); a skipped acceptance test is not evidence (§6.8), so this asserts
the ≤ 64 MiB cap unconditionally rather than only opt-in.

The benchmark writes a JSON + Markdown summary capturing the measured numbers
under ``PERF_RESULTS_DIR`` (default ``.docs/review-fix``) so a reviewer can
see the actual RSS rather than just a pass/fail bit (plan §5 op 7)."""

import asyncio
import gc
import hashlib
import json
import os
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Callable

import pytest

from linktools.ai.artifact.store import ArtifactStore
from linktools.ai.storage.filesystem.artifact import (
    FilesystemArtifactBlobStore,
    FilesystemArtifactRecordStore,
)

_GIB = 1024 ** 3
_MIB = 1024 ** 2
_KIB = 1024
# Plan §6.4 / RF-04: extra RSS ceiling for streaming a 1 GiB artifact.
_RSS_LIMIT_MIB = 64
# Default chunk size yielded by the source generator: 1 MiB. Small enough that
# peak RSS from a single in-flight chunk is trivially bounded; large enough
# that the generator overhead is negligible vs the hashing + I/O cost.
_CHUNK_KIB = 1024
# Sampler interval: ~5 ms gives ~200 samples per second of operation (the POC
# yielded ~800 samples for a 4 s upload), enough resolution to catch a transient
# peak without measurably perturbing the work.
_SAMPLE_INTERVAL_S = 0.005


def _read_vm_rss_kib() -> "int | None":
    """Read VmRSS (KiB) from ``/proc/self/status``. Returns None outside Linux
    or if the file is unexpectedly shaped -- the caller skips the test rather
    than fabricating a number."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (FileNotFoundError, ValueError, IndexError):
        return None
    return None


class _RssSampler:
    """Background thread sampling VmRSS at a fixed interval. Tracks the peak
    observed value so the caller can compute (peak - baseline) = the extra RSS
    attributable to the operation. The thread is a daemon and stops on demand
    via an Event so it does not outlive the test."""

    __slots__ = ("_interval", "_stop", "_thread", "_samples", "_baseline")

    def __init__(self, interval_s: float = _SAMPLE_INTERVAL_S) -> None:
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: "threading.Thread | None" = None
        self._samples: "list[int]" = []
        self._baseline: "int | None" = None

    @property
    def baseline_kib(self) -> "int | None":
        return self._baseline

    def start(self) -> None:
        # Snapshot the baseline BEFORE the work begins; the peak delta is
        # measured relative to this point.
        self._baseline = _read_vm_rss_kib()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = _read_vm_rss_kib()
            if rss is not None:
                self._samples.append(rss)
            # Use wait() not sleep() so stop() unblocks immediately.
            self._stop.wait(self._interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def peak_kib(self) -> "int | None":
        if not self._samples:
            return None
        return max(self._samples)

    @property
    def sample_count(self) -> int:
        return len(self._samples)


def _gen_non_repeating_gib(
    *, total_bytes: int = _GIB, chunk_kib: int = _CHUNK_KIB
) -> AsyncIterator[bytes]:
    """Async generator yielding ``total_bytes`` of NON-REPEATING content in
    ``chunk_kib``-sized chunks. Each chunk's bytes come from a PRNG seeded with
    its index, so (a) no two chunks share content (defeating page-cache dedup)
    and (b) within a chunk the bytes are pseudo-random (defeating compression).
    The generator never holds more than one chunk at a time, and explicitly
    drops the reference before allocating the next so peak RSS reflects a
    single chunk rather than the accumulated total."""
    chunk_size = chunk_kib * _KIB
    chunks_total = total_bytes // chunk_size
    assert chunks_total * chunk_size == total_bytes, "size must be chunk-aligned"

    async def _gen() -> AsyncIterator[bytes]:
        for i in range(chunks_total):
            chunk = random.Random(i).randbytes(chunk_size)
            yield chunk
            # Drop the reference before the next allocation; without this the
            # previous chunk's storage stays live until the next loop iteration
            # overwrites the name, slightly inflating peak RSS for no reason
            # related to the streaming implementation under test.
            chunk = None  # noqa: F841

    return _gen()


def _write_summary(payload: dict, *, out_dir: Path) -> None:
    """Persist the benchmark summary as JSON + Markdown for op 7. The JSON is
    machine-readable (a CI artifact); the Markdown is reviewer-readable (paste
    into a report). Both go to ``PERF_RESULTS_DIR`` so a reviewer can see the
    actual numbers, not just a pass/fail bit."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rss_benchmark.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    name = payload.get("name", "benchmark")
    upload = payload["upload"]
    download = payload["download"]
    md_lines = [
        f"# {name}",
        "",
        f"- scale: {payload['scale_gib']} GiB non-repeating bytes",
        f"- baseline RSS: {payload['baseline_rss_mib']:.1f} MiB",
        f"- sampled at ~{int(1 / _SAMPLE_INTERVAL_S)} Hz via /proc/self/status VmRSS",
        "",
        "| direction | baseline (MiB) | peak (MiB) | extra (MiB) | limit (MiB) | result |",
        "|---|---|---|---|---|---|",
        f"| upload | {upload['baseline_rss_mib']:.1f} | {upload['peak_rss_mib']:.1f} | "
        f"{upload['extra_rss_mib']:.1f} | {upload['limit_mib']} | "
        f"{'PASS' if upload['extra_rss_mib'] <= upload['limit_mib'] else 'FAIL'} |",
        f"| download | {download['baseline_rss_mib']:.1f} | {download['peak_rss_mib']:.1f} | "
        f"{download['extra_rss_mib']:.1f} | {download['limit_mib']} | "
        f"{'PASS' if download['extra_rss_mib'] <= download['limit_mib'] else 'FAIL'} |",
        "",
        f"- generated: {payload['generated_at']}",
        f"- overall: {'PASS' if payload['passed'] else 'FAIL'}",
    ]
    (out_dir / "rss_benchmark.md").write_text("\n".join(md_lines), encoding="utf-8")


def test_put_stream_and_open_stream_extra_rss_under_64mib(tmp_path: Path) -> None:
    """Streaming 1 GiB of non-repeating bytes through ``ArtifactStore.put_stream``
    and ``open_stream`` over a ``FilesystemArtifactBlobStore`` must keep the
    extra RSS ≤ 64 MiB in both directions.

    The scale is the FULL 1 GiB; the data is NON-REPEATING (per-chunk PRNG
    seed); the RSS is the real peak-during-minus-baseline from a background
    thread sampling /proc/self/status. The 64 MiB ceiling is the plan §6.4 /
    RF-04 acceptance threshold; the failure-handling line forbids any cheat
    (lower scale, bytes-API only, removed assertion) to make it green."""
    if _read_vm_rss_kib() is None:
        pytest.skip("/proc/self/status VmRSS unavailable -- requires Linux")

    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")
    records = FilesystemArtifactRecordStore(records_root=tmp_path / "records")
    store = ArtifactStore(blob, records)

    pre_baseline_kib = _read_vm_rss_kib()
    assert pre_baseline_kib is not None

    # ----- upload phase -----
    sampler_up = _RssSampler()
    sampler_up.start()
    upload_t0 = time.perf_counter()
    try:
        async def _upload() -> "object":
            return await store.put_stream(
                source=_gen_non_repeating_gib(total_bytes=_GIB, chunk_kib=_CHUNK_KIB),
                media_type="application/octet-stream",
                tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,
    )

        record = asyncio.run(_upload())
    finally:
        sampler_up.stop()
    upload_elapsed_s = time.perf_counter() - upload_t0

    # Sanity: a full 1 GiB was published (a regression that truncated the
    # stream would slip past the RSS assertion otherwise).
    assert record.ref.size == _GIB, (
        f"expected 1 GiB artifact, got {record.ref.size} bytes"
    )

    # Release any per-iteration state before measuring download RSS so the
    # download baseline is not inflated by leftover allocator state from the
    # upload (Python may not return memory to the OS immediately).
    gc.collect()

    # ----- download phase -----
    sampler_down = _RssSampler()
    sampler_down.start()
    download_t0 = time.perf_counter()
    hasher = hashlib.sha256()
    downloaded = 0
    try:
        async def _download() -> None:
            nonlocal downloaded
            async with store.open_stream(artifact_id=record.ref.id, tenant_id="t1") as chunks:
                async for chunk in chunks:
                    downloaded += len(chunk)
                    hasher.update(chunk)

        asyncio.run(_download())
    finally:
        sampler_down.stop()
    download_elapsed_s = time.perf_counter() - download_t0

    # Integrity: every byte round-tripped and the streamed digest matches the
    # pinned one (this is the same guarantee the at-exhaustion check enforces
    # in production; asserting it here catches a regression that quietly
    # truncated the stream behind a passing RSS number).
    assert downloaded == _GIB, f"downloaded {downloaded}, expected {_GIB}"
    assert hasher.hexdigest() == record.ref.sha256, (
        "streamed sha256 does not match the pinned record digest"
    )

    # ----- compute deltas + persist summary -----
    def _extra_mib(sampler: _RssSampler) -> "float | None":
        if sampler.peak_kib is None or sampler.baseline_kib is None:
            return None
        return (sampler.peak_kib - sampler.baseline_kib) / _KIB

    delta_up_mib = _extra_mib(sampler_up)
    delta_down_mib = _extra_mib(sampler_down)
    assert delta_up_mib is not None and delta_down_mib is not None, (
        "RSS sampler produced no samples -- measurement is unreliable"
    )

    out_dir = Path(os.environ.get("PERF_RESULTS_DIR", ".docs/review-fix"))
    summary = {
        "name": "artifact_1gib_streaming_rss",
        "scale_gib": 1,
        "scale_bytes": _GIB,
        "baseline_rss_mib": pre_baseline_kib / _KIB,
        "chunk_kib": _CHUNK_KIB,
        "sample_interval_s": _SAMPLE_INTERVAL_S,
        "upload": {
            "baseline_rss_mib": (sampler_up.baseline_kib or 0) / _KIB,
            "peak_rss_mib": (sampler_up.peak_kib or 0) / _KIB,
            "extra_rss_mib": delta_up_mib,
            "limit_mib": _RSS_LIMIT_MIB,
            "samples": sampler_up.sample_count,
            "elapsed_s": upload_elapsed_s,
        },
        "download": {
            "baseline_rss_mib": (sampler_down.baseline_kib or 0) / _KIB,
            "peak_rss_mib": (sampler_down.peak_kib or 0) / _KIB,
            "extra_rss_mib": delta_down_mib,
            "limit_mib": _RSS_LIMIT_MIB,
            "samples": sampler_down.sample_count,
            "elapsed_s": download_elapsed_s,
        },
        "passed": (
            delta_up_mib <= _RSS_LIMIT_MIB and delta_down_mib <= _RSS_LIMIT_MIB
        ),
        "integrity_sha256": record.ref.sha256,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_summary(summary, out_dir=out_dir)

    # ----- the load-bearing assertions -----
    assert delta_up_mib <= _RSS_LIMIT_MIB, (
        f"upload extra RSS {delta_up_mib:.1f} MiB exceeds the "
        f"{_RSS_LIMIT_MIB} MiB cap (baseline {sampler_up.baseline_kib / _KIB:.1f} MiB, "
        f"peak {sampler_up.peak_kib / _KIB:.1f} MiB)"
    )
    assert delta_down_mib <= _RSS_LIMIT_MIB, (
        f"download extra RSS {delta_down_mib:.1f} MiB exceeds the "
        f"{_RSS_LIMIT_MIB} MiB cap (baseline {sampler_down.baseline_kib / _KIB:.1f} MiB, "
        f"peak {sampler_down.peak_kib / _KIB:.1f} MiB)"
    )
