#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""event-loop responsiveness: artifact disk I/O runs on worker threads, so
a large upload/download/scan does NOT monopolize the event loop. The proof is a
latency bound on a concurrent heartbeat, not RSS: a regression that moves
file.read / file.write / hashlib.update back onto the event-loop thread would
starve the heartbeat for the whole operation (the loop blocked in the syscall).

The assertion is "the heartbeat ran DURING the operation" (more than the
blocking floor of ~1 beat), not a high count: a blocking implementation holds
the loop for the entire duration and the heartbeat cannot fire until it ends,
so any number of beats above the floor proves the loop was yielded to. The
upload test additionally forces a long operation via a slow source so a healthy
floor of beats is required there."""

import asyncio
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from linktools.ai.storage.filesystem.artifact import (
    FilesystemArtifactBlobStore,
    FilesystemArtifactRecordStore,
)
from linktools.ai.storage.orphan import OrphanSweepConfig, sweep_orphan_blobs


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def _heartbeat_until_cancelled(interval: float):
    beats = 0
    try:
        while True:
            await asyncio.sleep(interval)
            beats += 1
    except asyncio.CancelledError:
        return beats


def test_large_upload_does_not_block_event_loop_heartbeat(tmp_path: Path) -> None:
    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")
    chunk = b"x" * (256 * 1024)
    n_chunks = 8

    async def slow_source():
        for _ in range(n_chunks):
            yield chunk
            await asyncio.sleep(0.02)

    async def run() -> None:
        digest = _digest(chunk * n_chunks)
        hb = asyncio.create_task(_heartbeat_until_cancelled(0.005))
        try:
            await blob.put_if_absent(
                digest=digest, source=slow_source(), size=len(chunk) * n_chunks
            )
        finally:
            beats = await _cancel(hb)
        # ~160ms of source sleeps at a 5ms heartbeat is dozens of beats IF the
        # per-chunk write did not block the loop. A blocking write would starve
        # the heartbeat to a handful (only the between-chunk yields let it run).
        assert beats >= 20, (
            f"heartbeat fired only {beats} times during a ~160ms upload -- the "
            f"event loop was blocked by blocking artifact write I/O"
        )

    asyncio.run(run())


def test_large_download_does_not_block_event_loop(tmp_path: Path) -> None:
    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")
    payload = b"y" * (16 * 1024 * 1024)
    digest = _digest(payload)

    async def seed():
        async def _src():
            yield payload

        await blob.put_if_absent(digest=digest, source=_src(), size=len(payload))

    asyncio.run(seed())

    async def run() -> None:
        hb = asyncio.create_task(_heartbeat_until_cancelled(0.003))
        try:
            collected = 0
            async with blob.open(digest=digest) as chunks:
                async for c in chunks:
                    collected += len(c)
            assert collected == len(payload)
        finally:
            beats = await _cancel(hb)
        # A blocking read would hold the loop for the entire 16MiB download and
        # the heartbeat could not fire until it ended. Requiring beats above the
        # blocking floor proves each chunk read yielded the loop.
        assert beats >= 3, (
            f"heartbeat fired only {beats} times during a 16MiB download -- the "
            f"event loop was blocked by blocking artifact read I/O"
        )

    asyncio.run(run())


def test_orphan_scan_does_not_block_event_loop(tmp_path: Path) -> None:
    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")
    record_store = FilesystemArtifactRecordStore(records_root=tmp_path / "records")

    async def seed():
        async def _src(data):
            yield data

        for i in range(300):
            data = f"blob-{i}".encode()
            await blob.put_if_absent(
                digest=_digest(data), source=_src(data), size=len(data)
            )

    asyncio.run(seed())

    async def run() -> None:
        hb = asyncio.create_task(_heartbeat_until_cancelled(0.003))
        try:
            await sweep_orphan_blobs(
                blob,
                record_store,
                OrphanSweepConfig(
                    grace_period=__import__("datetime").timedelta(seconds=0)
                ),
                now=datetime.now(timezone.utc),
            )
        finally:
            beats = await _cancel(hb)
        assert beats >= 3, (
            f"heartbeat fired only {beats} times during the orphan scan -- the "
            f"event loop was blocked by blocking directory I/O"
        )

    asyncio.run(run())


async def _cancel(task: "asyncio.Task") -> int:
    task.cancel()
    try:
        return await task
    except asyncio.CancelledError:
        return 0


def test_parallel_runs_not_blocked_by_artifact_hash(tmp_path: Path) -> None:
    """Fourth responsiveness scenario: two concurrent artifact puts (parallel
    Runs) sharing one event loop BOTH make progress while a heartbeat ticks.
    A blocking on-loop hash would serialize them -- one put holds the loop for
    its whole hash and the other (plus the heartbeat) starves until it ends.
    The incremental per-chunk hash + the threaded re-verify yield between
    chunks, so the puts interleave and the heartbeat fires throughout."""
    blob = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")
    chunk = b"z" * (256 * 1024)

    async def one_put(tag: str) -> str:
        # Distinct payload per call so both puts run the full write+hash path
        # (rather than the second hitting the blob-exists re-verify path).
        payload = (tag.encode() * (len(chunk) * 4 // len(tag) + 1))[: len(chunk) * 4]
        digest = _digest(payload)

        async def src():
            for _ in range(4):
                yield payload[: len(chunk)]
                await asyncio.sleep(0.01)

        await blob.put_if_absent(
            digest=digest, source=src(), size=len(payload)
        )
        return tag

    async def run() -> None:
        hb = asyncio.create_task(_heartbeat_until_cancelled(0.005))
        try:
            done = await asyncio.gather(one_put("a"), one_put("b"))
        finally:
            beats = await _cancel(hb)
        assert set(done) == {"a", "b"}, f"both puts must complete, got {done}"
        # Each put yields per chunk; interleaved they still let the 5ms heartbeat
        # fire repeatedly across the combined source sleeps. A blocking on-loop
        # hash would drop this to the floor (~1 beat: the loop held for the whole
        # hash). Requiring well above the blocking floor proves the puts interleave.
        assert beats >= 6, (
            f"heartbeat fired only {beats} times during two concurrent artifact "
            f"hashes -- a put monopolized the event loop"
        )

    asyncio.run(run())
