#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared-root content budgets, mutation ownership, and crash recovery."""

import asyncio
import json
import os
import re
import sys
import threading
from pathlib import Path

import pytest
from linktools.ai.storage import FilesystemCacheBackend


def _charge(root: Path) -> int:
    return sum(
        max(1, item.stat().st_size)
        for item in root.iterdir()
        if re.fullmatch(r"[0-9a-f]{64}", item.name)
        and not item.is_symlink() and item.is_file()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent", (False, True))
async def test_instances_share_content_budget(tmp_path: Path, concurrent: bool) -> None:
    first = FilesystemCacheBackend(tmp_path, max_bytes=2)
    second = FilesystemCacheBackend(tmp_path / ".", max_bytes=2)
    await first.put("first", b"aa")
    await second.put("second", b"bb")
    assert _charge(tmp_path) == 2
    for index in range(4):
        writes = (first.put(f"first-{index}", b""), second.put(f"second-{index}", b""))
        if concurrent:
            await asyncio.gather(*writes)
        else:
            for write in writes:
                await write
                assert _charge(tmp_path) <= 2
        assert _charge(tmp_path) <= 2
    keys = tuple(f"{label}-{index}" for label in ("first", "second") for index in range(4))
    retained = await first.contains_many(keys)
    assert len(retained) == 2
    for key in retained:
        assert await first.get(key) == b""


@pytest.mark.asyncio
async def test_rescan_preserves_local_access_order(tmp_path: Path) -> None:
    first = FilesystemCacheBackend(tmp_path, max_bytes=3)
    second = FilesystemCacheBackend(tmp_path, max_bytes=3)
    await first.put("a", b"a")
    await first.put("b", b"b")
    await first.get("a")
    await second.put("c", b"c")
    await first.put("d", b"d")
    assert await first.contains_many(("a", "b", "c", "d")) == frozenset({"a", "c", "d"})
    assert _charge(tmp_path) == 3


@pytest.mark.asyncio
async def test_writer_budget_applies_to_put_but_delete_only_removes_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    large = FilesystemCacheBackend(tmp_path, max_bytes=20)
    small = FilesystemCacheBackend(tmp_path, max_bytes=2)
    await large.put("a", b"aaaa")
    await large.put("b", b"bbbb")
    await large.put("c", b"cccc")

    await small.delete("absent")

    def forbid_scan() -> tuple[tuple[str, int], ...]:
        raise AssertionError("delete must not scan the root")

    with monkeypatch.context() as patch:
        patch.setattr(small, "_scan_index", forbid_scan)
        await small.delete("a")
    assert await large.get("a") is None
    assert _charge(tmp_path) == 8
    await small.put("d", b"dd")
    assert _charge(tmp_path) == 2
    await large.put("e", b"eeee")
    assert _charge(tmp_path) == 6


@pytest.mark.asyncio
async def test_eviction_precedes_atomic_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = FilesystemCacheBackend(tmp_path, max_bytes=4)
    second = FilesystemCacheBackend(tmp_path, max_bytes=4)
    await first.put("a", b"aaaa")
    original_replace = os.replace
    observed: list[int] = []

    def replace(source: str, destination: str | Path) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", Path(destination).name):
            original_replace(source, destination)
            return
        assert _charge(tmp_path) == 0
        assert len(tuple(tmp_path.glob(".content-cache-*.tmp"))) == 1
        original_replace(source, destination)
        observed.append(_charge(tmp_path))

    monkeypatch.setattr(os, "replace", replace)
    await second.put("b", b"bbbb")
    assert observed == [4]
    assert await second.get("a") is None
    assert await first.get("b") == b"bbbb"


@pytest.mark.asyncio
async def test_only_recognized_regular_files_are_accounted_or_removed(tmp_path: Path) -> None:
    cache = FilesystemCacheBackend(tmp_path, max_bytes=2)
    target = tmp_path / "unrelated.txt"
    target.write_bytes(b"external-content")
    upper = tmp_path / ("A" * 64)
    upper.write_bytes(b"uppercase-is-not-a-cache-key")
    directory = tmp_path / cache._name("directory")
    directory.mkdir()
    link = tmp_path / cache._name("link")
    link.symlink_to(target)
    orphan = tmp_path / f".content-cache-{cache._name('orphan')}.ab12_xyz.tmp"
    orphan.write_bytes(b"unfinished")
    temp_link = tmp_path / f".content-cache-{cache._name('temp-link')}.abc123.tmp"
    temp_link.symlink_to(target)
    unknown_temp = tmp_path / ".content-cache-unrecognized.abc.tmp"
    unknown_temp.write_bytes(b"not-owned")
    await cache.put("owned", b"aa")
    assert not orphan.exists()
    await cache.put("next", b"bb")
    await cache.put("link", b"cc")
    await cache.put("directory", b"cc")
    await cache.delete("link")
    await cache.delete("directory")
    assert link.is_symlink() and temp_link.is_symlink()
    assert directory.is_dir()
    assert target.read_bytes() == b"external-content"
    assert upper.read_bytes() == b"uppercase-is-not-a-cache-key"
    assert unknown_temp.read_bytes() == b"not-owned"
    assert _charge(tmp_path) == 2
    assert await cache.get("next") == b"bb"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("scan", "eviction", "orphan"))
async def test_failed_preparation_does_not_publish_new_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    cache = FilesystemCacheBackend(tmp_path, max_bytes=4)
    await cache.put("old", b"old!")
    old = tmp_path / cache._name("old")
    orphan = tmp_path / f".content-cache-{cache._name('orphan')}.abc.tmp"
    if failure in {"scan", "orphan"}:
        (tmp_path / ".content-cache.control").write_text(
            json.dumps({"total": 4, "dirty": True}),
        )
    if failure == "orphan":
        orphan.write_bytes(b"incomplete")
    original_unlink = Path.unlink
    original_os_unlink = os.unlink

    def fail_scan() -> tuple[tuple[str, int], ...]:
        raise PermissionError("scan unavailable")

    def fail_unlink(path: Path, *, missing_ok: bool = False) -> None:
        if path == old:
            raise PermissionError("eviction unavailable")
        original_unlink(path, missing_ok=missing_ok)

    def fail_orphan(path: str, *, dir_fd: int | None = None) -> None:
        if Path(path) == orphan:
            raise PermissionError("orphan cleanup unavailable")
        original_os_unlink(path, dir_fd=dir_fd)

    with monkeypatch.context() as patch:
        if failure == "scan":
            patch.setattr(cache, "_scan_index", fail_scan)
        elif failure == "eviction":
            patch.setattr(Path, "unlink", fail_unlink)
        else:
            patch.setattr(os, "unlink", fail_orphan)
        await cache.put("new", b"new!")
    assert await cache.get("new") is None
    assert await cache.get("old") == b"old!"
    assert _charge(tmp_path) == 4
    assert len(tuple(tmp_path.glob(".content-cache-*.tmp"))) == int(failure == "orphan")
    await cache.put("new", b"new!")
    assert await cache.get("new") == b"new!"
    assert _charge(tmp_path) == 4
    assert not tuple(tmp_path.glob(".content-cache-*.tmp"))


@pytest.mark.asyncio
async def test_cancelled_owner_retains_shared_lock_until_worker_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = FilesystemCacheBackend(tmp_path, max_bytes=4)
    second = FilesystemCacheBackend(tmp_path, max_bytes=4)
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    original_put = first._put_sync
    second_entered = threading.Event()
    original_second_put = second._put_sync

    def blocked_put(key: str, content: bytes) -> None:
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        original_put(key, content)

    def second_put(key: str, content: bytes) -> None:
        second_entered.set()
        original_second_put(key, content)

    monkeypatch.setattr(first, "_put_sync", blocked_put)
    monkeypatch.setattr(second, "_put_sync", second_put)
    owner = asyncio.create_task(first.put("first", b"aaaa"))
    waiter: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(started.wait(), 2)
        waiter = asyncio.create_task(second.put("second", b"bbbb"))
        owner.cancel()
        await asyncio.sleep(0)
        owner.cancel()
        await asyncio.sleep(0.03)
        assert not owner.done()
        assert not second_entered.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await owner
        await asyncio.wait_for(waiter, 2)
        assert await first.get("second") == b"bbbb"
        assert _charge(tmp_path) == 4
    finally:
        release.set()
        await asyncio.gather(owner, *((waiter,) if waiter is not None else ()), return_exceptions=True)


@pytest.mark.asyncio
async def test_rescan_replaces_tokens_before_stale_missing_read_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FilesystemCacheBackend(tmp_path, max_bytes=8)
    other = FilesystemCacheBackend(tmp_path, max_bytes=8)
    await cache.put("a", b"old")
    target = tmp_path / cache._name("a")
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    original_read = Path.read_bytes

    def missing_read(path: Path) -> bytes:
        if path == target:
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5)
            raise FileNotFoundError(path)
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", missing_read)
    reader = asyncio.create_task(cache.get("a"))
    try:
        await asyncio.wait_for(started.wait(), 2)
        await other.put("a", b"fresh")
        await cache.put("b", b"b")
        release.set()
        assert await reader is None
        assert cache._total == _charge(tmp_path) == 6
    finally:
        release.set()
        await asyncio.gather(reader, return_exceptions=True)


_WRITER = """
import asyncio, sys
from pathlib import Path
from linktools.ai.storage import FilesystemCacheBackend
root = Path(sys.argv[1])
label = sys.argv[2]
async def main():
    cache = FilesystemCacheBackend(root, max_bytes=12)
    for index in range(20):
        await cache.put(f'{label}-{index}', b'xxx')
    print('done', flush=True)
asyncio.run(main())
"""


@pytest.mark.asyncio
async def test_processes_sharing_root_enforce_one_content_budget(tmp_path: Path) -> None:
    children = [
        await asyncio.create_subprocess_exec(
            sys.executable, "-c", _WRITER, str(tmp_path), str(index),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        for index in range(4)
    ]
    try:
        results = await asyncio.wait_for(asyncio.gather(*(child.communicate() for child in children)), 15)
        assert all(child.returncode == 0 for child in children), results
        assert all(output.strip() == b"done" for output, _ in results)
        assert _charge(tmp_path) == 12
        assert len(tuple(tmp_path.glob(".content-cache-*.tmp"))) == 0
    finally:
        for child in children:
            if child.returncode is None:
                child.kill()
        await asyncio.gather(*(child.wait() for child in children))


_CRASH_WRITER = """
import asyncio, os, sys, time
from pathlib import Path
from linktools.ai.storage import FilesystemCacheBackend
root = Path(sys.argv[1])
original_replace = os.replace
def blocked_replace(source, destination):
    if Path(destination).name.startswith('.'):
        original_replace(source, destination)
        return
    print('staged', flush=True)
    time.sleep(60)
    original_replace(source, destination)
os.replace = blocked_replace
asyncio.run(FilesystemCacheBackend(root, max_bytes=4).put('crashed', b'dead'))
"""

_WAITING_WRITER = """
import asyncio, sys
from pathlib import Path
from linktools.ai.storage import FilesystemCacheBackend
async def main():
    print('started', flush=True)
    await FilesystemCacheBackend(Path(sys.argv[1]), max_bytes=4).put('survivor', b'live')
    print('done', flush=True)
asyncio.run(main())
"""


@pytest.mark.asyncio
async def test_killed_writer_releases_lock_and_waiter_reclaims_only_orphan_temp(tmp_path: Path) -> None:
    child = await asyncio.create_subprocess_exec(
        sys.executable, "-c", _CRASH_WRITER, str(tmp_path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    waiter: asyncio.subprocess.Process | None = None
    try:
        assert child.stdout is not None
        assert await asyncio.wait_for(child.stdout.readline(), 5) == b"staged\n"
        temporary = tuple(tmp_path.glob(".content-cache-*.tmp"))
        assert len(temporary) == 1
        assert temporary[0].read_bytes() == b"dead"
        waiter = await asyncio.create_subprocess_exec(
            sys.executable, "-c", _WAITING_WRITER, str(tmp_path),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        assert waiter.stdout is not None
        assert await asyncio.wait_for(waiter.stdout.readline(), 5) == b"started\n"
        await asyncio.sleep(0.05)
        assert waiter.returncode is None
        assert temporary[0].exists()
        child.kill()
        await asyncio.wait_for(child.wait(), 5)
        output, error = await asyncio.wait_for(waiter.communicate(), 5)
        assert waiter.returncode == 0, error
        assert output == b"done\n"
        assert not temporary[0].exists()
        cache = FilesystemCacheBackend(tmp_path, max_bytes=4)
        assert await cache.get("crashed") is None
        assert await cache.get("survivor") == b"live"
        assert _charge(tmp_path) == 4
    finally:
        for process in (child, waiter):
            if process is not None and process.returncode is None:
                process.kill()
        await asyncio.gather(*(process.wait() for process in (child, waiter) if process is not None))


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_write_or_block_next_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = FilesystemCacheBackend(tmp_path, max_bytes=4)
    second = FilesystemCacheBackend(tmp_path, max_bytes=4)
    third = FilesystemCacheBackend(tmp_path, max_bytes=4)
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    original_put = first._put_sync

    def blocked_put(key: str, content: bytes) -> None:
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        original_put(key, content)

    monkeypatch.setattr(first, "_put_sync", blocked_put)
    owner = asyncio.create_task(first.put("first", b"aaaa"))
    waiter: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(started.wait(), 2)
        waiter = asyncio.create_task(second.put("cancelled", b"bbbb"))
        await asyncio.sleep(0.03)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiter, 2)
        release.set()
        await asyncio.wait_for(owner, 2)
        assert await third.get("cancelled") is None
        await asyncio.wait_for(third.put("third", b"cccc"), 2)
        assert await third.get("third") == b"cccc"
        assert _charge(tmp_path) == 4
    finally:
        release.set()
        await asyncio.gather(owner, *((waiter,) if waiter is not None else ()), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("after_replace", (False, True))
async def test_failed_publication_accounts_for_the_file_that_remains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, after_replace: bool,
) -> None:
    from linktools.ai.storage import _cache as cache_module

    cache = FilesystemCacheBackend(tmp_path, max_bytes=6)
    await cache.put("a", b"old")

    original_replace = os.replace

    def fail_replace(source: str, destination: str | Path) -> None:
        if Path(destination) == tmp_path / cache._name("a"):
            raise OSError("replace failed")
        original_replace(source, destination)

    def fail_sync(root: Path) -> None:
        raise OSError("directory sync failed")

    with monkeypatch.context() as patch:
        if after_replace:
            patch.setattr(cache_module, "sync_directory", fail_sync)
        else:
            patch.setattr(os, "replace", fail_replace)
        await cache.put("a", b"newest")
    expected = b"newest" if after_replace else b"old"
    assert await cache.get("a") == expected
    assert cache._total == _charge(tmp_path) == len(expected)
    assert not tuple(tmp_path.glob(".content-cache-*.tmp"))


@pytest.mark.asyncio
async def test_clean_mutations_reuse_shared_accounting_without_full_scans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FilesystemCacheBackend(tmp_path, max_bytes=8)
    await cache.put("a", b"aaaa")

    def forbid_scan() -> tuple[tuple[str, int], ...]:
        raise AssertionError("clean mutations must not scan all content metadata")

    monkeypatch.setattr(cache, "_scan_index", forbid_scan)
    await cache.put("b", b"bbbb")
    await cache.put("c", b"cccc")
    await cache.delete("b")
    await cache.delete("absent")
    assert await cache.get("c") == b"cccc"
    assert _charge(tmp_path) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("record", (None, b"{broken", b'{"total":true,"dirty":false}', b'{"total":0,"dirty":true}'))
async def test_missing_corrupt_or_dirty_accounting_recovers_from_content(
    tmp_path: Path, record: bytes | None,
) -> None:
    cache = FilesystemCacheBackend(tmp_path, max_bytes=6)
    await cache.put("old", b"old!")
    control = tmp_path / ".content-cache.control"
    if record is None:
        control.unlink()
    else:
        control.write_bytes(record)
    await cache.put("new", b"new!")
    assert await cache.get("new") == b"new!"
    assert _charge(tmp_path) == 4
    assert json.loads(control.read_bytes()) == {"total": 4, "dirty": False}


@pytest.mark.asyncio
async def test_new_instance_calibrates_even_when_control_looks_clean(tmp_path: Path) -> None:
    original = FilesystemCacheBackend(tmp_path, max_bytes=6)
    await original.put("old", b"old!")
    (tmp_path / ".content-cache.control").write_text('{"total":0,"dirty":false}')
    reopened = FilesystemCacheBackend(tmp_path, max_bytes=6)
    await reopened.put("new", b"new!")
    assert _charge(tmp_path) == 4
    assert await reopened.get("new") == b"new!"


@pytest.mark.asyncio
async def test_stale_local_charges_never_change_shared_accounting(tmp_path: Path) -> None:
    first = FilesystemCacheBackend(tmp_path, max_bytes=8)
    second = FilesystemCacheBackend(tmp_path, max_bytes=8)
    await first.put("a", b"aaaa")
    await first.put("b", b"bbbb")
    await second.put("a", b"a")
    await first.put("a", b"aa")
    assert _charge(tmp_path) == 6
    await second.put("b", b"b")
    await first.delete("b")
    assert _charge(tmp_path) == 2
    assert json.loads((tmp_path / ".content-cache.control").read_bytes())["total"] == 2


@pytest.mark.asyncio
async def test_stale_candidates_reconcile_names_without_statting_every_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = 128
    first = FilesystemCacheBackend(tmp_path, max_bytes=count)
    for index in range(count):
        await first.put(f"first-{index}", b"a")
    second = FilesystemCacheBackend(tmp_path, max_bytes=count)
    for index in range(count):
        await second.put(f"second-{index}", b"b")
    original = first._file_charge
    considered: list[str] = []

    def charge(path: Path) -> int | None:
        if re.fullmatch(r"[0-9a-f]{64}", path.name):
            considered.append(path.name)
        return original(path)

    monkeypatch.setattr(first, "_file_charge", charge)
    await first.put("last", b"c")
    assert len(considered) < 10
    assert _charge(tmp_path) == count
    assert await first.get("last") == b"c"


@pytest.mark.asyncio
async def test_oversized_put_does_not_shrink_existing_content(tmp_path: Path) -> None:
    large = FilesystemCacheBackend(tmp_path, max_bytes=8)
    small = FilesystemCacheBackend(tmp_path, max_bytes=0)
    await large.put("existing", b"existing")
    await small.put("empty", b"")
    assert await small.get("existing") == b"existing"
    assert await large.get("empty") is None
    assert _charge(tmp_path) == 8


_CONTROL_CRASH_WRITER = """
import asyncio, json, os, re, sys, time
from pathlib import Path
from linktools.ai.storage import FilesystemCacheBackend
root, stage = Path(sys.argv[1]), sys.argv[2]
def stop(where):
    if where == stage:
        print('staged', flush=True)
        time.sleep(60)
async def main():
    cache = FilesystemCacheBackend(root, max_bytes=4)
    await cache.delete('absent')
    original_replace, original_unlink = os.replace, Path.unlink
    def replace(source, destination):
        destination = Path(destination)
        if destination.name == '.content-cache.control':
            dirty = json.loads(Path(source).read_bytes())['dirty']
            phase = 'dirty' if dirty else 'clean'
            stop('before_' + phase)
            original_replace(source, destination)
            stop('after_' + phase)
        else:
            stop('content_temp')
            original_replace(source, destination)
            stop('content_publish')
    def unlink(path, *args, **kwargs):
        original_unlink(path, *args, **kwargs)
        if re.fullmatch(r'[0-9a-f]{64}', path.name):
            stop('eviction')
    os.replace, Path.unlink = replace, unlink
    await cache.put('crashed', b'dead')
asyncio.run(main())
"""


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", (
    "before_dirty", "after_dirty", "eviction", "content_temp",
    "content_publish", "before_clean", "after_clean",
))
async def test_surviving_instance_recovers_each_interrupted_mutation_stage(
    tmp_path: Path, stage: str,
) -> None:
    survivor = FilesystemCacheBackend(tmp_path, max_bytes=8)
    await survivor.put("a", b"aaaa")
    await survivor.put("b", b"bbbb")
    child = await asyncio.create_subprocess_exec(
        sys.executable, "-c", _CONTROL_CRASH_WRITER, str(tmp_path), stage,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        assert child.stdout is not None
        assert await asyncio.wait_for(child.stdout.readline(), 5) == b"staged\n"
        child.kill()
        await asyncio.wait_for(child.wait(), 5)
        await survivor.put("survivor", b"live")
        record = json.loads((tmp_path / ".content-cache.control").read_bytes())
        assert record == {"total": _charge(tmp_path), "dirty": False}
        assert record["total"] <= 8
        assert await survivor.get("survivor") == b"live"
        assert not (tmp_path / ".content-cache.control.tmp").exists()
        assert not tuple(tmp_path.glob(".content-cache-*.tmp"))
    finally:
        if child.returncode is None:
            child.kill()
        await child.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("dirty", (False, True))
@pytest.mark.parametrize("after_replace", (False, True))
async def test_failed_control_publication_is_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dirty: bool, after_replace: bool,
) -> None:
    cache = FilesystemCacheBackend(tmp_path, max_bytes=6)
    other = FilesystemCacheBackend(tmp_path, max_bytes=6)
    await cache.put("a", b"old")
    await other.delete("absent")
    original_replace = os.replace

    def replace(source: str, destination: str | Path) -> None:
        if Path(destination).name == ".content-cache.control":
            record = json.loads(Path(source).read_bytes())
            if record["dirty"] is dirty:
                if after_replace:
                    original_replace(source, destination)
                raise OSError("control publication failed")
        original_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", replace)
        await cache.put("a", b"newest")
    expected = b"old" if dirty else b"newest"
    assert await cache.get("a") == expected
    assert _charge(tmp_path) == len(expected)
    assert not (tmp_path / ".content-cache.control.tmp").exists()
    await other.put("b", b"second")
    record = json.loads((tmp_path / ".content-cache.control").read_bytes())
    assert record == {"total": _charge(tmp_path), "dirty": False}
    assert _charge(tmp_path) == 6
    assert await other.get("b") == b"second"


@pytest.mark.asyncio
@pytest.mark.parametrize("name", (".content-cache.control", ".content-cache.control.tmp"))
async def test_reserved_metadata_symlinks_are_never_followed_or_removed(
    tmp_path: Path, name: str,
) -> None:
    target = tmp_path / "unrelated"
    target.write_bytes(b"do-not-change")
    link = tmp_path / name
    link.symlink_to(target)
    cache = FilesystemCacheBackend(tmp_path, max_bytes=4)
    await cache.put("new", b"data")
    assert await cache.get("new") is None
    assert link.is_symlink()
    assert target.read_bytes() == b"do-not-change"


@pytest.mark.asyncio
async def test_remote_delete_churn_does_not_grow_local_candidates(tmp_path: Path) -> None:
    writer = FilesystemCacheBackend(tmp_path, max_bytes=1)
    remover = FilesystemCacheBackend(tmp_path, max_bytes=0)
    for index in range(128):
        key = f"entry-{index}"
        await writer.put(key, b"x")
        await remover.delete(key)
    assert _charge(tmp_path) == 0
    assert len(writer._entries) <= writer.max_bytes
    assert len(remover._entries) == 0
    await writer.put("retained", b"x")
    assert await writer.get("retained") == b"x"
    assert _charge(tmp_path) == 1


@pytest.mark.asyncio
async def test_calibration_and_recovery_keep_local_candidates_bounded(tmp_path: Path) -> None:
    large = FilesystemCacheBackend(tmp_path, max_bytes=16)
    for index in range(16):
        await large.put(f"entry-{index}", b"x")
    small = FilesystemCacheBackend(tmp_path, max_bytes=2)
    await small.delete("absent")
    assert len(small._entries) <= 2
    assert _charge(tmp_path) == 16
    (tmp_path / ".content-cache.control").write_text('{"total":0,"dirty":true}')
    await small.delete("absent")
    assert len(small._entries) <= 2
    assert _charge(tmp_path) == 16
    await small.put("accepted", b"xx")
    assert len(small._entries) <= 2
    assert _charge(tmp_path) == 2
    assert await small.get("accepted") == b"xx"
