#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline filesystem inventory retains every participating writer boundary."""

import asyncio
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence._filesystem import FilesystemStateStore

from .test_filesystem_live_reads import _key, _store

pytestmark = pytest.mark.asyncio


async def test_offline_scope_rejects_same_group_reentry_without_releasing_its_lock(tmp_path: Path) -> None:
    root = tmp_path / "state"
    writer = _store(root, grouped=True)
    await writer.initialize()
    await writer.close()
    offline = _store(root, grouped=True)
    contender = _store(root, grouped=True)
    try:
        async with offline.storage_group.offline_exclusivity():
            with pytest.raises(AIError) as nested:
                async with offline.storage_group.offline_exclusivity():
                    pytest.fail("a second scope reused the first scope's physical locks")
            assert nested.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
            with pytest.raises(AIError) as blocked:
                await contender.initialize()
            assert blocked.value.code is ErrorCode.STORAGE_CONFLICT
        await contender.initialize()
    finally:
        await contender.close()
        await offline.close()


async def test_offline_scope_discovers_partitions_after_group_lock_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "state"
    writer = _store(root, grouped=True)
    await writer.initialize()
    await writer.close()
    offline = _store(root, grouped=True)
    group = offline.storage_group
    entered, release = asyncio.Event(), asyncio.Event()
    original_acquire = group._group_lock.acquire

    async def gated_acquire() -> None:
        entered.set()
        await release.wait()
        await original_acquire()

    monkeypatch.setattr(group._group_lock, "acquire", gated_acquire)

    async def inspect() -> None:
        async with group.offline_exclusivity():
            pytest.fail("offline scope missed the late member's active standalone writer")

    checking = asyncio.create_task(inspect())
    late_owner = _store(root, grouped=True)
    standalone = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await late_owner.initialize()
        late = await late_owner.storage_group.open_partition(_key("late"), create=True)
        assert late is not None
        await late_owner.close()
        standalone = FilesystemStateStore(late.root, namespace="live", tenant_id="tenant",
                                          runtime_domain=f"group_partition:{_key('late').hex()}")
        await standalone.initialize()
        release.set()
        with pytest.raises(AIError) as blocked:
            await asyncio.wait_for(checking, 5)
        assert blocked.value.code is ErrorCode.STORAGE_CONFLICT
    finally:
        release.set()
        if not checking.done():
            checking.cancel()
            await asyncio.gather(checking, return_exceptions=True)
        if standalone is not None:
            await standalone.close()
        await late_owner.close()
        await offline.close()


async def test_offline_inventory_revalidates_complete_membership_after_all_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    root = tmp_path / "state"
    writer = _store(root, grouped=True)
    await writer.initialize()
    partition = await writer.storage_group.open_partition(_key("changed"), create=True)
    assert partition is not None
    await writer.close()
    manifest = root / ".state-groups/scope/manifest.json"
    complete = manifest.read_bytes()
    value = json.loads(complete)
    value["members"] = [member for member in value["members"] if member["runtime_domain"] == "execution"]
    manifest.write_text(json.dumps(value))
    offline = _store(root, grouped=True)
    acquire = offline._writer_lock.acquire

    async def change_inventory_after_lock() -> None:
        await acquire()
        manifest.write_bytes(complete)

    monkeypatch.setattr(offline._writer_lock, "acquire", change_inventory_after_lock)
    try:
        with pytest.raises(AIError) as rejected:
            async with offline.storage_group.offline_exclusivity():
                pytest.fail("changed inventory was accepted without all member locks")
        assert rejected.value.code is ErrorCode.STORAGE_CONFLICT
        assert not offline.storage_group._offline_locks
        with pytest.raises(AIError):
            offline.storage_group.describe_offline_members()
        assert manifest.read_bytes() == complete
    finally:
        await offline.close()


async def test_offline_inventory_rejects_member_added_before_async_validation_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    root = tmp_path / "state"
    writer = _store(root, grouped=True)
    await writer.initialize()
    await writer.close()
    offline = _store(root, grouped=True)
    group = offline.storage_group
    describe = group._describe_locked_members_sync
    loop = asyncio.get_running_loop()
    registered = threading.Event()
    late: list[FilesystemStateStore] = []

    def register_member() -> None:
        late.append(FilesystemStateStore(root / "late", namespace="live", tenant_id="tenant",
                                         runtime_domain="capability:late", group=group))
        registered.set()

    def delayed_return(members):
        result = describe(members)
        loop.call_soon_threadsafe(register_member)
        assert registered.wait(5)
        return result

    monkeypatch.setattr(group, "_describe_locked_members_sync", delayed_return)
    try:
        with pytest.raises(AIError) as rejected:
            async with group.offline_exclusivity():
                pytest.fail("member registered after threaded validation was not locked")
        assert rejected.value.code is ErrorCode.STORAGE_CONFLICT
        assert len(late) == 1 and not late[0].root.exists()
        assert not group._offline_locks
    finally:
        for member in late:
            await member.close()
        await offline.close()


@pytest.mark.parametrize("grouped", [False, True])
async def test_offline_members_are_task_scoped_immutable_facts_not_export_authority(
    tmp_path: Path, grouped: bool,
) -> None:
    from dataclasses import FrozenInstanceError

    root = tmp_path / "state"
    writer = _store(root, grouped=grouped)
    explicit = (
        FilesystemStateStore(root / "explicit", namespace="live", tenant_id="tenant",
                             runtime_domain="capability:explicit", group=writer.storage_group)
        if grouped else None
    )
    await writer.initialize()
    dynamic = await writer.storage_group.open_partition(_key("dynamic"), create=True)
    archive = await writer.storage_group.open_archive(create=True)
    assert dynamic is not None and archive is not None
    if explicit is not None:
        await explicit.close()
    await writer.close()
    offline = _store(root, grouped=grouped)
    group = offline.storage_group
    with pytest.raises(AIError):
        group.describe_offline_members()
    try:
        async with group.offline_exclusivity():
            described = group.describe_offline_members()
            members = {member.runtime_domain: member for member in described}
            expected = {"execution", "group_archive", f"group_partition:{_key('dynamic').hex()}"}
            if explicit is not None:
                expected.add("capability:explicit")
                assert members["capability:explicit"].root == explicit.root
                assert members["capability:explicit"].partition_key is None
            assert set(members) == expected
            assert members[f"group_partition:{_key('dynamic').hex()}"].partition_key == _key("dynamic")
            assert members["group_archive"].partition_key is None
            with pytest.raises(FrozenInstanceError):
                described[0].runtime_domain = "changed"

            async def unrelated_task() -> None:
                with pytest.raises(AIError) as denied:
                    group.describe_offline_members()
                assert denied.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
                with pytest.raises(AIError):
                    async with group.offline_exclusivity():
                        pytest.fail("another task reused this scope")
                with pytest.raises(AIError):
                    await group.close()

            await asyncio.create_task(unrelated_task())
            with pytest.raises(AIError):
                await group.initialize()
            assert group.describe_offline_members() is described
        with pytest.raises(AIError):
            group.describe_offline_members()
        assert described[0].root.is_dir()
    finally:
        await offline.close()


@pytest.mark.parametrize("grouped", [False, True])
async def test_offline_acquisition_cancellation_settles_owned_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, grouped: bool,
) -> None:
    root = tmp_path / "state"
    writer = _store(root, grouped=grouped)
    await writer.initialize()
    await writer.close()
    offline = _store(root, grouped=grouped)
    group = offline.storage_group
    entered, release = asyncio.Event(), asyncio.Event()
    acquire = offline._writer_lock.acquire

    async def delayed_acquire() -> None:
        entered.set()
        await release.wait()
        await acquire()

    monkeypatch.setattr(offline._writer_lock, "acquire", delayed_acquire)

    async def inspect() -> None:
        async with group.offline_exclusivity():
            pytest.fail("cancelled admission yielded an offline scope")

    task = asyncio.create_task(inspect())
    contender = _store(root, grouped=grouped)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        with pytest.raises(AIError):
            group.describe_offline_members()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert not group._offline_locks and group._offline_owner is None
        await contender.initialize()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await contender.close()
        await offline.close()


async def test_offline_release_failure_retains_locks_until_close_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "state"
    writer = _store(root, grouped=True)
    await writer.initialize()
    await writer.close()
    offline = _store(root, grouped=True)
    group = offline.storage_group
    contender = _store(root, grouped=True)
    release = offline._writer_lock.release
    calls = 0
    failure = OSError("release unavailable")

    async def fail_once() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise failure
        await release()

    monkeypatch.setattr(offline._writer_lock, "release", fail_once)
    try:
        with pytest.raises(OSError) as failed:
            async with group.offline_exclusivity():
                assert group.describe_offline_members()
        assert failed.value is failure
        assert len(group._offline_locks) == 2 and group._offline_owner is None
        with pytest.raises(AIError):
            group.describe_offline_members()
        with pytest.raises(AIError):
            async with group.offline_exclusivity():
                pytest.fail("unfinished release granted another scope")
        with pytest.raises(AIError):
            await group.initialize()
        with pytest.raises(AIError) as blocked:
            await contender.initialize()
        assert blocked.value.code is ErrorCode.STORAGE_CONFLICT
        await group.close()
        assert calls == 2 and not group._offline_locks
        await contender.initialize()
    finally:
        await contender.close()
        await offline.close()


@pytest.mark.parametrize("corruption", ["namespace", "format", "missing_member", "missing_archive_scope"])
async def test_offline_inventory_rejects_incomplete_metadata_without_provisioning_it(
    tmp_path: Path, corruption: str,
) -> None:
    import json
    import shutil

    root = tmp_path / "state"
    writer = _store(root, grouped=True)
    await writer.initialize()
    if corruption == "missing_archive_scope":
        assert await writer.storage_group.open_archive(create=True) is not None
    await writer.close()
    manifest = root / ".state-groups/scope/manifest.json"
    value = json.loads(manifest.read_text())
    if corruption == "namespace":
        value["namespace_digest"] = "0" * 64
        manifest.write_text(json.dumps(value))
    elif corruption == "format":
        value["format"] = "unrecognized"
        manifest.write_text(json.dumps(value))
    elif corruption == "missing_member":
        shutil.rmtree(root / "domain")
    else:
        (root / ".state-groups/scope/archive_scope.json").unlink()
    before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    offline = _store(root, grouped=True)
    try:
        with pytest.raises(AIError) as rejected:
            async with offline.storage_group.offline_exclusivity():
                pytest.fail("invalid offline inventory became available")
        assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        with pytest.raises(AIError):
            offline.storage_group.describe_offline_members()
        after = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
        assert after == before
    finally:
        await offline.close()


@pytest.mark.parametrize("grouped", [False, True])
async def test_offline_inventory_rejects_archive_scope_without_its_member(
    tmp_path: Path, grouped: bool,
) -> None:
    import shutil

    root = tmp_path / "state"
    writer = _store(root, grouped=grouped)
    await writer.initialize()
    archive = await writer.storage_group.open_archive(create=True)
    assert archive is not None
    await writer.close()
    shutil.rmtree(archive[1].root)
    before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    offline = _store(root, grouped=grouped)
    try:
        with pytest.raises(AIError) as rejected:
            async with offline.storage_group.offline_exclusivity():
                pytest.fail("archive scope was silently omitted from the offline inventory")
        assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert not archive[1].root.exists()
        after = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
        assert after == before
    finally:
        await offline.close()


@pytest.mark.parametrize("corruption", ["format", "version", "domain"])
async def test_offline_standalone_inventory_validates_its_partition_manifest(
    tmp_path: Path, corruption: str,
) -> None:
    import json

    root = tmp_path / "state"
    writer = _store(root, grouped=False)
    await writer.initialize()
    partition = await writer.storage_group.open_partition(_key("member"), create=True)
    assert partition is not None
    await writer.close()
    manifest = root / ".state-groups/scope/manifest.json"
    value = json.loads(manifest.read_text())
    if corruption == "domain":
        value["members"][0]["runtime_domain"] = "capability:renamed"
    else:
        value[corruption] = "invalid"
    manifest.write_text(json.dumps(value))
    before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    offline = _store(root, grouped=False)
    member = FilesystemStateStore(partition.root, namespace="live", tenant_id="tenant",
                                  runtime_domain=f"group_partition:{_key('member').hex()}",
                                  group=offline.storage_group)
    try:
        with pytest.raises(AIError) as rejected:
            async with offline.storage_group.offline_exclusivity():
                pytest.fail("standalone partition manifest was not validated")
        assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        after = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
        assert after == before
    finally:
        await member.close()
        await offline.close()


async def test_offline_release_cancellation_does_not_release_outer_owner_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "state"
    writer = _store(root, grouped=True)
    await writer.initialize()
    await writer.close()
    offline = _store(root, grouped=True)
    group = offline.storage_group
    entered, release = asyncio.Event(), asyncio.Event()
    original_release = offline._writer_lock.release

    async def delayed_release() -> None:
        entered.set()
        await release.wait()
        await original_release()

    monkeypatch.setattr(offline._writer_lock, "release", delayed_release)

    async def inspect() -> None:
        async with group.offline_exclusivity():
            assert group.describe_offline_members()

    task = asyncio.create_task(inspect())
    contender = _store(root, grouped=True)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and group._offline_owner is task
        with pytest.raises(AIError):
            await group.close()
        with pytest.raises(AIError):
            group.describe_offline_members()
        with pytest.raises(AIError) as blocked:
            await contender.initialize()
        assert blocked.value.code is ErrorCode.STORAGE_CONFLICT
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert group._offline_owner is None and not group._offline_locks
        await contender.initialize()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await contender.close()
        await offline.close()


async def test_offline_scope_preserves_primary_failure_and_close_retries_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "state"
    writer = _store(root, grouped=True)
    await writer.initialize()
    await writer.close()
    offline = _store(root, grouped=True)
    group = offline.storage_group
    primary, secondary = ValueError("source unavailable"), OSError("release unavailable")
    original_release = group._group_lock.release

    async def fail_release() -> None:
        raise secondary

    monkeypatch.setattr(group._group_lock, "release", fail_release)
    try:
        with pytest.raises(ValueError) as failed:
            async with group.offline_exclusivity():
                raise primary
        assert failed.value is primary and failed.value.__cause__ is secondary
        assert len(group._offline_locks) == 1
        assert group._offline_locks[0] is group._group_lock
        monkeypatch.setattr(group._group_lock, "release", original_release)
        await group.close()
        assert not group._offline_locks
    finally:
        monkeypatch.setattr(group._group_lock, "release", original_release)
        await offline.close()
