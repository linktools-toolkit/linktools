#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§16.1 / §16.4: the unified atomic-write helper is crash-safe (temp + fsync +
rename + parent-dir fsync) and a corrupt session JSON surfaces as
SessionCorruptionError rather than being masked as a missing session."""

import asyncio
from pathlib import Path

import pytest

from linktools.ai.errors import SessionCorruptionError
from linktools.ai.storage.file.atomic import atomic_write_bytes


def test_atomic_write_bytes_writes_content_and_cleans_temp(tmp_path) -> None:
    # The helper writes the exact bytes to the destination (creating parent
    # dirs) and leaves no temp file behind on success.
    target = tmp_path / "nested" / "out.json"
    atomic_write_bytes(target, b'{"x": 1}')
    assert target.read_bytes() == b'{"x": 1}'
    leftover = [p for p in (tmp_path / "nested").iterdir() if p.suffix == ".tmp"]
    assert leftover == []


def test_atomic_write_bytes_replaces_existing_atomically(tmp_path) -> None:
    target = tmp_path / "out.json"
    atomic_write_bytes(target, b"first")
    atomic_write_bytes(target, b"second")
    assert target.read_bytes() == b"second"


def test_atomic_write_bytes_fsyncs_parent_directory_after_replace(
    tmp_path, monkeypatch
) -> None:
    # The parent-dir fsync is the §16.1 crash-safety invariant: without it the
    # rename is not durable after a power loss. Pin it: a successful write
    # fsyncs the parent exactly once, AFTER os.replace; a write whose replace
    # raises does NOT fsync the parent (and still cleans its temp).
    import linktools.ai.storage.file.atomic as atomic_mod

    calls: "list[Path]" = []
    real = atomic_mod._fsync_directory

    def _record(directory: Path) -> None:
        calls.append(directory)
        real(directory)

    monkeypatch.setattr(atomic_mod, "_fsync_directory", _record)
    target = tmp_path / "out.json"
    atomic_write_bytes(target, b"payload")
    assert calls == [target.parent]

    # A write that fails at os.replace must not fsync the parent, and the temp
    # is cleaned (no leak).
    calls.clear()
    target2 = tmp_path / "fail.json"

    def _boom(src, dst):  # noqa: ARG001
        raise OSError("simulated replace failure")

    monkeypatch.setattr(atomic_mod.os, "replace", _boom)
    with pytest.raises(OSError):
        atomic_write_bytes(target2, b"x")
    assert calls == []
    leftover = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftover == []


def test_corrupt_session_record_raises_not_treated_as_missing(tmp_path) -> None:
    # A present-but-unparseable record.json is corruption, not a missing
    # session: get() raises SessionCorruptionError naming the path, and the
    # corrupt file is left in place for a repair tool.
    from linktools.ai.session.models import (
        SessionRecord,
        SessionStatus,
    )
    from datetime import datetime, timezone
    from linktools.ai.storage.file.session import FileSessionStore

    store = FileSessionStore(root=tmp_path)

    async def run() -> None:
        now = datetime.now(timezone.utc)
        await store.create(
            SessionRecord(
                id="s1",
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
        record_path = tmp_path / "s1" / "record.json"
        assert record_path.exists()
        # Corrupt the record in place.
        record_path.write_text("{ NOT VALID JSON")
        with pytest.raises(SessionCorruptionError):
            await store.get("s1")
        # The corrupt file is preserved (not deleted / masked).
        assert record_path.exists()

    asyncio.run(run())


def test_corrupt_session_message_raises(tmp_path) -> None:
    from linktools.ai.session.models import (
        MessageRole,
        NewSessionMessage,
        SessionRecord,
        SessionStatus,
    )
    from datetime import datetime, timezone
    from linktools.ai.storage.file.session import FileSessionStore

    store = FileSessionStore(root=tmp_path)

    async def run() -> None:
        now = datetime.now(timezone.utc)
        await store.create(
            SessionRecord(
                id="s1",
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
        await store.append_messages(
            "s1",
            (
                NewSessionMessage(
                    role=MessageRole.USER, content="hi", run_id=None, metadata={}
                ),
            ),
        )
        # Corrupt the one message file in place.
        msg = next((tmp_path / "s1" / "messages").glob("*.json"))
        msg.write_text("<<<garbage>>>")
        with pytest.raises(SessionCorruptionError):
            await store.list_messages("s1")

    asyncio.run(run())
