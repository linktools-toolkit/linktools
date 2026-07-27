#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Soak tests too expensive for the default ``tests/ai`` loop.

This module holds ONLY tests whose runtime is disproportionate to what they
verify (seconds-to-minutes per test, vs. the sub-second norm elsewhere) --
they are split out of their normal ``tests/ai`` home so a routine run of
``tests/ai`` stays fast; run this directory explicitly when the contract it
covers actually changed."""

from __future__ import annotations

import asyncio

from linktools.ai.storage.backends.filesystem.object import FilesystemObjectBackend
from linktools.ai.storage.backends.filesystem.secure_directory import (
    FilesystemSecurityMode,
)
from linktools.ai.storage.object.models import StorageKey, WriteOptions


def _key(v: str) -> StorageKey:
    return StorageKey(v)


def _active(backend) -> "tuple[str, ...]":
    return backend._sd.list_names(".storage", "operations")


def test_100k_completed_mutations_leave_no_active_journals(tmp_path):
    """100,000 completed mutations must leave zero active operation
    directories. Uses TRUSTED_LOCAL mode (no per-op fsync) so the volume is
    exercisable in-test; the lifecycle contract (dir removed on commit) is
    identical across modes -- see
    tests/ai/storage/filesystem/test_object_journal_lifecycle.py for the
    fast, small-N version of this same contract."""
    backend = FilesystemObjectBackend(
        root=tmp_path / "root", mode=FilesystemSecurityMode.TRUSTED_LOCAL
    )
    total = 100_000

    async def _run():
        for i in range(total):
            await backend.raw_put_checked(
                _key(f"/k/{i}"),
                b"x",
                options=WriteOptions(),
                request_hash=f"h{i}",
            )

    asyncio.run(_run())
    # The assertion the spec demands: zero active journals regardless of how
    # many mutations ran.
    assert _active(backend) == ()
