#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Owner scans preserve transaction overlays without routing known keys again."""

from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.persistence import _memory
from linktools.ai.persistence.state import RecordQuery, record_key
from .test_state_batch_delete_backends import _open_store, _record


@pytest.mark.asyncio
@pytest.mark.parametrize("rollback", (False, True))
async def test_memory_owner_iteration_preserves_overlay_and_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rollback: bool,
) -> None:
    kinds = ("probe", "tool_operation", "agent_run")
    originals = tuple(replace(_record(index), kind=kind,
        key=record_key("iteration", "tenant", "probe", kind, (str(index),)), subject_id=None if kind == "agent_run" else "subject")
        for index, kind in enumerate(kinds))
    changed = replace(originals[0], storage_revision=1, data={"changed": True})
    added = replace(_record(4), subject_id="subject")
    expected = {record.key: record for record in (changed, originals[2], added)}
    async with _open_store("memory", tmp_path) as store:
        await store.mutate(lambda tx: tx.insert_records(originals))

        async def mutate(tx):
            assert await tx.replace_record(changed, expected_storage_revision=0)
            assert await tx.delete_record(originals[1].key)
            await tx.insert_record(added)
            def reroute(*args, **kwargs):
                raise AssertionError("a physical-owner scan must not reroute every known key")
            with monkeypatch.context() as patch:
                patch.setattr(_memory, "record_owner_table", reroute)
                assert {r.key: r for r in await tx.list_records(RecordQuery(scope_key=changed.scope_key))} == expected
                assert {r.key: r for r in await tx.scan_records()} == expected
                assert {r.key: r for r in await tx.scan_records_page(after=None, limit=10)} == expected
                assert {r.key for r in await tx.read_record_subjects(
                    scope_key=changed.scope_key, kind="probe", subject_ids=("subject",))} == {changed.key, added.key}
            if rollback:
                raise RuntimeError("abort staged scan changes")

        if rollback:
            with pytest.raises(RuntimeError, match="abort staged"):
                await store.mutate(mutate)
        else:
            await store.mutate(mutate)
        assert {r.key: r for r in await store.read(lambda tx: tx.scan_records())} == (
            {r.key: r for r in originals} if rollback else expected)
