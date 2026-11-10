#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable snapshots preserve the submission owner's identity projection."""

import hashlib
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.codec import _encode_persisted_domain, decode_record, encode_envelope, encode_record
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.storage import InMemoryObjectStore, ObjectRef, read_object
from linktools.ai.task import Task, TaskGraphSubmission
from linktools.ai.inputs import TaskGraphBindingCaptureStore
from linktools.ai.runtime import Runtime, RuntimeFeatures
from ._runtime_test_helpers import model_features
from .test_evaluation_consumers import FixtureModels
from .test_task_structured_identity import _request, _seed_legacy, _service


_LIMITS = SnapshotLimits(max_entries=1000, max_bytes=1024 * 1024)


async def _snapshot(
    root: Path, objects: InMemoryObjectStore, *, version: int, phase: str,
) -> tuple[ObjectRef, TaskGraphSubmission]:
    async with Persistence.open("structured-identity", storage=PersistenceStorage.from_root(root), tenant_id="tenant") as state:
        submission = await _seed_legacy(state) if version == 1 else await _service(state).prepare_submission(_request())
        features = RuntimeFeatures("snapshot-agent")
        features.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
        async with Runtime.attach(persistence=state, features=model_features(FixtureModels(), features)) as runtime:
            task = Task.from_agent("identity.agent", runtime.agents.get())
            captures = TaskGraphBindingCaptureStore(state.namespace, state.task.tasks.state_store)
            await captures.capture(submission.admission, submission.graph, tasks=(task,))
        if phase in {"admitted", "cancelled_admitted"}:
            await state.task.admissions.admit_prepared(submission)
        if phase in {"cancelled", "cancelled_admitted"}:
            await _service(state).cancel_submission(submission.ref, principal=submission.ref.principal, idempotency_key="cancel")
    async with Persistence.open("structured-identity", storage=PersistenceStorage.from_root(root), tenant_id="tenant", read_only=True) as state:
        return await state.export_snapshot(object_store=objects, limits=_LIMITS), submission


@pytest.mark.asyncio
@pytest.mark.parametrize("version", (1, 2))
@pytest.mark.parametrize("phase", ("prepared", "admitted", "cancelled", "cancelled_admitted"))
async def test_task_identity_snapshot_roundtrip_preserves_owner(
    tmp_path: Path, version: int, phase: str,
) -> None:
    objects = InMemoryObjectStore("snapshots")
    reference, submission = await _snapshot(tmp_path / "source", objects, version=version, phase=phase)
    restored = tmp_path / "restored"
    await Persistence.restore_snapshot(reference, object_store=objects, root=restored, limits=_LIMITS)
    async with Persistence.open("structured-identity", storage=PersistenceStorage.from_root(restored), tenant_id="tenant") as state:
        assert await state.task.admissions.get_submission_ref("structured", tenant_id="tenant") == submission.ref
        replay = await _service(state).prepare_submission(_request())
        assert replay.ref == submission.ref


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("prepared", "admitted", "cancelled_admitted"))
async def test_snapshot_restore_rejects_disagreeing_submission_version_before_writing(
    tmp_path: Path, phase: str,
) -> None:
    objects = InMemoryObjectStore("snapshots")
    reference, submission = await _snapshot(tmp_path / "source", objects, version=2, phase=phase)
    manifest = json.loads(await read_object(objects, reference.key, expected_digest=reference.digest,
                                          expected_size=reference.size))
    records = manifest["domains"]["task"]["records"]
    for index, item in enumerate(records):
        if item["kind"] == "task_submission":
            record = decode_record(item)
            mismatched = replace(submission.ref, request_identity_version=1)
            data = encode_envelope({"type": "task_submission_ref", "payload": _encode_persisted_domain(mismatched)})
            records[index] = encode_record(replace(record, data=data))
            break
    else:
        pytest.fail("snapshot has no submission owner")
    payload = canonical_json_bytes(manifest)
    digest = hashlib.sha256(payload).hexdigest()
    key = "tampered/" + digest

    async def chunks() -> AsyncIterator[bytes]:
        yield payload

    await objects.put(key, chunks(), expected_size=len(payload), expected_digest=digest)
    restored = tmp_path / "restored"
    with pytest.raises(AIError) as caught:
        await Persistence.restore_snapshot(ObjectRef(objects.store_id, key, digest, len(payload)),
                                           object_store=objects, root=restored, limits=_LIMITS)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert not restored.exists()
