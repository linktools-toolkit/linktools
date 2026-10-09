#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation read reuse remains subordinate to each canonical storage read."""

import asyncio
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from linktools.ai.core import ImmutableJsonMapping, canonical_json_bytes
from linktools.ai.core import _json as json_module
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import ScoreAttemptView, ScoreBundle, TargetTrialRef
from linktools.ai.runtime.state._codec import encode_domain
from linktools.ai.runtime.state._evaluation_records import EvaluationLaunchIntent
from linktools.ai.task import TaskGraph, TaskGraphAdmission, TaskGraphRequest, TaskGraphSubmission, TaskNode, TaskRef

from .test_evaluation_persistence import NOW, PRINCIPAL, experiment, storage_for


async def _repository(path: Path, backend: str = "memory", record=None):
    state = storage_for(backend, path)
    await state.initialize(namespace="evaluation", tenant_id="tenant")
    repository = state.evaluation.records
    await repository.reserve_experiment(experiment() if record is None else record)
    return state, repository


def test_immutable_mapping_canonical_bytes_before_and_after_read(monkeypatch):
    data = {"unicode": "你好", "nested": [1, True, -0.0, {"a": None}]}
    expected = canonical_json_bytes(data)
    value = ImmutableJsonMapping(data)
    original = json_module.json.dumps
    calls = []

    def dumps(value, **kwargs):
        calls.append(value)
        return original(value, **kwargs)

    monkeypatch.setattr(json_module.json, "dumps", dumps)
    assert canonical_json_bytes(value) == expected
    assert calls == []
    detached = value["nested"]
    detached.append("caller mutation")
    assert canonical_json_bytes(value) == expected
    assert len(calls) == 1
    assert calls[0] is not detached
    assert canonical_json_bytes(ImmutableJsonMapping({"value": 1})) != canonical_json_bytes(
        ImmutableJsonMapping({"value": True})
    )


def test_immutable_mapping_canonical_fast_path_excludes_subclasses():
    class MappingSubclass(ImmutableJsonMapping):
        pass

    with pytest.raises(TypeError):
        canonical_json_bytes(MappingSubclass({"value": 1}))


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_evaluation_unchanged_fresh_storage_records_decode_once(tmp_path, monkeypatch, backend):
    state, repository = await _repository(tmp_path, backend)
    original_read, original_decode = repository._record, repository._decode
    reads, decodes = [], []

    async def read(key):
        reads.append(key)
        value = await original_read(key)
        return None if value is None else replace(value, data=dict(value.data))

    async def decode(stored, target):
        decodes.append(stored.storage_version)
        return await original_decode(stored, target)

    monkeypatch.setattr(repository, "_record", read)
    monkeypatch.setattr(repository, "_decode", decode)
    try:
        first = await repository.get("experiment", tenant_id="tenant")
        second = await repository.get("experiment", tenant_id="tenant")
        assert first is second
        assert len(reads) == 2
        assert len(decodes) == 1
        await repository.close_reservation_gate("experiment")
        decodes.clear()
        changed = await repository.get("experiment", tenant_id="tenant")
        assert changed.gate == "closed_cancel"
        assert changed.revision == first.revision + 1
        assert len(decodes) == 1
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_evaluation_read_reuse_keeps_custom_mapping_subclasses_readable(tmp_path, monkeypatch):
    class MappingSubclass(ImmutableJsonMapping):
        def __init__(self, data):
            super().__init__(data)
            self.current = data

        def __getitem__(self, key):
            return self.current[key]

        def __iter__(self):
            return iter(self.current)

        def __len__(self):
            return len(self.current)

    state, repository = await _repository(tmp_path)
    stored = await repository._record(repository._key("evaluation", "experiment"))
    data = MappingSubclass(dict(stored.data))
    stored = replace(stored, data=data)

    async def read(key):
        return stored

    monkeypatch.setattr(repository, "_record", read)
    try:
        before = await repository.get("experiment", tenant_id="tenant")
        data.current["value"]["payload"]["fields"]["gate"] = "closed_cancel"
        after = await repository.get("experiment", tenant_id="tenant")
        assert before.gate == "open" and after.gate == "closed_cancel"
        assert after is not before
        data.current["value"]["payload"]["fields"]["gate"] = object()
        with pytest.raises(AIError) as caught:
            await repository.get("experiment", tenant_id="tenant")
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert repository._last_evaluation_read is None
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ("payload", "bool", "corrupt", "missing", "error"))
async def test_evaluation_read_reuse_rejects_changed_or_failed_storage(tmp_path, monkeypatch, change):
    record = experiment()
    manifest = replace(record.manifest, policy=replace(record.manifest.policy, max_trials=1))
    state, repository = await _repository(tmp_path, record=replace(record, manifest=manifest, manifest_digest=manifest.digest))
    try:
        before = await repository.get("experiment", tenant_id="tenant")
        stored = repository._last_evaluation_read[0]
        data = dict(stored.data)
        fields = data["value"]["payload"]["fields"]
        if change == "payload":
            fields["gate"] = "closed_cancel"
        elif change == "bool":
            fields["manifest"]["fields"]["policy"]["fields"]["max_trials"] = True
        elif change == "corrupt":
            del fields["manifest"]
        failure = AIError(ErrorCode.STORAGE_UNAVAILABLE)

        async def read(key):
            if change == "error":
                raise failure
            if change == "missing":
                return None
            return replace(stored, data=data)

        monkeypatch.setattr(repository, "_record", read)
        if change == "payload":
            after = await repository.get("experiment", tenant_id="tenant")
            assert after.gate == "closed_cancel" and before.gate == "open"
            assert after.revision == before.revision
        elif change == "missing":
            assert await repository.get("experiment", tenant_id="tenant") is None
            assert repository._last_evaluation_read is None
        else:
            with pytest.raises(AIError) as caught:
                await repository.get("experiment", tenant_id="tenant")
            if change == "error":
                assert caught.value is failure
            else:
                assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            assert repository._last_evaluation_read is None
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", (
    ("scope_digest", b"s" * 32), ("parent_digest", b"p" * 32),
    ("state", "changed"), ("storage_version", 19),
    ("lease_owner", "other"), ("lease_fence", 7),
    ("lease_expires_at", NOW + timedelta(hours=1)),
    ("key_digest", b"k" * 32), ("kind", "other"), ("sort_key", "different"),
))
async def test_evaluation_read_reuse_matches_every_storage_header_field(tmp_path, monkeypatch, field, value):
    state, repository = await _repository(tmp_path)
    try:
        await repository.get("experiment", tenant_id="tenant")
        stored = repository._last_evaluation_read[0]
        changed = replace(stored, **{field: value})
        original_decode = repository._decode
        calls = []

        async def read(key):
            return changed

        async def decode(record, target):
            calls.append(record)
            return await original_decode(record, target)

        monkeypatch.setattr(repository, "_record", read)
        monkeypatch.setattr(repository, "_decode", decode)
        await repository.get("experiment", tenant_id="tenant")
        assert calls == [changed]
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_evaluation_read_reuse_has_one_entry_and_preserves_concurrent_results(tmp_path, monkeypatch):
    state, repository = await _repository(tmp_path)
    original = experiment()
    manifest = replace(original.manifest, experiment_id="other")
    await repository.reserve_experiment(replace(original, manifest=manifest,
        manifest_digest=manifest.digest, idempotency_key_digest="c" * 64))
    original_read, original_decode = repository._record, repository._decode
    entered, release = asyncio.Event(), asyncio.Event()
    paused = False
    calls = []

    async def read(key):
        nonlocal paused
        value = await original_read(key)
        if key == repository._key("evaluation", "experiment") and not paused:
            paused = True
            entered.set()
            await release.wait()
        return value

    async def decode(stored, target):
        calls.append(stored.key_digest)
        return await original_decode(stored, target)

    monkeypatch.setattr(repository, "_record", read)
    monkeypatch.setattr(repository, "_decode", decode)
    pending = asyncio.create_task(repository.get("experiment", tenant_id="tenant"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        other = await repository.get("other", tenant_id="tenant")
        release.set()
        first = await asyncio.wait_for(pending, timeout=5)
        assert first.experiment_id == "experiment" and other.experiment_id == "other"
        assert repository._last_evaluation_read[2] is first
        assert (await repository.get("other", tenant_id="tenant")).experiment_id == "other"
        assert len(calls) == 3
        await repository.close()
        assert repository._last_evaluation_read is None
        await repository.get("other", tenant_id="tenant")
        assert len(calls) == 4
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        await state.close()


@pytest.mark.asyncio
async def test_evaluation_read_reuse_clears_purged_content(tmp_path):
    state, repository = await _repository(tmp_path, record=replace(experiment(),
        gate="closed_cancel", content_expires_at=NOW + timedelta(hours=1),
        metadata_expires_at=NOW + timedelta(hours=2)))
    try:
        assert await repository.get("experiment", tenant_id="tenant") is not None
        await repository.purge("experiment", now=NOW + timedelta(hours=3))
        assert repository._last_evaluation_read is None
        with pytest.raises(AIError) as caught:
            await repository.get("experiment", tenant_id="tenant")
        assert caught.value.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE
        assert repository._last_evaluation_read is None
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_reused_evaluation_record_does_not_expose_mutable_nested_values(tmp_path):
    record = experiment()
    manifest = replace(record.manifest,
        policy=replace(record.manifest.policy, environment={"nested": [{"value": 1}]},
                       model_fixtures=({"nested": [{"value": 1}]},)),
        candidates=(replace(record.manifest.candidates[0], definition_contracts=({"nested": [{"value": 1}]},)),),
        scorers=(replace(record.manifest.scorers[0], config={"nested": [{"value": 1}]}),))
    graph = TaskGraph("trial", (TaskNode("target", task=TaskRef("target", 1), input={"nested": [{"value": 1}]}),))
    admission = TaskGraphAdmission.from_request(TaskGraphRequest(graph, PRINCIPAL, "trial"))
    launch = EvaluationLaunchIntent("target:trial", TargetTrialRef("experiment", "trial"), None,
        TaskGraphSubmission("evaluation", admission, graph), None)
    score = ScoreAttemptView("experiment", "score", launch.trial, "score", TaskRef("score", 1),
        "valid", ScoreBundle(dimensions={"quality": 1.0}, rationale="original"))
    state, repository = await _repository(tmp_path, record=replace(record, manifest=manifest,
        manifest_digest=manifest.digest, intents=(launch,), scores=(score,)))
    try:
        first = await repository.get("experiment", tenant_id="tenant")
        before = canonical_json_bytes(encode_domain(first))
        mappings = (first.manifest.policy.environment, first.manifest.policy.model_fixtures[0],
                    first.manifest.candidates[0].definition_contracts[0], first.manifest.scorers[0].config,
                    first.intents[0].submission.graph.nodes[0].input)
        for mapping in mappings:
            mapping["nested"][0]["value"] = "changed"
        with pytest.raises(TypeError):
            first.scores[0].score.dimensions["quality"] = 2
        with pytest.raises(ValueError):
            first.scores[0].score.rationale = "changed"
        assert canonical_json_bytes(encode_domain(first)) == before
        assert await repository.get("experiment", tenant_id="tenant") is first
    finally:
        await state.close()
