#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from linktools.ai.artifact import ANONYMOUS_PROVENANCE
"""DirectEvalExecutor tests: reads the case input artifact, runs the target via
Runtime, and wraps the result (or a target error) in an EvalExecution."""

import asyncio
from types import SimpleNamespace

from linktools.ai.artifact import ArtifactStore
from linktools.ai.evaluation.executors import DirectEvalExecutor
from linktools.ai.evaluation.models import EvalCase, EvalTarget
from linktools.ai.storage.filesystem.artifact import (
    FilesystemArtifactBlobStore,
    FilesystemArtifactRecordStore,
)


def _artifacts(tmp_path) -> ArtifactStore:
    return ArtifactStore(
        FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs"),
        FilesystemArtifactRecordStore(records_root=tmp_path / "records"),
    )


class _Resolver:
    async def resolve(self, target):
        return f"spec:{target.id}"


class _FakeRuntime:
    def __init__(self):
        self.calls = []

    async def run(self, spec, prompt, *, run_id=None, user_id=None, tenant_id=None, **kw):
        self.calls.append({"spec": spec, "prompt": prompt, "run_id": run_id})
        return SimpleNamespace(
            output=f"did:{prompt}",
            token_usage={"total_tokens": 100, "total_cost": 0.5},
        )


def test_direct_executor_reads_input_and_runs(tmp_path) -> None:
    artifacts = _artifacts(tmp_path)
    rt = _FakeRuntime()
    executor = DirectEvalExecutor(rt, _Resolver(), artifacts, tenant_id="t1")

    async def run():
        record = await artifacts.put(content=b"hello", media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,)
        case = EvalCase(id="c1", input_artifact_id=record.ref.id)
        execution = await executor.execute(
            EvalTarget(kind="agent", id="a1"), case
        )
        assert execution.error is None
        assert execution.case_id == "c1"
        assert execution.output == "did:hello"
        assert rt.calls[0]["prompt"] == "hello"
        assert rt.calls[0]["spec"] == "spec:a1"
        assert execution.run_id is not None
        # Output is sealed to a content-addressed artifact + usage captured.
        assert execution.output_artifact_id is not None
        assert execution.model_usage["total_tokens"] == 100

    asyncio.run(run())


def test_direct_executor_captures_target_error(tmp_path) -> None:
    artifacts = _artifacts(tmp_path)

    class _BoomRuntime:
        async def run(self, spec, prompt, **kw):
            raise RuntimeError("target failed")

    executor = DirectEvalExecutor(
        _BoomRuntime(), _Resolver(), artifacts, tenant_id="t1"
    )

    async def run():
        record = await artifacts.put(content=b"x", media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,)
        case = EvalCase(id="c1", input_artifact_id=record.ref.id)
        execution = await executor.execute(EvalTarget(kind="agent", id="a1"), case)
        # The target raised; the execution records the error rather than crashing.
        assert execution.error == "RuntimeError"
        assert execution.output is None

    asyncio.run(run())


def test_direct_executor_captures_full_runsnapshot(tmp_path) -> None:
    """When Run/Definition/Event stores are wired, the executor seals the run
    record, run definition, and event stream into artifacts and returns a full
    RunSnapshot + its artifact id on the EvalExecution."""
    from linktools.ai.evaluation.snapshot import RunSnapshot
    from linktools.ai.events.store import EventPage

    artifacts = _artifacts(tmp_path)

    class _FakeRunStore:
        def __init__(self):
            self.runs: "dict" = {}

        async def get(self, run_id):
            return self.runs.get(run_id)

    class _FakeRunDefStore:
        def __init__(self):
            self.defs: "dict" = {}

        async def get(self, run_id):
            return self.defs.get(run_id)

    class _FakeEventStore:
        def __init__(self):
            self.by_stream: "dict" = {}

        async def list(self, stream_id, *, after_sequence=0, limit=100):
            return EventPage(
                items=tuple(self.by_stream.get(stream_id, [])), cursor=None
            )

    run_store = _FakeRunStore()
    run_def_store = _FakeRunDefStore()
    event_store = _FakeEventStore()

    class _SnapshotRuntime:
        async def run(self, spec, prompt, *, run_id=None, user_id=None, tenant_id=None, **kw):
            # Populate the stores the way the real Runtime does.
            run_store.runs[run_id] = SimpleNamespace(
                id=run_id, status="succeeded", input=SimpleNamespace(prompt=prompt)
            )
            run_def_store.defs[run_id] = SimpleNamespace(
                run_id=run_id, runnable_id="a1", spec_fingerprint="fp"
            )
            event_store.by_stream.setdefault(run_id, []).append(
                SimpleNamespace(event_id="e1", stream_id=run_id, sequence=1)
            )
            return SimpleNamespace(
                output=f"did:{prompt}", token_usage={"total_tokens": 50}
            )

    executor = DirectEvalExecutor(
        _SnapshotRuntime(),
        _Resolver(),
        artifacts,
        tenant_id="t1",
        run_store=run_store,
        run_definition_store=run_def_store,
        event_store=event_store,
    )

    async def run():
        record = await artifacts.put(content=b"hello", media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,)
        case = EvalCase(id="c1", input_artifact_id=record.ref.id)
        execution = await executor.execute(
            EvalTarget(kind="agent", id="a1"), case
        )
        assert execution.error is None
        assert execution.snapshot_artifact_id is not None
        assert isinstance(execution.snapshot, RunSnapshot)
        snap = execution.snapshot
        assert snap.run_record_artifact_id is not None
        assert snap.run_definition_artifact_id is not None
        assert len(snap.event_artifact_ids) == 1
        assert snap.input_artifact_id == record.ref.id
        assert snap.output_artifact_id is not None
        assert snap.model_usage["total_tokens"] == 50

    asyncio.run(run())


def test_direct_executor_normalizes_total_tokens_from_runtime_usage(tmp_path) -> None:
    """The real Runtime reports input_tokens + output_tokens (not total_tokens);
    the executor derives total_tokens so the eval avg-tokens metric populates."""
    artifacts = _artifacts(tmp_path)

    class _RuntimeUsage:
        async def run(self, spec, prompt, *, run_id=None, user_id=None, tenant_id=None, **kw):
            return SimpleNamespace(
                output="done",
                token_usage={"input_tokens": 30, "output_tokens": 20},
            )

    executor = DirectEvalExecutor(_RuntimeUsage(), _Resolver(), artifacts, tenant_id="t1")

    async def run():
        record = await artifacts.put(content=b"in", media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,)
        case = EvalCase(id="c1", input_artifact_id=record.ref.id)
        execution = await executor.execute(
            EvalTarget(kind="agent", id="a1"), case
        )
        # Derived from input + output; avg_tokens now has real data.
        assert execution.model_usage["total_tokens"] == 50
        assert execution.model_usage["input_tokens"] == 30

    asyncio.run(run())


def test_direct_executor_captures_safety_refusal_on_policy_error(tmp_path) -> None:
    """When the Runtime's security pipeline refuses (raises a PolicyError), the
    direct executor records a safety refusal so the rate is populated in direct
    mode too (not only task mode)."""
    artifacts = _artifacts(tmp_path)

    class PolicyError(Exception):
        pass

    class _DenyingRuntime:
        async def run(self, spec, prompt, *, run_id=None, user_id=None, tenant_id=None, **kw):
            raise PolicyError("blocked by security pipeline")

    executor = DirectEvalExecutor(
        _DenyingRuntime(), _Resolver(), artifacts, tenant_id="t1"
    )

    async def run():
        record = await artifacts.put(content=b"in", media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,)
        case = EvalCase(id="c1", input_artifact_id=record.ref.id)
        execution = await executor.execute(
            EvalTarget(kind="agent", id="a1"), case
        )
        assert execution.error == "PolicyError"
        assert execution.model_usage.get("safety_refusal") == 1.0

    asyncio.run(run())

