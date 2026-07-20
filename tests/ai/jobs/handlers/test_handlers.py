#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CallableTaskHandler + RuntimeTaskHandler tests."""

import asyncio

from linktools.ai.jobs.handlers import (
    CallableTaskHandler,
    MappingRunnableResolver,
    RuntimeTaskHandler,
)
from linktools.ai.jobs.models import (
    ActorChain,
    ActorRef,
    TaskBudget,
    TaskFailureKind,
    TaskPrincipal,
)
from linktools.ai.identity.principal import ScopeSet
from linktools.ai.jobs.protocols import (
    CancellationToken,
    TaskContext,
    TaskRequest,
    TaskSuccess,
)


def _ctx() -> TaskContext:
    return TaskContext(
        job_id="j1",
        task_id="t1",
        attempt_id="a1",
        fencing_token=1,
        worker_id="w1",
        principal=TaskPrincipal(tenant_id="t1", user_id="alice"),
        actor_chain=ActorChain(actors=(ActorRef("user", "alice"),)),
        delegated_scopes=ScopeSet.allow_all(),
        budget=TaskBudget(),
        resource_snapshots=(),
        cancellation=CancellationToken(),
    )


# ---- CallableTaskHandler ----


def test_callable_returns_success() -> None:
    async def echo(request, context):
        return "hello"

    handler = CallableTaskHandler(echo)

    async def run():
        outcome = await handler.execute(
            TaskRequest(input_artifact=None, metadata={}), _ctx()
        )
        assert isinstance(outcome, TaskSuccess)
        assert outcome.metadata == {"result": "hello"}

    asyncio.run(run())


def test_callable_permanent_exception() -> None:
    async def bad(request, context):
        raise ValueError("bad input")

    handler = CallableTaskHandler(bad)

    async def run():
        outcome = await handler.execute(
            TaskRequest(input_artifact=None, metadata={}), _ctx()
        )
        assert outcome.kind == TaskFailureKind.PERMANENT
        assert outcome.error_type == "ValueError"

    asyncio.run(run())


def test_callable_transient_exception() -> None:
    async def flaky(request, context):
        raise ConnectionError("network")

    handler = CallableTaskHandler(flaky)

    async def run():
        outcome = await handler.execute(
            TaskRequest(input_artifact=None, metadata={}), _ctx()
        )
        assert outcome.kind == TaskFailureKind.TRANSIENT

    asyncio.run(run())


# ---- RuntimeTaskHandler ----


class _FakeRuntime:
    def __init__(self):
        self.calls = []

    async def run(
        self,
        spec,
        prompt,
        *,
        session_id=None,
        run_id=None,
        user_id=None,
        tenant_id=None,
        agents=None,
        context_metadata=None,
    ):
        self.calls.append(
            {
                "spec": spec,
                "prompt": prompt,
                "run_id": run_id,
                "user_id": user_id,
                "tenant_id": tenant_id,
                "context_metadata": context_metadata,
            }
        )
        return {"output": "done"}


class _FakeTaskStore:
    async def bind_runnable(self, **kwargs):
        return None

    async def bind_run(self, **kwargs):
        return None


def _stores():
    from linktools.ai.storage.artifact_backends import build_artifact_store_from_assets
    from linktools.ai.asset.memory import MemoryAssetBackend
    from linktools.ai.asset.store import AssetStore
    return _FakeTaskStore(), build_artifact_store_from_assets(AssetStore(primary=MemoryAssetBackend()))


def _runtime_handler(runtime, resolver, **kwargs):
    task_store, artifacts = _stores()
    return RuntimeTaskHandler(runtime, resolver,
        task_store=kwargs.pop("task_store", task_store),
        artifact_store=kwargs.pop("artifact_store", artifacts), **kwargs)


def test_runtime_handler_calls_runtime_with_principal() -> None:
    rt = _FakeRuntime()
    resolver = MappingRunnableResolver({"agent-1": "fake-spec"})
    handler = _runtime_handler(rt, resolver)

    async def run():
        outcome = await handler.execute(
            TaskRequest(
                input_artifact=None,
                metadata={"runnable_id": "agent-1", "prompt": "do X"},
            ),
            _ctx(),
        )
        assert isinstance(outcome, TaskSuccess)
        assert "run_id" in outcome.metadata
        assert rt.calls[0]["spec"] == "fake-spec"
        assert rt.calls[0]["prompt"] == "do X"
        assert rt.calls[0]["user_id"] == "alice"
        assert rt.calls[0]["tenant_id"] == "t1"
        # Task lineage is threaded into Runtime.run as context_metadata so the
        # RunRecord carries its job/task/attempt correlation.
        meta = rt.calls[0]["context_metadata"]
        assert meta["job_id"] == "j1"
        assert meta["task_id"] == "t1"
        assert meta["attempt_id"] == "a1"
        assert meta["fencing_token"] == 1

    asyncio.run(run())


def test_runtime_handler_mid_run_exception_is_side_effect_unknown() -> None:
    # An agent run that raises after starting may have issued side effects, so
    # the failure must be SIDE_EFFECT_UNKNOWN (non-retryable), not TRANSIENT.
    class _BoomRuntime:
        async def run(self, spec, prompt, **kw):
            raise RuntimeError("model 500 mid-tool-call")

    resolver = MappingRunnableResolver({"agent-1": "fake-spec"})
    handler = _runtime_handler(_BoomRuntime(), resolver)

    async def run():
        outcome = await handler.execute(
            TaskRequest(
                input_artifact=None,
                metadata={"runnable_id": "agent-1", "prompt": "do X"},
            ),
            _ctx(),
        )
        assert outcome.kind == TaskFailureKind.SIDE_EFFECT_UNKNOWN

    asyncio.run(run())


def test_runtime_handler_missing_input() -> None:
    handler = _runtime_handler(_FakeRuntime(), MappingRunnableResolver({}))

    async def run():
        outcome = await handler.execute(
            TaskRequest(input_artifact=None, metadata={}),
            _ctx(),
        )
        assert outcome.kind == TaskFailureKind.INVALID_INPUT

    asyncio.run(run())


def test_runtime_handler_resolver_miss() -> None:
    handler = _runtime_handler(
        _FakeRuntime(), MappingRunnableResolver({"only": "spec"})
    )

    async def run():
        outcome = await handler.execute(
            TaskRequest(
                input_artifact=None, metadata={"runnable_id": "missing", "prompt": "x"}
            ),
            _ctx(),
        )
        assert outcome.kind == TaskFailureKind.PERMANENT

    asyncio.run(run())


def test_runtime_handler_seals_run_result_to_artifact() -> None:
    """When an ArtifactStore is wired, the handler seals the RunResult into a
    content-addressed artifact and returns its ref as the task output."""
    from linktools.ai.storage.artifact_backends import build_artifact_store_from_assets
    from linktools.ai.asset.memory import MemoryAssetBackend
    from linktools.ai.asset.store import AssetStore

    rt = _FakeRuntime()
    artifacts = build_artifact_store_from_assets(AssetStore(primary=MemoryAssetBackend()))
    handler = _runtime_handler(
        rt,
        MappingRunnableResolver({"agent-1": "fake-spec"}),
        artifact_store=artifacts,
    )

    async def run():
        outcome = await handler.execute(
            TaskRequest(
                input_artifact=None,
                metadata={"runnable_id": "agent-1", "prompt": "do X"},
            ),
            _ctx(),
        )
        assert isinstance(outcome, TaskSuccess)
        assert outcome.output_artifact is not None
        content = await artifacts.get(outcome.output_artifact.id, tenant_id="t1")
        assert content is not None
        assert b"done" in content

    asyncio.run(run())


def test_runtime_handler_rejects_oversized_output_before_writing() -> None:
    """An output exceeding the per-task payload cap is rejected BEFORE the
    oversized blob is written to the content-addressed store (fail-closed, not
    post-hoc)."""
    from linktools.ai.storage.artifact_backends import build_artifact_store_from_assets
    from linktools.ai.asset.memory import MemoryAssetBackend
    from linktools.ai.asset.store import AssetStore

    class _HugeRuntime:
        async def run(self, spec, prompt, **kw):
            return {"output": "x" * (2 * 1024 * 1024)}  # 2 MiB > 1 MiB cap

    artifacts = build_artifact_store_from_assets(AssetStore(primary=MemoryAssetBackend()))
    handler = _runtime_handler(
        _HugeRuntime(),
        MappingRunnableResolver({"agent-1": "fake-spec"}),
        artifact_store=artifacts,
    )

    async def run():
        outcome = await handler.execute(
            TaskRequest(
                input_artifact=None,
                metadata={"runnable_id": "agent-1", "prompt": "do X"},
            ),
            _ctx(),
        )
        assert outcome.kind == TaskFailureKind.PERMANENT
        assert outcome.error_type == "OutputTooLarge"

    asyncio.run(run())


def test_runtime_handler_rejects_stale_resource_snapshot() -> None:
    """A pinned resource snapshot whose artifact is missing fails fast with
    INVALID_INPUT -- the runtime is never invoked against possibly-changed
    resources."""
    from linktools.ai.storage.artifact_backends import build_artifact_store_from_assets
    from linktools.ai.asset.memory import MemoryAssetBackend
    from linktools.ai.asset.store import AssetStore
    from linktools.ai.jobs.models import ResourceSnapshotRef

    rt = _FakeRuntime()
    artifacts = build_artifact_store_from_assets(AssetStore(primary=MemoryAssetBackend()))
    handler = _runtime_handler(
        rt,
        MappingRunnableResolver({"agent-1": "fake-spec"}),
        artifact_store=artifacts,
    )
    ctx = TaskContext(
        job_id="j1",
        task_id="t1",
        attempt_id="a1",
        fencing_token=1,
        worker_id="w1",
        principal=TaskPrincipal(tenant_id="t1", user_id="alice"),
        actor_chain=ActorChain(actors=(ActorRef("user", "alice"),)),
        delegated_scopes=ScopeSet.allow_all(),
        budget=TaskBudget(),
        resource_snapshots=(
            ResourceSnapshotRef(
                path="/data/x",
                version=1,
                etag="e",
                artifact_id="nonexistent-sha",
                sha256="nonexistent-sha",
            ),
        ),
        cancellation=CancellationToken(),
    )

    async def run():
        outcome = await handler.execute(
            TaskRequest(
                input_artifact=None,
                metadata={"runnable_id": "agent-1", "prompt": "do X"},
            ),
            ctx,
        )
        assert outcome.kind == TaskFailureKind.INVALID_INPUT
        assert rt.calls == [], "runtime must not be invoked on a stale snapshot"

    asyncio.run(run())
