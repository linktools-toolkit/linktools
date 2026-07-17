#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TaskEvalExecutor tests: routes a case through TaskRuntime and captures
retry_count + the sealed output (section 23.4 task mode)."""

import asyncio

from linktools.ai.artifact import ArtifactStore
from linktools.ai.evaluation.executors import TaskEvalExecutor
from linktools.ai.evaluation.models import EvalCase, EvalTarget
from linktools.ai.storage.facade import FileStorage
from linktools.ai.task.handlers import CallableTaskHandler
from linktools.ai.task.runtime import TaskRuntime, TaskRuntimeOptions

FAST = TaskRuntimeOptions(
    poll_interval_seconds=0.01, lease_seconds=2.0, heartbeat_seconds=0.1
)


def _envelope_bytes(output: str, tokens: int = 5) -> bytes:
    import json

    return json.dumps(
        {"output": output, "token_usage": {"total_tokens": tokens}}
    ).encode("utf-8")


def test_task_eval_executor_captures_output_and_retry_count_zero(tmp_path) -> None:
    artifacts: "object | None" = None

    async def handler(request, context):
        rec = await artifacts.put(  # type: ignore[union-attr]
            _envelope_bytes("done"), media_type="application/json", tenant_id="t1"
        )
        from linktools.ai.task.protocols import TaskSuccess

        return TaskSuccess(output_artifact=rec.ref)

    async def run():
        nonlocal artifacts
        storage = FileStorage(root=tmp_path)
        artifacts = ArtifactStore(storage.resources)
        runtime = TaskRuntime(
            storage=storage, handlers={"eval": CallableTaskHandler(handler)}, options=FAST
        )
        executor = TaskEvalExecutor(
            runtime, artifacts, tenant_id="t1", handler_name="eval", wait_timeout=10
        )
        inp = await artifacts.put(b"case-in", media_type="text/plain", tenant_id="t1")
        case = EvalCase(id="c1", input_artifact_id=inp.ref.id)
        execution = await executor.execute(EvalTarget(kind="agent", id="a1"), case)
        assert execution.error is None
        assert execution.model_usage["retry_count"] == 0
        assert execution.output == "done"
        assert execution.model_usage["total_tokens"] == 5

    asyncio.run(run())


def test_task_eval_executor_captures_retry_after_transient(tmp_path) -> None:
    """A handler that fails once (transient) then succeeds yields retry_count=1,
    when the one-shot task is given a retry policy that allows a second attempt."""
    from linktools.ai.task.models import RetryPolicy

    artifacts: "object | None" = None
    calls = {"n": 0}

    async def flaky(request, context):
        calls["n"] += 1
        if calls["n"] == 1:
            from linktools.ai.task.models import TaskFailureKind
            from linktools.ai.task.protocols import TaskFailure

            return TaskFailure(
                kind=TaskFailureKind.TRANSIENT, error_type="Flaky", message="retry"
            )
        rec = await artifacts.put(  # type: ignore[union-attr]
            _envelope_bytes("ok"), media_type="application/json", tenant_id="t1"
        )
        from linktools.ai.task.protocols import TaskSuccess

        return TaskSuccess(output_artifact=rec.ref)

    async def run():
        nonlocal artifacts
        storage = FileStorage(root=tmp_path)
        artifacts = ArtifactStore(storage.resources)
        runtime = TaskRuntime(
            storage=storage, handlers={"eval": CallableTaskHandler(flaky)}, options=FAST
        )
        executor = TaskEvalExecutor(
            runtime,
            artifacts,
            tenant_id="t1",
            handler_name="eval",
            retry_policy=RetryPolicy(max_attempts=2, initial_delay_seconds=0.01),
            wait_timeout=10,
        )
        inp = await artifacts.put(b"case-in", media_type="text/plain", tenant_id="t1")
        case = EvalCase(id="c1", input_artifact_id=inp.ref.id)
        execution = await executor.execute(EvalTarget(kind="agent", id="a1"), case)
        # First attempt failed transiently, second succeeded -> 1 retry.
        assert execution.error is None
        assert execution.model_usage["retry_count"] == 1
        assert calls["n"] == 2

    asyncio.run(run())


def test_task_eval_executor_captures_safety_refusal_on_policy_denial(tmp_path) -> None:
    """A handler that the security pipeline denies (POLICY_DENIED) surfaces as a
    safety refusal so the safety_refusal_rate metric is populated."""
    from linktools.ai.task.models import TaskFailureKind
    from linktools.ai.task.protocols import TaskFailure

    async def denied(request, context):
        return TaskFailure(
            kind=TaskFailureKind.POLICY_DENIED,
            error_type="Policy",
            message="blocked by security pipeline",
        )

    async def run():
        storage = FileStorage(root=tmp_path)
        artifacts = ArtifactStore(storage.resources)
        runtime = TaskRuntime(
            storage=storage,
            handlers={"eval": CallableTaskHandler(denied)},
            options=FAST,
        )
        executor = TaskEvalExecutor(
            runtime, artifacts, tenant_id="t1", handler_name="eval", wait_timeout=10
        )
        inp = await artifacts.put(b"in", media_type="text/plain", tenant_id="t1")
        case = EvalCase(id="c1", input_artifact_id=inp.ref.id)
        execution = await executor.execute(EvalTarget(kind="agent", id="a1"), case)
        assert execution.error == "TaskFailed"
        assert execution.model_usage["safety_refusal"] == 1.0

    asyncio.run(run())
