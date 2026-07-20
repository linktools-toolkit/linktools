#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TaskEvalExecutor: run an eval case as a reliable task (task mode).

The counterpart to :class:`DirectEvalExecutor`: instead of running the target
inline, it submits the case as a one-shot task through JobRuntime (gaining
retries / lease / recovery) and reads back the sealed output artifact plus the
task's attempt history. ``retry_count`` (attempts beyond the first) is captured
into ``EvalExecution.model_usage`` so the runner can thread it into the result
metrics (retry rate).

This executor is the bridge between the evaluation and task planes. It depends
on JobRuntime only through the duck-typed ``run_one_task`` surface and reads
the result via attributes + the ArtifactStore, so it imports NO task-domain
module -- the evaluation package itself never reaches into task."""

import json

from ..models import EvalCase, EvalExecution, EvalTarget, normalize_usage


class TaskEvalExecutor:
    def __init__(
        self,
        task_runtime,
        artifact_store,
        *,
        tenant_id: str,
        handler_name: str,
        user_id: "str | None" = None,
        retry_policy: "object | None" = None,
        wait_timeout: float = 60.0,
    ) -> None:
        self._runtime = task_runtime
        self._artifact_store = artifact_store
        self._tenant_id = tenant_id
        self._handler_name = handler_name
        self._user_id = user_id
        # Opaque RetryPolicy forwarded to run_one_task; the caller (which owns
        # the task domain) constructs it, so this module imports no task type.
        self._retry_policy = retry_policy
        self._wait_timeout = wait_timeout

    async def execute(self, target: EvalTarget, case: EvalCase) -> EvalExecution:
        try:
            task = await self._runtime.run_one_task(
                self._handler_name,
                tenant_id=self._tenant_id,
                user_id=self._user_id,
                input_artifact_id=case.input_artifact_id,
                metadata={"target_kind": target.kind, "target_id": target.id},
                retry_policy=self._retry_policy,
                wait_timeout=self._wait_timeout,
            )
        except Exception as exc:  # noqa: BLE001 - submission / drive failure
            return EvalExecution(
                case_id=case.id, run_id=None, output=None, error=type(exc).__name__
            )

        if task is None:
            return EvalExecution(
                case_id=case.id, run_id=None, output=None, error="TaskMissing"
            )

        # Duck-typed reads (no task-domain import): attempts beyond the first
        # are retries.
        attempt_count = getattr(task, "attempt_count", 1) or 1
        retry_count = max(0, attempt_count - 1)
        status_value = getattr(getattr(task, "status", None), "value", "")
        task_id = getattr(task, "id", None)
        output_artifact_id = getattr(task, "output_artifact_id", None)
        output, captured_usage = await self._read_output(output_artifact_id)

        # A policy-denied failure (the security pipeline blocked the run) is the
        # available safety-refusal signal; surface it so aggregate() can compute
        # the safety refusal rate.
        safety_refusal = 0.0
        if status_value == "failed" and task_id is not None:
            safety_refusal = await self._was_policy_denied(task_id)

        error = "TaskFailed" if status_value == "failed" else None
        model_usage = normalize_usage(captured_usage)
        model_usage["retry_count"] = retry_count
        if safety_refusal:
            model_usage["safety_refusal"] = 1.0
        return EvalExecution(
            case_id=case.id,
            run_id=None,
            output=output,
            output_artifact_id=output_artifact_id,
            model_usage=model_usage,
            error=error,
        )

    async def _read_output(self, output_artifact_id):
        """Read the sealed RunResult artifact ({"output", "token_usage", ...})
        the handler produced. Returns (output, token_usage)."""
        if not output_artifact_id:
            return None, {}
        try:
            content = await self._artifact_store.get(
                output_artifact_id, tenant_id=self._tenant_id
            )
        except Exception:  # noqa: BLE001 - read is best-effort
            return None, {}
        if not content:
            return None, {}
        try:
            envelope = json.loads(content)
        except (TypeError, ValueError):
            return None, {}
        return envelope.get("output"), dict(envelope.get("token_usage") or {})

    async def _was_policy_denied(self, task_id) -> float:
        """True (1.0) if ANY of the task's attempts was policy-denied -- the
        available safety-refusal signal. Duck-typed over runtime.list_attempts;
        matches the failure_kind by string so no task-domain import is needed."""
        try:
            attempts = await self._runtime.list_attempts(task_id)
        except Exception:  # noqa: BLE001 - non-fatal: no signal if unreadable
            return 0.0
        for attempt in attempts:
            kind = getattr(getattr(attempt, "failure_kind", None), "value", "")
            if kind == "policy_denied":
                return 1.0
        return 0.0


__all__: "list[str]" = ["TaskEvalExecutor"]
