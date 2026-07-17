#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EvalRunner: drive a suite of cases through an executor + evaluators.

Direct mode: the executor runs the target inline (not through
TaskRuntime). For each case: execute → collect evaluator scores → produce an
EvalResult. One failing case does not abort the suite.
"""

import asyncio
import time
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone

from .models import (
    EvalCase,
    EvalResult,
    EvalRun,
    EvalRunStatus,
    EvalScore,
    EvalSuite,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EvalRunner:
    def __init__(
        self,
        executor,
        evaluators: "Mapping[str, object]",
        *,
        store=None,
    ) -> None:
        self._executor = executor
        self._evaluators = dict(evaluators)
        self._store = store

    async def run_suite(
        self,
        suite: EvalSuite,
        cases: "Mapping[str, EvalCase]",
    ) -> "tuple[EvalResult, ...]":
        run_id = f"eval-{uuid.uuid4().hex[:12]}"
        started = _utcnow()
        if self._store is not None:
            await self._store.create_run(
                EvalRun(
                    id=run_id,
                    suite_id=suite.id,
                    target=suite.target,
                    status=EvalRunStatus.RUNNING,
                    baseline_target=suite.baseline_target,
                    created_at=started,
                    started_at=started,
                    metadata=dict(suite.metadata),
                )
            )
        sem = asyncio.Semaphore(max(suite.max_concurrency, 1))

        async def run_case(case_id: str):
            case = cases.get(case_id)
            if case is None:
                return None
            async with sem:
                return await self._run_one(suite, case, run_id)

        raw = await asyncio.gather(*(run_case(cid) for cid in suite.case_ids))
        results = tuple(r for r in raw if r is not None)
        if self._store is not None:
            for result in results:
                await self._store.append_result(result)
            final = (
                EvalRunStatus.FAILED
                if any(r.error_type is not None for r in results)
                else EvalRunStatus.SUCCEEDED
            )
            await self._store.transition_run(
                run_id, status=final, finished_at=_utcnow()
            )
        return results

    async def _run_one(
        self, suite: EvalSuite, case: EvalCase, run_id: str
    ) -> EvalResult:
        start = time.monotonic()
        try:
            execution = await self._executor.execute(suite.target, case)
        except Exception as exc:  # noqa: BLE001
            return EvalResult(
                id=f"{run_id}-{case.id}",
                eval_run_id=run_id,
                case_id=case.id,
                run_id=None,
                output_artifact_id=None,
                snapshot_artifact_id=None,
                scores={},
                metrics={"latency_seconds": time.monotonic() - start},
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        # The target ran but reported an error without raising (e.g.
        # DirectEvalExecutor wraps a target failure as EvalExecution.error).
        # Record it so error_rate and the run's FAILED transition fire --
        # otherwise a fully-crashing suite wrongly reports SUCCEEDED.
        if execution.error is not None:
            return EvalResult(
                id=f"{run_id}-{case.id}",
                eval_run_id=run_id,
                case_id=case.id,
                run_id=execution.run_id,
                output_artifact_id=execution.output_artifact_id,
                snapshot_artifact_id=None,
                scores={},
                metrics={"latency_seconds": time.monotonic() - start},
                error_type=execution.error,
                error_message=f"target error: {execution.error}",
            )
        latency = time.monotonic() - start

        scores: dict[str, float] = {}
        metrics: dict[str, object] = {"latency_seconds": latency}
        # Thread the executor's captured stats (tokens / cost / retries /
        # safety refusal) into the result metrics so aggregate() can compute
        # avg tokens, avg cost, retry rate, safety refusal rate.
        usage = dict(execution.model_usage or {})
        for _usage_key in ("total_tokens", "total_cost", "retry_count", "safety_refusal"):
            _v = usage.get(_usage_key)
            if isinstance(_v, (int, float)) and not isinstance(_v, bool):
                metrics[_usage_key] = _v
        for name in suite.evaluator_names:
            evaluator = self._evaluators.get(name)
            if evaluator is None:
                continue
            try:
                score: EvalScore = await evaluator.evaluate(
                    case, execution, snapshot=execution.snapshot
                )
                scores[name] = score.score
                if score.details:
                    metrics[f"{name}_details"] = dict(score.details)
            except Exception:  # noqa: BLE001 - one evaluator failure does not abort
                scores[name] = 0.0

        return EvalResult(
            id=f"{run_id}-{case.id}",
            eval_run_id=run_id,
            case_id=case.id,
            run_id=execution.run_id,
            output_artifact_id=execution.output_artifact_id,
            snapshot_artifact_id=execution.snapshot_artifact_id,
            scores=scores,
            metrics=metrics,
        )


__all__: "list[str]" = ["EvalRunner"]
