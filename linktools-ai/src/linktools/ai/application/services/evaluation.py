#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluation aggregate, comparison and replay validation."""

from ...domain.evaluation import EvaluationAggregate, EvaluationComparison
from ...domain.trace import RunSnapshot
from ...foundation.errors import ErrorCode, LinktoolsAIError
from ...foundation.ids import deterministic_id

class EvaluationService:
    """Compute reports and create replay requests through explicit ports."""

    def __init__(
        self,
        snapshots: "object | None" = None,
        execution: "object | None" = None,
        artifacts: "object | None" = None,
    ) -> None:
        self._snapshots = snapshots
        self._execution = execution
        self._artifacts = artifacts

    def aggregate(
        self,
        scores: "tuple[float, ...]",
        *,
        errors: int = 0,
        refusals: int = 0,
        retries: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model_tokens: int = 0,
        cost_microusd: int = 0,
        latencies_ms: "tuple[float, ...]" = (),
    ) -> EvaluationAggregate:
        if not scores:
            return EvaluationAggregate(count=0, metrics={})
        ordered = sorted(scores)
        p50 = ordered[(len(ordered) - 1) // 2]
        p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
        latency_values = sorted(latencies_ms)
        latency_p50 = latency_values[(len(latency_values) - 1) // 2] if latency_values else None
        latency_p95 = latency_values[max(0, int(len(latency_values) * 0.95) - 1)] if latency_values else None
        count = len(scores)
        return EvaluationAggregate(
            count=count,
            metrics={
                "mean": sum(scores) / count,
                "pass_rate": sum(score >= 1 for score in scores) / count,
                "error_rate": errors / count,
                "refusal_rate": refusals / count,
                "retry_count": float(retries),
                "input_tokens": float(input_tokens),
                "output_tokens": float(output_tokens),
                "model_tokens": float(model_tokens),
                "cost_microusd": float(cost_microusd),
            },
            pass_rate=sum(score >= 1 for score in scores) / count,
            error_rate=errors / count,
            refusal_rate=refusals / count,
            retry_count=retries,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_tokens=model_tokens,
            cost_microusd=cost_microusd,
            latency_p50_ms=latency_p50,
            latency_p95_ms=latency_p95,
            p50=p50,
            p95=p95,
        )

    def compare(self, baseline: EvaluationAggregate, candidate: EvaluationAggregate) -> EvaluationComparison:
        return EvaluationComparison.compare(baseline, candidate)

    async def replay(self, request: object) -> object:
        """Verify a snapshot and start a fresh execution without old effects."""
        if self._snapshots is None or self._execution is None:
            raise LinktoolsAIError(ErrorCode.STORAGE_CAPABILITY_MISSING, "replay ports are not configured")
        snapshot: RunSnapshot = await self._snapshots.get(request.snapshot_id)
        if snapshot is None or not snapshot.verify():
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "snapshot is invalid")
        if request.verify_artifacts:
            available = request.artifact_digests
            if self._artifacts is not None:
                available = await self._artifacts.available(snapshot.artifact_digests)
            if not snapshot.verify_artifacts(tuple(available)):
                raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "snapshot artifact is unavailable")
        replay_request = {
            "idempotency_key": deterministic_id(b"evaluation-replay", snapshot.snapshot_id),
            "source_execution_id": snapshot.execution_id,
            "snapshot_id": snapshot.snapshot_id,
            "input_digest": snapshot.input_digest,
            "release_digest": snapshot.release_digest,
            "bundle_digest": snapshot.bundle_digest,
            "prompt_digest": snapshot.prompt_digest,
        }
        return await self._execution.run(replay_request)


__all__ = ["EvaluationService"]
