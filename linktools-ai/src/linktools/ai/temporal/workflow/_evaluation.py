"""Deterministic evaluation workflow boundary."""

from dataclasses import dataclass
from typing import Protocol

try:
    from temporalio import workflow as _temporal_workflow
    from temporalio.common import RetryPolicy as _TemporalRetryPolicy
except ModuleNotFoundError as error:
    if error.name != "temporalio":
        raise
    _temporal_workflow = None
    _TemporalRetryPolicy = None

from ._execution import (
    ExecutionWorkflow,
    ExecutionWorkflowInput,
    ExecutionWorkflowResult,
)


@dataclass(frozen=True, slots=True)
class EvaluationWorkflowInput:
    evaluation_id: str
    dataset_digest: str
    target_revision: int
    case_ids: "tuple[str, ...]"
    worker_build: str
    binding_digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.worker_build, str) or not self.worker_build.strip():
            raise ValueError("evaluation workflow worker build is required")


@dataclass(frozen=True, slots=True)
class EvaluationWorkflowResult:
    evaluation_id: str
    status: str
    completed_case_ids: "tuple[str, ...]"


class EvaluationActivity(Protocol):
    async def run(self, request: EvaluationWorkflowInput) -> EvaluationWorkflowResult: ...


class EvaluationWorkflow:
    def __init__(self, activity: 'EvaluationActivity | None' = None) -> None:
        self._activity = activity

    async def run(self, request: EvaluationWorkflowInput) -> EvaluationWorkflowResult:
        if not request.evaluation_id or not request.dataset_digest or request.target_revision < 1:
            raise ValueError("evaluation workflow input is incomplete")
        if len(set(request.case_ids)) != len(request.case_ids):
            raise ValueError("evaluation workflow contains duplicate cases")
        if self._activity is not None:
            result = await self._activity.run(request)
        elif _temporal_workflow is not None and _TemporalRetryPolicy is not None:
            result = await _run_evaluation_children(request)
        else:
            raise RuntimeError("Temporal SDK is required for production workflow execution")
        if result.evaluation_id != request.evaluation_id:
            raise ValueError("evaluation activity returned mismatched identity")
        return result


async def _run_evaluation_children(request: EvaluationWorkflowInput) -> EvaluationWorkflowResult:
    if _temporal_workflow is None or _TemporalRetryPolicy is None:
        raise RuntimeError("Temporal SDK is required for production workflow execution")
    binding_digest = request.binding_digest or f"evaluation:{request.evaluation_id}:{request.target_revision}"
    handles = tuple(
        _temporal_workflow.start_child_workflow(
            ExecutionWorkflow,
            ExecutionWorkflowInput(
                execution_id=f"{request.evaluation_id}:{case_id}",
                tenant_id=f"evaluation:{request.evaluation_id}",
                binding_digest=binding_digest,
                bundle_digest=binding_digest,
                request_ref=f"dataset:{request.dataset_digest}:case:{case_id}",
                worker_build=request.worker_build,
            ),
            id=f"{request.evaluation_id}:{case_id}",
            retry_policy=_TemporalRetryPolicy(maximum_attempts=2),
        )
        for case_id in request.case_ids
    )
    results_list: list[ExecutionWorkflowResult] = []
    for handle in handles:
        results_list.append(await handle.result())
    results = tuple(results_list)
    _validate_case_results(request, results)
    completed = tuple(
        case_id
        for case_id, result in zip(request.case_ids, results)
        if result.status == "SUCCEEDED"
    )
    status = "SUCCEEDED" if len(completed) == len(request.case_ids) else "FAILED"
    return EvaluationWorkflowResult(request.evaluation_id, status, completed)


def _validate_case_results(
    request: EvaluationWorkflowInput,
    results: "tuple[ExecutionWorkflowResult, ...]",
) -> None:
    if len(results) != len(request.case_ids):
        raise ValueError("evaluation case workflow count does not match the evaluation")
    for result, case_id in zip(results, request.case_ids):
        if result.execution_id != f"{request.evaluation_id}:{case_id}":
            raise ValueError("evaluation case workflow returned mismatched identity")
        if result.status not in {"SUCCEEDED", "FAILED", "CANCELLED", "WAITING"}:
            raise ValueError("evaluation case workflow returned an invalid status")


if _temporal_workflow is not None:
    EvaluationWorkflow.run = _temporal_workflow.run(EvaluationWorkflow.run)
    EvaluationWorkflow = _temporal_workflow.defn(name="EvaluationWorkflow")(EvaluationWorkflow)


__all__ = [
    "EvaluationActivity",
    "EvaluationWorkflow",
    "EvaluationWorkflowInput",
    "EvaluationWorkflowResult",
]
