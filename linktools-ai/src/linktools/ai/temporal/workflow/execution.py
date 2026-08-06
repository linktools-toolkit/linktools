#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic execution workflow state machine."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Protocol

try:
    from temporalio import workflow as _temporal_workflow
    from temporalio.common import RetryPolicy as _TemporalRetryPolicy
except ModuleNotFoundError as error:
    if error.name != "temporalio":
        raise
    _temporal_workflow = None
    _TemporalRetryPolicy = None

from ...core.json import JsonValue

CONTINUE_EVENT_THRESHOLD = 1000


@dataclass(frozen=True, slots=True)
class ExecutionWorkflowInput:
    execution_id: str
    tenant_id: str
    binding_digest: str
    bundle_digest: str
    request_ref: str
    worker_build: str


@dataclass(frozen=True, slots=True)
class ExecutionWorkflowState:
    status: str
    conversation_id: str
    run_id: str
    session_id: str
    session_revision: int
    resource_generation: int
    binding_digest: str
    bundle_digest: str
    prompt_digest: str
    model_registry_revision: int
    budget_reservation_id: str
    pending_approval_ids: "tuple[str, ...]"
    pending_external_ids: "tuple[str, ...]"
    last_event_sequence: int
    result_ref: 'str | None'
    continue_count: int
    operation_ids: "tuple[str, ...]"
    last_stage: str = ""
    external_results: "tuple[tuple[str, JsonValue], ...]" = ()


@dataclass(frozen=True, slots=True)
class ExecutionWorkflowResult:
    execution_id: str
    status: str
    result_ref: "str | None"
    last_event_sequence: int
    state: 'ExecutionWorkflowState | None' = None


class ExecutionActivity(Protocol):
    async def run(self, request: ExecutionWorkflowInput) -> ExecutionWorkflowResult: ...


class ExecutionWorkflow:
    def __init__(self, activity: 'ExecutionActivity | None' = None) -> None:
        self._activity = activity
        self._state: 'ExecutionWorkflowState | None' = None

    async def run(
        self,
        request: ExecutionWorkflowInput,
        resume_state: "ExecutionWorkflowState | None" = None,
    ) -> ExecutionWorkflowResult:
        _require_request(request)
        self._state = _resume_state(request, resume_state)
        if self._activity is not None:
            result = await self._activity.run(request)
        elif _temporal_workflow is not None and _TemporalRetryPolicy is not None:
            state = self._require_state()
            stages = (
                "load_input",
                "fix_bundle_route",
                "fix_binding",
                "load_prompt",
                "reserve_budget",
                "run_agent",
                "process_deferred",
                "append_event",
                "commit_result",
                "settle_budget",
                "append_terminal_event",
            )
            start = stages.index(state.last_stage) + 1 if state.last_stage in stages else 0
            for name in stages[start:]:
                next_state = await _execute_stage(name, state)
                _validate_stage_transition(state, next_state, name)
                state = replace(next_state, last_stage=name)
                self._state = state
                if name == "process_deferred" and _has_pending_deferred(state):
                    await _temporal_workflow.wait_condition(self._deferred_resolved)
                    state = self._require_state()
            if state.last_event_sequence >= CONTINUE_EVENT_THRESHOLD:
                _temporal_workflow.continue_as_new(args=(request, state))
            result = ExecutionWorkflowResult(
                request.execution_id,
                state.status,
                state.result_ref,
                state.last_event_sequence,
                state,
            )
        else:
            raise RuntimeError("Temporal SDK is required for production workflow execution")
        if result.execution_id != request.execution_id:
            raise ValueError("execution activity returned a different execution id")
        if result.status not in {"SUCCEEDED", "FAILED", "CANCELLED", "WAITING"}:
            raise ValueError("execution activity returned an invalid terminal status")
        if result.state is not None:
            _validate_stage_transition(self._require_state(), result.state, "run")
            self._state = result.state
        state = replace(
            self._require_state(),
            status=result.status,
            result_ref=result.result_ref,
            last_event_sequence=result.last_event_sequence,
        )
        self._state = state
        return replace(result, state=state)

    def inspect(self) -> ExecutionWorkflowState:
        return self._require_state()

    def pending_approvals(self) -> 'tuple[str, ...]':
        return self._require_state().pending_approval_ids

    def pending_external_calls(self) -> 'tuple[str, ...]':
        return self._require_state().pending_external_ids

    def approve(self, operation_id: str, approval_id: str) -> ExecutionWorkflowState:
        state = self._require_state()
        if operation_id in state.operation_ids:
            return state
        if state.status in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            return state
        if approval_id not in state.pending_approval_ids:
            raise ValueError("approval is not pending")
        return self._record_operation(
            state,
            operation_id,
            pending_approval_ids=tuple(item for item in state.pending_approval_ids if item != approval_id),
        )

    def supply_external_result(
        self,
        operation_id: str,
        external_id: str,
        payload: 'Mapping[str, JsonValue]',
    ) -> ExecutionWorkflowState:
        if not payload:
            raise ValueError("external result payload is empty")
        state = self._require_state()
        if operation_id in state.operation_ids:
            return state
        if state.status in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            return state
        if external_id not in state.pending_external_ids:
            raise ValueError("external call is not pending")
        return self._record_operation(
            state,
            operation_id,
            external_results=(*state.external_results, (external_id, dict(payload))),
            pending_external_ids=tuple(item for item in state.pending_external_ids if item != external_id),
        )

    def cancel(self, operation_id: str) -> ExecutionWorkflowState:
        state = self._require_state()
        if operation_id in state.operation_ids:
            return state
        if state.status in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            return state
        return self._record_operation(state, operation_id, status="CANCELLED")

    def continue_as_new(self) -> ExecutionWorkflowState:
        state = self._require_state()
        updated = replace(state, continue_count=state.continue_count + 1)
        self._state = updated
        return updated

    def continue_snapshot(self) -> ExecutionWorkflowState:
        """Return the complete deterministic snapshot used for a new run."""
        return self._require_state()

    def _deferred_resolved(self) -> bool:
        state = self._state
        return state is not None and (
            state.status in {"SUCCEEDED", "FAILED", "CANCELLED"}
            or not state.pending_approval_ids and not state.pending_external_ids
        )

    def _record_operation(
        self,
        state: ExecutionWorkflowState,
        operation_id: str,
        *,
        status: 'str | None' = None,
        pending_approval_ids: 'tuple[str, ...] | None' = None,
        pending_external_ids: 'tuple[str, ...] | None' = None,
        external_results: 'tuple[tuple[str, JsonValue], ...] | None' = None,
    ) -> ExecutionWorkflowState:
        if not operation_id.strip():
            raise ValueError("operation id is required")
        updated = replace(
            state,
            status=state.status if status is None else status,
            pending_approval_ids=state.pending_approval_ids if pending_approval_ids is None else pending_approval_ids,
            pending_external_ids=state.pending_external_ids if pending_external_ids is None else pending_external_ids,
            operation_ids=(*state.operation_ids, operation_id),
            external_results=state.external_results if external_results is None else external_results,
        )
        self._state = updated
        return updated

    def _require_state(self) -> ExecutionWorkflowState:
        if self._state is None:
            raise ValueError("execution workflow has not started")
        return self._state


def _require_request(request: ExecutionWorkflowInput) -> None:
    values = (
        request.execution_id,
        request.tenant_id,
        request.binding_digest,
        request.bundle_digest,
        request.request_ref,
        request.worker_build,
    )
    if any(not value.strip() for value in values):
        raise ValueError("execution workflow input is incomplete")


def _initial_state(request: ExecutionWorkflowInput) -> ExecutionWorkflowState:
    return ExecutionWorkflowState(
        status="PENDING",
        conversation_id="",
        run_id="",
        session_id="",
        session_revision=0,
        resource_generation=0,
        binding_digest=request.binding_digest,
        bundle_digest=request.bundle_digest,
        prompt_digest="",
        model_registry_revision=0,
        budget_reservation_id="",
        pending_approval_ids=(),
        pending_external_ids=(),
        last_event_sequence=0,
        result_ref=None,
        continue_count=0,
        operation_ids=(),
    )


def _resume_state(
    request: ExecutionWorkflowInput,
    state: "ExecutionWorkflowState | None",
) -> ExecutionWorkflowState:
    if state is None:
        return _initial_state(request)
    if state.binding_digest != request.binding_digest or state.bundle_digest != request.bundle_digest:
        raise ValueError("execution continue snapshot does not match the workflow input")
    return state


def _has_pending_deferred(state: ExecutionWorkflowState) -> bool:
    return bool(state.pending_approval_ids or state.pending_external_ids)


def _validate_stage_transition(
    previous: ExecutionWorkflowState,
    current: ExecutionWorkflowState,
    stage: str,
) -> None:
    if current.binding_digest != previous.binding_digest or current.bundle_digest != previous.bundle_digest:
        raise ValueError(f"activity {stage} changed the pinned execution binding")
    if current.last_event_sequence < previous.last_event_sequence:
        raise ValueError(f"activity {stage} moved the event sequence backwards")
    if current.continue_count < previous.continue_count:
        raise ValueError(f"activity {stage} moved the continue count backwards")
    if current.status not in {"PENDING", "SUCCEEDED", "FAILED", "CANCELLED", "WAITING"}:
        raise ValueError(f"activity {stage} returned an invalid status")
    if current.operation_ids[: len(previous.operation_ids)] != previous.operation_ids:
        raise ValueError(f"activity {stage} removed an operation record")
    if len(set(current.operation_ids)) != len(current.operation_ids):
        raise ValueError(f"activity {stage} returned duplicate operation ids")
    if current.external_results[: len(previous.external_results)] != previous.external_results:
        raise ValueError(f"activity {stage} removed an external result")
    if len({external_id for external_id, _ in current.external_results}) != len(current.external_results):
        raise ValueError(f"activity {stage} returned duplicate external results")


async def _execute_stage(name: str, state: ExecutionWorkflowState) -> ExecutionWorkflowState:
    if _temporal_workflow is None or _TemporalRetryPolicy is None:
        raise RuntimeError("Temporal SDK is required for production workflow execution")
    result = await _temporal_workflow.execute_activity(
        name,
        state,
        start_to_close_timeout=timedelta(seconds=300),
        retry_policy=_TemporalRetryPolicy(maximum_attempts=3),
    )
    if not isinstance(result, ExecutionWorkflowState):
        raise ValueError(f"activity {name} returned an invalid execution snapshot")
    return result


if _temporal_workflow is not None:
    ExecutionWorkflow.run = _temporal_workflow.run(ExecutionWorkflow.run)
    ExecutionWorkflow.inspect = _temporal_workflow.query(name="inspect")(ExecutionWorkflow.inspect)
    ExecutionWorkflow.pending_approvals = _temporal_workflow.query(name="pending_approvals")(ExecutionWorkflow.pending_approvals)
    ExecutionWorkflow.pending_external_calls = _temporal_workflow.query(name="pending_external_calls")(ExecutionWorkflow.pending_external_calls)
    ExecutionWorkflow.approve = _temporal_workflow.update(name="approve")(ExecutionWorkflow.approve)
    ExecutionWorkflow.supply_external_result = _temporal_workflow.update(name="supply_external_result")(ExecutionWorkflow.supply_external_result)
    ExecutionWorkflow.cancel = _temporal_workflow.update(name="cancel")(ExecutionWorkflow.cancel)
    ExecutionWorkflow.continue_snapshot = _temporal_workflow.query(name="continue_snapshot")(ExecutionWorkflow.continue_snapshot)
    ExecutionWorkflow = _temporal_workflow.defn(name="ExecutionWorkflow")(ExecutionWorkflow)


__all__ = [
    "ExecutionActivity",
    "ExecutionWorkflow",
    "ExecutionWorkflowInput",
    "ExecutionWorkflowResult",
    "ExecutionWorkflowState",
]
