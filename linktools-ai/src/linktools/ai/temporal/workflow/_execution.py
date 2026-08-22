#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic execution workflow state machine."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import timedelta
from enum import StrEnum
from typing import Protocol

try:
    from temporalio import workflow as _temporal_workflow
    from temporalio.common import RetryPolicy as _TemporalRetryPolicy
except ModuleNotFoundError as error:
    if error.name != "temporalio":
        raise
    _temporal_workflow = None
    _TemporalRetryPolicy = None

from ...core import (
    ApprovalDecision,
    ExecutionStatus,
    JsonValue,
    validate_lease_owner,
    validate_tenant_id,
)
from ...errors import AIError, ErrorCode

CONTINUE_EVENT_THRESHOLD = 10000


class WorkflowPhase(StrEnum):
    LOADING = "LOADING"
    RESERVING_BUDGET = "RESERVING_BUDGET"
    RUNNING_AGENT = "RUNNING_AGENT"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    WAITING_EXTERNAL = "WAITING_EXTERNAL"
    COMMITTING_RESULT = "COMMITTING_RESULT"
    SETTLING_BUDGET = "SETTLING_BUDGET"
    CANCELLING = "CANCELLING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class StagePolicy:
    timeout_seconds: int
    maximum_attempts: int
    heartbeat: bool = False


STAGE_POLICIES: Mapping[str, StagePolicy] = {
    "load_input": StagePolicy(30, 3),
    "reserve_budget": StagePolicy(30, 3),
    "run_agent": StagePolicy(900, 1, True),
    "persist_deferred": StagePolicy(30, 3),
    "load_resume_input": StagePolicy(30, 3),
    "commit_result": StagePolicy(60, 3, True),
    "settle_budget": StagePolicy(30, 3),
    "cancel_effect": StagePolicy(60, 3, True),
    "blob_put": StagePolicy(300, 3, True),
}


@dataclass(frozen=True, slots=True)
class ExecutionWorkflowInput:
    execution_id: str
    tenant_id: str
    binding_digest: str
    bundle_digest: str
    request_ref: str
    worker_build: str
    owner: str = ""
    fence: int = 0
    operation_id: str = ""

    def __post_init__(self) -> None:
        _require_request(self)


@dataclass(frozen=True, slots=True)
class ExecutionWorkflowState:
    execution_id: str
    tenant_id: str
    request_ref: str
    worker_build: str
    owner: str
    fence: int
    operation_id: str
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
    result_ref: "str | None"
    continue_count: int
    operation_ledger_ref: str
    last_stage: str = ""
    external_result_refs: "tuple[tuple[str, str, str], ...]" = ()
    approval_decisions: "tuple[tuple[str, ApprovalDecision], ...]" = ()
    approval_idempotency_keys: "tuple[tuple[str, str], ...]" = ()
    last_operation_id: str = ""


@dataclass(frozen=True, slots=True)
class ExecutionWorkflowResult:
    execution_id: str
    status: str
    result_ref: "str | None"
    last_event_sequence: int
    state: "ExecutionWorkflowState | None" = None


class ExecutionActivity(Protocol):
    async def run(self, request: ExecutionWorkflowInput) -> ExecutionWorkflowResult: ...


class ExecutionWorkflow:
    def __init__(self, activity: "ExecutionActivity | None" = None) -> None:
        self._activity = activity
        self._state: "ExecutionWorkflowState | None" = None

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
            state = await self._run_stages(request)
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
        if self._require_state().status == WorkflowPhase.CANCELLING.value:
            cancelled = replace(
                self._require_state(),
                status=WorkflowPhase.CANCELLED.value,
            )
            self._state = cancelled
            return ExecutionWorkflowResult(
                request.execution_id,
                cancelled.status,
                cancelled.result_ref,
                cancelled.last_event_sequence,
                cancelled,
            )
        state = replace(
            self._require_state(),
            status=result.status,
            result_ref=result.result_ref,
            last_event_sequence=result.last_event_sequence,
        )
        self._state = state
        return replace(result, state=state)

    async def _run_stages(
        self,
        request: ExecutionWorkflowInput,
    ) -> ExecutionWorkflowState:
        state = self._require_state()
        while True:
            if state.status == WorkflowPhase.CANCELLING.value:
                state = await _cancel_effect(state)
            elif state.last_stage == "":
                state = await _load_input(state)
            elif state.last_stage == "load_input":
                state = await _reserve_budget(state)
            elif state.last_stage in {"reserve_budget", "load_resume_input"}:
                state = await _run_agent(state)
            elif state.last_stage == "run_agent":
                state = await _persist_deferred(state)
                if _has_pending_deferred(state):
                    state = replace(
                        state,
                        status=(
                            WorkflowPhase.WAITING_APPROVAL.value
                            if state.pending_approval_ids
                            else WorkflowPhase.WAITING_EXTERNAL.value
                        ),
                    )
            elif state.last_stage == "persist_deferred":
                if _has_pending_deferred(state):
                    state = _waiting_state(state)
                    self._state = state
                    self._continue_as_new_if_needed(request, state)
                    await _temporal_workflow.wait_condition(self._deferred_resolved)
                    state = self._require_state()
                    if state.status == WorkflowPhase.CANCELLING.value:
                        continue
                    state = await _load_resume_input(state)
                else:
                    state = await _commit_result(state)
            elif state.last_stage == "commit_result":
                state = await _settle_budget(state)
            elif state.last_stage in {"settle_budget", "cancel_effect"}:
                if state.status not in {
                    WorkflowPhase.SUCCEEDED.value,
                    WorkflowPhase.FAILED.value,
                    WorkflowPhase.CANCELLED.value,
                }:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                self._state = state
                return state
            else:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._state = state
            self._continue_as_new_if_needed(request, state)

    def _continue_as_new_if_needed(
        self,
        request: ExecutionWorkflowInput,
        state: ExecutionWorkflowState,
    ) -> None:
        if not _should_continue_as_new(state):
            return
        continued = replace(state, continue_count=state.continue_count + 1)
        self._state = continued
        _temporal_workflow.continue_as_new(args=(request, continued))

    def inspect(self) -> ExecutionWorkflowState:
        return self._require_state()

    def pending_approvals(self) -> "tuple[str, ...]":
        return self._require_state().pending_approval_ids

    def pending_external_calls(self) -> "tuple[str, ...]":
        return self._require_state().pending_external_ids

    def approve(
        self,
        operation_id: str,
        approval_id: str,
        decision: ApprovalDecision = ApprovalDecision.APPROVE,
        idempotency_key: str = "",
    ) -> ExecutionWorkflowState:
        state = self._require_state()
        if state.status in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            return state
        if approval_id not in state.pending_approval_ids:
            if any(
                item[0] == approval_id
                and (not idempotency_key or item[1] == idempotency_key)
                for item in state.approval_idempotency_keys
            ):
                return state
            raise ValueError("approval is not pending")
        return self._record_operation(
            state,
            operation_id,
            pending_approval_ids=tuple(
                item for item in state.pending_approval_ids if item != approval_id
            ),
            approval_decisions=(*state.approval_decisions, (approval_id, decision)),
            approval_idempotency_keys=(
                *state.approval_idempotency_keys,
                (approval_id, idempotency_key or operation_id),
            ),
        )

    def supply_external_result(
        self,
        operation_id: str,
        external_id: str,
        object_ref: str,
        payload_digest: str,
    ) -> ExecutionWorkflowState:
        state = self._require_state()
        if state.status in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            return state
        if external_id not in state.pending_external_ids:
            if any(
                item[0] == external_id
                and item[1] == object_ref
                and item[2] == payload_digest
                for item in state.external_result_refs
            ):
                return state
            raise ValueError("external call is not pending")
        if not object_ref.strip() or not payload_digest.strip():
            raise ValueError("external result reference and digest are required")
        return self._record_operation(
            state,
            operation_id,
            external_result_refs=(
                *state.external_result_refs,
                (external_id, object_ref, payload_digest),
            ),
            pending_external_ids=tuple(
                item for item in state.pending_external_ids if item != external_id
            ),
        )

    def cancel(self, operation_id: str) -> ExecutionWorkflowState:
        state = self._require_state()
        if state.status in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            return state
        return self._record_operation(
            state,
            operation_id,
            status=WorkflowPhase.CANCELLING.value,
        )

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
            or not state.pending_approval_ids
            and not state.pending_external_ids
        )

    def _record_operation(
        self,
        state: ExecutionWorkflowState,
        operation_id: str,
        *,
        status: "str | None" = None,
        pending_approval_ids: "tuple[str, ...] | None" = None,
        pending_external_ids: "tuple[str, ...] | None" = None,
        external_result_refs: "tuple[tuple[str, str, str], ...] | None" = None,
        approval_decisions: "tuple[tuple[str, ApprovalDecision], ...] | None" = None,
        approval_idempotency_keys: "tuple[tuple[str, str], ...] | None" = None,
    ) -> ExecutionWorkflowState:
        if not operation_id.strip():
            raise ValueError("operation id is required")
        if state.last_operation_id == operation_id:
            return state
        updated = replace(
            state,
            status=state.status if status is None else status,
            pending_approval_ids=(
                state.pending_approval_ids
                if pending_approval_ids is None
                else pending_approval_ids
            ),
            pending_external_ids=(
                state.pending_external_ids
                if pending_external_ids is None
                else pending_external_ids
            ),
            operation_ledger_ref=state.operation_ledger_ref or operation_id,
            external_result_refs=(
                state.external_result_refs
                if external_result_refs is None
                else external_result_refs
            ),
            approval_decisions=(
                state.approval_decisions
                if approval_decisions is None
                else approval_decisions
            ),
            approval_idempotency_keys=(
                state.approval_idempotency_keys
                if approval_idempotency_keys is None
                else approval_idempotency_keys
            ),
            last_operation_id=operation_id,
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
    try:
        validate_tenant_id(request.tenant_id)
    except AIError as error:
        raise ValueError("execution workflow tenant is invalid") from error
    if request.owner:
        try:
            validate_lease_owner(request.owner)
        except AIError as error:
            raise ValueError("execution workflow lease identity is invalid") from error
    if (
        request.fence < 0
        or (request.fence and not request.owner.strip())
        or (request.operation_id and not request.owner.strip())
    ):
        raise ValueError("execution workflow lease identity is incomplete")


def _initial_state(request: ExecutionWorkflowInput) -> ExecutionWorkflowState:
    return ExecutionWorkflowState(
        execution_id=request.execution_id,
        tenant_id=request.tenant_id,
        request_ref=request.request_ref,
        worker_build=request.worker_build,
        owner=request.owner,
        fence=request.fence,
        operation_id=request.operation_id,
        status=WorkflowPhase.LOADING.value,
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
        operation_ledger_ref="",
    )


def _resume_state(
    request: ExecutionWorkflowInput,
    state: "ExecutionWorkflowState | None",
) -> ExecutionWorkflowState:
    if state is None:
        return _initial_state(request)
    pinned_state = (
        state.execution_id,
        state.tenant_id,
        state.request_ref,
        state.worker_build,
        state.owner,
        state.fence,
        state.operation_id,
        state.binding_digest,
        state.bundle_digest,
    )
    pinned_request = (
        request.execution_id,
        request.tenant_id,
        request.request_ref,
        request.worker_build,
        request.owner,
        request.fence,
        request.operation_id,
        request.binding_digest,
        request.bundle_digest,
    )
    if pinned_state != pinned_request:
        raise ValueError("execution continue snapshot does not match the workflow input")
    return state


def _has_pending_deferred(state: ExecutionWorkflowState) -> bool:
    return bool(state.pending_approval_ids or state.pending_external_ids)


def _waiting_state(state: ExecutionWorkflowState) -> ExecutionWorkflowState:
    if not _has_pending_deferred(state):
        return state
    return replace(
        state,
        status=(
            WorkflowPhase.WAITING_APPROVAL.value
            if state.pending_approval_ids
            else WorkflowPhase.WAITING_EXTERNAL.value
        ),
    )


def _should_continue_as_new(state: ExecutionWorkflowState) -> bool:
    if state.status in {
        WorkflowPhase.SUCCEEDED.value,
        WorkflowPhase.FAILED.value,
        WorkflowPhase.CANCELLED.value,
        WorkflowPhase.CANCELLING.value,
    }:
        return False
    return state.last_event_sequence >= (
        state.continue_count + 1
    ) * CONTINUE_EVENT_THRESHOLD


def _validate_stage_transition(
    previous: ExecutionWorkflowState,
    current: ExecutionWorkflowState,
    stage: str,
) -> None:
    previous_identity = (
        previous.execution_id,
        previous.tenant_id,
        previous.request_ref,
        previous.worker_build,
        previous.owner,
        previous.fence,
        previous.operation_id,
        previous.binding_digest,
        previous.bundle_digest,
    )
    current_identity = (
        current.execution_id,
        current.tenant_id,
        current.request_ref,
        current.worker_build,
        current.owner,
        current.fence,
        current.operation_id,
        current.binding_digest,
        current.bundle_digest,
    )
    if current_identity != previous_identity:
        raise ValueError(f"activity {stage} changed the pinned execution identity")
    if current.last_event_sequence < previous.last_event_sequence:
        raise ValueError(f"activity {stage} moved the event sequence backwards")
    if current.continue_count < previous.continue_count:
        raise ValueError(f"activity {stage} moved the continue count backwards")
    if current.status not in {item.value for item in ExecutionStatus} | {
        item.value for item in WorkflowPhase
    } | {"WAITING"}:
        raise ValueError(f"activity {stage} returned an invalid status")
    if (
        current.operation_ledger_ref != previous.operation_ledger_ref
        and previous.operation_ledger_ref
    ):
        raise ValueError(f"activity {stage} changed the operation ledger reference")
    if (
        current.external_result_refs[: len(previous.external_result_refs)]
        != previous.external_result_refs
    ):
        raise ValueError(f"activity {stage} removed an external result reference")
    if (
        current.approval_decisions[: len(previous.approval_decisions)]
        != previous.approval_decisions
    ):
        raise ValueError(f"activity {stage} removed an approval decision")
    if len(current.external_result_refs) > 1000 or len(current.approval_decisions) > 1000:
        raise AIError(ErrorCode.TOO_MANY_PENDING_OPERATIONS)


async def _execute_activity(
    name: str,
    state: ExecutionWorkflowState,
) -> ExecutionWorkflowState:
    if _temporal_workflow is None or _TemporalRetryPolicy is None:
        raise RuntimeError("Temporal SDK is required for production workflow execution")
    policy = STAGE_POLICIES.get(name)
    if policy is None:
        raise ValueError(f"unknown execution stage: {name}")
    kwargs: dict[str, JsonValue] = {
        "start_to_close_timeout": timedelta(seconds=policy.timeout_seconds),
        "retry_policy": _TemporalRetryPolicy(maximum_attempts=policy.maximum_attempts),
    }
    if policy.heartbeat:
        kwargs["heartbeat_timeout"] = timedelta(seconds=30)
    result = await _temporal_workflow.execute_activity(
        name,
        state,
        **kwargs,
    )
    if not isinstance(result, ExecutionWorkflowState):
        raise ValueError(f"activity {name} returned an invalid execution snapshot")
    return result


async def _load_input(state: ExecutionWorkflowState) -> ExecutionWorkflowState:
    return _validate_activity_result(
        state,
        await _execute_activity("load_input", state),
        "load_input",
    )


async def _reserve_budget(state: ExecutionWorkflowState) -> ExecutionWorkflowState:
    return _validate_activity_result(
        state,
        await _execute_activity("reserve_budget", state),
        "reserve_budget",
    )


async def _run_agent(state: ExecutionWorkflowState) -> ExecutionWorkflowState:
    return _validate_activity_result(
        state,
        await _execute_activity("run_agent", state),
        "run_agent",
    )


async def _persist_deferred(state: ExecutionWorkflowState) -> ExecutionWorkflowState:
    return _validate_activity_result(
        state,
        await _execute_activity("persist_deferred", state),
        "persist_deferred",
    )


async def _load_resume_input(state: ExecutionWorkflowState) -> ExecutionWorkflowState:
    return _validate_activity_result(
        state,
        await _execute_activity("load_resume_input", state),
        "load_resume_input",
    )


async def _commit_result(state: ExecutionWorkflowState) -> ExecutionWorkflowState:
    return _validate_activity_result(
        state,
        await _execute_activity("commit_result", state),
        "commit_result",
    )


async def _settle_budget(state: ExecutionWorkflowState) -> ExecutionWorkflowState:
    return _validate_activity_result(
        state,
        await _execute_activity("settle_budget", state),
        "settle_budget",
    )


async def _cancel_effect(state: ExecutionWorkflowState) -> ExecutionWorkflowState:
    result = _validate_activity_result(
        state,
        await _execute_activity("cancel_effect", state),
        "cancel_effect",
    )
    return replace(result, status=WorkflowPhase.CANCELLED.value)


def _validate_activity_result(
    previous: ExecutionWorkflowState,
    current: ExecutionWorkflowState,
    stage: str,
) -> ExecutionWorkflowState:
    _validate_stage_transition(previous, current, stage)
    return replace(current, last_stage=stage)


if _temporal_workflow is not None:
    ExecutionWorkflow.run = _temporal_workflow.run(ExecutionWorkflow.run)
    ExecutionWorkflow.inspect = _temporal_workflow.query(name="inspect")(
        ExecutionWorkflow.inspect
    )
    ExecutionWorkflow.pending_approvals = _temporal_workflow.query(
        name="pending_approvals"
    )(ExecutionWorkflow.pending_approvals)
    ExecutionWorkflow.pending_external_calls = _temporal_workflow.query(
        name="pending_external_calls"
    )(ExecutionWorkflow.pending_external_calls)
    ExecutionWorkflow.approve = _temporal_workflow.update(name="approve")(
        ExecutionWorkflow.approve
    )
    ExecutionWorkflow.supply_external_result = _temporal_workflow.update(
        name="supply_external_result"
    )(ExecutionWorkflow.supply_external_result)
    ExecutionWorkflow.cancel = _temporal_workflow.update(name="cancel")(
        ExecutionWorkflow.cancel
    )
    ExecutionWorkflow.continue_snapshot = _temporal_workflow.query(
        name="continue_snapshot"
    )(ExecutionWorkflow.continue_snapshot)
    ExecutionWorkflow = _temporal_workflow.defn(name="ExecutionWorkflow")(
        ExecutionWorkflow
    )


__all__ = [
    "ExecutionActivity",
    "ExecutionWorkflow",
    "ExecutionWorkflowInput",
    "ExecutionWorkflowResult",
    "ExecutionWorkflowState",
    "WorkflowPhase",
]
