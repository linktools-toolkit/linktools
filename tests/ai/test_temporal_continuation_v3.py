#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for Temporal continue-as-new stage restoration."""

from dataclasses import replace

import pytest

from linktools.ai.core import ApprovalDecision, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.temporal.workflow._execution import (
    CONTINUE_EVENT_THRESHOLD,
    ExecutionWorkflow,
    ExecutionWorkflowInput,
    WorkflowPhase,
    _initial_state,
    _resume_state,
    _should_continue_as_new,
)


def _request() -> ExecutionWorkflowInput:
    return ExecutionWorkflowInput(
        execution_id="execution",
        tenant_id="tenant",
        binding_digest="a" * 64,
        request_ref="request",
        worker_build="build",
    )


def _decision_digest(
    approval_id: str,
    idempotency_key: str,
    decision: ApprovalDecision,
    principal_id: str,
) -> str:
    return canonical_sha256(
        {
            "approval_id": approval_id,
            "idempotency_key": idempotency_key,
            "decision": decision.value,
            "principal_id": principal_id,
        }
    )


def _approve(
    workflow: ExecutionWorkflow,
    *,
    idempotency_key: str = "approval-key",
    decision: ApprovalDecision = ApprovalDecision.APPROVE,
) -> object:
    principal_id = "principal"
    return workflow.approve(
        "approval-operation",
        "approval",
        idempotency_key,
        decision,
        principal_id,
        _decision_digest("approval", idempotency_key, decision, principal_id),
    )


def test_terminal_workflow_never_continues_as_new() -> None:
    state = replace(
        _initial_state(_request()),
        status=WorkflowPhase.SUCCEEDED.value,
        last_stage="settle_budget",
        last_event_sequence=CONTINUE_EVENT_THRESHOLD,
    )
    assert _should_continue_as_new(state) is False


def test_continue_threshold_advances_with_continue_count() -> None:
    state = replace(
        _initial_state(_request()),
        status=WorkflowPhase.WAITING_EXTERNAL.value,
        last_stage="persist_deferred",
        last_event_sequence=CONTINUE_EVENT_THRESHOLD,
    )
    assert _should_continue_as_new(state) is True
    continued = replace(state, continue_count=1)
    assert _should_continue_as_new(continued) is False
    assert _should_continue_as_new(
        replace(continued, last_event_sequence=CONTINUE_EVENT_THRESHOLD * 2)
    ) is True


def test_resume_preserves_exact_stage_checkpoint() -> None:
    request = _request()
    state = replace(
        _initial_state(request),
        status=WorkflowPhase.WAITING_APPROVAL.value,
        last_stage="persist_deferred",
        pending_approval_ids=("approval",),
        last_event_sequence=CONTINUE_EVENT_THRESHOLD,
        continue_count=1,
    )
    assert _resume_state(request, state) == state


def test_cancel_wakes_deferred_wait_immediately() -> None:
    workflow = ExecutionWorkflow()
    workflow._state = replace(
        _initial_state(_request()),
        status=WorkflowPhase.WAITING_APPROVAL.value,
        last_stage="persist_deferred",
        pending_approval_ids=("approval",),
    )

    cancelled = workflow.cancel("cancel-operation")

    assert cancelled.status == WorkflowPhase.CANCELLING.value
    assert workflow._deferred_resolved() is True


def test_last_approval_switches_to_external_wait_phase() -> None:
    workflow = ExecutionWorkflow()
    workflow._state = replace(
        _initial_state(_request()),
        status=WorkflowPhase.WAITING_APPROVAL.value,
        last_stage="persist_deferred",
        pending_approval_ids=("approval",),
        pending_external_ids=("external",),
    )

    updated = _approve(workflow)

    assert updated.pending_approval_ids == ()
    assert updated.pending_external_ids == ("external",)
    assert updated.status == WorkflowPhase.WAITING_EXTERNAL.value
    assert workflow._deferred_resolved() is False


def test_approval_does_not_overwrite_cancelling_phase() -> None:
    workflow = ExecutionWorkflow()
    workflow._state = replace(
        _initial_state(_request()),
        status=WorkflowPhase.CANCELLING.value,
        last_stage="persist_deferred",
        pending_approval_ids=("approval",),
        pending_external_ids=("external",),
    )

    updated = _approve(workflow)

    assert updated.status == WorkflowPhase.CANCELLING.value


def test_approval_exact_replay_is_idempotent() -> None:
    workflow = ExecutionWorkflow()
    workflow._state = replace(
        _initial_state(_request()),
        status=WorkflowPhase.WAITING_APPROVAL.value,
        last_stage="persist_deferred",
        pending_approval_ids=("approval",),
    )

    first = _approve(workflow)
    second = _approve(workflow)

    assert second == first


def test_approval_same_key_different_decision_conflicts() -> None:
    workflow = ExecutionWorkflow()
    workflow._state = replace(
        _initial_state(_request()),
        status=WorkflowPhase.WAITING_APPROVAL.value,
        last_stage="persist_deferred",
        pending_approval_ids=("approval",),
    )
    _approve(workflow)

    with pytest.raises(AIError) as error:
        _approve(workflow, decision=ApprovalDecision.DENY)

    assert error.value.code is ErrorCode.APPROVAL_CONFLICT


def test_approval_same_decision_different_key_conflicts() -> None:
    workflow = ExecutionWorkflow()
    workflow._state = replace(
        _initial_state(_request()),
        status=WorkflowPhase.WAITING_APPROVAL.value,
        last_stage="persist_deferred",
        pending_approval_ids=("approval",),
    )
    _approve(workflow)

    with pytest.raises(AIError) as error:
        _approve(workflow, idempotency_key="other-key")

    assert error.value.code is ErrorCode.APPROVAL_CONFLICT
