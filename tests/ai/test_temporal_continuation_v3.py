#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for Temporal continue-as-new stage restoration."""

from dataclasses import replace

from linktools.ai.temporal.workflow._execution import (
    CONTINUE_EVENT_THRESHOLD,
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
        bundle_digest="b" * 64,
        request_ref="request",
        worker_build="build",
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
