#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/run/test_models.py"""
from datetime import datetime, timezone

from linktools.ai.run.models import (
    ALLOWED_RUN_TRANSITIONS,
    RunErrorInfo,
    RunInput,
    RunnableType,
    RunRecord,
    RunResult,
    RunStatus,
)


def test_run_status_values():
    assert RunStatus.PENDING == "pending"
    assert RunStatus.RUNNING == "running"
    assert RunStatus.WAITING_APPROVAL == "waiting_approval"
    assert RunStatus.PAUSED == "paused"
    assert RunStatus.SUCCEEDED == "succeeded"
    assert RunStatus.FAILED == "failed"
    assert RunStatus.CANCELLED == "cancelled"


def test_allowed_transitions_match_spec():
    assert ALLOWED_RUN_TRANSITIONS[RunStatus.PENDING] == frozenset({RunStatus.RUNNING})
    assert ALLOWED_RUN_TRANSITIONS[RunStatus.RUNNING] == frozenset({
        RunStatus.WAITING_APPROVAL, RunStatus.PAUSED, RunStatus.SUCCEEDED,
        RunStatus.FAILED, RunStatus.CANCELLED,
    })
    assert ALLOWED_RUN_TRANSITIONS[RunStatus.WAITING_APPROVAL] == frozenset({RunStatus.RUNNING, RunStatus.CANCELLED})
    assert ALLOWED_RUN_TRANSITIONS[RunStatus.PAUSED] == frozenset({RunStatus.RUNNING, RunStatus.CANCELLED})
    assert ALLOWED_RUN_TRANSITIONS[RunStatus.SUCCEEDED] == frozenset()
    assert ALLOWED_RUN_TRANSITIONS[RunStatus.FAILED] == frozenset()
    assert ALLOWED_RUN_TRANSITIONS[RunStatus.CANCELLED] == frozenset()


def test_runnable_type_values():
    assert RunnableType.AGENT == "agent"
    assert RunnableType.SWARM == "swarm"


def test_run_input_defaults():
    ri = RunInput(prompt="hello")
    assert ri.prompt == "hello"
    assert dict(ri.metadata) == {}


def test_run_result_defaults():
    rr = RunResult(output={"ok": True})
    assert rr.output == {"ok": True}
    assert dict(rr.token_usage) == {}
    assert dict(rr.metadata) == {}


def test_run_error_info():
    err = RunErrorInfo(error_type="ModelOutputError", message="boom")
    assert err.error_type == "ModelOutputError"
    assert err.message == "boom"
    assert dict(err.detail) == {}


def test_run_record_construction():
    now = datetime.now(timezone.utc)
    record = RunRecord(
        id="run-1", root_run_id="run-1", parent_run_id=None, session_id="session-1",
        runnable_id="agent-1", runnable_type=RunnableType.AGENT, status=RunStatus.PENDING,
        input=RunInput(prompt="hi"), result=None, error=None, version=1,
        created_at=now, started_at=None, finished_at=None,
    )
    assert record.id == "run-1"
    assert record.status == RunStatus.PENDING
    assert dict(record.metadata) == {}
