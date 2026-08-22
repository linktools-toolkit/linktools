#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for durable execution binding invariants."""

from datetime import datetime, timezone

import pytest

from linktools.ai.core import ExecutionLineageKind, ExecutionStatus
from linktools.ai.runtime.state._contracts import (
    ExecutionRecord,
    RecoveryExecutionInput,
    RecoveryIdempotencyInput,
)


def _execution(*, planning: bool = False, thinking: bool = False) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    return ExecutionRecord(
        execution_id="execution",
        tenant_id="tenant",
        session_id=None,
        binding_digest="a" * 64,
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.PENDING_START,
        revision=0,
        event_sequence=0,
        agent_run_sequence=0,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        planning=planning,
        thinking=thinking,
        binding=None,
    )


def _recovery(*, planning: bool = False, thinking: bool = False) -> RecoveryExecutionInput:
    return RecoveryExecutionInput(
        user_prompt="prompt",
        principal_id="principal",
        principal_kind="service",
        session_id=None,
        memory_scope=None,
        agent_id="agent",
        binding_digest="a" * 64,
        lineage_kind=ExecutionLineageKind.RUN.value,
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        conversation_step_run_id=None,
        idempotency=RecoveryIdempotencyInput("scope", "key", "request"),
        planning=planning,
        thinking=thinking,
        binding=None,
    )


def test_legacy_execution_shape_requires_modes_disabled() -> None:
    assert _execution().binding is None
    with pytest.raises(ValueError, match="legacy execution"):
        _execution(planning=True)
    with pytest.raises(ValueError, match="legacy execution"):
        _execution(thinking=True)


def test_legacy_recovery_shape_requires_modes_disabled() -> None:
    assert _recovery().binding is None
    with pytest.raises(ValueError, match="legacy recovery"):
        _recovery(planning=True)
    with pytest.raises(ValueError, match="legacy recovery"):
        _recovery(thinking=True)
