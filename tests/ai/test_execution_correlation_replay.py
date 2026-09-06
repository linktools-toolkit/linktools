#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution replay correlation contract regressions."""

from types import SimpleNamespace

import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._execution import DefaultExecutionService


def _replay_values(
    *,
    execution_correlation: dict[str, str | int],
    request_correlation: dict[str, str | int],
) -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    snapshot = object()
    execution = SimpleNamespace(
        binding_digest="a" * 64,
        planning=False,
        thinking=False,
        binding=snapshot,
        correlation=execution_correlation,
    )
    binding = SimpleNamespace(digest="a" * 64, snapshot=snapshot)
    request = SimpleNamespace(
        planning=False,
        thinking=False,
        correlation=request_correlation,
    )
    return execution, binding, request


def test_execution_replay_accepts_matching_durable_correlation() -> None:
    service = object.__new__(DefaultExecutionService)
    execution, binding, request = _replay_values(
        execution_correlation={"trace_id": "durable", "attempt": 1},
        request_correlation={"trace_id": "durable", "attempt": 1},
    )

    service._validate_replayed_execution(execution, binding, request)


def test_execution_replay_rejects_correlation_drift() -> None:
    service = object.__new__(DefaultExecutionService)
    execution, binding, request = _replay_values(
        execution_correlation={"trace_id": "durable", "attempt": 1},
        request_correlation={"trace_id": "request", "attempt": 1},
    )

    with pytest.raises(AIError) as raised:
        service._validate_replayed_execution(execution, binding, request)

    assert raised.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
    assert execution.correlation == {"trace_id": "durable", "attempt": 1}
