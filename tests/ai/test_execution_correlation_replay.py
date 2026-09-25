#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution replay correlation contract regressions."""

from types import SimpleNamespace

from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.runtime._execution import DefaultExecutionService
from linktools.ai.spec import AgentSpec


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
    binding = SimpleNamespace(binding_digest="a" * 64, snapshot=snapshot)
    request = SimpleNamespace(
        planning=False,
        thinking=False,
        correlation=request_correlation,
    )
    return execution, binding, request


def test_execution_replay_uses_binding_digest() -> None:
    durable = AgentBindingSnapshot(
        agent_spec=AgentSpec("agent", description="durable label"),
        model_contract={"model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={"type": "string"},
    )
    replayed = AgentBindingSnapshot(
        agent_spec=AgentSpec("agent", description="request label"),
        model_contract={"model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={"type": "string"},
    )
    assert durable != replayed
    assert durable.binding_digest == replayed.binding_digest

    service = object.__new__(DefaultExecutionService)
    execution = SimpleNamespace(
        binding_digest=durable.binding_digest,
        planning=False,
        thinking=False,
        binding=durable,
    )
    binding = SimpleNamespace(
        digest=replayed.binding_digest,
        snapshot=replayed,
    )
    request = SimpleNamespace(planning=False, thinking=False)

    service._validate_replayed_execution(execution, binding, request)


def test_execution_replay_accepts_matching_durable_correlation() -> None:
    service = object.__new__(DefaultExecutionService)
    execution, binding, request = _replay_values(
        execution_correlation={"trace_id": "durable", "attempt": 1},
        request_correlation={"trace_id": "durable", "attempt": 1},
    )

    service._validate_replayed_execution(execution, binding, request)


def test_execution_replay_ignores_correlation_drift() -> None:
    service = object.__new__(DefaultExecutionService)
    execution, binding, request = _replay_values(
        execution_correlation={"trace_id": "durable", "attempt": 1},
        request_correlation={"trace_id": "request", "attempt": 2},
    )

    service._validate_replayed_execution(execution, binding, request)

    assert execution.correlation == {"trace_id": "durable", "attempt": 1}
