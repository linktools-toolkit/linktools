#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for swarm.engine.SwarmEngine: the orchestrator that compiles
member agents, creates the driving RunRecord + SwarmRun, builds the
SwarmExecutionContext, delegates the round loop to the resolved strategy, writes
ONLY the final aggregate to the shared Session, and transitions the driving Run
to SUCCEEDED. Plus resume() (explicit, caller-driven) and cancel() (store-level).

PROGRAMMATIC -- workers are real CompiledAgents driven by FunctionModel; no real
model calls. Mirrors the test_strategy.py harness conventions."""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.engine import AgentEngine
from tests.ai.swarm._dispatch_double import StrategyTestDispatcher
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.run.controller import RunController
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.errors import (
    SwarmLimitExceededError,
    SwarmRunNotFoundError,
)
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.resolver import ModelResolver
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import (
    RunErrorInfo,
    RunInput,
    RunRecord,
    RunResult,
    RunStatus,
    RunnableType,
)
from linktools.ai.session.models import (
    MessageRole,
    SessionRecord,
    SessionStatus,
)
from linktools.ai.agent.persistence.filesystem import FilesystemApprovalStore
from linktools.ai.run.persistence.checkpoint import FilesystemCheckpointStore
from linktools.ai.run.persistence.commit import FilesystemRunCommitCoordinator
from linktools.ai.run.persistence.definition import FilesystemRunDefinitionStore
from linktools.ai.events.persistence.filesystem import FilesystemEventStore
from linktools.ai.run.persistence.run import FilesystemRunStore
from linktools.ai.session.persistence.filesystem import FilesystemSessionStore
from linktools.ai.swarm.persistence.filesystem import FilesystemSwarmStore
from linktools.ai.swarm.aggregation import AggregationPolicy
from linktools.ai.swarm.limits import SwarmLimits
from linktools.ai.swarm.models import (
    AgentRef,
    SwarmRun,
    SwarmStatus,
    SwarmStep,
    SwarmStepStatus,
    TaskInput,
    TokenUsage,
)
from linktools.ai.swarm.spec import (
    SwarmContextPolicy,
    SwarmSpec,
    SwarmStrategySpec,
)
from linktools.ai.governance.policy.engine import PolicyEngine
from linktools.ai.tool.executor import GovernedToolInvoker


_NOW = datetime.now(timezone.utc)


class _FakeClock:
    """Deterministic Clock for lease/timestamp tests: a fixed ``now`` the test
    controls (no wall clock, no real sleep), so a lease's expiry is decided
    purely by the fixed time the FakeClock reports, never by suite-load timing."""

    def __init__(self, start: datetime) -> None:
        self._now = start

    def now(self) -> datetime:
        return self._now

    async def sleep(self, seconds: float) -> None:  # noqa: D401 - no-op by design
        # Deterministic tests never advance real time; ``advance`` moves the
        # logical clock explicitly.
        return None

    def advance(self, delta: "timedelta") -> None:
        self._now = self._now + delta


# --- helpers ----------------------------------------------------------------


def _make_model(output_text: str) -> FunctionModel:
    def _fn(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=output_text)])

    return FunctionModel(_fn)


def _make_model_with_usage(
    output_text: str, *, input_tokens: int, output_tokens: int
) -> FunctionModel:
    """Variant of _make_model that also reports token usage on each response --
    needed for max_total_tokens enforcement (the swarm accumulates
    RunResult.token_usage which AgentEngine populates from run_result.usage)."""
    from pydantic_ai.usage import RunUsage

    usage = RunUsage(input_tokens=input_tokens, output_tokens=output_tokens)

    def _fn(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=output_text)], usage=usage)

    return FunctionModel(_fn)


def _build_compiler(*outputs: str) -> AgentCompiler:
    """Build an AgentCompiler with one registered model per output string. The
    model_type for the i-th output is ``f"model-{i}"`` so test specs can request
    a deterministic output by referencing that model_type."""
    registry = ModelRegistry()
    for i, out in enumerate(outputs):
        registry.register(f"model-{i}", model=_make_model(out))
    return AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=registry),
    )


def _agent_spec(agent_id: str, model_type: str) -> AgentSpec:
    return AgentSpec(
        id=agent_id,
        name=agent_id,
        model=ModelPolicy(primary=model_type),
        instructions=PromptSpec(instructions=f"you are {agent_id}"),
        output_schema=str,
    )


def _limits(**overrides) -> SwarmLimits:
    base = dict(
        max_rounds=10,
        max_tasks=50,
        max_delegations=20,
        max_depth=5,
        max_concurrency=4,
        max_total_tokens=None,
        max_total_cost=None,
        timeout_seconds=None,
    )
    base.update(overrides)
    return SwarmLimits(**base)


def _spec(
    *,
    kind: str,
    limits: "SwarmLimits | None" = None,
    agents: "tuple[AgentRef, ...]",
    coordinator: AgentRef,
    config: "dict[str, Any] | None" = None,
) -> SwarmSpec:
    return SwarmSpec(
        id="swarm-spec-1",
        name="test-swarm",
        agents=agents,
        coordinator=coordinator,
        strategy=SwarmStrategySpec(kind=kind, config=config or {}),
        limits=limits or _limits(),
        context_policy=SwarmContextPolicy(),
        aggregation=AggregationPolicy(),
    )


class _Stores:
    """Wires the five file-backed stores under tmp_path subdirs, plus one
    Runtime-style AgentEngine + RunController -- SwarmEngine no longer builds
    its own AgentEngine (scenario: it must reuse the one build_runtime()
    assembles), so tests build it here exactly like build_runtime() does."""

    def __init__(self, tmp_path: Path, *, clock=None) -> None:
        from linktools.ai.clock import SystemClock

        self.clock = clock if clock is not None else SystemClock()
        self.run_store = FilesystemRunStore(root=tmp_path / "runs")
        self.session_store = FilesystemSessionStore(root=tmp_path / "sessions")
        self.event_store = FilesystemEventStore(root=tmp_path / "events")
        self.checkpoint_store = FilesystemCheckpointStore(root=tmp_path / "checkpoints")
        self.swarm_store = FilesystemSwarmStore(root=tmp_path / "swarm")
        self.run_definitions = FilesystemRunDefinitionStore(root=tmp_path / "definitions")
        self.run_controller = RunController()
        self.agent_runner = StrategyTestDispatcher(
            AgentEngine(),
            session_store=self.session_store,
            run_store=self.run_store,
            run_definitions=self.run_definitions,
        )

    def seed_shared_session(self, session_id: str) -> None:
        asyncio.run(
            self.session_store.create(
                SessionRecord(
                    id=session_id,
                    parent_id=None,
                    status=SessionStatus.ACTIVE,
                    version=1,
                    created_at=_NOW,
                    updated_at=_NOW,
                )
            )
        )


def _driving_context(run_id: str, session_id: str) -> RunContext:
    return RunContext(
        run_id=run_id,
        root_run_id=run_id,
        parent_run_id=None,
        session_id=session_id,
        runnable_id="swarm-spec-1",
        runnable_type=RunnableType.SWARM,
        user_id=None,
        tenant_id=None,
        workspace=None,
    )


# --- 1. End-to-end run() with ParallelFanOutStrategy ------------------------
