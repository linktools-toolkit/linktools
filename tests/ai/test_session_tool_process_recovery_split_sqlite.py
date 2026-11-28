#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed-approval session recovery after a process exits across split SQLite stores."""

import asyncio
import os
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from linktools.ai.capability import AgentContext, CapabilityContribution, SessionCapability
from linktools.ai.core import ApprovalDecision, ApprovalStatus, ExecutionStatus, Principal, ToolOperationStatus
from linktools.ai.execution import ApprovalDecisionRequest
from linktools.ai.persistence import Persistence
from linktools.ai.persistence.turns import SessionTurnKey
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import _Models, _UsageFunctionModel, _wait_for_committed, model_features
from .test_session_terminal_process_recovery import _storage
from .test_session_tool_process_recovery import _HostAuthorization, _SESSION, _append, _run_until_crash

_NAMESPACE = "split-sqlite-approval-crash"
_PRINCIPAL = Principal("reader", "default")


async def _send(runtime: Runtime):
    return await runtime.sessions.start(
        _SESSION, "inspect", agent=runtime.agents.get(), principal=_PRINCIPAL, idempotency_key="original",
    )


def _features(root: Path) -> RuntimeFeatures:
    def lookup(_context: AgentContext[None], value: str) -> str:
        _append(root / "effects.log", value)
        return value

    async def reply(messages, _info):
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if not returns:
            _append(root / "provider.log", "initial")
            return ModelResponse(parts=[ToolCallPart("lookup", {"value": "fixture"}, "approve")])
        assert [(part.tool_call_id, part.content) for part in returns] == [("approve", "fixture")]
        _append(root / "provider.log", "continuation")
        return ModelResponse(parts=[TextPart("approved lookup finished")])

    features = RuntimeFeatures(_NAMESPACE)
    tool = features.tool(lookup, effect_policy="none")
    tool.requires_approval = True
    features._register(CapabilityContribution.from_opaque("tool", "lookup", tool))
    features.capability(SessionCapability(checkpoints=True))
    features.agent("default", model="default", allow_tools=("lookup",))
    return model_features(_Models(_UsageFunctionModel(reply)), features)


async def _approve(runtime: Runtime, key: SessionTurnKey, approval_id: str) -> None:
    await runtime.approvals.decide(key, ApprovalDecisionRequest(
        _PRINCIPAL, approval_id, "approve-original", ApprovalDecision.APPROVE,
    ))


def _exit_after_split_approval_frontier(root_path: str, phase: str) -> None:
    async def run():
        root = Path(root_path)
        async with (
            _storage(root, "split_sqlite", provision=True) as storage,
            Persistence.open(_NAMESPACE, storage=storage) as state,
            Runtime.open(persistence=state, features=_features(root), authorization=_HostAuthorization()) as runtime,
        ):
            await runtime.sessions.create_turn_session(_SESSION, principal=_PRINCIPAL)
            turn = await _send(runtime)
            if not isinstance(turn.key, SessionTurnKey):
                raise AssertionError("native turn required")
            waiting = await _wait_for_committed(
                lambda: state.session_turns.get_turn(turn.key), lambda row: row is not None and row["state"] == "waiting",
            )
            assert waiting["split_source_receipt"] is not None
            [approval] = await runtime.approvals.list(turn.key, principal=_PRINCIPAL)
            _append(root / "original.log", turn.key.turn_id)
            _append(root / "original.log", approval.approval_id)
            if phase == "ready":
                await _approve(runtime, turn.key, approval.approval_id)
                assert (await state.session_turns.get_turn(turn.key))["state"] == "ready"
            os._exit(91)

    asyncio.run(run())


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("waiting", pytest.param("ready", marks=pytest.mark.merge)))
async def test_split_session_recovers_its_original_approval_after_process_exit(tmp_path: Path, phase: str) -> None:
    await _run_until_crash(_exit_after_split_approval_frontier, str(tmp_path), phase)
    turn_id, approval_id = (tmp_path / "original.log").read_text().splitlines()
    key = SessionTurnKey(_SESSION, turn_id)
    assert (tmp_path / "provider.log").read_text().splitlines() == ["initial"]
    assert not (tmp_path / "effects.log").exists()

    async with (
        _storage(tmp_path, "split_sqlite", provision=False) as storage,
        Persistence.open(_NAMESPACE, storage=storage) as state,
        Runtime.open(persistence=state, features=_features(tmp_path), authorization=_HostAuthorization()) as runtime,
    ):
        tenant, principal = runtime.tenant_id, _PRINCIPAL
        turn = await state.session_turns.get_turn(key)
        assert turn["state"] == phase and turn["generation"] == 1
        assert turn["split_source_receipt"] is not None
        [original] = await state.checkpoint.tools.list_by_execution(turn_id, tenant_id=tenant)
        assert original.status is ToolOperationStatus.PENDING and original.owner is None
        assert original.tool_call_id == "approve"
        repeated = await _send(runtime)
        assert repeated.key == key and await state.session_turns.get_turn(key) == turn

        if phase == "waiting":
            [approval] = await runtime.approvals.list(key, principal=principal)
            assert approval.approval_id == approval_id and approval.status is ApprovalStatus.PENDING
            assert not await runtime.sessions.recover_turn(_SESSION, turn_id, principal=principal)
            await _approve(runtime, key, approval_id)
        assert (await state.session_turns.get_turn(key))["state"] == "ready"
        assert not (tmp_path / "effects.log").exists()
        assert await runtime.sessions.recover_turn(_SESSION, turn_id, principal=principal)
        result = await repeated.wait_final(timeout_seconds=15)
        assert result.key == key and result.status is ExecutionStatus.SUCCEEDED
        assert result.view["output"] == {"text": "approved lookup finished"}
        assert (await state.session_turns.get_turn(key))["generation"] == 2
        assert (await state.session_turns.get_session(_SESSION))["active_turn_id"] is None
        [settled] = await state.checkpoint.tools.list_by_execution(turn_id, tenant_id=tenant)
        assert settled.tool_operation_id == original.tool_operation_id
        assert settled.status is ToolOperationStatus.COMPLETED
        assert not await runtime.sessions.recover_turn(_SESSION, turn_id, principal=principal)

    assert (tmp_path / "effects.log").read_text().splitlines() == ["fixture"]
    assert (tmp_path / "provider.log").read_text().splitlines() == ["initial", "continuation"]
