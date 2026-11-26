#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native parent claims fence explicit approval of the original child."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.exceptions import ApprovalRequired
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from linktools.ai.capability import AgentContext, ExecutionCapability, StepPersistenceCapability
from linktools.ai.core import ApprovalDecision, ApprovalStatus, ExecutionStatus, Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ApprovalDecisionRequest
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime._native_subagent import NativeSubagentAdmission

from ._runtime_test_helpers import _wait_for_committed
from .test_canonical_turn_subagents import _HostAuthorization, _model


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("approved", "unconfirmed-child-resume", "parent-stop"))
async def test_native_parent_approves_exact_child_without_shadow_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    child_entries, child_models, parent_models, child_effects, returned_ids = [], [], [], [], []

    async def inspect_fixture(context: AgentContext[None]) -> str:
        child_entries.append(context.execution_id)
        if len(child_entries) == 1:
            raise ApprovalRequired()
        child_effects.append(context.execution_id)
        return "checked"

    async def child(messages, _info):
        child_models.append(messages)
        results = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        if results:
            assert [(part.tool_call_id, part.content) for part in results] == [("child-approval", "checked")]
            return ModelResponse(parts=[TextPart("child complete")])
        return ModelResponse(parts=[ToolCallPart("inspect_fixture", {}, tool_call_id="child-approval")])

    async def parent(messages, _info):
        parent_models.append(messages)
        results = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        if results:
            [result] = results
            assert result.tool_call_id == "delegate" and result.content["status"] == "SUCCEEDED"
            returned_ids.append(result.content["execution_id"])
            return ModelResponse(parts=[TextPart("parent complete")])
        return ModelResponse(parts=[ToolCallPart(
            "delegate_task", {"subagent_id": "child", "task": "Inspect test fixture"}, tool_call_id="delegate",
        )])

    features = RuntimeFeatures("native-child-approval")
    features.model("parent", _model("parent", parent))
    features.model("child", _model("child", child))
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.capability(StepPersistenceCapability(), scope="runtime")
    features.tool(inspect_fixture, effect_policy="replay_safe")
    features.agent("default", model="parent", allow_subagents=("child",), allow_tools=(), allow_skills=())
    features.agent("child", model="child", allow_subagents=(), allow_tools=("inspect_fixture",), allow_skills=())
    storage = PersistenceStorage.from_root(tmp_path)
    async with (
        Persistence.open("native-child-approval", storage=storage) as state,
        Runtime.open(persistence=state, features=features, authorization=_HostAuthorization(), auto_recover=False) as runtime,
    ):
        actor = Principal("owner", runtime.tenant_id)
        await runtime.sessions.create_turn_session("chat", principal=actor)
        turn = await runtime.agents.get().start(
            "Delegate once", session_id="chat", principal=actor, idempotency_key="once",
        )

        async def child_parked():
            children = await state.execution.executions.list_children(turn.key.turn_id, tenant_id=runtime.tenant_id)
            assert len(children) <= 1
            return children[0] if children else None

        parked = await _wait_for_committed(
            child_parked, lambda row: row is not None and row.status is ExecutionStatus.WAITING_DEFERRED,
        )
        child_id = parked.execution_id
        assert child_entries == [child_id] and not child_effects
        assert parked.root_execution_id == parked.parent_execution_id == turn.key.turn_id
        [pending] = await runtime.approvals.list(child_id, principal=actor)
        assert pending.status is ApprovalStatus.PENDING and pending.tool_name == "inspect_fixture"

        outsider = Principal("other-same-tenant-user", runtime.tenant_id)
        with pytest.raises(AIError) as denied:
            await runtime.approvals.list(child_id, principal=outsider)
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        with pytest.raises(AIError) as denied:
            await runtime.approvals.decide(
                child_id, ApprovalDecisionRequest(outsider, pending.approval_id, "other", ApprovalDecision.APPROVE),
            )
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED

        if outcome == "unconfirmed-child-resume":
            claim = NativeSubagentAdmission.guard_deferred_resume

            async def unconfirmed(proof, child_store, write):
                await claim(proof, child_store, write)  # Real fenced checkpoint transaction commits.
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

            monkeypatch.setattr(NativeSubagentAdmission, "guard_deferred_resume", unconfirmed)

        request = ApprovalDecisionRequest(actor, pending.approval_id, "approve-once", ApprovalDecision.APPROVE)
        if outcome == "unconfirmed-child-resume":
            with pytest.raises(AIError) as error:
                await runtime.approvals.decide(child_id, request)
            assert error.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
            current = await child_parked()
            assert current.execution_id == child_id and current.status is ExecutionStatus.STARTED
            assert current.agent_run_seq == parked.agent_run_seq + 1
            assert len(parent_models) == len(child_models) == 1
            assert child_entries == [child_id] and not child_effects and not returned_ids
            await runtime.handoff()
        elif outcome == "parent-stop":
            before_resume, continue_resume = asyncio.Event(), asyncio.Event()
            resume = LocalExecutionBackend._run_owned_action

            async def pause_native_resume(backend, execution, generation, action, *, native_parent=None):
                if native_parent is not None and execution.execution_id == child_id:
                    before_resume.set()
                    await continue_resume.wait()
                return await resume(backend, execution, generation, action, native_parent=native_parent)

            monkeypatch.setattr(LocalExecutionBackend, "_run_owned_action", pause_native_resume)
            approval_task = asyncio.create_task(runtime.approvals.decide(child_id, request))
            phase = "wait for approved child resume"
            try:
                await asyncio.wait_for(before_resume.wait(), timeout=10)
                phase = "request parent Stop"
                assert await turn.cancel()
                phase = "wait for cancelled parent"
                assert (await turn.wait_final(timeout_seconds=10)).status is ExecutionStatus.CANCELLED
            except (AIError, TimeoutError) as error:
                detail = error.code.value if isinstance(error, AIError) else type(error).__name__
                parent = await state.session_turns.get_turn(turn.key)
                current = await child_parked()
                parent_status = None if parent is None else (parent["state"], parent["generation"])
                child_status = None if current is None else (current.status.value, current.producer_generation)
                raise AssertionError(f"{phase}: {detail} (parent={parent_status}, child={child_status})") from error
            finally:
                continue_resume.set()
                [decided] = await asyncio.gather(approval_task, return_exceptions=True)
            if isinstance(decided, BaseException):
                detail = decided.code.value if isinstance(decided, AIError) else type(decided).__name__
                raise AssertionError(f"finish competing child approval: {detail}") from decided
            assert decided.approval_id == pending.approval_id and decided.decision is ApprovalDecision.APPROVE
            assert len(parent_models) == len(child_models) == 1
            assert child_entries == [child_id] and not child_effects and not returned_ids
            current = await child_parked()
            assert current.status is ExecutionStatus.CANCELLED
            assert current.agent_run_seq == parked.agent_run_seq
        else:
            decided = await runtime.approvals.decide(child_id, request)
            assert decided.approval_id == pending.approval_id and decided.decision is ApprovalDecision.APPROVE
            assert (await turn.wait_final(timeout_seconds=10)).status is ExecutionStatus.SUCCEEDED
            current = await child_parked()
            assert current.execution_id == child_id and current.status is ExecutionStatus.SUCCEEDED
            assert current.agent_run_seq == parked.agent_run_seq + 1
            assert child_entries == [child_id, child_id] and child_effects == returned_ids == [child_id]
            assert len(parent_models) == len(child_models) == 2

        approval = await state.checkpoint.approvals.get(pending.approval_id, tenant_id=runtime.tenant_id)
        assert approval.status is ApprovalStatus.APPROVED and approval.execution_id == child_id
        assert await state.execution.executions.get(turn.key.turn_id, tenant_id=runtime.tenant_id) is None
