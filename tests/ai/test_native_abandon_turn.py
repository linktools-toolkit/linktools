#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""An explicit abandon releases only an expired exact native session occupant."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, UserPromptPart

from linktools.ai.capability import StepPersistenceCapability
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, TenantAuthorizationPolicy, ToolOperationStatus
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.contracts import StoredPayload, ToolOperationRecord
from linktools.ai.persistence.turns import SessionTurnKey
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._regenerable_turn import _RegenerableTurnRunner

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features, _wait_for_committed
from .test_history_projection_conformance import _record


class _HostPolicy(TenantAuthorizationPolicy):
    """Enforce ownership on explicit canonical controls."""


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("no-child", "cancel-race", "child-present"))
async def test_expired_ordinary_turn_can_be_released_without_replay(tmp_path, monkeypatch, boundary):
    prompts = []
    stopped = asyncio.Event()
    original = _RegenerableTurnRunner._settle_with_lease

    async def crash_before_dispatch(self, turn, **kwargs):
        if turn.prompt == "discard this prompt":
            stopped.set()
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        return await original(self, turn, **kwargs)

    async def model(messages, _info):
        prompts.extend(part.content for message in messages for part in message.parts if isinstance(part, UserPromptPart))
        return ModelResponse(parts=[TextPart("retained answer")])

    features = model_features(_Models(_UsageFunctionModel(model)), RuntimeFeatures("abandon"))
    features.capability(StepPersistenceCapability())
    features.agent("default", model="default", allow_subagents=("child",), allow_tools=(), allow_skills=())
    features.agent("child", model="default", allow_subagents=(), allow_tools=(), allow_skills=())
    monkeypatch.setattr(_RegenerableTurnRunner, "_settle_with_lease", crash_before_dispatch)
    async with (
        Persistence.open("abandon", storage=PersistenceStorage.from_root(tmp_path)) as state,
        Runtime.open(persistence=state, features=features, authorization=_HostPolicy()) as runtime,
    ):
        principal = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=principal)
        turn_id = await runtime.sessions.start_turn(
            "session", "discard this prompt", agent="default", principal=principal, idempotency_key="old",
        )
        await asyncio.wait_for(stopped.wait(), timeout=5)
        key = SessionTurnKey("session", turn_id)
        turns = state.session_turns

        with pytest.raises(AIError) as denied:
            await runtime.sessions.abandon_turn("session", turn_id, expected_generation=1, principal=replace(principal, principal_id="other"))
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        before = await turns.get_session("session")
        assert not await runtime.sessions.abandon_turn("session", turn_id, expected_generation=1, principal=principal)
        assert (await turns.get_turn(key))["state"] == "accepted"
        assert await turns.get_session("session") == before

        async with turns._context.engine.begin() as connection:
            changed = await connection.execute(
                turns._turns.update().filter_by(**turns._turn_values(key)).values(lease_until=0),
            )
            assert changed.rowcount == 1

        if boundary == "cancel-race":
            original_finish = turns.finish

            async def cancel_before_abandon(*args, **kwargs):
                assert await turns.request_cancel(key)
                return await original_finish(*args, **kwargs)

            monkeypatch.setattr(turns, "finish", cancel_before_abandon)
            assert not await runtime.sessions.abandon_turn("session", turn_id, expected_generation=1, principal=principal)
            assert (await turns.get_turn(key))["state"] == "cancelling"
            assert await turns.get_session("session") == before
            assert prompts == []
            return

        now = datetime.now(timezone.utc)
        args = StoredPayload.inline_json({"subagent_id": "child", "task": "verify a fixture"})
        unknown = ToolOperationRecord(
            "original-call", turn_id, turn_id, "delegate-call", "delegate_task",
            args.digest, "binding", False, ToolOperationStatus.EFFECT_UNKNOWN, "previous-owner", 1,
            ErrorCode.TOOL_EFFECT_UNKNOWN.value, now, now, producer_generation=1, arguments_payload=args,
        )
        await state.checkpoint.tools.reserve(unknown)
        if boundary == "child-present":
            # A real indexed child Execution models an original reservation
            # that committed before the parent lost its acknowledgement.
            repository = state.execution.executions
            child = replace(
                _record(ExecutionStatus.PENDING_START, 0),
                execution_id="original-child", parent_execution_id=turn_id, root_execution_id=turn_id,
                parent_invocation_id=unknown.tool_call_id, lineage_kind=ExecutionLineageKind.SUBAGENT,
                principal_id=principal.principal_id, principal_kind=principal.kind,
            )
            await repository.create_execution(child)
            previous = await turns.get_turn(key)
            with pytest.raises(AIError) as blocked:
                await runtime.sessions.abandon_turn("session", turn_id, expected_generation=1, principal=principal)
            assert blocked.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
            assert blocked.value.safe_details["reason"] == "abandon_child_disposition_required"
            assert await turns.get_turn(key) == previous
            assert await turns.get_session("session") == before
            assert await repository.get(child.execution_id, tenant_id=principal.tenant_id) == child
            assert await state.checkpoint.tools.get_operation(
                unknown.tool_operation_id, tenant_id=principal.tenant_id,
            ) == unknown
            assert prompts == []
            return
        assert not await runtime.sessions.abandon_turn("session", turn_id, expected_generation=2, principal=principal)
        assert (await turns.get_turn(key))["generation"] == 1
        pytest.importorskip("starlette")
        from linktools.ai.web import create_app
        from .test_web_console import _HEADERS, client

        async with client(create_app(runtime=runtime)) as http:
            released = await http.post("/api/session/abandon?session_id=session", headers=_HEADERS, json={
                "turn_id": turn_id, "expected_generation": 1, "request_id": "explicit-abandon",
            })
            assert released.status_code == 200 and released.json() == {"accepted": True}
            shown = await http.get("/api/session", params={"session_id": "session", "turn_id": turn_id})
            assert shown.status_code == 200 and shown.json()["abandoned"] is True
        assert await state.checkpoint.tools.get_operation(
            unknown.tool_operation_id, tenant_id=principal.tenant_id,
        ) == unknown
        assert not await runtime.sessions.abandon_turn("session", turn_id, expected_generation=1, principal=principal)
        old = await turns.get_turn(key)
        assert (old["state"], old["generation"], old["result_json"]) == (
            "failed", 1, {"error_code": ErrorCode.STORAGE_RECOVERY_REQUIRED.value, "abandoned": True},
        )
        assert await state.execution.executions.list_children(turn_id, tenant_id=principal.tenant_id) == ()
        assert (await turns.get_session("session"))["active_turn_id"] is None

        next_id = await runtime.sessions.start_turn(
            "session", "new prompt", agent="default", principal=principal, idempotency_key="new",
        )
        new_key = SessionTurnKey("session", next_id)
        row = await _wait_for_committed(lambda: turns.get_turn(new_key), lambda value: value["state"] != "accepted")
        assert row["state"] == "succeeded"
        assert prompts == ["new prompt"]
        assert (await turns.get_turn(key)) == old
