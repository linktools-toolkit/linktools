#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A configured split Step route retains the canonical conversation transcript."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import StepPersistenceCapability
from linktools.ai.core import Principal, TenantAuthorizationPolicy
from linktools.ai.history import History
from linktools.ai.persistence import Persistence, PersistencePlan, PersistenceStorage
from linktools.ai.persistence.turns import SessionTurnKey
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import StorageRoute

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


class _SessionPolicy(TenantAuthorizationPolicy):
    """Explicit host policy for this fixture's session."""


@pytest.mark.asyncio
async def test_split_step_text_turn_reopens_through_canonical_history(tmp_path: Path) -> None:
    namespace = "split-text"
    principal = Principal("reader", "default")
    policy = _SessionPolicy(principal.tenant_id)
    conversation = StorageRoute.sqlite(tmp_path / "conversation.db")
    step = StorageRoute.sqlite(tmp_path / "step.db")
    storage = PersistenceStorage(PersistencePlan(conversation=conversation, execution=step, checkpoint=step))
    model_calls = 0

    async def reply(_messages, _info):
        nonlocal model_calls
        model_calls += 1
        return ModelResponse(parts=[TextPart("retained in the source database")])

    features = RuntimeFeatures(namespace)
    features.capability(StepPersistenceCapability())
    features.agent("default", model="default", allow_tools=())
    async with Persistence.open(namespace, storage=storage) as state:
        async with Runtime.open(
            persistence=state, features=model_features(_Models(_UsageFunctionModel(reply)), features),
            authorization=policy,
        ) as runtime:
            await runtime.sessions.create_turn_session("chat", principal=principal)
            frames = asyncio.Queue()
            detach = runtime.sessions.observe_turn_session("chat", frames.put_nowait)
            try:
                turn_id = await runtime.sessions.start_turn(
                    "chat", "question", agent="default", principal=principal, idempotency_key="first",
                )
                async with asyncio.timeout(10):
                    while (frame := await frames.get())["type"] != "terminal":
                        pass
                assert frame["turn_id"] == turn_id and frame["state"] == "succeeded", frame
            finally:
                detach()

            committed = await state.session_turns.get_turn(SessionTurnKey("chat", turn_id))
            assert committed is not None and committed["split_source_receipt"] is not None
            assert await state.execution.executions.get(turn_id, tenant_id=principal.tenant_id) is None

    async with Persistence.open(namespace, storage=storage, read_only=True) as state:
        async with History.open(persistence=state, authorization=policy) as history:
            detail = await history.turn_result("chat", turn_id, principal=principal)
            assert detail["turn_id"] == turn_id and detail["conversation_committed"]
            assert detail["output"] == {"text": "retained in the source database"}
            page = await history.turn_session("chat", principal=principal, limit=1)
            assert page["timeline"]["items"] == [detail]
        assert await state.execution.executions.get(turn_id, tenant_id=principal.tenant_id) is None
    assert model_calls == 1
