#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The split waiting handoff only grants its original durable source pin."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord, StepEvent

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain
from linktools.ai.persistence.turns import CanonicalDeferredFrontier, SessionTurnKey, SplitStepPin, SplitStepWaiting
from linktools.ai.runtime._canonical_split_step import _CanonicalSplitStep

from ._session_tool_test_helpers import _split_sqlite_storage
from .test_session_turn_sql import accept


@pytest.mark.asyncio
async def test_split_waiting_adopts_only_its_settled_source_once(tmp_path: Path) -> None:
    key = SessionTurnKey("chat", "turn")
    waiting = SplitStepWaiting("approval", "operation", "approval", 1)
    frontier = CanonicalDeferredFrontier.from_split_waiting(key, checkpoint_id="wait", waiting=waiting)
    pin = SplitStepPin(frontier.split_receipt_id, checkpoint_id=frontier.checkpoint_id, waiting=waiting)
    async with Persistence.open("split-wait", storage=_split_sqlite_storage(tmp_path / "state.db")) as state:
        turns = state.session_turns
        assert turns is not None
        source_store = state._stores[PersistenceDomain.EXECUTION]
        assert source_store.storage_group is not state._stores[PersistenceDomain.CONVERSATION].storage_group
        native = await state.create_step_store()
        try:
            owner = _CanonicalSplitStep(turns, source_store, native)
            source = owner._source
            await turns.create(key.session_id, config={})
            assert await accept(turns, key.turn_id)
            identity = owner._identity(key, 1)
            prepared = await native.prepare_step(
                run=RunRecord(run_id=key.turn_id, conversation_id=key.session_id, registration_id="register"),
                events=(StepEvent(run_id=key.turn_id, conversation_id=key.session_id, kind="run_started",
                                  step_index=0, idempotency_key="start"),),
            )
            await owner.commit_step(prepared, key=key, generation=1)
            prepared = await native.prepare_step(snapshot=ContinuableSnapshot(
                run_id=key.turn_id, conversation_id=key.session_id, step_index=1,
                state="interrupted", idempotency_key=frontier.checkpoint_id,
                messages=[ModelRequest(parts=[UserPromptPart("question")]),
                          ModelResponse(parts=[ToolCallPart("read", {}, "call")])],
            ))
            await owner.commit_step(prepared, key=key, generation=1)

            foreign_pin = replace(pin, receipt_id="foreign")
            with pytest.raises(AIError) as invalid:
                await owner._write(lambda db: source.seal_waiting(db, **identity, pin=foreign_pin))
            assert invalid.value.code is ErrorCode.STORAGE_CAPABILITY_MISSING
            assert await owner._write(lambda db: source.seal_waiting(db, **identity, pin=pin))
            parked = await owner._read_pin(identity)
            assert parked is not None and parked.phase == "waiting" and parked.pin == pin
            assert await turns.publish_split_waiting(frontier, source=parked, max_bytes=4096)
            assert await owner._write(lambda db: source.mark_waiting_ready(db, parked))
            ready = await owner._read_pin(identity)
            assert ready == replace(parked, phase="ready")
            assert await turns.publish_split_ready(frontier, source=ready)
            assert await turns.read_split_ready_frontier(key, source=ready) == frontier

            for invalid_source in (replace(ready, generation=2), replace(ready, pin=foreign_pin)):
                with pytest.raises(AIError) as refused:
                    await turns.take_over_split_ready(frontier, source=invalid_source, lease_seconds=60)
                assert refused.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
                assert not await owner._write(lambda db: source.adopt_ready(db, invalid_source))
            granted, _ = await turns.take_over_split_ready(frontier, source=ready, lease_seconds=60)
            assert granted
            assert await owner._write(lambda db: source.adopt_ready(db, ready))
            assert not await owner._write(lambda db: source.adopt_ready(db, ready))
            assert await owner._read_pin(identity) is None
            assert await owner._read_pin(owner._identity(key, 2)) == replace(ready, generation=2, phase="resuming")
        finally:
            await native.close()
