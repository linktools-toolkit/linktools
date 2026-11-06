#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent recovery effect identity verification."""

import asyncio

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.tools import ToolDefinition
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord, ToolEffectRecord
from linktools.ai.persistence import Persistence
from linktools.ai.runtime._step_store import RuntimeStepStore
from .test_native_tool_effect_owner import _bridge, state
from ._runtime_test_helpers import tool_run_context
from .test_official_checkpoint_owner import _ObservedStepStore, _runtime_store


@pytest.mark.asyncio
async def test_new_business_call_after_recovery_uses_its_observed_operation(state: Persistence) -> None:
    bridge = _bridge(state, run_id="resumed", source_run_id="run")
    store = RuntimeStepStore(_ObservedStepStore(), bridge)
    await store.record_tool_effect(ToolEffectRecord(
        run_id="resumed", tool_call_id="new-call", tool_name="create", status="started",
    ))
    decision = await bridge.begin(
        tool_run_context(), ToolCallPart("create", {}, tool_call_id="new-call"),
        ToolDefinition(name="create"), {}, False,
    )
    assert decision is not None
    await bridge.complete(decision, {"created": True})
    effect = await bridge.get_tool_effect(run_id="resumed", tool_call_id="new-call")
    assert effect is not None and effect.status == "completed"



@pytest.mark.asyncio
@pytest.mark.parametrize("invalidation", ["cancel_after_commit", "new_generation"])
async def test_invalidated_native_frontier_returns_to_its_source(
    invalidation: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _ObservedStepStore()
    store = _runtime_store(native)
    await store.initialize()
    store.bind_execution_producer("attempt", execution_id="execution", producer_generation=1)
    view = RuntimeStepStore(native, native, working_set=store)
    await view.register_run(RunRecord(run_id="attempt", metadata={"linktools.transcript_start": "0"}))
    first = ContinuableSnapshot(
        run_id="attempt", step_index=1, messages=[ModelResponse(parts=[TextPart("first")], run_id="attempt")],
        idempotency_key="1:1:complete",
    )
    second = ContinuableSnapshot(
        run_id="attempt", step_index=2, messages=[ModelResponse(parts=[TextPart("second")], run_id="attempt")],
        idempotency_key="2:1:complete",
    )
    await view.save_snapshot(first)
    prior = await store.latest_checkpoint(agent_run_id="attempt")
    assert prior is not None and prior.context_messages == first.messages
    source_reads = native.reads
    if invalidation == "cancel_after_commit":
        original = native.save_snapshot

        async def committed_then_cancelled(snapshot: ContinuableSnapshot) -> None:
            await original(snapshot)
            raise asyncio.CancelledError

        monkeypatch.setattr(native, "save_snapshot", committed_then_cancelled)
        with pytest.raises(asyncio.CancelledError):
            await view.save_snapshot(second)
    else:
        store.bind_execution_producer("attempt", execution_id="execution", producer_generation=2)
        await native.save_snapshot(second)
    observed = await store.latest_checkpoint(agent_run_id="attempt")
    assert observed is not None and observed.context_messages == second.messages
    assert observed.messages == second.messages
    assert native.reads > source_reads
    await store.release_staging_many(candidate_agent_run_ids=("attempt",), execution_id="execution")
    await store.preflight_close()
    await store.close()
