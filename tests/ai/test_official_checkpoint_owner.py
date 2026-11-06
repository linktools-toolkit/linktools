#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recovery reads the same official snapshots that the native capability wrote."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot, InMemoryStepStore, RunRecord, SqliteStepStore, StepPersistence, continue_run,
)

from linktools.ai.runtime._step_store import RuntimeStepStore

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence.runs import (
    AgentRunCheckpoint, AgentRunRecord, InMemoryStepArchive, OfficialStepReader,
    RuntimeAgentRunStore, StagingAgentRunStore,
)


def _runtime_store(native=None) -> RuntimeAgentRunStore:
    store = RuntimeAgentRunStore(
        StagingAgentRunStore(),
        conversation_archive=InMemoryStepArchive(PersistenceDomain.CONVERSATION),
        execution_archive=None,
    )
    if native is not None:
        store.bind_step_store(native)
    return store


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_native_checkpoint_is_the_only_recovery_source(tmp_path: Path, backend: str) -> None:
    native = InMemoryStepStore() if backend == "memory" else SqliteStepStore(database=tmp_path / "steps.db")
    initial = [ModelRequest(parts=[UserPromptPart("earlier")]), ModelResponse(parts=[TextPart("prior answer")])]
    agent = Agent(
        FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart("native answer")])),
        capabilities=[StepPersistence(store=native, run_id="attempt", metadata={"linktools.transcript_start": "2"})],
    )
    result = await agent.run("new question", message_history=initial, run_id="attempt")
    store = _runtime_store(native)
    await store.initialize()
    store.bind_execution_producer("attempt", execution_id="execution", producer_generation=3)
    reader = store.read_store(PersistenceDomain.CHECKPOINT)
    assert reader is store.read_store(PersistenceDomain.CHECKPOINT, durable=False)
    checkpoint = await store.latest_checkpoint(agent_run_id="attempt")
    assert checkpoint is not None and checkpoint.context_messages == result.all_messages()
    assert checkpoint.messages == result.new_messages()
    await store.register_agent_run(await reader.get_agent_run(agent_run_id="attempt"))
    await store.save_checkpoint(AgentRunCheckpoint(
        agent_run_id="attempt", step_index=999, messages=[ModelResponse(parts=[TextPart("uncommitted mirror")])],
    ))
    assert await store.latest_checkpoint(agent_run_id="attempt") == checkpoint
    before = await native.latest_snapshot(run_id="attempt", include_interrupted=True)
    await store.checkpoint_persistence.materialize_recovery_checkpoint(agent_run_id="attempt", require_complete=True)
    assert await native.latest_snapshot(run_id="attempt", include_interrupted=True) == before
    payload = await store.capture_agent_run_checkpoint_payload(
        "attempt", execution_id="execution", producer_generation=3,
    )
    assert json.loads(payload)["snapshot_key"] == checkpoint.idempotency_key
    assert b"native answer" not in payload and b"earlier" not in payload
    run, restored = await store.decode_agent_run_checkpoint_payload(payload, agent_run_id="attempt", producer_generation=3)
    assert run.agent_run_id == "attempt" and restored == replace(checkpoint, messages=[])
    context = await store.load_loaded_model_context(PersistenceDomain.CHECKPOINT, "attempt")
    assert list(context.model_messages()) == result.all_messages()
    await store.preflight_close()
    await store.close()


@pytest.mark.asyncio
async def test_cold_checkpoint_retains_settled_return_without_repeating_tool(tmp_path: Path) -> None:
    database = tmp_path / "cold.db"
    effects = []

    def failing_model(messages, _info):
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            raise RuntimeError("interrupted after settled effect")
        return ModelResponse(parts=[ToolCallPart("effect", {}, tool_call_id="effect-1")])

    native = SqliteStepStore(database=database)
    agent = Agent(FunctionModel(failing_model), capabilities=[StepPersistence(
        store=native, run_id="source", capture_frontier=True,
        metadata={"linktools.transcript_start": "0"},
    )])

    @agent.tool_plain
    def effect() -> str:
        effects.append("performed")
        return "saved tool return"

    with pytest.raises(RuntimeError, match="interrupted after settled effect"):
        await agent.run("question", run_id="source")
    cold = SqliteStepStore(database=database)
    reader = OfficialStepReader(cold)
    checkpoint = await reader.latest_checkpoint(agent_run_id="source", include_interrupted=True)
    assert checkpoint is not None
    assert [part.content for message in checkpoint.messages for part in message.parts
            if isinstance(part, ToolReturnPart)] == ["saved tool return"]
    context = await reader.load_loaded_model_context(owner_id="source")
    assert list(context.model_messages()) == await continue_run(cold, run_id="source", include_interrupted=True)
    resumed = Agent(
        FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart("recovered")])),
        capabilities=[StepPersistence(store=cold, run_id="resumed", metadata={
            "linktools.transcript_start": str(len(context.messages)),
        })],
    )
    result = await resumed.run(None, message_history=list(context.model_messages()), run_id="resumed")
    assert result.output == "recovered" and effects == ["performed"]
    assert {run.run_id for run in await cold.list_runs()} == {"source", "resumed"}


@pytest.mark.asyncio
async def test_deferred_checkpoint_keeps_unsettled_native_tool_frontier(tmp_path: Path) -> None:
    native = SqliteStepStore(database=tmp_path / "deferred.db")
    agent = Agent(
        FunctionModel(lambda _messages, _info: ModelResponse(parts=[ToolCallPart("approved", {}, tool_call_id="approval")])),
        output_type=[str, DeferredToolRequests],
        capabilities=[StepPersistence(store=native, run_id="deferred", capture_frontier=True,
                                      metadata={"linktools.transcript_start": "0"})],
    )

    @agent.tool_plain(requires_approval=True)
    def approved() -> str:
        raise AssertionError("approval was not granted")

    result = await agent.run("question", run_id="deferred")
    assert isinstance(result.output, DeferredToolRequests)
    cold = SqliteStepStore(database=tmp_path / "deferred.db")
    reader = OfficialStepReader(cold)
    checkpoint = await reader.latest_checkpoint(agent_run_id="deferred", include_interrupted=True)
    assert checkpoint is not None and checkpoint.state == "interrupted"
    assert any(isinstance(part, ToolCallPart) and part.tool_call_id == "approval"
               for message in checkpoint.context_messages for part in message.parts)
    store = _runtime_store(cold)
    await store.checkpoint_persistence.materialize_recovery_checkpoint(agent_run_id="deferred", require_complete=False)
    with pytest.raises(AIError) as invalid:
        await store.checkpoint_persistence.materialize_recovery_checkpoint(agent_run_id="deferred", require_complete=True)
    assert invalid.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_checkpoint_disabled_buffer_belongs_to_trace_and_conversation() -> None:
    store = _runtime_store()
    await store.initialize()
    store.bind_execution_producer("attempt", execution_id="execution", producer_generation=1,
                                  restart_checkpoints=False, durable_trace=False)
    await store.register_agent_run(AgentRunRecord(agent_run_id="attempt"))
    checkpoint = AgentRunCheckpoint("attempt", 1, [ModelResponse(parts=[TextPart("conversation result")])])
    await store.save_checkpoint(checkpoint)
    payload = await store.capture_agent_run_checkpoint_payload(
        "attempt", execution_id="execution", producer_generation=1,
    )
    cold = _runtime_store()
    cold.bind_execution_persistence("attempt", conversation=True, restart_checkpoints=False, durable_trace=False)
    assert cold.execution_producer_generation("attempt", execution_id="execution") is None
    _, restored = await cold.decode_agent_run_checkpoint_payload(payload, agent_run_id="attempt", producer_generation=1)
    assert restored is not None and restored.messages == checkpoint.messages
    await store.checkpoint_persistence.materialize_recovery_checkpoint(agent_run_id="attempt", require_complete=True)
    await store.preflight_close()
    await store.close()


@pytest.mark.asyncio
async def test_failure_before_native_run_has_no_fabricated_snapshot_reference() -> None:
    native = InMemoryStepStore()
    store = _runtime_store(native)
    await store.initialize()
    store.bind_execution_producer("attempt", execution_id="execution", producer_generation=1)
    await store.register_agent_run(AgentRunRecord(agent_run_id="attempt"))
    assert await store.capture_agent_run_checkpoint_payload(
        "attempt", execution_id="execution", producer_generation=1, include_interrupted=True,
    ) is None
    await store.checkpoint_persistence.materialize_recovery_checkpoint(agent_run_id="attempt", require_complete=False)
    assert await native.get_run(run_id="attempt") is None
    with pytest.raises(AIError) as missing:
        await store.checkpoint_persistence.materialize_recovery_checkpoint(agent_run_id="attempt", require_complete=True)
    assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    await store.preflight_close()
    await store.close()


@pytest.mark.asyncio
async def test_snapshot_reference_rejects_a_newer_frontier_instead_of_substituting_it() -> None:
    native = InMemoryStepStore()
    model_entered, model_release = asyncio.Event(), asyncio.Event()

    async def model(messages, _info):
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            model_entered.set()
            await model_release.wait()
            return ModelResponse(parts=[TextPart("completed")])
        return ModelResponse(parts=[ToolCallPart("read", {}, tool_call_id="read-1")])

    agent = Agent(FunctionModel(model), capabilities=[StepPersistence(
        store=native, run_id="attempt", metadata={"linktools.transcript_start": "0"},
    )])

    @agent.tool_plain
    def read() -> str:
        return "tool result"

    store = _runtime_store(native)
    await store.initialize()
    store.bind_execution_producer("attempt", execution_id="execution", producer_generation=1)
    task = asyncio.create_task(agent.run("question", run_id="attempt"))
    try:
        await asyncio.wait_for(model_entered.wait(), timeout=5)
        payload = await store.capture_agent_run_checkpoint_payload(
            "attempt", execution_id="execution", producer_generation=1,
        )
        _, captured = await store.decode_agent_run_checkpoint_payload(payload, agent_run_id="attempt", producer_generation=1)
        assert captured is not None
        model_release.set()
        assert (await task).output == "completed"
        with pytest.raises(AIError) as changed:
            await store.decode_agent_run_checkpoint_payload(payload, agent_run_id="attempt", producer_generation=1)
        assert changed.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        model_release.set()
        await task
        await store.preflight_close()
        await store.close()


class _ObservedStepStore(InMemoryStepStore):
    def __init__(self) -> None:
        super().__init__()
        self.reads = 0
        self.saved: list[ContinuableSnapshot] = []
        self.failure: str | None = None

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        self.reads += 1
        return await super().get_run(run_id=run_id)

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        self.reads += 1
        return await super().latest_snapshot(run_id=run_id, include_interrupted=include_interrupted)

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        if self.failure == "before_commit":
            raise OSError("source unavailable")
        await super().save_snapshot(snapshot)
        self.saved.append(snapshot)
        if self.failure == "after_commit":
            raise OSError("acknowledgment lost")


@pytest.mark.asyncio
async def test_confirmed_native_frontier_is_reused_until_owned_run_release() -> None:
    native = _ObservedStepStore()
    store = _runtime_store(native)
    await store.initialize()
    store.bind_execution_producer("attempt", execution_id="execution", producer_generation=1)
    native_view = RuntimeStepStore(native, native, working_set=store)
    agent = Agent(
        FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart("answer")])),
        capabilities=[StepPersistence(store=native_view, run_id="attempt", capture_frontier=True,
                                      metadata={"linktools.transcript_start": "0"})],
    )
    result = await agent.run("question", run_id="attempt")
    native.reads = 0
    checkpoint = await store.latest_checkpoint(agent_run_id="attempt")
    assert checkpoint is not None and checkpoint.context_messages == result.all_messages()
    payload = await store.capture_agent_run_checkpoint_payload(
        "attempt", execution_id="execution", producer_generation=1,
    )
    _, restored = await store.decode_agent_run_checkpoint_payload(payload, agent_run_id="attempt", producer_generation=1)
    assert restored == replace(checkpoint, messages=[])
    await store.checkpoint_persistence.materialize_recovery_checkpoint(agent_run_id="attempt", require_complete=True)
    await store.verify_terminal_attempts(candidate_agent_run_ids=("attempt",), required_agent_run_id="attempt")
    assert list((await store.load_loaded_model_context(PersistenceDomain.CHECKPOINT, "attempt")).model_messages()) == result.all_messages()
    await native_view.save_snapshot(native.saved[0])
    assert await store.latest_checkpoint(agent_run_id="attempt") == checkpoint
    assert native.reads == 0
    await store.release_staging_many(candidate_agent_run_ids=("attempt",), execution_id="execution")
    store.confirm_step_snapshot(native.saved[-1])
    assert await store.read_store(PersistenceDomain.CHECKPOINT).latest_checkpoint(agent_run_id="attempt") == checkpoint
    assert native.reads > 0
    await store.preflight_close()
    await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("before_commit", "after_commit"))
async def test_unacknowledged_snapshot_invalidates_the_owned_view(failure: str) -> None:
    native = _ObservedStepStore()
    store = _runtime_store(native)
    await store.initialize()
    store.bind_execution_producer("attempt", execution_id="execution", producer_generation=1)
    native_view = RuntimeStepStore(native, native, working_set=store)
    agent = Agent(
        FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart("answer")])),
        capabilities=[StepPersistence(store=native_view, run_id="attempt",
                                      metadata={"linktools.transcript_start": "0"})],
    )
    await agent.run("question", run_id="attempt")
    previous = native.saved[-1]
    candidate = replace(previous, messages=[ModelResponse(parts=[TextPart("next frontier")], run_id="attempt")],
                        idempotency_key="100:2:complete")
    native.failure = failure
    with pytest.raises(OSError):
        await native_view.save_snapshot(candidate)
    native.failure = None
    native.reads = 0
    observed = await store.latest_checkpoint(agent_run_id="attempt")
    assert observed.context_messages == (previous.messages if failure == "before_commit" else candidate.messages)
    assert native.reads > 0
    await native_view.save_snapshot(candidate)
    native.reads = 0
    assert (await store.latest_checkpoint(agent_run_id="attempt")).context_messages == candidate.messages
    assert native.reads > 0
    await store.release_staging_many(candidate_agent_run_ids=("attempt",), execution_id="execution")
    await store.preflight_close()
    await store.close()


@pytest.mark.asyncio
async def test_sticky_history_snapshot_is_context_not_an_untrimmed_transcript() -> None:
    from pydantic_ai.capabilities import ProcessHistory

    native = InMemoryStepStore()
    initial = [ModelRequest(parts=[UserPromptPart("earlier")]), ModelResponse(parts=[TextPart("prior")])]
    agent = Agent(
        FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart("answer")])),
        capabilities=[ProcessHistory(lambda messages: messages[-1:]), StepPersistence(
            store=native, run_id="trimmed", capture_frontier=True,
            metadata={"linktools.transcript_start": "2"},
        )],
    )
    result = await agent.run("question", message_history=initial, run_id="trimmed")
    snapshot = await native.latest_snapshot(run_id="trimmed")
    assert snapshot.messages == result.all_messages()
    assert len(snapshot.messages) == 2 and "earlier" not in str(snapshot.messages)
    reader = OfficialStepReader(native)
    # Runtime's view is native currently visible history, not a reconstructed
    # immutable transcript of messages deliberately deleted by the processor.
    checkpoint = await reader.latest_checkpoint(agent_run_id="trimmed")
    assert checkpoint.messages == snapshot.messages
    context = await reader.load_loaded_model_context(owner_id="trimmed")
    assert list(context.model_messages()) == await continue_run(native, run_id="trimmed")
    continued = await Agent(FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart("continued")]))).run(
        "next", message_history=list(context.model_messages()),
    )
    assert continued.output == "continued"


@pytest.mark.asyncio
async def test_compaction_snapshot_retains_sdk_history_and_continues_with_projection() -> None:
    from linktools.ai.core import PromptLimits
    from linktools.ai.runtime._compaction import CompactionCapability
    from linktools.ai.runtime._harness import SummaryProviderModel
    from .test_compaction_summary_identity import _history, _summaries

    native = InMemoryStepStore()
    history = _history()
    requests = []

    async def respond(messages, _info):
        if "context summarization assistant" in str(messages):
            return ModelResponse(parts=[TextPart("trusted-summary")])
        requests.append(messages)
        return ModelResponse(parts=[TextPart("done")])

    model = SummaryProviderModel(FunctionModel(respond))
    agent = Agent(model, capabilities=[CompactionCapability(1, limits=PromptLimits()), StepPersistence(
        store=native, run_id="compacted", metadata={"linktools.transcript_start": str(len(history))},
    )])
    result = await agent.run("continue", message_history=history, run_id="compacted")
    snapshot = await native.latest_snapshot(run_id="compacted")
    assert snapshot.messages == result.all_messages()
    assert not _summaries(snapshot.messages)
    assert "trusted-summary" in str(requests[-1]) and len(requests[-1]) < len(snapshot.messages)
    assert (await OfficialStepReader(native).latest_checkpoint(agent_run_id="compacted")).messages == result.new_messages()
    continuation = await continue_run(native, run_id="compacted")
    resumed = Agent(model, capabilities=[CompactionCapability(1, limits=PromptLimits())])
    assert (await resumed.run("next", message_history=continuation)).output == "done"
    assert "trusted-summary" in str(requests[-1])


@pytest.mark.asyncio
async def test_repeated_conversation_publication_does_not_reuse_native_snapshot_key() -> None:
    from linktools.ai.persistence import Persistence, PersistenceStorage

    native = InMemoryStepStore()
    agent = Agent(FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart("answer")])),
                  capabilities=[StepPersistence(store=native, run_id="conversation-repeat", metadata={
                      "linktools.transcript_start": "0", "history_id": "history",
                  })])
    await agent.run("question", run_id="conversation-repeat")
    async with Persistence.open("conversation-key", storage=PersistenceStorage.in_memory()) as persistence:
        store = persistence.run_store
        store.bind_step_store(native)
        store.bind_execution_persistence("conversation-repeat", conversation=True, restart_checkpoints=True, durable_trace=False)
        reader = store.read_store(PersistenceDomain.CHECKPOINT)
        checkpoint = await reader.latest_checkpoint(agent_run_id="conversation-repeat")
        assert checkpoint.idempotency_key is not None
        await store.register_agent_run(await reader.get_agent_run(agent_run_id="conversation-repeat"))
        await store.save_checkpoint(checkpoint)
        for _ in range(2):
            await store.materialize_conversation(agent_run_id="conversation-repeat")
            await store.materialize_from_recovery(target=PersistenceDomain.CONVERSATION, agent_run_id="conversation-repeat")
        conversation = await store.read_store(PersistenceDomain.CONVERSATION).latest_checkpoint(agent_run_id="conversation-repeat")
        assert conversation.messages == checkpoint.messages and conversation.idempotency_key is None
        assert (await reader.latest_checkpoint(agent_run_id="conversation-repeat")).idempotency_key == checkpoint.idempotency_key
