#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harness summarization must remain visible to Runtime request observability."""

import asyncio
from collections.abc import Sequence
from types import SimpleNamespace
from copy import deepcopy

import pytest
from linktools.ai.core import PromptLimits
from linktools.ai.runtime._compaction import (
    CompactionCapability,
    CompactionPolicy,
    _ObservedCompactionModel,
)
from linktools.ai.runtime._journal import ModelRequestFact, ModelRequestJournal
from linktools.ai.runtime._metric_capability import ModelObservationCapability
from linktools.ai.observe import Observation
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage


async def _provider_context(
    capability: CompactionCapability,
    ctx: RunContext[object],
    request_context: ModelRequestContext,
) -> ModelRequestContext:
    captured: list[ModelRequestContext] = []

    async def handler(current: ModelRequestContext) -> ModelResponse:
        captured.append(current)
        return ModelResponse(parts=[TextPart("done")])

    await capability.wrap_model_request(
        ctx,
        request_context=request_context,
        handler=handler,
    )
    assert len(captured) == 1
    return captured[0]


@pytest.mark.asyncio
async def test_harness_summary_request_uses_runtime_journal_and_observer() -> None:
    model = TestModel(custom_output_text="summary")
    ctx = RunContext(
        deps=None,
        model=model,
        usage=RunUsage(),
        run_id="run",
    )
    messages: list[ModelMessage] = []
    for index in range(15):
        messages.append(
            ModelRequest(parts=[UserPromptPart(f"user {index} " + "x" * 200)])
        )
        messages.append(
            ModelResponse(parts=[TextPart(f"assistant {index} " + "y" * 200)])
        )
    original_messages = deepcopy(messages)
    ctx.messages = list(messages)
    request_context = ModelRequestContext(
        model=model,
        messages=list(messages),
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    journal = ModelRequestJournal(
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        agent_run_id="run",
    )
    observed: list[tuple[str, ModelRequestFact, ModelResponse | None]] = []
    projections: list[
        tuple[tuple[ModelMessage, ...], tuple[ModelMessage, ...] | None]
    ] = []

    async def observer(
        _ctx: RunContext[object],
        fact: ModelRequestFact,
        phase: str,
        _model: object,
        response: ModelResponse | None,
        error: BaseException | None,
    ) -> None:
        assert error is None
        observed.append((phase, fact, response))

    def projection_sink(
        source: Sequence[ModelMessage],
        projected: Sequence[ModelMessage] | None,
    ) -> None:
        projections.append(
            (
                tuple(source),
                None if projected is None else tuple(projected),
            )
        )

    capability = CompactionCapability(
        1,
        limits=PromptLimits(),
        journal=journal,
        observer=observer,  # type: ignore[arg-type]
        projection_sink=projection_sink,
    )
    provider_context = await _provider_context(
        capability,
        ctx,  # type: ignore[arg-type]
        request_context,
    )

    assert [phase for phase, _, _ in observed] == ["started", "completed"]
    assert all(fact.purpose == "compaction" for _, fact, _ in observed)
    assert observed[-1][2] is not None
    assert projections
    assert projections[-1][0] == tuple(messages)
    assert projections[-1][1] is not None
    assert len(provider_context.messages) < len(messages)
    assert request_context.messages == original_messages
    assert ctx.messages == original_messages
    compaction_sequence = observed[-1][1].model_request_seq
    with pytest.raises(RuntimeError, match="missing"):
        journal.current(compaction_sequence)
    agent = journal.begin(3, purpose="agent")
    assert compaction_sequence == 1
    assert agent.model_request_seq == 2
    journal.finish(agent.model_request_seq, status="SUCCEEDED")
    journal.consume(agent.model_request_seq)


def test_journal_keeps_agent_and_compaction_requests_distinct_on_same_step() -> None:
    journal = ModelRequestJournal(
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        agent_run_id="run",
    )
    compaction = journal.begin(3, purpose="compaction")
    agent = journal.begin(3, purpose="agent")

    assert compaction.model_request_seq == 1
    assert agent.model_request_seq == 2
    assert agent.observation_id != compaction.observation_id
    journal.finish(compaction.model_request_seq, status="SUCCEEDED")
    assert journal.consume(compaction.model_request_seq).purpose == "compaction"
    assert journal.current(agent.model_request_seq) == agent
    journal.finish(agent.model_request_seq, status="SUCCEEDED")
    assert journal.consume(agent.model_request_seq).purpose == "agent"


def test_journal_rejects_double_finish() -> None:
    journal = ModelRequestJournal(
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        agent_run_id="run",
    )
    fact = journal.begin(1)
    journal.finish(fact.model_request_seq, status="SUCCEEDED")

    with pytest.raises(RuntimeError, match="already finished"):
        journal.finish(fact.model_request_seq, status="FAILED")

    journal.consume(fact.model_request_seq)


def _duplicate_file_history() -> list[ModelMessage]:
    return [
        ModelResponse(
            parts=[
                ToolCallPart(
                    "inspect_document",
                    {"path": "same.txt"},
                    tool_call_id="read-1",
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "inspect_document",
                    {"files": [], "texts": [{
                        "path": "same.txt", "hash": "a" * 64, "content": "same content",
                        "offset": 0, "end_offset": 12, "byte_offset": 0, "end_byte_offset": 12,
                        "line_offset": 0, "end_line_offset": 1, "lossy": False,
                    }]},
                    tool_call_id="read-1",
                )
            ]
        ),
        ModelResponse(
            parts=[
                ToolCallPart(
                    "inspect_document",
                    {"path": "same.txt"},
                    tool_call_id="read-2",
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "inspect_document",
                    {"files": [], "texts": [{
                        "path": "same.txt", "hash": "a" * 64, "content": "same content",
                        "offset": 0, "end_offset": 12, "byte_offset": 0, "end_byte_offset": 12,
                        "line_offset": 0, "end_line_offset": 1, "lossy": False,
                    }]},
                    tool_call_id="read-2",
                )
            ]
        ),
    ]


@pytest.mark.asyncio
async def test_compaction_target_does_not_rewrite_history_below_threshold() -> None:
    model = TestModel()
    ctx = RunContext(
        deps=None,
        model=model,
        usage=RunUsage(),
        run_id="run",
    )
    messages = _duplicate_file_history()
    request_context = ModelRequestContext(
        model=model,
        messages=list(messages),
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )

    provider_context = await _provider_context(
        CompactionCapability(
            1_000_000,
            limits=PromptLimits(),
            policy=CompactionPolicy(
                context_dedupe_by_tool={
                    "inspect_document": "workspace_file_read_v1",
                },
            ),
        ),
        ctx,  # type: ignore[arg-type]
        request_context,
    )

    assert provider_context.messages == messages
    assert request_context.messages == messages


@pytest.mark.asyncio
async def test_compaction_without_target_still_deduplicates_file_reads() -> None:
    model = TestModel()
    ctx = RunContext(
        deps=None,
        model=model,
        usage=RunUsage(),
        run_id="run",
    )
    messages = _duplicate_file_history()
    request_context = ModelRequestContext(
        model=model,
        messages=list(messages),
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )

    provider_context = await _provider_context(
        CompactionCapability(
            None,
            limits=PromptLimits(),
            policy=CompactionPolicy(
                context_dedupe_by_tool={
                    "inspect_document": "workspace_file_read_v1",
                },
            ),
        ),
        ctx,  # type: ignore[arg-type]
        request_context,
    )

    assert provider_context.messages != messages
    assert "[superseded file read]" in str(provider_context.messages)
    assert request_context.messages == messages


@pytest.mark.asyncio
async def test_compaction_keeps_semantic_control_results() -> None:
    model = TestModel()
    ctx = RunContext(
        deps=None,
        model=model,
        usage=RunUsage(),
        run_id="run",
    )
    messages: list[ModelMessage] = []
    for index in range(5):
        name = "hidden_control" if index == 0 else "ordinary"
        call_id = f"call-{index}"
        messages.extend(
            (
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            name,
                            {"value": index},
                            tool_call_id=call_id,
                        )
                    ]
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            name,
                            f"result-{index}",
                            tool_call_id=call_id,
                        )
                    ]
                ),
            )
        )
    request_context = ModelRequestContext(
        model=model,
        messages=messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    provider_context = await _provider_context(
        CompactionCapability(
            1,
            limits=PromptLimits(),
            policy=CompactionPolicy(
                keep_result_tools=frozenset({"hidden_control"}),
            ),
        ),
        ctx,  # type: ignore[arg-type]
        request_context,
    )

    returns = [
        part
        for message in provider_context.messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert returns[0].content == "result-0"
    assert returns[1].content == "[tool result cleared]"


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_during_observer", (False, True))
async def test_compaction_metric_is_interruptible_after_observer_handoff(
    cancel_during_observer: bool,
) -> None:
    observer_entered = asyncio.Event()
    observer_release = asyncio.Event()
    metric_entered = asyncio.Event()
    phases: list[str] = []
    metric_tasks: list[asyncio.Task[object] | None] = []

    class BlockingRecorder:
        async def try_record(self, observation: Observation) -> bool:
            assert observation.status == "SUCCEEDED"
            assert observation.correlation["linktools.request_purpose"] == "compaction"
            metric_tasks.append(asyncio.current_task())
            metric_entered.set()
            await asyncio.Event().wait()
            return True

    model = TestModel(custom_output_text="summary")
    ctx = RunContext(
        deps=SimpleNamespace(correlation={}), model=model, usage=RunUsage(),
        run_id="run", run_step=1,
    )
    journal = ModelRequestJournal(
        source_namespace="workspace", tenant_id="tenant",
        execution_id="execution", agent_run_id="run",
    )
    metrics = ModelObservationCapability(
        BlockingRecorder(), source_namespace="workspace", tenant_id="tenant",
        execution_id="execution", session_id=None, agent_run_id="run", agent_id="agent",
        journal=journal,
    )

    async def event(fact: object, *, phase: str, **kwargs: object) -> None:
        phases.append("recorded:" + phase)

    async def publish(fact: object, *, phase: str, **kwargs: object) -> None:
        phases.append("published:" + phase)

    metrics._record_request_event = event
    metrics._publish_request_event = publish

    async def observer(_ctx: object, fact: ModelRequestFact, phase: str,
                       wrapped: object, response: object, error: object) -> None:
        if phase == "completed":
            observer_entered.set()
            await observer_release.wait()
        phases.append("observed:" + phase)

    observed = _ObservedCompactionModel(
        model, ctx=ctx, journal=journal, observer=observer,
        recorder=metrics.record_external_model_request, source_messages=(),
    )
    task = asyncio.create_task(observed.request([], None, ModelRequestParameters()))
    await asyncio.wait_for(observer_entered.wait(), timeout=1)
    if cancel_during_observer:
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    observer_release.set()
    await asyncio.wait_for(metric_entered.wait(), timeout=1)
    assert phases[-3:] == ["observed:completed", "recorded:completed", "published:completed"]
    assert metric_tasks == [task]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(RuntimeError, match="missing"):
        journal.current(1)
