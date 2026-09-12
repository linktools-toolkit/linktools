#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harness summarization must remain visible to Runtime request observability."""

from collections.abc import Sequence

import pytest
from linktools.ai.runtime._compaction import (
    RuntimeCompaction,
    RuntimeCompactionPolicy,
)
from linktools.ai.runtime._journal import ModelRequestFact, ModelRequestJournal
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
    capability: RuntimeCompaction,
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
        step_run_id="run",
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

    capability = RuntimeCompaction(
        1,
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
    assert request_context.messages == messages
    with pytest.raises(RuntimeError, match="missing"):
        journal.current(observed[-1][1].request_sequence)


def test_journal_keeps_agent_and_compaction_requests_distinct_on_same_step() -> None:
    journal = ModelRequestJournal(
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id="run",
    )
    agent = journal.begin(3, purpose="agent")
    compaction = journal.begin(3, purpose="compaction")

    assert agent.request_sequence != compaction.request_sequence
    assert agent.observation_id != compaction.observation_id
    journal.finish(compaction.request_sequence, status="SUCCEEDED")
    assert journal.consume(compaction.request_sequence).purpose == "compaction"
    assert journal.current(agent.request_sequence) == agent
    journal.finish(agent.request_sequence, status="SUCCEEDED")
    assert journal.consume(agent.request_sequence).purpose == "agent"


def test_journal_rejects_double_finish() -> None:
    journal = ModelRequestJournal(
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id="run",
    )
    fact = journal.begin(1)
    journal.finish(fact.request_sequence, status="SUCCEEDED")

    with pytest.raises(RuntimeError, match="already finished"):
        journal.finish(fact.request_sequence, status="FAILED")

    journal.consume(fact.request_sequence)


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
                    "old content",
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
                    "new content",
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
        RuntimeCompaction(
            1_000_000,
            policy=RuntimeCompactionPolicy(
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
        RuntimeCompaction(
            None,
            policy=RuntimeCompactionPolicy(
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
        RuntimeCompaction(
            1,
            policy=RuntimeCompactionPolicy(
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
