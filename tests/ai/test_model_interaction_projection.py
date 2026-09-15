#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contract tests for model interaction projection primitives."""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel

from linktools.ai.core import JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.observe import Metrics
from linktools.ai.runtime import Runtime
from linktools.ai.runtime._harness import HarnessStepStoreAdapter
from linktools.ai.runtime._journal import ModelRequestJournal
from linktools.ai.runtime._model_interaction import (
    StagedContextInline,
    StagedContextProjection,
    StagedContextSpan,
    StagedModelInteraction,
    build_context_projection,
    message_prefix_digest,
    model_identity,
    project_public_messages,
    request_envelope,
)
from linktools.ai.runtime.state._step_contracts import RunRecord
from linktools.ai.runtime.state._steps import StagingStepStore
from linktools.ai.spec import AgentSpec, AgentSpecCodec
from linktools.ai.workspace import Workspace


class _TextModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:test"
    vision = False
    fingerprint = "d" * 64
    semantic_payload: dict[str, JsonValue] = {
        "provider": "test",
        "model": "test",
    }

    def materialize(self) -> TestModel:
        return TestModel(custom_output_text="ok")


class _TextModels:
    def snapshot(self) -> "_TextModels":
        return self

    def resolve(self, route_id: str) -> _TextModelBinding:
        if route_id != "default":
            raise AssertionError(route_id)
        return _TextModelBinding()

    def restore(
        self,
        payload: Mapping[str, JsonValue],
        *,
        route_id: str | None = None,
    ) -> _TextModelBinding:
        if route_id not in {None, "default"} or dict(payload) != {
            "provider": "test",
            "model": "test",
        }:
            raise AssertionError("unexpected model snapshot")
        return _TextModelBinding()


def _write_default_agent(root: Path) -> None:
    path = root / ".linktools" / "agents" / "default"
    path.parent.mkdir(parents=True)
    path.write_bytes(
        AgentSpecCodec().encode(AgentSpec("default", model="default", allow_tools=()))
    )


def _journal() -> ModelRequestJournal:
    return ModelRequestJournal(
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id="run",
    )


def _cancelled_interaction(sequence: int = 1) -> StagedModelInteraction:
    return StagedModelInteraction(
        run_id="run",
        step_index=1,
        request_sequence=sequence,
        purpose="agent",
        output_retry_index=None,
        model={"route_id": "default"},
        request_context=StagedContextProjection(
            0,
            message_prefix_digest(()),
            (),
        ),
        request_envelope_digest="a" * 64,
        response_context=None,
        status="CANCELLED",
        error_code=None,
        duration_ns=0,
        usage=None,
    )


def test_context_projection_retains_spans_and_deduplicates_inline_payloads() -> None:
    source = tuple(
        ModelRequest(parts=[UserPromptPart(value)]) for value in ("one", "two")
    )
    payloads: dict[str, bytes] = {}

    def intern(value: bytes) -> tuple[str, int]:
        import hashlib

        digest = hashlib.sha256(value).hexdigest()
        payloads.setdefault(digest, value)
        return digest, len(value)

    summary = ModelRequest(parts=[UserPromptPart("summary")])
    projection = build_context_projection(
        source,
        (
            source[0],
            summary,
            summary,
        ),
        intern,
    )

    assert projection.source_message_count == 2
    assert projection.source_prefix_digest == message_prefix_digest(source)
    assert isinstance(projection.items[0], StagedContextSpan)
    assert isinstance(projection.items[1], StagedContextInline)
    assert projection.items[1].payload_digest == next(iter(payloads))
    assert len(payloads) == 1


def test_public_model_messages_replace_binary_body_with_metadata() -> None:
    message = ModelRequest(
        parts=[
            UserPromptPart(
                [BinaryContent(b"image", media_type="image/png")],
            )
        ]
    )

    projected = project_public_messages((message,))

    assert "data" not in projected[0]["parts"][0]["content"][0]  # type: ignore[index]
    assert projected[0]["parts"][0]["content"][0]["size"] == 5  # type: ignore[index]


def test_request_envelope_keeps_capability_visibility_inputs() -> None:
    envelope, raw = request_envelope(
        model_settings={"temperature": 0.2},
        parameters=ModelRequestParameters(
            deferred_capability_ids={"skill-a"},
            revealed_tool_names={"tool-a"},
        ),
        streaming=True,
    )

    assert envelope["version"] == 1
    assert envelope["streaming"] is True
    parameters = envelope["parameters"]
    assert isinstance(parameters, Mapping)
    assert parameters["deferred_capability_ids"] == ["skill-a"]
    assert parameters["revealed_tool_names"] == ["tool-a"]
    assert b'"version":1' in raw


def test_model_identity_preserves_selection_route() -> None:
    identity = model_identity(TestModel(), route_id="tenant-alias")
    assert identity["route_id"] == "tenant-alias"


def test_cancelled_model_interaction_has_no_synthetic_error() -> None:
    interaction = _cancelled_interaction()
    assert interaction.status == "CANCELLED"
    assert interaction.error_code is None


@pytest.mark.asyncio
async def test_staging_interaction_identity_is_idempotent() -> None:
    store = StagingStepStore()
    await store.initialize()
    await store.register_run(RunRecord("run"))
    interaction = _cancelled_interaction()

    store.stage_model_interaction(interaction)
    store.stage_model_interaction(interaction)

    assert await store.list_model_interactions(run_id="run") == [interaction]
    with pytest.raises(AIError) as raised:
        store.stage_model_interaction(
            replace(
                interaction,
                status="FAILED",
                error_code=ErrorCode.MODEL_API_ERROR.value,
            )
        )
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_success_request_does_not_stage_full_message_payloads() -> None:
    store = StagingStepStore()
    await store.initialize()
    await store.register_run(RunRecord("run"))
    adapter = HarnessStepStoreAdapter(
        store,
        execution_id="execution",
        step_run_id="run",
    )
    message = ModelRequest(parts=[UserPromptPart("hello")])
    journal = _journal()
    fact = journal.begin(1)

    adapter.begin_model_interaction(
        fact,
        TestModel(),
        (message,),
        None,
        ModelRequestParameters(),
        False,
        "alias",
    )

    assert len(store._payloads["run"]) == 1
    finished = journal.finish(fact.request_sequence, status="SUCCEEDED")
    adapter.finish_model_interaction(
        finished,
        model=TestModel(),
        response=ModelResponse(parts=[TextPart("done")]),
        status="SUCCEEDED",
        error_code=None,
        duration_ns=1,
        usage=None,
    )
    staged = await store.list_model_interactions(run_id="run")
    assert len(staged) == 1
    assert all(
        isinstance(item, StagedContextSpan)
        for item in staged[0].request_context.items  # type: ignore[union-attr]
    )
    assert len(store._payloads["run"]) == 1


@pytest.mark.asyncio
async def test_failed_request_inlines_context_only_after_failure() -> None:
    store = StagingStepStore()
    await store.initialize()
    await store.register_run(RunRecord("run"))
    adapter = HarnessStepStoreAdapter(
        store,
        execution_id="execution",
        step_run_id="run",
    )
    message = ModelRequest(parts=[UserPromptPart("hello")])
    journal = _journal()
    fact = journal.begin(1)

    adapter.begin_model_interaction(
        fact,
        TestModel(),
        (message,),
        None,
        ModelRequestParameters(),
        False,
    )
    assert len(store._payloads["run"]) == 1

    finished = journal.finish(fact.request_sequence, status="FAILED")
    adapter.finish_model_interaction(
        finished,
        model=TestModel(),
        response=None,
        status="FAILED",
        error_code=ErrorCode.MODEL_API_ERROR.value,
        duration_ns=1,
        usage=None,
    )
    staged = await store.list_model_interactions(run_id="run")
    request = staged[0].request_context  # type: ignore[union-attr]
    assert request.source_message_count == 0
    assert all(isinstance(item, StagedContextInline) for item in request.items)
    assert len(store._payloads["run"]) == 2


@pytest.mark.asyncio
async def test_compaction_request_does_not_reuse_agent_projection_source() -> None:
    store = StagingStepStore()
    await store.initialize()
    await store.register_run(RunRecord("run"))
    adapter = HarnessStepStoreAdapter(
        store,
        execution_id="execution",
        step_run_id="run",
    )
    agent_source = (ModelRequest(parts=[UserPromptPart("agent")]),)
    agent_projected = (ModelRequest(parts=[UserPromptPart("agent projected")]),)
    adapter.remember_context_projection(agent_source, agent_projected)

    compaction_request = ModelRequest(parts=[UserPromptPart("summarize")])
    journal = _journal()
    fact = journal.begin(2, purpose="compaction")
    adapter.begin_model_interaction(
        fact,
        TestModel(),
        (compaction_request,),
        None,
        ModelRequestParameters(),
        False,
    )
    finished = journal.finish(fact.request_sequence, status="SUCCEEDED")
    adapter.finish_model_interaction(
        finished,
        model=TestModel(),
        response=ModelResponse(parts=[TextPart("summary")]),
        status="SUCCEEDED",
        error_code=None,
        duration_ns=1,
        usage=None,
    )

    staged = await store.list_model_interactions(run_id="run")
    interaction = staged[0]
    assert interaction.purpose == "compaction"  # type: ignore[union-attr]
    assert interaction.request_context.source_message_count == 0  # type: ignore[union-attr]
    assert all(
        isinstance(item, StagedContextInline)
        for item in interaction.request_context.items  # type: ignore[union-attr]
    )


@pytest.mark.asyncio
async def test_execution_model_interactions_are_durable_and_public(tmp_path: Path) -> None:
    _write_default_agent(tmp_path)

    async with Runtime.open(
        Workspace.load(tmp_path, workspace_id="workspace"),
        models=_TextModels(),  # type: ignore[arg-type]
        metrics=Metrics.in_memory(),
    ) as runtime:
        execution = await runtime.agent("default").start("hello")
        result = await execution.wait()
        page = await execution.model_interactions()

    assert result.status == "SUCCEEDED"
    assert len(page.items) == 1
    assert page.items[0].purpose == "agent"
    assert page.items[0].status == "SUCCEEDED"
    assert page.items[0].request["messages"]
    assert page.items[0].response is not None
