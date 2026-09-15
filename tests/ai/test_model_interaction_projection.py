#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contract tests for model interaction projection primitives."""

from pathlib import Path
from collections.abc import Mapping

import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel

from linktools.ai.core import JsonValue
from linktools.ai.observe import Metrics
from linktools.ai.runtime import Runtime
from linktools.ai.runtime._model_interaction import (
    StagedContextProjection,
    StagedContextInline,
    StagedContextSpan,
    StagedModelInteraction,
    build_context_projection,
    message_prefix_digest,
    project_public_messages,
    request_envelope,
)
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


def test_request_envelope_is_versioned_json_without_provider_objects() -> None:
    envelope, raw = request_envelope(
        model_settings={"temperature": 0.2},
        parameters=ModelRequestParameters(),
        streaming=True,
    )

    assert envelope["version"] == 1
    assert envelope["streaming"] is True
    assert b'"version":1' in raw


def test_cancelled_model_interaction_has_no_synthetic_error() -> None:
    interaction = StagedModelInteraction(
        run_id="run",
        step_index=1,
        request_sequence=1,
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

    assert interaction.status == "CANCELLED"
    assert interaction.error_code is None


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
