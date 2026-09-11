#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable deferred tool-result contract tests."""

from types import SimpleNamespace

import pytest
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.messages import BinaryContent, ImageUrl
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolResults
from pydantic_ai.usage import RunUsage, UsageLimits

from linktools.ai.agent import AssistantTextOutput
from linktools.ai.agent._output import bind_output
from linktools.ai.capability import SkillSourceRegistry
from linktools.ai.runtime import _agent_executor as agent_executor
from linktools.ai.runtime._agent_executor import AgentExecutor, _RunScope
from linktools.ai.runtime._tool_return_codec import (
    decode_tool_return_content,
    encode_tool_return_content,
    rehydrate_deferred_tool_results,
    tool_return_content_digest,
)


def test_tool_return_content_round_trips_nested_multimodal_values() -> None:
    value = {
        "items": [
            BinaryContent(data=b"image-bytes", media_type="image/png"),
            ImageUrl(url="https://example.com/image.png"),
        ],
        "plain": {"kind": "binary", "label": "not-multimodal"},
    }

    encoded = encode_tool_return_content(value)
    restored = decode_tool_return_content(encoded)

    assert isinstance(restored, dict)
    items = restored["items"]
    assert isinstance(items, list)
    assert isinstance(items[0], BinaryContent)
    assert items[0].data == b"image-bytes"
    assert items[0].media_type == "image/png"
    assert isinstance(items[1], ImageUrl)
    assert items[1].url == "https://example.com/image.png"
    assert restored["plain"] == {"kind": "binary", "label": "not-multimodal"}


def test_deferred_rehydrate_preserves_control_results() -> None:
    portable = encode_tool_return_content(
        {"image": BinaryContent(data=b"binary", media_type="image/png")}
    )
    retry = ModelRetry("retry later")
    failed = ToolFailed("failed")
    source = DeferredToolResults(
        calls={"success": portable, "retry": retry, "failed": failed},
        metadata={"success": {"source": "test"}},
    )

    restored = rehydrate_deferred_tool_results(source)

    success = restored.calls["success"]
    assert isinstance(success, dict)
    assert isinstance(success["image"], BinaryContent)
    assert restored.calls["retry"] is retry
    assert restored.calls["failed"] is failed
    assert restored.metadata == {"success": {"source": "test"}}


@pytest.mark.asyncio
async def test_agent_executor_rehydrates_deferred_results_before_pydantic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, DeferredToolResults] = {}

    class _Agent:
        async def run(self, *args: object, **kwargs: object) -> object:
            del args
            results = kwargs.get("deferred_tool_results")
            assert isinstance(results, DeferredToolResults)
            captured["results"] = results
            return SimpleNamespace(
                output=AssistantTextOutput(text="ok"),
                run_id="step-run",
                all_messages=lambda: [],
            )

    async def materialize(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return _Agent(), (), ()

    monkeypatch.setattr(agent_executor, "_materialize_agent", materialize)

    class _StepStore:
        def __init__(self) -> None:
            self.get_run_calls = 0

        async def get_run(self, *, run_id: str) -> object | None:
            assert run_id == "step-run"
            self.get_run_calls += 1
            if self.get_run_calls == 1:
                return None
            return SimpleNamespace(conversation_id="conversation")

        async def latest_snapshot(self, *, run_id: str) -> object:
            assert run_id == "step-run"
            return object()

    definition = SimpleNamespace(
        digest="definition",
        model=SimpleNamespace(materialize=lambda: TestModel()),
        spec=SimpleNamespace(id="agent"),
    )
    binding = SimpleNamespace(
        definition=definition,
        output_binding=bind_output(),
    )
    context = SimpleNamespace(
        workspace=SimpleNamespace(workspace_id="workspace"),
        principal=SimpleNamespace(tenant_id="tenant"),
        execution_id="execution",
    )

    async def sink(_emission: object) -> None:
        return None

    portable = encode_tool_return_content(
        {"image": BinaryContent(data=b"binary", media_type="image/png")}
    )
    scope = _RunScope(
        binding=binding,  # type: ignore[arg-type]
        context=context,  # type: ignore[arg-type]
        user_prompt=None,
        history=[],
        conversation_id="conversation",
        step_store=_StepStore(),  # type: ignore[arg-type]
        step_run_id="step-run",
        segment_sequence=1,
        event_sink=sink,
        deferred_tool_results=DeferredToolResults(calls={"success": portable}),
    )
    executor = AgentExecutor(SkillSourceRegistry())

    result = await executor._execute(
        scope,
        run_usage=RunUsage(),
        usage_limits=UsageLimits(),
    )

    success = captured["results"].calls["success"]
    assert isinstance(success, dict)
    assert isinstance(success["image"], BinaryContent)
    assert result.output == {"text": "ok"}


def test_tool_return_content_digest_is_canonical() -> None:
    left = {
        "b": [BinaryContent(data=b"same", media_type="image/png")],
        "a": 1,
    }
    right = {
        "a": 1,
        "b": [BinaryContent(data=b"same", media_type="image/png")],
    }

    assert tool_return_content_digest(left) == tool_return_content_digest(right)
