#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable deferred tool-result contract tests."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from pydantic import BaseModel
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.messages import BinaryContent, ImageUrl
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolResults
from pydantic_ai.usage import RunUsage, UsageLimits

from linktools.ai.agent import AssistantTextOutput
from linktools.ai.agent._output import bind_output
from linktools.ai.capability import SkillSourceRegistry
from linktools.ai.core import PromptLimits
from linktools.ai.runtime import _agent_executor as agent_executor
from linktools.ai.runtime._agent_executor import AgentExecutor, _AgentRunScope
from linktools.ai.runtime.state._contracts import LoadedModelContext
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


def test_tool_return_content_snapshots_arbitrary_python_values_as_json() -> None:
    class Result(BaseModel):
        count: int

    model_encoded = encode_tool_return_content(Result(count=2))
    bytes_encoded = encode_tool_return_content(b"abc")

    assert model_encoded["value"] == {
        "type": "json-snapshot",
        "value": {"count": 2},
    }
    assert decode_tool_return_content(model_encoded) == {"count": 2}
    assert bytes_encoded["value"] == {
        "type": "json-snapshot",
        "value": "YWJj",
    }
    assert decode_tool_return_content(bytes_encoded) == "YWJj"


def test_tool_return_content_snapshots_non_string_mapping_keys() -> None:
    encoded = encode_tool_return_content({1: "one"})

    assert encoded["value"] == {
        "type": "json-snapshot",
        "value": {"1": "one"},
    }
    assert decode_tool_return_content(encoded) == {"1": "one"}


def test_tool_return_content_uses_explicit_linktools_envelope() -> None:
    encoded = encode_tool_return_content({"type": "business", "value": 1})

    assert encoded["contract"] == "linktools.tool-return"
    assert encoded["version"] == 1
    assert encoded["value"]["type"] == "mapping"


def test_deferred_rehydrate_does_not_guess_plain_business_mapping() -> None:
    plain = {"type": "business", "value": 1}
    source = DeferredToolResults(calls={"plain": plain})

    restored = rehydrate_deferred_tool_results(source)

    assert restored.calls["plain"] is plain


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
                agent_run_id="agent-run",
                all_messages=lambda: [],
            )

    async def materialize(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return _Agent(), ()

    monkeypatch.setattr(agent_executor, "_materialize_agent", materialize)

    class _AgentRunStore:
        def __init__(self) -> None:
            self.get_agent_run_calls = 0

        async def get_agent_run(self, *, agent_run_id: str) -> object | None:
            assert agent_run_id == "agent-run"
            self.get_agent_run_calls += 1
            if self.get_agent_run_calls == 1:
                return None
            return SimpleNamespace(conversation_id="conversation")

        async def latest_snapshot(self, *, agent_run_id: str) -> object:
            assert agent_run_id == "agent-run"
            return object()

        async def model_interaction_count(self, *, agent_run_id: str) -> int:
            assert agent_run_id == "agent-run"
            return 0

    compiled_agent = SimpleNamespace(
        digest="compiled-agent",
        model=SimpleNamespace(materialize=lambda: TestModel()),
        spec=SimpleNamespace(id="agent"),
        selected_tools=(),
    )
    binding = SimpleNamespace(
        compiled_agent=compiled_agent,
        output_binding=bind_output(),
    )
    context = SimpleNamespace(
        namespace="workspace",
        principal=SimpleNamespace(tenant_id="tenant"),
        execution_id="execution",
    )

    async def sink(_emission: object) -> None:
        return None

    portable = encode_tool_return_content(
        {"image": BinaryContent(data=b"binary", media_type="image/png")}
    )
    scope = _AgentRunScope(
        binding=binding,  # type: ignore[arg-type]
        context=context,  # type: ignore[arg-type]
        workspace=None,
        limits=PromptLimits(),
        execution_cwd="",
        user_prompt=None,
        history=[],
        initial_context=LoadedModelContext(()),
        agent_conversation_id="conversation",
        run_store=_AgentRunStore(),  # type: ignore[arg-type]
        agent_run_id="agent-run",
        agent_run_sequence=1,
        event_sink=sink,
        deferred_tool_results=DeferredToolResults(calls={"success": portable}),
    )
    executor = AgentExecutor(SkillSourceRegistry())

    result = await executor._execute(
        scope,
        run_usage=RunUsage(),
        usage_limits=UsageLimits(),
        skill_sources=SkillSourceRegistry(),
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


def test_tool_return_content_digest_is_stable_across_hash_seeds() -> None:
    script = (
        "from pydantic import create_model;"
        "from linktools.ai.runtime._tool_return_codec "
        "import tool_return_content_digest;"
        "Result=create_model('Result',values=(set[str],...));"
        "print(tool_return_content_digest({'values':{'alpha','beta','gamma'}}),"
        "tool_return_content_digest(Result(values={'alpha','beta','gamma'})))"
    )
    values = []
    for seed in ("1", "2"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        values.append(
            subprocess.check_output(
                [sys.executable, "-c", script],
                env=env,
                text=True,
            ).strip()
        )

    assert len(set(values)) == 1
