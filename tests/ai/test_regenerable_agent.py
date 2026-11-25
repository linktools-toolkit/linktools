#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The existing Agent executor can finish ordinary turns without SQL Step writes."""

import json
from pathlib import Path

import pytest
from linktools.ai.capability import AgentContext, SkillSourceRegistry
from linktools.ai.core import PromptLimits
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._agent_executor import AgentEmission, AgentExecutor
from linktools.ai.runtime._regenerable_transcript import (
    decode_regenerable_turn,
    encode_regenerable_turn,
    reply_from_turn,
)
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.function import FunctionModel
from sqlalchemy import event
from sqlalchemy.engine import Engine

from ._runtime_test_helpers import _Models, model_features


@pytest.mark.asyncio
async def test_model_stream_continues_committed_context_without_sql(tmp_path: Path) -> None:
    prompts: list[list[str]] = []
    events: list[AgentEmission] = []
    sql: list[str] = []

    async def stream(messages: list[ModelMessage], _info):
        prompts.append([part.content for message in messages if isinstance(message, ModelRequest)
                        for part in message.parts if isinstance(part, UserPromptPart)])
        yield f"reply {len(prompts)}"

    async def emit(value: AgentEmission) -> None:
        events.append(value)

    def capture(_connection, _cursor, statement, *_rest):
        sql.append(statement)

    features = RuntimeFeatures("plain")
    features.agent("default", model="default", allow_tools=())
    model_features(_Models(FunctionModel(stream_function=stream)), features)
    async with Runtime.open("plain", storage=PersistenceStorage.sqlite(tmp_path / "state.db"),
                            features=features, metrics=None) as runtime:
        binding = await runtime._binding_resolver.resolve_agent("default")
        executor = AgentExecutor(SkillSourceRegistry())
        event.listen(Engine, "before_cursor_execute", capture)
        try:
            history: list[ModelMessage] = []
            for sequence, prompt in enumerate(("first", "second"), 1):
                context = AgentContext(
                    None, runtime.default_principal, "plain", "chat", f"turn-{sequence}", {},
                )
                outcome = await executor.execute_regenerable(
                    binding=binding, context=context, prompt=prompt, history=history,
                    limits=PromptLimits(), event_sink=emit,
                )
                assert outcome.output == {"text": f"reply {sequence}"}
                assert len(outcome.messages) == sequence * 2
                payload = encode_regenerable_turn(prompt, history, outcome.messages, outcome.output)
                encoded = json.dumps(payload)
                assert f'"{prompt}"' not in encoded
                assert encoded.count(f"reply {sequence}") == 1
                if sequence == 2:
                    assert '"first"' not in encoded and "reply 1" not in encoded
                restored = decode_regenerable_turn(prompt, payload)
                assert tuple(history) + restored == tuple(outcome.messages)
                assert reply_from_turn(restored) == outcome.output
                history.extend(restored)
        finally:
            event.remove(Engine, "before_cursor_execute", capture)
    assert prompts == [["first"], ["first", "second"]]
    assert events and not sql
