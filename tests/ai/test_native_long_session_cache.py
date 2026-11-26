#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Warm native session context retains only candidates the model can consume."""

import hashlib

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from linktools.ai.core import ExecutionStatus, Principal, TenantAuthorizationPolicy
from linktools.ai.model.messages import SummaryPromptPart, encode_model_messages
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.turns import SessionTurnKey
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime import _regenerable_turn
from linktools.ai.persistence import _session_turn_sql

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_session_turn_sql import _BINDING, store


class _HostAuthorization(TenantAuthorizationPolicy):
    pass


@pytest.mark.asyncio
async def test_adjacent_long_turns_do_not_reload_warm_sql_history(tmp_path, monkeypatch):
    monkeypatch.setattr(_regenerable_turn, "_CONTEXT_BYTES", 9_000)
    monkeypatch.setattr(_session_turn_sql, "_HISTORY_CACHE_BYTES", 24_000)
    captures = []
    offered = []

    async def model(messages, _info):
        prompts = [part.content for msg in messages if isinstance(msg, ModelRequest)
                   for part in msg.parts if isinstance(part, UserPromptPart)]
        offered.append((prompts, list(captures)))
        current = prompts[-1]
        body = " ".join(hashlib.sha256(current.encode() + i.to_bytes(4, "big")).hexdigest()[:20]
                        for i in range(320))
        return ModelResponse(parts=[TextPart(f"{current} {body}")])

    features = RuntimeFeatures("long-session-cache")
    features.agent("default", model="default", allow_tools=())
    model_features(_Models(_UsageFunctionModel(model)), features)
    async with (
        Persistence.open("long-session-cache", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as state,
        Runtime.open(persistence=state, features=features, authorization=_HostAuthorization()) as runtime,
    ):
        owner = Principal("owner", runtime.tenant_id)
        await runtime.sessions.create_turn_session("chat", principal=owner)
        turns = state.session_turns
        source, provider, capture = turns._transcript_source

        async def capture_history(roots, *, transaction):
            load = await capture(roots, transaction=transaction)

            async def read():
                bodies = await load()
                captures.append((len(roots), sum(len(body) for root in bodies for body in root)))
                return bodies
            return read

        turns.bind_transcript_reader(source, provider, capture_history)
        for turn_no in range(1, 7):
            captures.clear()
            prompt = f"turn-{turn_no}"
            turn = await runtime.sessions.start("chat", prompt, agent="default", principal=owner,
                                               idempotency_key=prompt)
            assert (await turn.wait_final(timeout_seconds=10)).status is ExecutionStatus.SUCCEEDED
        assert [value for value, _ in offered] == [
            [f"turn-{n}"] if n == 1 else [f"turn-{n - 1}", f"turn-{n}"] for n in range(1, 7)
        ]
        assert all(not loads for _, loads in offered), offered[-5:]


@pytest.mark.asyncio
async def test_warm_suffix_retains_compaction_boundary_and_exact_history(store):
    _engine, _metadata, turns = store
    for number in range(1, 4):
        key = SessionTurnKey("chat", f"turn-{number}")
        prompt = f"prompt-{number}"
        assert await turns.accept(
            key, expected_revision=2 * (number - 1), context_from=1, context_through=number - 1,
            request_key=key.turn_id, request={"prompt": prompt}, binding=_BINDING, lease_seconds=60,
        )
        response = ModelResponse(parts=[TextPart("short")])
        messages = ((ModelRequest(parts=[UserPromptPart("x" * 1024)]), response) if number == 2 else (response,))
        compacted = (
            (ModelRequest(parts=[SummaryPromptPart("compact")]), messages[-1]) if number == 2 else ()
        )
        assert await turns.finish(
            key, generation=1, turn_no=number, state="succeeded", transcript=messages,
            compacted_context=compacted or None,
        )
        turns.note_terminal_context(
            key, turn_no=number, prompt=prompt, stored_input=None, messages=messages, succeeded=True,
            max_context_bytes=2300, compacted_context=compacted,
        )
    warm = await turns.recent_context("chat", through=3, max_context_bytes=2300)
    assert [candidate.turn_no for candidate in warm] == [2, 3]
    assert warm[0].compacted_context[0].parts[0].content == "compact"
    remaining = 2300 - warm[-1].transcript_bytes - len(warm[-1].request["prompt"])
    assert len(encode_model_messages(warm[0].compacted_context)) <= remaining < warm[0].transcript_bytes
    exact = await turns.read_context_range("chat", after=0, through=3)
    assert [candidate.turn_no for candidate in exact] == [1, 2, 3]
