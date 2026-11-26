#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicitly granted exact native turn tool returns, independent of Trace."""

import pytest
from pydantic_ai.messages import ModelResponse, TextContent, TextPart, ToolCallPart, ToolReturn, ToolReturnPart

from linktools.ai.capability import AgentContext, SessionCapability
from linktools.ai.core import ExecutionStatus, Principal, ResourceRef, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence.turns import SqlSessionTurnReader
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


class _Policy(TenantAuthorizationPolicy):
    allowed_turn: str | None = None

    async def session_tool_return_names(
        self, principal: Principal, session_id: str, turn: ResourceRef,
    ) -> frozenset[str]:
        return frozenset({"read_evidence"}) if session_id == "chat" and turn.id == self.allowed_turn else frozenset()


@pytest.mark.asyncio
async def test_native_tool_returns_require_exact_grant_and_owner(tmp_path, monkeypatch):
    def read_evidence(_context: AgentContext[None]) -> ToolReturn:
        return ToolReturn(return_value=[{"evidence": "approved"}, TextContent("excluded native content"), None])

    def read_internal(_context: AgentContext[None]) -> dict[str, str]:
        return {"private": "excluded"}

    async def model(messages, _info):
        if any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[
            ToolCallPart("read_evidence", {}, tool_call_id="result"),
            ToolCallPart("read_internal", {}, tool_call_id="excluded"),
        ])

    features = RuntimeFeatures("native-turn-tool-returns")
    features.capability(SessionCapability(checkpoints=True), scope="runtime")
    features.tool(read_evidence, effect_policy="replay_safe")
    features.tool(read_internal, effect_policy="replay_safe")
    features.agent("default", model="default", allow_tools=("read_evidence", "read_internal"))
    model_features(_Models(_UsageFunctionModel(model)), features)
    policy = _Policy()
    async with Runtime.open(
        "native-turn-tool-returns", storage=PersistenceStorage.sqlite(tmp_path / "state.db"),
        features=features, authorization=policy,
    ) as runtime:
        owner = Principal("owner", runtime.tenant_id)
        other = Principal("other", runtime.tenant_id)
        for session in ("chat", "foreign"):
            await runtime.sessions.create_turn_session(session, principal=owner)
        turns = []
        for key in ("first", "next"):
            turn = await runtime.sessions.start("chat", key, agent="default", principal=owner, idempotency_key=key)
            assert (await turn.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
            turns.append(turn.key.turn_id)
        policy.allowed_turn, excluded_turn = turns
        read = runtime.history.turn_tool_returns
        names = frozenset({"read_evidence"})
        assert await read("chat", policy.allowed_turn, principal=owner, tool_names=names) == (
            {"tool_name": "read_evidence", "tool_call_id": "result", "value": [{"evidence": "approved"}, None]},
        )
        assert await read("chat", policy.allowed_turn, principal=owner, tool_names=names,
                          tool_call_ids=frozenset({"excluded"})) == ()
        default = await runtime.history.turn_result("chat", policy.allowed_turn, principal=owner)
        assert all(item["item_kind"] != "tool_result" for item in default["items"])
        async def forbidden_read(*_args, **_kwargs):
            pytest.fail("unauthorized call opened the transcript")
        monkeypatch.setattr(SqlSessionTurnReader, "get_history_turn", forbidden_read)
        for session, turn, actor, requested in (
            ("chat", excluded_turn, owner, names),
            ("foreign", policy.allowed_turn, owner, names),
            ("chat", policy.allowed_turn, other, names),
            ("chat", policy.allowed_turn, owner, frozenset({"read_internal"})),
        ):
            with pytest.raises(AIError) as denied:
                await read(session, turn, principal=actor, tool_names=requested)
            assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
