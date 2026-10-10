#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Response fixtures bypass live MCP resources before model execution."""

from types import SimpleNamespace

import pytest
from pydantic_ai.usage import RunUsage, UsageLimits

from linktools.ai.capability import SkillSourceRegistry
from linktools.ai.core import PromptLimits, canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._agent_executor import AgentExecutor, _AgentRunScope
from linktools.ai.runtime._mcp import _MCPBinding
from linktools.ai.runtime._tool_response_fixture import ToolResponseFixture
from linktools.ai.spec import MCPServerSpec, mcp_server_selector

from .test_tool_response_fixture import _manifest, _store


def _scope(ref: object) -> _AgentRunScope:
    async def sink(value: object) -> None:
        raise AssertionError("the test stops before model execution")

    return _AgentRunScope(
        binding=SimpleNamespace(compiled_agent=SimpleNamespace(
            selected_tools=(), skill_definitions=(),
            mcp_servers=(MCPServerSpec("server", "never-start", revision=3),),
            mcp_policy=(mcp_server_selector("server"),),
        )),
        context=None, workspace=None, limits=PromptLimits(), execution_cwd=None,
        user_prompt=None, history=[], initial_context=None,
        agent_conversation_id="conversation", run_store=None,
        agent_run_id="run", agent_run_seq=1, event_sink=sink,
        tool_response_ref=ref,
    )


@pytest.mark.asyncio
async def test_executor_uses_saved_fixture_and_never_projects_live_stdio_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, original = await _store(_manifest())
    try:
        changed = _manifest()
        changed["servers"][0]["tools"][0]["responses"][0]["outcome"]["value"] = "new-case"
        await store.put(original.ref.key, canonical_json_bytes(changed))
        new_ref, = await store.resolve_versions((original.ref.key,))
        configured = ToolResponseFixture(new_ref, store)
        seen: list[object] = []

        monkeypatch.setattr(
            "linktools.ai.runtime._agent_executor._mcp_bindings",
            lambda binding: {"server": _MCPBinding(None, None, {"boundary": "saved"})},
        )

        async def project(servers: object, *args: object, **kwargs: object) -> dict:
            assert servers == ()
            return {}

        async def execute(self: AgentExecutor, scope: _AgentRunScope, **kwargs: object) -> str:
            assert scope.sandbox_session is None
            assert scope.mcp_projections == {}
            tool = scope.response_fixture.server("server").tools[0]
            assert tool.result({"query": "ready", "options": {"a": 1, "b": 2}}) == {
                "url": "https://example.test/info",
            }
            seen.append("saved-case")
            return "model-boundary"

        monkeypatch.setattr("linktools.ai.runtime._mcp.prepare_mcp_projections", project)
        monkeypatch.setattr(AgentExecutor, "_execute", execute)
        executor = AgentExecutor(SkillSourceRegistry(()), tool_responses=configured)
        result = await executor._execute_with_sandbox(
            _scope(original.ref), run_usage=RunUsage(), usage_limits=UsageLimits(),
        )
        assert result == "model-boundary"
        assert seen == ["saved-case"]
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("historical_fixture", (False, True))
async def test_executor_rejects_live_fixture_mode_mismatch_before_any_model_or_transport(
    monkeypatch: pytest.MonkeyPatch, historical_fixture: bool,
) -> None:
    store, fixture = await _store(_manifest())
    try:
        monkeypatch.setattr("linktools.ai.runtime._agent_executor._mcp_bindings", lambda binding: {})

        async def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("mismatched fixture mode cannot reach live resources or a model")

        monkeypatch.setattr("linktools.ai.runtime._mcp.prepare_mcp_projections", forbidden)
        monkeypatch.setattr(AgentExecutor, "_execute", forbidden)
        executor = AgentExecutor(
            SkillSourceRegistry(()), tool_responses=None if historical_fixture else fixture,
        )
        with pytest.raises(AIError) as raised:
            await executor._execute_with_sandbox(
                _scope(fixture.ref if historical_fixture else None),
                run_usage=RunUsage(), usage_limits=UsageLimits(),
            )
        assert raised.value.code is (
            ErrorCode.AGENT_BINDING_UNAVAILABLE if historical_fixture else ErrorCode.BINDING_CONFLICT
        )
        if historical_fixture:
            assert raised.value.safe_details["cause_code"] == ErrorCode.CAPABILITY_REQUIRED_MISSING.value
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("code", (ErrorCode.ASSET_VERSION_NOT_FOUND, ErrorCode.STORAGE_INTEGRITY_ERROR))
async def test_fixture_dependency_failure_remains_recoverable_before_model_execution(
    monkeypatch: pytest.MonkeyPatch, code: ErrorCode,
) -> None:
    from linktools.ai.runtime._local import _is_infrastructure_error

    store, fixture = await _store(_manifest())
    try:
        async def unavailable(refs: object) -> tuple[bytes, ...]:
            raise AIError(code)

        monkeypatch.setattr(store, "read_versions", unavailable)
        executor = AgentExecutor(SkillSourceRegistry(()), tool_responses=fixture)
        with pytest.raises(AIError) as raised:
            await executor.validate_recovery_inputs(_scope(fixture.ref).binding, fixture.ref)
        assert _is_infrastructure_error(raised.value)
        if code is ErrorCode.STORAGE_INTEGRITY_ERROR:
            assert raised.value.code is code
        else:
            assert raised.value.code is ErrorCode.AGENT_BINDING_UNAVAILABLE
            assert raised.value.safe_details["cause_code"] == code.value
    finally:
        await store.close()
