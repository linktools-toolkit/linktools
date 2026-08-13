#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transport-only ACP adapter over the public Runtime."""

import asyncio
from dataclasses import dataclass
from types import ModuleType
from typing import Protocol
from uuid import uuid4

from linktools.core import environ

try:
    import acp as _acp
    import acp.schema as _acp_schema
except ModuleNotFoundError:
    _acp = None
    _acp_schema = None

from .core import ExecutionEventType, JsonValue, Principal, validate_memory_scope
from .errors import AIError
from .runtime import CancelExecutionRequest, ListSessionRequest, Runtime
from .workspace import Workspace, open_workspace_runtime, trusted_workspace_principal

_logger = environ.get_logger("ai.acp")


class ACPConnection(Protocol):
    async def session_update(self, session_id: str, update: JsonValue) -> None: ...


class ACPTextContent(Protocol):
    text: str


class ACPAgent:
    """Translate ACP requests into Runtime calls and durable event reads."""

    def __init__(self, runtime: Runtime, *, principal: Principal, memory_scope: str) -> None:
        try:
            validate_memory_scope(memory_scope)
        except AIError as error:
            raise ValueError("memory namespace is invalid") from error
        self._runtime = runtime
        self._principal = principal
        self._memory_scope = memory_scope
        self._connection: ACPConnection | None = None
        self._initialized = False

    def on_connect(self, connection: ACPConnection) -> None:
        self._connection = connection

    async def initialize(self, protocol_version: int, **kwargs: JsonValue) -> JsonValue:
        acp, schema = _require_acp()
        if protocol_version != acp.PROTOCOL_VERSION:
            raise acp.RequestError("no_common_protocol_version")
        self._initialized = True
        return schema.InitializeResponse(protocolVersion=protocol_version, agentCapabilities=schema.AgentCapabilities(loadSession=True), authMethods=[], agentInfo=schema.Implementation(name="linktools-ai", version="0.1"))

    async def new_session(self, cwd: str, **kwargs: JsonValue) -> JsonValue:
        self._require_initialized()
        _, schema = _require_acp()
        session = await self._runtime.create_session(uuid4().hex, principal=self._principal, cwd=cwd)
        return schema.NewSessionResponse(sessionId=session.session_id)

    async def load_session(self, cwd: str, session_id: str, **kwargs: JsonValue) -> JsonValue:
        self._require_initialized()
        _, schema = _require_acp()
        await self._runtime.session.get(session_id, principal=self._principal)
        return schema.LoadSessionResponse()

    async def list_sessions(self, cwd: "str | None" = None, **kwargs: JsonValue) -> JsonValue:
        self._require_initialized()
        _, schema = _require_acp()
        page = await self._runtime.session.list(ListSessionRequest(self._principal, limit=200))
        return schema.ListSessionsResponse(sessions=[schema.SessionInfo(sessionId=item.session_id, cwd=item.cwd or cwd or "") for item in page.items])

    async def resume_session(self, session_id: str, cwd: str, **kwargs: JsonValue) -> JsonValue:
        return await self.load_session(cwd, session_id, **kwargs)

    async def fork_session(self, session_id: str, cwd: str, **kwargs: JsonValue) -> JsonValue:
        self._require_initialized()
        _, schema = _require_acp()
        from .runtime import ForkSessionRequest

        session = await self._runtime.session.fork(session_id, ForkSessionRequest(self._principal, uuid4().hex, uuid4().hex, cwd))
        return schema.ForkSessionResponse(sessionId=session.session_id)

    async def close_session(self, session_id: str, **kwargs: JsonValue) -> JsonValue:
        self._require_initialized()
        _, schema = _require_acp()
        from .runtime import CloseSessionRequest

        await self._runtime.session.close(session_id, CloseSessionRequest(self._principal, uuid4().hex))
        return schema.CloseSessionResponse()

    async def set_session_mode(self, session_id: str, mode_id: str, **kwargs: JsonValue) -> None:
        self._require_initialized()

    async def set_config_option(self, config_id: str, session_id: str, value: JsonValue, **kwargs: JsonValue) -> None:
        self._require_initialized()

    async def authenticate(self, method_id: str, **kwargs: JsonValue) -> None:
        acp, _ = _require_acp()
        raise acp.RequestError("unknown_auth_method")

    async def prompt(self, session_id: str, prompt: "list[ACPTextContent]", **kwargs: JsonValue) -> JsonValue:
        self._require_initialized()
        _, schema = _require_acp()
        text = "".join(item.text for item in prompt)
        async for event in self._runtime.stream(text, principal=self._principal, session_id=session_id, memory_scope=self._memory_scope):
            if self._connection is not None:
                update = _acp_update(schema, event.event_type, event.payload)
                if update is not None:
                    await self._connection.session_update(session_id, update)
        return schema.PromptResponse(stopReason="end_turn")

    async def cancel(self, session_id: str, **kwargs: JsonValue) -> None:
        loaded = await self._runtime.session.load(session_id, principal=self._principal)
        for execution_id in loaded.active_execution_ids:
            await self._runtime.execution.cancel(execution_id, CancelExecutionRequest(self._principal, uuid4().hex, True))

    async def ext_method(self, method: str, params: "dict[str, JsonValue]") -> None:
        acp, _ = _require_acp()
        raise acp.RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: "dict[str, JsonValue]") -> None:
        return None

    def _require_initialized(self) -> None:
        if not self._initialized:
            acp, _ = _require_acp()
            raise acp.RequestError("initialize_required")


@dataclass(frozen=True, slots=True)
class ACPApplication:
    workspace: Workspace

    @classmethod
    def for_workspace(cls, workspace: Workspace) -> "ACPApplication":
        return cls(workspace)

    async def serve(self, *, memory_scope: str) -> None:
        principal = trusted_workspace_principal(self.workspace.workspace_id)
        async with open_workspace_runtime(self.workspace) as runtime:
            await serve_stdio(ACPAgent(runtime, principal=principal, memory_scope=memory_scope))


async def serve_stdio(agent: ACPAgent) -> None:
    acp, _ = _require_acp()
    await acp.run_agent(agent, use_unstable_protocol=True)


def run_stdio(agent: ACPAgent) -> None:
    asyncio.run(serve_stdio(agent))


def _require_acp() -> "tuple[ModuleType, ModuleType]":
    if _acp is None or _acp_schema is None:
        raise ModuleNotFoundError("agent-client-protocol")
    return _acp, _acp_schema


def _acp_update(schema: ModuleType, event_type: ExecutionEventType, payload: JsonValue) -> "JsonValue | None":
    if not isinstance(payload, dict):
        return None
    if event_type is ExecutionEventType.ASSISTANT_TEXT_DELTA:
        return schema.AgentMessageChunk(content=schema.TextContentBlock(type="text", text=str(payload.get("text", ""))), sessionUpdate="agent_message_chunk")
    if event_type is ExecutionEventType.ASSISTANT_THINKING_DELTA:
        return schema.AgentThoughtChunk(content=schema.TextContentBlock(type="text", text=str(payload.get("text", ""))), sessionUpdate="agent_thought_chunk")
    if event_type is ExecutionEventType.TOOL_CALL_STARTED:
        return schema.ToolCallStart(toolCallId=str(payload.get("call_id", "")), title=str(payload.get("tool_name", "tool")), kind="execute", status="in_progress", sessionUpdate="tool_call")
    if event_type is ExecutionEventType.TOOL_CALL_FINISHED:
        return schema.ToolCallProgress(toolCallId=str(payload.get("call_id", "")), kind="execute", status="completed" if payload.get("status") == "SUCCEEDED" else "failed", sessionUpdate="tool_call_update")
    return None


__all__ = ["ACPAgent", "ACPApplication", "ACPConnection", "run_stdio", "serve_stdio"]
