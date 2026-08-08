#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACP transport for the Workspace application runtime."""

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

from ..core import JsonValue
from ..workspace import Workspace
from ._workbench import WorkspaceAgentRuntime, open_workspace_runtime

_logger = environ.get_logger("ai.app.acp")


class ACPConnection(Protocol):
    async def session_update(self, session_id: str, update: JsonValue) -> None: ...


class ACPTextContent(Protocol):
    text: str


class ACPAgent:
    def __init__(self, runtime: WorkspaceAgentRuntime) -> None:
        self._runtime = runtime
        self._connection: ACPConnection | None = None
        self._initialized = False

    def on_connect(self, connection: ACPConnection) -> None:
        self._connection = connection

    async def initialize(self, protocol_version: int, **kwargs: JsonValue) -> JsonValue:
        acp, schema = _require_acp()
        if protocol_version != acp.PROTOCOL_VERSION:
            raise acp.RequestError("no_common_protocol_version")
        self._initialized = True
        _logger.info("ACP initialized: protocol_version=%s", protocol_version)
        return schema.InitializeResponse(
            protocolVersion=protocol_version,
            agentCapabilities=schema.AgentCapabilities(loadSession=True),
            authMethods=[],
            agentInfo=schema.Implementation(name="linktools-ai", version="0.1"),
        )

    async def new_session(self, cwd: str, **kwargs: JsonValue) -> JsonValue:
        self._require_initialized()
        _, schema = _require_acp()
        session = await self._runtime.open_session(uuid4().hex, cwd=cwd)
        return schema.NewSessionResponse(sessionId=session.session_id)

    async def load_session(self, cwd: str, session_id: str, **kwargs: JsonValue) -> JsonValue:
        self._require_initialized()
        _, schema = _require_acp()
        await self._runtime.open_session(session_id, cwd=cwd)
        return schema.LoadSessionResponse()

    async def list_sessions(self, cwd: 'str | None' = None, **kwargs: JsonValue) -> JsonValue:
        self._require_initialized()
        _, schema = _require_acp()
        sessions = await self._runtime.list_sessions(cwd=cwd)
        return schema.ListSessionsResponse(
            sessions=[schema.SessionInfo(sessionId=item.session_id, cwd=item.cwd.as_posix()) for item in sessions]
        )

    async def resume_session(self, session_id: str, cwd: str, **kwargs: JsonValue) -> JsonValue:
        self._require_initialized()
        _, schema = _require_acp()
        await self._runtime.open_session(session_id, cwd=cwd)
        return schema.ResumeSessionResponse()

    async def fork_session(self, session_id: str, cwd: str, **kwargs: JsonValue) -> JsonValue:
        self._require_initialized()
        _, schema = _require_acp()
        session = await self._runtime.fork_session(session_id, cwd=cwd)
        return schema.ForkSessionResponse(sessionId=session.session_id)

    async def close_session(self, session_id: str, **kwargs: JsonValue) -> JsonValue:
        self._require_initialized()
        _, schema = _require_acp()
        await self._runtime.close_session(session_id)
        return schema.CloseSessionResponse()

    async def set_session_mode(self, session_id: str, mode_id: str, **kwargs: JsonValue) -> None:
        self._require_initialized()

    async def set_config_option(self, config_id: str, session_id: str, value: JsonValue, **kwargs: JsonValue) -> None:
        self._require_initialized()

    async def authenticate(self, method_id: str, **kwargs: JsonValue) -> None:
        acp, _ = _require_acp()
        raise acp.RequestError("unknown_auth_method")

    async def prompt(self, session_id: str, prompt: 'list[ACPTextContent]', **kwargs: JsonValue) -> JsonValue:
        self._require_initialized()
        _, schema = _require_acp()
        text = "".join(item.text for item in prompt)

        async def on_event(event: 'dict[str, JsonValue]') -> None:
            if self._connection is None:
                return
            update = _acp_update(schema, event)
            if update is not None:
                await self._connection.session_update(session_id, update)

        await self._runtime.run(session_id, text, on_event=on_event)
        _logger.info("ACP prompt completed: session=%s", session_id)
        return schema.PromptResponse(stopReason="end_turn")

    async def cancel(self, session_id: str, **kwargs: JsonValue) -> None:
        self._require_initialized()
        await self._runtime.cancel(session_id)

    async def ext_method(self, method: str, params: 'dict[str, JsonValue]') -> None:
        acp, _ = _require_acp()
        raise acp.RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: 'dict[str, JsonValue]') -> None:
        return None

    def _require_initialized(self) -> None:
        if not self._initialized:
            acp, _ = _require_acp()
            raise acp.RequestError("initialize_required")


@dataclass(frozen=True, slots=True)
class ACPApplication:
    workspace: Workspace
    runtime: WorkspaceAgentRuntime | None = None

    @classmethod
    def for_workspace(cls, workspace: Workspace) -> "ACPApplication":
        return cls(workspace)

    def agent(self) -> ACPAgent:
        if self.runtime is None:
            raise RuntimeError("ACP runtime is not open")
        return ACPAgent(self.runtime)

    async def serve(self) -> None:
        async with open_workspace_runtime(self.workspace) as runtime:
            self.runtime = runtime
            await serve_stdio(self.agent())


async def serve_stdio(agent: ACPAgent) -> None:
    acp, _ = _require_acp()
    await acp.run_agent(agent, use_unstable_protocol=True)


def run_stdio(agent: ACPAgent) -> None:
    asyncio.run(serve_stdio(agent))


def _require_acp() -> 'tuple[ModuleType, ModuleType]':
    if _acp is None or _acp_schema is None:
        raise ModuleNotFoundError("agent-client-protocol")
    return _acp, _acp_schema


def _acp_update(schema, event: 'dict[str, JsonValue]') -> 'JsonValue | None':
    event_type = event.get("type")
    if event_type == "text":
        return schema.AgentMessageChunk(
            content=schema.TextContentBlock(type="text", text=str(event.get("text", ""))),
            sessionUpdate="agent_message_chunk",
        )
    if event_type == "thinking":
        return schema.AgentThoughtChunk(
            content=schema.TextContentBlock(type="text", text=str(event.get("text", ""))),
            sessionUpdate="agent_thought_chunk",
        )
    if event_type != "tool":
        return None
    title = str(event.get("name", "tool"))
    kind = "execute" if title == "bash" else "read" if title in {"list_dir", "read_file"} else "edit"
    if event.get("phase") == "start":
        return schema.ToolCallStart(
            toolCallId=str(event.get("id", "")),
            title=title,
            kind=kind,
            status="in_progress",
            rawInput=event.get("arguments"),
            sessionUpdate="tool_call",
        )
    return schema.ToolCallProgress(
        toolCallId=str(event.get("id", "")),
        kind=kind,
        status="completed" if event.get("ok") else "failed",
        rawOutput=event.get("detail"),
        sessionUpdate="tool_call_update",
    )


__all__ = ["ACPAgent", "ACPApplication", "ACPConnection", "run_stdio", "serve_stdio"]
