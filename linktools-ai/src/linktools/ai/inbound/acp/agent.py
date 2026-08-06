#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ACP method adapter for the local Agent runtime."""

from typing import Any
from uuid import uuid4

from linktools.core import environ

from ...local.runtime import LocalAgentRuntime

logger = environ.get_logger("ai.inbound.acp.agent")


class LocalACPAgent:
    def __init__(self, runtime: LocalAgentRuntime) -> None:
        self._runtime = runtime
        self._connection: Any = None
        self._initialized = False

    def on_connect(self, connection: Any) -> None:
        self._connection = connection

    async def initialize(self, protocol_version: int, **kwargs: Any) -> Any:
        import acp
        import acp.schema as schema

        if protocol_version != acp.PROTOCOL_VERSION:
            raise acp.RequestError("no_common_protocol_version")
        self._initialized = True
        logger.info("ACP initialized: protocol_version=%s", protocol_version)
        return schema.InitializeResponse(
            protocolVersion=protocol_version,
            agentCapabilities=schema.AgentCapabilities(loadSession=True),
            authMethods=[],
            agentInfo=schema.Implementation(name="linktools-ai", version="0.1"),
        )

    async def new_session(self, cwd: str, **kwargs: Any) -> Any:
        self._require_initialized()
        import acp.schema as schema

        session = await self._runtime.open_session(uuid4().hex, cwd=cwd)
        logger.info("ACP session created: session=%s cwd=%s", session.session_id, cwd)
        return schema.NewSessionResponse(sessionId=session.session_id)

    async def load_session(self, cwd: str, session_id: str, **kwargs: Any) -> Any:
        self._require_initialized()
        import acp.schema as schema

        await self._runtime.open_session(session_id, cwd=cwd)
        return schema.LoadSessionResponse()

    async def list_sessions(self, cwd: "str | None" = None, **kwargs: Any) -> Any:
        self._require_initialized()
        import acp.schema as schema

        sessions = await self._runtime.list_sessions(cwd=cwd)
        return schema.ListSessionsResponse(
            sessions=[schema.SessionInfo(sessionId=item.session_id, cwd=item.cwd.as_posix()) for item in sessions]
        )

    async def resume_session(self, session_id: str, cwd: str, **kwargs: Any) -> Any:
        self._require_initialized()
        import acp.schema as schema

        await self._runtime.open_session(session_id, cwd=cwd)
        return schema.ResumeSessionResponse()

    async def fork_session(self, session_id: str, cwd: str, **kwargs: Any) -> Any:
        self._require_initialized()
        import acp.schema as schema

        session = await self._runtime.fork_session(session_id, cwd=cwd)
        return schema.ForkSessionResponse(sessionId=session.session_id)

    async def close_session(self, session_id: str, **kwargs: Any) -> Any:
        self._require_initialized()
        import acp.schema as schema

        await self._runtime.close_session(session_id)
        return schema.CloseSessionResponse()

    async def set_session_mode(self, session_id: str, mode_id: str, **kwargs: Any) -> None:
        self._require_initialized()

    async def set_config_option(self, config_id: str, session_id: str, value: Any, **kwargs: Any) -> None:
        self._require_initialized()

    async def authenticate(self, method_id: str, **kwargs: Any) -> None:
        import acp

        raise acp.RequestError("unknown_auth_method")

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> Any:
        self._require_initialized()
        import acp.schema as schema

        text = "".join(item.text for item in prompt if isinstance(item, schema.TextContentBlock))
        logger.info("ACP prompt received: session=%s", session_id)
        async def on_event(event: "dict[str, object]") -> None:
            if self._connection is None:
                return
            update = _acp_update(schema, event)
            if update is not None:
                await self._connection.session_update(session_id, update)

        await self._runtime.run(session_id, text, on_event=on_event)
        logger.info("ACP prompt completed: session=%s", session_id)
        return schema.PromptResponse(stopReason="end_turn")

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        self._require_initialized()
        await self._runtime.cancel(session_id)

    async def ext_method(self, method: str, params: dict[str, Any]) -> Any:
        import acp

        raise acp.RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        return None

    def _require_initialized(self) -> None:
        if not self._initialized:
            import acp

            raise acp.RequestError("initialize_required")


def _acp_update(schema: Any, event: "dict[str, object]") -> Any:
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
    phase = event.get("phase")
    tool_call_id = str(event.get("id", ""))
    title = str(event.get("name", "tool"))
    kind = "execute" if title == "bash" else "read" if title in {"list_dir", "read_file"} else "edit"
    if phase == "start":
        return schema.ToolCallStart(
            toolCallId=tool_call_id,
            title=title,
            kind=kind,
            status="in_progress",
            rawInput=event.get("arguments"),
            sessionUpdate="tool_call",
        )
    return schema.ToolCallProgress(
        toolCallId=tool_call_id,
        kind=kind,
        status="completed" if event.get("ok") else "failed",
        rawOutput=event.get("detail"),
        sessionUpdate="tool_call_update",
    )


__all__ = ["LocalACPAgent"]
