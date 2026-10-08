#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transport-only ACP adapter over the public Runtime."""

import asyncio
from dataclasses import asdict, dataclass
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

from .capability import CapabilityGroup
from .core import (
    ExecutionDeltaType,
    ExecutionEventType,
    ExecutionStatus,
    JsonValue,
    Principal,
    validate_memory_scope,
)
from .errors import AIError, ErrorCode
from .model import ModelRegistry
from .runtime import (
    CancelExecutionRequest,
    ExecutionTreeEvent,
    ListSessionRequest,
    Runtime,
    RuntimeStorage,
)
from .workspace import Workspace

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
            raise ValueError("memory scope is invalid") from error
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
            raise acp.RequestError.invalid_request({"reason": "no_common_protocol_version"})
        self._initialized = True
        return schema.InitializeResponse(
            protocolVersion=protocol_version,
            agentCapabilities=schema.AgentCapabilities(
                loadSession=True,
                sessionCapabilities=schema.SessionCapabilities(
                    list=schema.SessionListCapabilities(),
                ),
            ),
            authMethods=[],
            agentInfo=schema.Implementation(name="linktools-ai", version="0.1"),
        )

    async def new_session(self, cwd: str, **kwargs: JsonValue) -> JsonValue:
        self._require_initialized()
        _, schema = _require_acp()
        session = await self._runtime.agents.get().create_session(
            uuid4().hex,
            principal=self._principal,
            cwd=cwd,
        )
        return schema.NewSessionResponse(sessionId=session.session_id)

    async def load_session(self, cwd: str, session_id: str, **kwargs: JsonValue) -> JsonValue:
        self._require_initialized()
        _, schema = _require_acp()
        await self._runtime.sessions.get(session_id, principal=self._principal)
        return schema.LoadSessionResponse()

    async def list_sessions(
        self,
        cwd: "str | None" = None,
        cursor: "str | None" = None,
        **kwargs: JsonValue,
    ) -> JsonValue:
        self._require_initialized()
        _, schema = _require_acp()
        page = await self._runtime.sessions.list(
            ListSessionRequest(self._principal, cursor=cursor, limit=200),
        )
        return schema.ListSessionsResponse(
            sessions=[
                schema.SessionInfo(sessionId=item.session_id, cwd=item.cwd or "")
                for item in page.items
                if cwd is None or item.cwd == cwd
            ],
            nextCursor=page.next_cursor,
        )

    async def resume_session(self, session_id: str, cwd: str, **kwargs: JsonValue) -> JsonValue:
        return await self.load_session(cwd, session_id, **kwargs)

    async def fork_session(self, session_id: str, cwd: str, **kwargs: JsonValue) -> JsonValue:
        self._require_initialized()
        _, schema = _require_acp()
        from .runtime import ForkSessionRequest

        source = await self._runtime.sessions.get(session_id, principal=self._principal)
        session = await self._runtime.sessions.fork(
            source.agent_id,
            session_id,
            ForkSessionRequest(self._principal, uuid4().hex, uuid4().hex, cwd),
        )
        return schema.ForkSessionResponse(sessionId=session.session_id)

    async def close_session(self, session_id: str, **kwargs: JsonValue) -> JsonValue:
        self._require_initialized()
        _, schema = _require_acp()
        from .runtime import CloseSessionRequest

        await self._runtime.sessions.close(session_id, CloseSessionRequest(self._principal, uuid4().hex))
        return schema.CloseSessionResponse()

    async def set_session_mode(self, session_id: str, mode_id: str, **kwargs: JsonValue) -> None:
        self._require_initialized()

    async def set_config_option(self, config_id: str, session_id: str, value: JsonValue, **kwargs: JsonValue) -> None:
        self._require_initialized()

    async def authenticate(self, method_id: str, **kwargs: JsonValue) -> None:
        acp, _ = _require_acp()
        raise acp.RequestError.invalid_params({"reason": "unknown_auth_method"})

    async def prompt(self, session_id: str, prompt: "list[ACPTextContent]", **kwargs: JsonValue) -> JsonValue:
        self._require_initialized()
        acp, schema = _require_acp()
        text = "".join(item.text for item in prompt)
        loaded = await self._runtime.sessions.get(session_id, principal=self._principal)
        execution = await self._runtime.agents.get(loaded.agent_id).start(
            text,
            principal=self._principal,
            session_id=session_id,
            memory_scope=self._memory_scope,
        )
        async def on_event(item: ExecutionTreeEvent) -> None:
            if item.depth != 0:
                return
            event = item.event
            if self._connection is not None:
                update = _acp_update(schema, event.event_type, event.payload)
                if update is not None:
                    await self._connection.session_update(session_id, update)

        try:
            outcome = await execution.wait(on_event=on_event, include_event_content=True)
            result = outcome.result
            if outcome.observation_error is not None:
                _logger.warning(
                    "ACP prompt observation incomplete: execution_id=%s code=%s",
                    execution.execution_id, outcome.observation_error.code.value,
                )
            if result.status is ExecutionStatus.FAILED:
                try:
                    code = ErrorCode(result.error_code)
                except (TypeError, ValueError):
                    code = ErrorCode.EXECUTION_FAILED
                raise AIError(code)
        except AIError as error:
            failure = AIError(
                error.code,
                retryable=error.retryable,
                safe_details={"execution_id": execution.execution_id},
            )
            raise acp.RequestError.internal_error(asdict(
                failure.to_safe_error(operation_id=execution.execution_id)
            )) from error
        stop_reason = "cancelled" if result.status is ExecutionStatus.CANCELLED else "end_turn"
        return schema.PromptResponse(stopReason=stop_reason)

    async def cancel(self, session_id: str, **kwargs: JsonValue) -> None:
        loaded = await self._runtime.sessions.reconcile(
            session_id,
            principal=self._principal,
        )
        if loaded.active_execution_id is not None:
            await self._runtime.executions.cancel(
                loaded.active_execution_id,
                CancelExecutionRequest(self._principal, uuid4().hex, True),
            )

    async def ext_method(self, method: str, params: "dict[str, JsonValue]") -> None:
        acp, _ = _require_acp()
        raise acp.RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: "dict[str, JsonValue]") -> None:
        return None

    def _require_initialized(self) -> None:
        if not self._initialized:
            acp, _ = _require_acp()
            raise acp.RequestError.invalid_request({"reason": "initialize_required"})


@dataclass(frozen=True, slots=True)
class ACPApplication:
    workspace: Workspace
    models: ModelRegistry
    storage: RuntimeStorage

    @classmethod
    def for_workspace(
        cls,
        workspace: Workspace,
        *,
        models: ModelRegistry,
        storage: RuntimeStorage,
    ) -> "ACPApplication":
        return cls(workspace, models, storage)

    async def serve(self, *, memory_scope: str) -> None:
        async with Runtime.open(
            "default",
            models=self.models,
            storage=self.storage,
            capabilities=(CapabilityGroup("workspace", workspace=self.workspace),),
        ) as runtime:
            await serve_stdio(
                ACPAgent(
                    runtime,
                    principal=runtime.default_principal,
                    memory_scope=memory_scope,
                )
            )


async def serve_stdio(agent: ACPAgent) -> None:
    acp, _ = _require_acp()
    await acp.run_agent(agent, use_unstable_protocol=True)


def run_stdio(agent: ACPAgent) -> None:
    asyncio.run(serve_stdio(agent))


def _require_acp() -> "tuple[ModuleType, ModuleType]":
    if _acp is None or _acp_schema is None:
        raise ModuleNotFoundError("agent-client-protocol")
    return _acp, _acp_schema


def _acp_update(
    schema: ModuleType,
    event_type: str,
    payload: JsonValue,
) -> "JsonValue | None":
    if not isinstance(payload, dict):
        return None
    if event_type == ExecutionDeltaType.ASSISTANT_TEXT_DELTA.value:
        return schema.AgentMessageChunk(content=schema.TextContentBlock(type="text", text=str(payload.get("text", ""))), sessionUpdate="agent_message_chunk")
    if event_type == ExecutionDeltaType.ASSISTANT_THINKING_DELTA.value:
        return schema.AgentThoughtChunk(content=schema.TextContentBlock(type="text", text=str(payload.get("text", ""))), sessionUpdate="agent_thought_chunk")
    if event_type == ExecutionEventType.TOOL_CALL_STARTED.value:
        return schema.ToolCallStart(toolCallId=str(payload.get("call_id", "")), title=str(payload.get("tool_name", "tool")), kind="execute", status="in_progress", sessionUpdate="tool_call")
    if event_type == ExecutionEventType.TOOL_CALL_FINISHED.value:
        return schema.ToolCallProgress(toolCallId=str(payload.get("call_id", "")), kind="execute", status="completed" if payload.get("status") == "SUCCEEDED" else "failed", sessionUpdate="tool_call_update")
    return None


__all__ = ["ACPAgent", "ACPApplication", "ACPConnection", "run_stdio", "serve_stdio"]
