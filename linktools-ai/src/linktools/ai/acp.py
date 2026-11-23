#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transport-only ACP adapter over the public Runtime."""

import asyncio
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

from linktools.core import environ

try:
    import acp as _acp
    import acp.schema as _acp_schema
except ModuleNotFoundError:
    _acp = None
    _acp_schema = None

from .capability import ExecutionCapability, FileSystem
from .runtime import RuntimeFeatures
from .persistence import PersistenceStorage
from .core import (
    AuthorizationPolicy,
    ExecutionDeltaType,
    ExecutionStatus,
    JsonValue,
    Principal,
)
from .errors import AIError, ErrorCode
from .execution import ExecutionTreeEvent, ListSessionRequest
from .runtime import Runtime
from .session import project_tool_process_event
from .workspace import Workspace

if TYPE_CHECKING:
    from acp.schema import ResourceContentBlock

    from .execution import Execution
    from .session import ToolProcessItem

_logger = environ.get_logger("ai.acp")


class ACPConnection(Protocol):
    async def session_update(self, session_id: str, update: JsonValue) -> None: ...


class ACPTextContent(Protocol):
    text: str


class ACPAgent:
    """Translate ACP requests into Runtime calls and durable event reads."""

    def __init__(self, runtime: Runtime, *, principal: Principal, workspace: Workspace | None = None) -> None:
        self._runtime = runtime
        self._principal = principal
        self._workspace = workspace
        self._connection: ACPConnection | None = None
        self._initialized = False
        self._prompts: dict[str, asyncio.Future[Execution | None]] = {}

    def on_connect(self, connection: ACPConnection) -> None:
        if self._connection is not connection:
            self._prompts.clear()
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
        acp, schema = _require_acp()
        session_id = uuid4().hex
        try:
            logical_cwd = self._workspace_cwd(cwd)
            session = await self._runtime.agents.get().create_session(
                session_id, principal=self._principal, cwd=logical_cwd,
            )
        except AIError as error:
            raise _acp_error(acp, error, session_id=session_id) from error
        return schema.NewSessionResponse(sessionId=session.session_id)

    async def load_session(self, cwd: str, session_id: str, **kwargs: JsonValue) -> JsonValue:
        self._require_initialized()
        acp, schema = _require_acp()
        connection = self._connection
        cursor = None
        pages = []
        try:
            logical_cwd = None if self._workspace is None else self._workspace_cwd(cwd)
            while True:
                detail = await self._runtime.history.session_detail(
                    session_id, principal=self._principal, cursor=cursor,
                )
                if logical_cwd is not None and (detail.session.cwd or ".") != logical_cwd:
                    raise acp.RequestError.invalid_params({"reason": "workspace_cwd_mismatch"})
                page = detail.timeline
                pages.append(page.items)
                cursor = page.next_cursor
                if cursor is None:
                    break
        except AIError as error:
            raise _acp_error(acp, error, session_id=session_id) from error
        if connection is not None:
            for turns in reversed(pages):
                for turn in turns:
                    if not turn.conversation_committed:
                        continue
                    for text in _history_text(turn.user_input):
                        await connection.session_update(session_id, schema.UserMessageChunk(
                            content=schema.TextContentBlock(type="text", text=text), sessionUpdate="user_message_chunk",
                        ))
                    process_by_key = {item.key: item for item in turn.process}
                    for item in turn.items:
                        if item.item_kind in ("assistant", "thinking") and isinstance(item.content, str) and item.content:
                            await connection.session_update(session_id, _assistant_chunk(
                                schema, item.content, thinking=item.item_kind == "thinking",
                            ))
                        elif item.item_kind == "tool_call" and item.tool_call_id is not None:
                            process = process_by_key.pop(f"tool:{item.tool_call_id}", None)
                            if process is not None:
                                await connection.session_update(
                                    session_id, _acp_tool_update(schema, turn.execution_id, process, replay=True),
                                )
        return schema.LoadSessionResponse()

    async def list_sessions(
        self,
        cwd: "str | None" = None,
        cursor: "str | None" = None,
        **kwargs: JsonValue,
    ) -> JsonValue:
        self._require_initialized()
        _, schema = _require_acp()
        workspace = self._workspace
        try:
            logical_cwd = None if cwd is None else self._workspace_cwd(cwd)
        except AIError:
            return schema.ListSessionsResponse(sessions=[])
        page = await self._runtime.sessions.list(
            ListSessionRequest(self._principal, cursor=cursor, limit=200),
        )
        sessions = []
        for item in page.items:
            directory = item.cwd if workspace is None else item.cwd or "."
            if logical_cwd is None or directory == logical_cwd:
                sessions.append(schema.SessionInfo(
                    sessionId=item.session_id,
                    cwd=(directory or "") if workspace is None else str(workspace.root / (directory or ".")),
                ))
        return schema.ListSessionsResponse(sessions=sessions, nextCursor=page.next_cursor)

    async def resume_session(self, session_id: str, cwd: str, **kwargs: JsonValue) -> JsonValue:
        return await self.load_session(cwd, session_id, **kwargs)

    async def fork_session(self, session_id: str, cwd: str, **kwargs: JsonValue) -> JsonValue:
        self._require_initialized()
        acp, schema = _require_acp()
        from .execution import ForkSessionRequest

        try:
            logical_cwd = self._workspace_cwd(cwd)
            source = await self._runtime.sessions.get(session_id, principal=self._principal)
            session = await self._runtime.sessions.fork(
                source.agent_id, session_id,
                ForkSessionRequest(self._principal, uuid4().hex, uuid4().hex, logical_cwd),
            )
        except AIError as error:
            raise _acp_error(acp, error, session_id=session_id) from error
        return schema.ForkSessionResponse(sessionId=session.session_id)

    def _workspace_cwd(self, cwd: str) -> str:
        workspace = self._workspace
        if workspace is None:
            return cwd
        try:
            path = Path(cwd)
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError
            resolved = path.resolve()
            relative = resolved.relative_to(workspace.root)
            # Keep an in-workspace logical alias after checking physical containment.
            if path.is_relative_to(workspace.root):
                relative = path.relative_to(workspace.root)
            return workspace.validate_path(relative.as_posix())
        except (OSError, RuntimeError, ValueError) as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID,
                          safe_details={"field": "cwd", "reason": "invalid_workspace_cwd"}) from error

    async def close_session(self, session_id: str, **kwargs: JsonValue) -> JsonValue:
        self._require_initialized()
        _, schema = _require_acp()
        from .execution import CloseSessionRequest

        await self._runtime.sessions.close(session_id, CloseSessionRequest(self._principal, uuid4().hex))
        return schema.CloseSessionResponse()

    async def set_session_mode(self, session_id: str, mode_id: str, **kwargs: JsonValue) -> None:
        self._require_initialized()

    async def set_config_option(self, config_id: str, session_id: str, value: JsonValue, **kwargs: JsonValue) -> None:
        self._require_initialized()

    async def authenticate(self, method_id: str, **kwargs: JsonValue) -> None:
        acp, _ = _require_acp()
        raise acp.RequestError.invalid_params({"reason": "unknown_auth_method"})

    async def prompt(self, session_id: str, prompt: "list[ACPTextContent | ResourceContentBlock]", **kwargs: JsonValue) -> JsonValue:
        self._require_initialized()
        acp, schema = _require_acp()
        parts: list[str] = []
        for item in prompt:
            kind = getattr(item, "type", "text")
            if kind == "text" and isinstance(getattr(item, "text", None), str):
                parts.append(item.text)
            elif kind == "resource_link":
                # Links are context, not permission to read or fetch their URI.
                fields = [f"Resource: {item.name}", f"URI: {item.uri}"]
                for name, label in (("title", "Title"), ("description", "Description"),
                                    ("mime_type", "MIME type"), ("size", "Size")):
                    value = getattr(item, name, None)
                    if value is not None:
                        fields.append(f"{label}: {value}")
                parts.append("\n" + "\n".join(fields) + "\n")
            else:
                raise acp.RequestError.invalid_params({"reason": "unsupported_prompt_content"})
        if session_id in self._prompts:
            raise _acp_error(acp, AIError(ErrorCode.SESSION_BUSY), session_id=session_id)
        connection = self._connection
        admitted: asyncio.Future[Execution | None] = asyncio.get_running_loop().create_future()
        self._prompts[session_id] = admitted
        execution_id = None
        request_id = uuid4().hex

        async def on_event(item: ExecutionTreeEvent) -> None:
            if item.depth != 0 or connection is None or connection is not self._connection:
                return
            event = item.event
            payload = event.payload
            thinking = event.event_type == ExecutionDeltaType.ASSISTANT_THINKING_DELTA.value
            if thinking or event.event_type == ExecutionDeltaType.ASSISTANT_TEXT_DELTA.value:
                if isinstance(payload, dict) and isinstance(text := payload.get("text"), str) and text:
                    await connection.session_update(session_id, _assistant_chunk(schema, text, thinking=thinking))
            else:
                process = project_tool_process_event(event.event_type, payload)
                if process is not None:
                    await connection.session_update(session_id, _acp_tool_update(schema, item.execution_id, process))

        try:
            execution = await self._runtime.sessions.start(
                session_id, "".join(parts), principal=self._principal, idempotency_key=request_id,
            )
            execution_id = execution.execution_id
            admitted.set_result(execution)
            outcome = await execution.wait(on_event=on_event, include_event_content=True)
            result = outcome.result
            if outcome.observation_error is not None:
                _logger.warning(
                    "ACP prompt observation incomplete: execution_id=%s code=%s",
                    execution_id, outcome.observation_error.code.value,
                )
            if result.status is ExecutionStatus.FAILED:
                try:
                    code = ErrorCode(result.error_code)
                except (TypeError, ValueError):
                    code = ErrorCode.EXECUTION_FAILED
                raise AIError(code)
        except AIError as error:
            raise _acp_error(acp, error, session_id=session_id, operation_id=request_id,
                             execution_id=execution_id) from error
        finally:
            if self._prompts.get(session_id) is admitted:
                del self._prompts[session_id]
            if not admitted.done():
                admitted.set_result(None)
        stop_reason = "cancelled" if result.status is ExecutionStatus.CANCELLED else "end_turn"
        return schema.PromptResponse(stopReason=stop_reason)

    async def cancel(self, session_id: str, **kwargs: JsonValue) -> None:
        self._require_initialized()
        acp, _ = _require_acp()
        admitted = self._prompts.get(session_id)
        if admitted is None:
            return
        execution = await asyncio.shield(admitted)
        if execution is None or self._prompts.get(session_id) is not admitted:
            return
        try:
            await execution.cancel()
        except AIError as error:
            raise _acp_error(acp, error, session_id=session_id, execution_id=execution.execution_id) from error

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
    features: RuntimeFeatures
    storage: PersistenceStorage
    execution: ExecutionCapability | None = None
    authorization: AuthorizationPolicy | None = None

    @classmethod
    def for_workspace(
        cls,
        workspace: Workspace,
        *,
        features: RuntimeFeatures,
        storage: PersistenceStorage,
        execution: ExecutionCapability | None = None,
        authorization: AuthorizationPolicy | None = None,
    ) -> "ACPApplication":
        """Assemble workspace and execution capabilities into the given features."""
        return cls(workspace, features, storage, execution, authorization)

    def __post_init__(self) -> None:
        """Complete this application's feature declarations once, before serving."""
        self.features.capability(FileSystem(self.workspace))
        from .capability import ShellCapability
        self.features.capability(ShellCapability())
        self.features.capability(
            ExecutionCapability(retain_history=True) if self.execution is None else self.execution
        )

    async def serve(self) -> None:
        async with Runtime.open(
            "default", storage=self.storage, features=self.features,
            authorization=self.authorization,
        ) as runtime:
            await serve_stdio(
                ACPAgent(
                    runtime,
                    principal=runtime.default_principal,
                    workspace=self.workspace,
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


def _acp_error(
    acp: ModuleType, error: AIError, *, session_id: str,
    operation_id: str | None = None, execution_id: str | None = None,
) -> Exception:
    if error.code in {ErrorCode.SESSION_NOT_FOUND, ErrorCode.AUTHORIZATION_DENIED}:
        return acp.RequestError(-32002, "Resource not found", {"sessionId": session_id})
    safe = AIError(
        error.code, retryable=error.retryable,
        safe_details={} if execution_id is None else {"execution_id": execution_id},
    )
    data = asdict(safe.to_safe_error(operation_id=operation_id or execution_id or uuid4().hex))
    if error.code is ErrorCode.SESSION_BUSY:
        return acp.RequestError(-32901, "Session busy", data)
    if error.code is ErrorCode.REQUEST_FIELD_INVALID:
        return acp.RequestError.invalid_params(data)
    return acp.RequestError.internal_error(data)


def _history_text(value: JsonValue) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, dict):
        if value.get("version") == 1:
            return _history_text(value.get("prompt"))
        if value.get("kind") == "text" and isinstance(text := value.get("text"), str) and text:
            return (text,)
        if value.get("kind") == "items" and isinstance(items := value.get("items"), list):
            return tuple(text for item in items for text in _history_text(item))
    return ()


def _assistant_chunk(schema: ModuleType, text: str, *, thinking: bool) -> JsonValue:
    if thinking:
        return schema.AgentThoughtChunk(
            content=schema.TextContentBlock(type="text", text=text), sessionUpdate="agent_thought_chunk",
        )
    return schema.AgentMessageChunk(
        content=schema.TextContentBlock(type="text", text=text), sessionUpdate="agent_message_chunk",
    )


def _acp_tool_update(
    schema: ModuleType, execution_id: str, process: "ToolProcessItem", *, replay: bool = False,
) -> JsonValue:
    call_id = f"{execution_id}:{process.key.removeprefix('tool:')}"
    status = {
        "RUNNING": "in_progress",
        "UNCONFIRMED": "pending",
        "SUCCEEDED": "completed",
        "FAILED": "failed",
        "INTERRUPTED": "failed",
        "CANCELLED": "failed",
    }[process.status]
    if replay or process.status == "RUNNING":
        return schema.ToolCallStart(
            toolCallId=call_id, title=process.tool_name, kind="execute", status=status, sessionUpdate="tool_call",
        )
    return schema.ToolCallProgress(toolCallId=call_id, status=status, sessionUpdate="tool_call_update")


__all__ = ["ACPAgent", "ACPApplication", "ACPConnection", "run_stdio", "serve_stdio"]
