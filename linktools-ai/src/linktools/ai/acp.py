#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transport-only ACP adapter over the public Runtime."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from secrets import token_urlsafe
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

from .capability import FileSystemCapability
from .runtime import RuntimeFeatures
from .persistence import PersistenceStorage
from .core import (
    AuthorizationPolicy,
    ExecutionDeltaType,
    ExecutionStatus,
    JsonValue,
    Principal,
    PrincipalKind,
)
from .errors import AIError, ErrorCode
from .execution import ExecutionTreeEvent
from .session import ListSessionRequest
from .runtime import Runtime
from .runtime.turn_clients import TurnObserver, uses_turn_sessions
from .session import ToolProcessItem, project_tool_process_event
from .workspace import Workspace

if TYPE_CHECKING:
    from acp.schema import ResourceContentBlock

    from .execution import Execution

_logger = environ.get_logger("ai.acp")


class ACPConnection(Protocol):
    async def session_update(self, session_id: str, update: JsonValue) -> None: ...


class ACPTextContent(Protocol):
    text: str


class ACPAgent:
    """Translate ACP requests into Runtime calls and durable event reads."""

    def __init__(
        self, runtime: Runtime, *, principal: Principal, workspace: Workspace | None = None,
        list_session_ids: Callable[[Principal], Awaitable[Sequence[str]]] | None = None,
    ) -> None:
        """Bind a host's explicit session scope for embedded SQL listing."""
        self._runtime = runtime
        self._principal = principal
        self._workspace = workspace
        self._list_session_ids = list_session_ids
        self._connection: ACPConnection | None = None
        self._initialized = False
        self._prompts: dict[str, asyncio.Future[Execution | str | None]] = {}
        self._local_workspace = principal.kind == PrincipalKind.LOCAL_TRUSTED.value

    def _can_list_sessions(self) -> bool:
        return self._local_workspace or self._list_session_ids is not None or not uses_turn_sessions(self._runtime)

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
                    list=schema.SessionListCapabilities() if self._can_list_sessions() else None,
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
            if uses_turn_sessions(self._runtime):
                await self._runtime.sessions.create_turn_session(
                    session_id, principal=self._principal, cwd=logical_cwd,
                )
                return schema.NewSessionResponse(sessionId=session_id)
            session = await self._runtime.agents.get().create_session(
                session_id, principal=self._principal, cwd=logical_cwd,
            )
        except AIError as error:
            raise _acp_error(acp, error, session_id=session_id) from error
        return schema.NewSessionResponse(sessionId=session.session_id)

    async def load_session(self, cwd: str, session_id: str, **kwargs: JsonValue) -> JsonValue:
        self._require_initialized()
        acp, schema = _require_acp()
        if uses_turn_sessions(self._runtime):
            return await self._load_turn_session(cwd, session_id, acp, schema)
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

    async def _load_turn_session(self, cwd: str, session_id: str, acp: ModuleType, schema: ModuleType) -> JsonValue:
        connection = self._connection
        cursor = None
        pages = []
        try:
            logical_cwd = None if self._workspace is None else self._workspace_cwd(cwd)
            while True:
                detail = await self._runtime.history.turn_session(
                    session_id, principal=self._principal, cursor=cursor, local_workspace=self._local_workspace,
                )
                if logical_cwd is not None and (detail["session"].get("cwd") or ".") != logical_cwd:
                    raise acp.RequestError.invalid_params({"reason": "workspace_cwd_mismatch"})
                page = detail["timeline"]
                pages.append(page["items"])
                cursor = page["next_cursor"]
                if cursor is None or connection is None or connection is not self._connection:
                    break
        except AIError as error:
            raise _acp_error(acp, error, session_id=session_id) from error
        for turns in reversed(pages):
            for turn in turns:
                if connection is None or connection is not self._connection:
                    break
                if not turn["conversation_committed"]:
                    continue
                for text in _history_text(turn["user_input"]):
                    await connection.session_update(session_id, schema.UserMessageChunk(
                        content=schema.TextContentBlock(type="text", text=text), sessionUpdate="user_message_chunk",
                    ))
                for update in _turn_replay(schema, turn["turn_id"], turn):
                    await connection.session_update(session_id, update)
        return schema.LoadSessionResponse()

    async def list_sessions(
        self,
        cwd: "str | None" = None,
        cursor: "str | None" = None,
        **kwargs: JsonValue,
    ) -> JsonValue:
        self._require_initialized()
        acp, schema = _require_acp()
        if not self._can_list_sessions():
            raise acp.RequestError.internal_error({"reason": "host_session_scope_required"})
        workspace = self._workspace
        try:
            logical_cwd = None if cwd is None else self._workspace_cwd(cwd)
        except AIError:
            return schema.ListSessionsResponse(sessions=[])
        if uses_turn_sessions(self._runtime):
            source = self._list_session_ids
            session_ids = None if source is None else await source(self._principal)
            if source is not None and session_ids is None:
                raise acp.RequestError.internal_error({"reason": "invalid_host_session_scope"})
            page = await self._runtime.history.turn_sessions(
                principal=self._principal, cursor=cursor, limit=64,
                session_ids=session_ids, local_workspace=self._local_workspace,
            )
            entries = ((item["session_id"], item.get("cwd")) for item in page["items"])
            next_cursor = page["next_cursor"]
        else:
            page = await self._runtime.sessions.list(
                ListSessionRequest(self._principal, cursor=cursor, limit=200),
            )
            entries = ((item.session_id, item.cwd) for item in page.items)
            next_cursor = page.next_cursor
        sessions = []
        for session_id, item_cwd in entries:
            directory = item_cwd if workspace is None else item_cwd or "."
            if logical_cwd is None or directory == logical_cwd:
                sessions.append(schema.SessionInfo(
                    sessionId=session_id,
                    cwd=(directory or "") if workspace is None else str(workspace.root / (directory or ".")),
                ))
        return schema.ListSessionsResponse(sessions=sessions, nextCursor=next_cursor)

    async def resume_session(self, session_id: str, cwd: str, **kwargs: JsonValue) -> JsonValue:
        return await self.load_session(cwd, session_id, **kwargs)

    async def fork_session(self, session_id: str, cwd: str, **kwargs: JsonValue) -> JsonValue:
        self._require_initialized()
        acp, schema = _require_acp()
        from .session import ForkSessionRequest

        try:
            logical_cwd = self._workspace_cwd(cwd)
            if uses_turn_sessions(self._runtime):
                source = await self._turn_session_state(session_id)
                if logical_cwd is not None and (source.get("cwd") or ".") != logical_cwd:
                    raise acp.RequestError.invalid_params({"reason": "workspace_cwd_mismatch"})
                target = uuid4().hex
                if not await self._runtime.sessions.fork_turn_session(
                    session_id, target, principal=self._principal,
                    expected_revision=source["revision"], idempotency_key=uuid4().hex,
                ):
                    raise AIError(ErrorCode.SESSION_REVISION_CONFLICT)
                return schema.ForkSessionResponse(sessionId=target)
            source = await self._runtime.sessions.get(session_id, principal=self._principal)
            session = await self._runtime.sessions.fork(
                source.agent_id, session_id,
                ForkSessionRequest(self._principal, uuid4().hex, uuid4().hex, logical_cwd),
            )
        except AIError as error:
            raise _acp_error(acp, error, session_id=session_id) from error
        return schema.ForkSessionResponse(sessionId=session.session_id)

    async def _turn_session_state(self, session_id: str) -> dict[str, JsonValue]:
        return await self._runtime.history.turn_session_state(
            session_id, principal=self._principal, local_workspace=self._local_workspace,
        )

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
        acp, schema = _require_acp()
        from .session import CloseSessionRequest

        if uses_turn_sessions(self._runtime):
            try:
                source = await self._turn_session_state(session_id)
                if not await self._runtime.sessions.close_turn_session(
                    session_id, principal=self._principal, expected_revision=source["revision"],
                ):
                    raise AIError(ErrorCode.SESSION_REVISION_CONFLICT)
            except AIError as error:
                raise _acp_error(acp, error, session_id=session_id) from error
        else:
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
        admitted: asyncio.Future[Execution | str | None] = asyncio.get_running_loop().create_future()
        self._prompts[session_id] = admitted
        execution_id = None
        native = uses_turn_sessions(self._runtime)
        request_id = token_urlsafe(9) if native else uuid4().hex

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
            if native:
                with TurnObserver(
                    self._runtime, session_id, self._principal, local_workspace=self._local_workspace,
                ) as observer:
                    turn = await self._runtime.sessions.start_turn(
                        session_id, "".join(parts), agent="default",
                        principal=self._principal, idempotency_key=request_id,
                    )
                    execution_id = turn.key.turn_id
                    admitted.set_result(execution_id)
                    sent_text = False
                    async for frame in observer.events(execution_id):
                        if frame["type"] == "terminal":
                            break
                        if connection is None or connection is not self._connection:
                            continue
                        if frame["type"] == "delta" and isinstance(text := frame.get("text"), str) and text:
                            thinking = frame.get("delta_kind") == "thinking"
                            sent_text |= not thinking
                            await connection.session_update(session_id, _assistant_chunk(schema, text, thinking=thinking))
                        elif frame["type"] == "tool" and isinstance(item := frame.get("item"), Mapping):
                            if (process := ToolProcessItem.from_notice(item)) is not None:
                                await connection.session_update(session_id, _acp_tool_update(schema, execution_id, process))
                    status = ExecutionStatus(str(frame["state"]).upper())
                    error_code = frame.get("error_code")
                    output = frame.get("output")
                    if (
                        not sent_text and connection is not None and connection is self._connection
                        and status is ExecutionStatus.SUCCEEDED and isinstance(output, Mapping)
                        and isinstance(text := output.get("text"), str) and text
                    ):
                        await connection.session_update(session_id, _assistant_chunk(schema, text, thinking=False))
                    if observer.incomplete and sent_text:
                        # ACP text chunks are append-only; silently appending a
                        # complete answer would duplicate a partial one.
                        raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            else:
                execution = await self._runtime.executions.start_in_session(
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
                status, error_code = result.status, result.error_code if result.status is ExecutionStatus.FAILED else None
            if status in (ExecutionStatus.WAITING_DEFERRED, ExecutionStatus.WAITING_RETRY):
                raise AIError(ErrorCode.EXECUTION_NOT_READY)
            if status is ExecutionStatus.RECOVERY_REQUIRED:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            if status is ExecutionStatus.FAILED:
                try:
                    code = ErrorCode(error_code)
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
        stop_reason = "cancelled" if status is ExecutionStatus.CANCELLED else "end_turn"
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
        execution_id = execution if isinstance(execution, str) else execution.execution_id
        try:
            if isinstance(execution, str):
                await self._runtime.sessions.cancel_turn(session_id, execution_id, principal=self._principal)
            else:
                await execution.cancel()
        except AIError as error:
            raise _acp_error(acp, error, session_id=session_id, execution_id=execution_id) from error

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
    authorization: AuthorizationPolicy | None = None
    list_session_ids: Callable[[Principal], Awaitable[Sequence[str]]] | None = None

    @classmethod
    def for_workspace(
        cls,
        workspace: Workspace,
        *,
        features: RuntimeFeatures,
        storage: PersistenceStorage,
        authorization: AuthorizationPolicy | None = None,
        list_session_ids: Callable[[Principal], Awaitable[Sequence[str]]] | None = None,
    ) -> "ACPApplication":
        """Assemble workspace capabilities into the given features."""
        return cls(workspace, features, storage, authorization, list_session_ids)

    def __post_init__(self) -> None:
        """Complete this application's feature declarations once, before serving."""
        defaults = dict(self.features.defaults)
        execution = defaults.get("execution", {})
        assert isinstance(execution, dict)
        if "retain_history" not in execution:
            defaults["execution"] = {**execution, "retain_history": True}
            object.__setattr__(self, "features", self.features.with_defaults(defaults))
        self.features.capability(FileSystemCapability(self.workspace))
        from .capability import ShellCapability
        self.features.capability(ShellCapability())

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
                    list_session_ids=self.list_session_ids,
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


def _turn_replay(schema: ModuleType, turn_id: str, turn: Mapping[str, JsonValue]):
    process = {
        tool.key: tool for item in turn["process"]
        if (tool := ToolProcessItem.from_notice(item)) is not None
    }
    for item in turn["items"]:
        kind, content = item["item_kind"], item.get("content")
        if kind in ("assistant", "thinking") and isinstance(content, str) and content:
            yield _assistant_chunk(schema, content, thinking=kind == "thinking")
        elif kind == "tool_call" and (tool := process.pop(f"tool:{item['tool_call_id']}", None)) is not None:
            yield _acp_tool_update(schema, turn_id, tool, replay=True)


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
