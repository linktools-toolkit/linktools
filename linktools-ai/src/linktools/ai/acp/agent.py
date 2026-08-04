#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ACP v1 Agent implementation over the protocol-neutral Runtime facade."""

import asyncio
import functools
import logging
from dataclasses import replace
from typing import Any, Awaitable, Callable
from uuid import uuid4

from ..execution.domain import ApprovalDecision, RunStatus
from ..execution.live_events import (
    ExecutionCancelled,
    ExecutionCompleted,
    ExecutionEventHub,
    ExecutionFailed,
    ExecutionPaused,
)
from ..governance.identity import PrincipalContext
from ..runtime.facade import Runtime
from .capabilities import AcpMode, CapabilityBuilder, CapabilityInput
from .client_services import AcpClientServices
from .content_mapper import AcpContentMapper
from .errors import internal_error, request_error
from .event_mapper import AcpEventMapper
from .execution import AcpExecutionAdapter
from .persistence import AcpSessionRepository
from .sessions import AcpSessionService, ActiveAcpSession

logger = logging.getLogger("linktools.ai.acp.agent")


def _protocol_handler(function: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(function)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return await function(*args, **kwargs)
        except Exception as exc:
            import acp

            if isinstance(exc, acp.RequestError):
                raise
            logger.warning(
                "ACP handler failed method=%s error_type=%s",
                function.__name__,
                type(exc).__name__,
            )
            raise internal_error("internal_error") from exc

    return wrapped

STANDARD_AGENT_METHODS = {
    "initialize": "initialize",
    "authenticate": "authenticate",
    "session/new": "new_session",
    "session/list": "list_sessions",
    "session/load": "load_session",
    "session/resume": "resume_session",
    "session/fork": "fork_session",
    "session/close": "close_session",
    "session/prompt": "prompt",
    "session/cancel": "cancel",
    "session/set_mode": "set_session_mode",
    "session/set_config_option": "set_config_option",
}


class LinktoolsAcpAgent:
    def __init__(
        self,
        *,
        runtime: Runtime,
        state_root: str,
        project_root: str,
        principal: PrincipalContext,
        spec_resolver: Callable[[str], Awaitable[Any]],
        modes: "tuple[AcpMode, ...]" = (),
        capability_input: "CapabilityInput | None" = None,
        version: str = "0.0.0",
    ) -> None:
        self.runtime = runtime
        self.version = version
        self.spec_resolver = spec_resolver
        self.capability_input = capability_input or CapabilityInput(
            modes=modes or (AcpMode("default", "Default"),)
        )
        if not self.capability_input.modes:
            self.capability_input = replace(
                self.capability_input,
                modes=(AcpMode("default", "Default"),),
            )
        self.capability_builder = CapabilityBuilder()
        self.event_hub = ExecutionEventHub()
        self.event_mapper = AcpEventMapper()
        self.client_services = AcpClientServices(project_root=project_root)
        self.sessions = AcpSessionService(
            runtime=runtime,
            repository=AcpSessionRepository(state_root),
            project_root=project_root,
            principal=principal,
            default_mode_id=self.capability_input.modes[0].id,
            mode_ids=tuple(mode.id for mode in self.capability_input.modes),
        )
        self._initialized = False
        self._client_capabilities: Any = None
        self._connection: Any = None
        self.event_hub.set_cancel_callback(self._cancel_execution_from_hub)

    async def _cancel_execution_from_hub(self, execution_id: str) -> None:
        for active in tuple(self.sessions.active_sessions.values()):
            if active.active_execution_id == execution_id:
                try:
                    await self.runtime.cancel(execution_id, principal=self.sessions.principal)
                except Exception:
                    logger.debug("ACP event consumer cancellation failed execution=%s", execution_id)
                return

    def handler_registry(self) -> dict[str, Callable[..., Any]]:
        return {method: getattr(self, name) for method, name in STANDARD_AGENT_METHODS.items()}

    def on_connect(self, connection: Any) -> None:
        self._connection = connection
        self.client_services.set_connection(connection, self._client_capabilities)

    @_protocol_handler
    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any = None,
        client_info: Any = None,
        **kwargs: Any,
    ) -> Any:
        import acp

        if protocol_version != acp.PROTOCOL_VERSION:
            raise acp.RequestError.invalid_params({"reason": "no_common_protocol_version"})
        self._client_capabilities = client_capabilities
        self.client_services.set_connection(self._connection, client_capabilities)
        self._initialized = True
        logger.info("ACP initialized protocol=%s client=%s", protocol_version, getattr(client_info, "name", None))
        return acp.InitializeResponse(
            protocolVersion=protocol_version,
            agentCapabilities=self.capability_builder.build(
                self.capability_input,
                client_capabilities=client_capabilities,
            ),
            authMethods=[],
            agentInfo=self.capability_builder.agent_info(version=self.version),
        )

    @_protocol_handler
    async def authenticate(self, method_id: str, **kwargs: Any) -> Any:
        self._require_initialized()
        raise request_error("unknown_auth_method")

    @_protocol_handler
    async def new_session(self, cwd: str, additional_directories: "list[str] | None" = None, mcp_servers: "list[Any] | None" = None, **kwargs: Any) -> Any:
        self._require_initialized()
        import acp.schema as schema

        active = await self.sessions.create(
            cwd=cwd,
            additional_directories=additional_directories,
            mcp_servers=mcp_servers,
        )
        logger.info("ACP session created session=%s cwd=%s", active.record.session_id, active.record.cwd)
        return schema.NewSessionResponse(
            sessionId=active.record.session_id,
            modes=self._mode_state(active.record.mode_id),
            configOptions=[],
        )

    @_protocol_handler
    async def list_sessions(self, cwd: "str | None" = None, cursor: "str | None" = None, **kwargs: Any) -> Any:
        self._require_initialized()
        import acp.schema as schema

        records, next_cursor = await self.sessions.list(cwd=cwd, cursor=cursor)
        return schema.ListSessionsResponse(
            sessions=[
                schema.SessionInfo(
                    sessionId=record.session_id,
                    cwd=record.cwd,
                    additionalDirectories=list(record.additional_directories),
                    title=record.title,
                    updatedAt=record.updated_at.isoformat(),
                )
                for record in records
            ],
            nextCursor=next_cursor,
        )

    @_protocol_handler
    async def load_session(self, cwd: str, session_id: str, mcp_servers: "list[Any] | None" = None, additional_directories: "list[str] | None" = None, **kwargs: Any) -> Any:
        self._require_initialized()
        import acp.schema as schema

        active = await self.sessions.load_or_resume(
            session_id=session_id,
            cwd=cwd,
            additional_directories=additional_directories,
            mcp_servers=mcp_servers,
            replay=True,
        )
        await self._replay_history(active)
        return schema.LoadSessionResponse(modes=self._mode_state(active.record.mode_id), configOptions=[])

    @_protocol_handler
    async def resume_session(self, session_id: str, cwd: str, additional_directories: "list[str] | None" = None, mcp_servers: "list[Any] | None" = None, **kwargs: Any) -> Any:
        self._require_initialized()
        import acp.schema as schema

        active = await self.sessions.load_or_resume(
            session_id=session_id,
            cwd=cwd,
            additional_directories=additional_directories,
            mcp_servers=mcp_servers,
            replay=False,
        )
        return schema.ResumeSessionResponse(modes=self._mode_state(active.record.mode_id), configOptions=[])

    @_protocol_handler
    async def fork_session(self, session_id: str, cwd: str, additional_directories: "list[str] | None" = None, mcp_servers: "list[Any] | None" = None, **kwargs: Any) -> Any:
        self._require_initialized()
        import acp.schema as schema

        active = await self.sessions.fork(
            session_id,
            cwd=cwd,
            additional_directories=additional_directories,
            mcp_servers=mcp_servers,
        )
        return schema.ForkSessionResponse(
            sessionId=active.record.session_id,
            modes=self._mode_state(active.record.mode_id),
            configOptions=[],
        )

    @_protocol_handler
    async def close_session(self, session_id: str, **kwargs: Any) -> Any:
        self._require_initialized()
        import acp.schema as schema

        await self.sessions.close(session_id)
        return schema.CloseSessionResponse()

    @_protocol_handler
    async def set_session_mode(self, session_id: str, mode_id: str, **kwargs: Any) -> Any:
        self._require_initialized()
        await self.sessions.set_mode(session_id, mode_id)
        return None

    @_protocol_handler
    async def set_config_option(self, config_id: str, session_id: str, value: "str | bool", **kwargs: Any) -> Any:
        self._require_initialized()
        await self.sessions.set_config(session_id, config_id, value)
        return None

    @_protocol_handler
    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        self._require_initialized()
        try:
            active = await self.sessions.get(session_id)
        except Exception as exc:
            import acp

            if isinstance(exc, acp.RequestError):
                logger.info("ACP cancel ignored unknown session=%s", session_id)
                return
            raise
        async with active.lock:
            if active.active_execution_id is not None:
                try:
                    await self.runtime.cancel(active.active_execution_id, principal=self.sessions.principal)
                except (KeyError, ValueError):
                    logger.debug("ACP cancel raced execution creation session=%s", session_id)

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        from .errors import require_sdk

        acp = require_sdk()
        raise acp.RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        return None

    @_protocol_handler
    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> Any:
        self._require_initialized()
        active = await self.sessions.get(session_id)
        async with active.lock:
            if active.record.closed:
                raise request_error("session_closed", session_id=session_id)
            if active.active_execution_id is not None:
                raise request_error("session_busy", session_id=session_id)
            mapped = AcpContentMapper(
                image=self.capability_input.image,
                audio=self.capability_input.audio,
                embedded=self.capability_input.embedded_context,
            ).map(prompt)
            execution_id = uuid4().hex
            active.active_execution_id = execution_id
            logger.info("ACP prompt started session=%s execution=%s", session_id, execution_id)
        try:
            spec = await self.spec_resolver(active.record.mode_id)
            subscription = await self.event_hub.subscribe(execution_id)
            return await self._run_and_stream(
                active,
                execution_id,
                subscription,
                spec,
                mapped,
            )
        finally:
            async with active.lock:
                active.active_execution_id = None
            logger.info("ACP prompt detached session=%s execution=%s", session_id, execution_id)

    async def _run_and_stream(self, active: ActiveAcpSession, execution_id: str, subscription: Any, spec: Any, prompt: Any) -> Any:
        import acp.schema as schema

        task = asyncio.create_task(
            self.runtime.run(
                spec,
                prompt,
                principal=self.sessions.principal,
                session_id=active.record.session_id,
                execution_id=execution_id,
                extra_toolsets=await active.mcp_resources.toolsets(),
            )
        )
        try:
            await self._drain_until_done(active, execution_id, subscription, task)
            await task
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
            raise
        record = await self.runtime.get_execution_record(execution_id, principal=self.sessions.principal)
        if record is None:
            raise internal_error("execution_record_missing", session_id=active.record.session_id, execution_id=execution_id)
        if record.status is RunStatus.PAUSED:
            await subscription.release()
            await self._request_permission(active, record)
            return await self._resume_after_permission(active, execution_id, spec)
        if record.status is RunStatus.CANCELLED:
            return schema.PromptResponse(stopReason="cancelled")
        if record.status is RunStatus.FAILED:
            raise internal_error("execution_failed", session_id=active.record.session_id, execution_id=execution_id)
        return schema.PromptResponse(
            stopReason=AcpExecutionAdapter.stop_reason(record.status)
        )

    async def _drain_until_done(self, active: ActiveAcpSession, execution_id: str, subscription: Any, task: asyncio.Task) -> None:
        connection = self._connection
        while True:
            event_task = asyncio.create_task(subscription.__anext__())
            done, _ = await asyncio.wait({task, event_task}, return_when=asyncio.FIRST_COMPLETED)
            if event_task in done:
                event = event_task.result()
                update = self.event_mapper.map(event)
                if update is not None and connection is not None:
                    await connection.session_update(active.record.session_id, update)
                if isinstance(
                    event,
                    (ExecutionPaused, ExecutionCompleted, ExecutionFailed, ExecutionCancelled),
                ):
                    break
                continue
            event_task.cancel()
            try:
                await event_task
            except asyncio.CancelledError:
                pass
            if task.exception() is not None:
                await self.event_hub.close(execution_id, ExecutionFailed(execution_id=execution_id, error_id=uuid4().hex, error_type=type(task.exception()).__name__))
                raise internal_error("execution_failed", session_id=active.record.session_id, execution_id=execution_id)
            record = await self.runtime.get_execution_record(execution_id, principal=self.sessions.principal)
            if record is not None and record.status is RunStatus.PAUSED:
                await self.event_hub.publish(
                    execution_id,
                    ExecutionPaused(execution_id=execution_id),
                )
                return
            terminal: Any = ExecutionCompleted(execution_id=execution_id)
            if record is not None and record.status is RunStatus.CANCELLED:
                terminal = ExecutionCancelled(execution_id=execution_id)
            await self.event_hub.close(execution_id, terminal)

    async def _request_permission(self, active: ActiveAcpSession, record: Any) -> None:
        if self._connection is None:
            raise internal_error("client_connection_missing", session_id=active.record.session_id, execution_id=record.id)
        import acp.schema as schema

        async with active.lock:
            if active.active_execution_id != record.id:
                return
            detail = await self.runtime.inspect(
                run_id=record.id,
                principal=self.sessions.principal,
            )
            tool = detail.tool_calls[-1] if detail is not None and detail.tool_calls else None
            tool_call = schema.ToolCallUpdate(
                toolCallId=record.approval.tool_call_id if record.approval else "",
                title=record.approval.tool_name if record.approval else "approval",
                status="pending",
                rawInput=tool.arguments if tool is not None else None,
            )
            options = [
                schema.PermissionOption(
                    optionId="allow_once",
                    name="Allow once",
                    kind="allow_once",
                ),
                schema.PermissionOption(
                    optionId="reject_once",
                    name="Reject once",
                    kind="reject_once",
                ),
            ]
            response = await self._connection.request_permission(
                active.record.session_id,
                tool_call,
                options,
            )
            outcome = response.outcome
            active.pending_elicitation_ids.discard(record.id)
            if getattr(outcome, "outcome", None) != "selected":
                await self.runtime.cancel(record.id, principal=self.sessions.principal)
                return
            decision = (
                ApprovalDecision.ALLOW
                if outcome.option_id == "allow_once"
                else ApprovalDecision.DENY
            )
            await self.runtime.decide_approval(
                record.id,
                approval_id=record.approval.approval_id,
                decision=decision,
                principal=self.sessions.principal,
            )

    async def _resume_after_permission(self, active: ActiveAcpSession, execution_id: str, spec: Any) -> Any:
        import acp.schema as schema

        record = await self.runtime.get_execution_record(execution_id, principal=self.sessions.principal)
        if record is None or record.status is RunStatus.CANCELLED:
            return schema.PromptResponse(stopReason="cancelled")
        subscription = await self.event_hub.subscribe(execution_id)
        task = asyncio.create_task(
            self.runtime.resume(
                execution_id,
                principal=self.sessions.principal,
                extra_toolsets=await active.mcp_resources.toolsets(),
            )
        )
        await self._drain_until_done(active, execution_id, subscription, task)
        await task
        final = await self.runtime.get_execution_record(execution_id, principal=self.sessions.principal)
        if final is None or final.status is RunStatus.FAILED:
            raise internal_error("execution_failed", session_id=active.record.session_id, execution_id=execution_id)
        return schema.PromptResponse(
            stopReason=AcpExecutionAdapter.stop_reason(final.status)
        )

    async def _replay_history(self, active: ActiveAcpSession) -> None:
        if self._connection is None:
            return
        views = await self.runtime.get_session_messages(session_id=active.record.session_id, principal=self.sessions.principal)
        import acp.schema as schema

        for view in views:
            for message in view.messages:
                if not isinstance(message, dict):
                    continue
                for part in message.get("parts", ()):
                    if part.get("type") == "text":
                        await self._connection.session_update(
                            active.record.session_id,
                            schema.AgentMessageChunk(
                                content=schema.TextContentBlock(type="text", text=part.get("content", "")),
                                sessionUpdate="agent_message_chunk",
                            ),
                        )

    def _mode_state(self, current_mode_id: str) -> Any:
        return self.capability_builder.modes(self.capability_input, current_mode_id)

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise request_error("initialize_required")


__all__ = ["LinktoolsAcpAgent", "STANDARD_AGENT_METHODS"]
