#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ACP v1 Agent implementation over the protocol-neutral Runtime facade."""

import functools
import logging
from dataclasses import replace
from typing import Any, Awaitable, Callable
from uuid import uuid4

from ..execution.live_events import ExecutionEventHub
from ..runtime.facade import Runtime
from .capabilities import AcpMode, CapabilityBuilder, CapabilityInput
from .client_services import AcpClientServices
from .content_mapper import AcpContentMapper
from .errors import internal_error, request_error
from .history_mapper import AcpHistoryMapper
from .prompt_service import AcpPromptService
from .session_models import ActiveAcpSession
from .sessions import AcpSessionService
from .session_state import SessionOperationKind

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
        event_hub: ExecutionEventHub,
        session_service: AcpSessionService,
        project_root: str,
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
        if runtime.execution_event_hub is not event_hub:
            raise ValueError("ACP Agent and Runtime must share one ExecutionEventHub")
        if session_service.runtime is not runtime:
            raise ValueError("ACP Agent and SessionService must share one Runtime")
        self.event_hub = event_hub
        self.client_services = session_service.client_services or AcpClientServices(
            project_root=project_root
        )
        session_service.client_services = self.client_services
        self.sessions = session_service
        self.prompt_service = AcpPromptService(
            runtime=runtime,
            event_hub=event_hub,
            principal=session_service.principal,
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

    def handler_registry(self) -> "dict[str, Callable[..., Any]]":
        return {method: getattr(self, name) for method, name in STANDARD_AGENT_METHODS.items()}

    def on_connect(self, connection: Any) -> None:
        self._connection = connection
        self.client_services.set_connection(connection, self._client_capabilities)
        self.prompt_service.set_connection(connection)

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
        self.prompt_service.set_connection(self._connection)
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
        views = await self.runtime.get_session_messages(
            session_id=session_id,
            principal=self.sessions.principal,
        )
        history_updates = AcpHistoryMapper().preflight(session_id, views)
        await self._replay_history(active, history_updates)
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

        result = await self.sessions.close_session_resources(session_id, reason="client")
        if not result.closed:
            raise internal_error(
                "session_cleanup_failed",
                session_id=session_id,
                details={
                    "errorIds": [failure.error_id for failure in result.failures],
                    "remainingResourceCount": len(result.failures),
                },
            )
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
        await self.sessions.cancel_prompt(session_id)

    async def ext_method(self, method: str, params: "dict[str, Any]") -> "dict[str, Any]":
        from .errors import require_sdk

        acp = require_sdk()
        raise acp.RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: "dict[str, Any]") -> None:
        return None

    @_protocol_handler
    async def prompt(self, session_id: str, prompt: "list[Any]", **kwargs: Any) -> Any:
        self._require_initialized()
        active = await self.sessions.get(session_id)
        mapped = AcpContentMapper(
            image=self.capability_input.image,
            audio=self.capability_input.audio,
            embedded=self.capability_input.embedded_context,
        ).map(prompt)
        execution_id = uuid4().hex
        operation = await self.sessions.coordinator.reserve(
            active,
            SessionOperationKind.PROMPT,
            execution_id=execution_id,
        )
        logger.info(
            "event=acp.session.operation_reserved session_id=%s execution_id=%s operation_id=%s kind=%s epoch=%s",
            session_id,
            execution_id,
            operation.operation_id,
            operation.kind.value,
            operation.epoch,
        )
        try:
            async with active.lock:
                mode_id = active.record.mode_id
            spec = await self.spec_resolver(mode_id)
            return await self.prompt_service.execute(
                active,
                execution_id,
                spec,
                mapped,
            )
        finally:
            await self.prompt_service.clear_permission(active, execution_id)
            await self.sessions.coordinator.release(active, operation)
            logger.info(
                "event=acp.session.operation_released session_id=%s execution_id=%s operation_id=%s kind=%s epoch=%s",
                session_id,
                execution_id,
                operation.operation_id,
                operation.kind.value,
                operation.epoch,
            )
            logger.info("ACP prompt detached session=%s execution=%s", session_id, execution_id)

    async def _replay_history(self, active: ActiveAcpSession, updates: "tuple[Any, ...] | None" = None) -> None:
        if self._connection is None:
            return
        if updates is None:
            views = await self.runtime.get_session_messages(session_id=active.record.session_id, principal=self.sessions.principal)
            updates = AcpHistoryMapper().preflight(active.record.session_id, views)
        for update in updates:
            await self._connection.session_update(active.record.session_id, update)

    def _mode_state(self, current_mode_id: str) -> Any:
        return self.capability_builder.modes(self.capability_input, current_mode_id)

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise request_error("initialize_required")


__all__ = ["LinktoolsAcpAgent", "STANDARD_AGENT_METHODS"]
