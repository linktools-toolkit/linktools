#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Thin ACP handlers over the protocol-neutral Runtime facade."""

from dataclasses import replace
from typing import Any
from uuid import uuid4

from linktools.core import environ

from ..execution.domain import ApprovalDecision
from ..errors import InvalidSessionConfigValueError, UnknownSessionConfigOptionError
from ..runtime.interaction import ApprovalRequest, InteractionObserver, InteractionResult, InteractionStopReason
from ..runtime.session import SessionSettings, SessionWorkspace
from ..runtime.facade import Runtime
from .client import AcpClient
from .codec import AcpCodec
from .protocol import (
    AcpProtocol,
    STANDARD_AGENT_METHODS,
    internal_error,
    protocol_handler,
    request_error,
    require_sdk,
)

logger = environ.get_logger("ai.acp.agent")


class _ClientObserver(InteractionObserver):
    def __init__(self, agent: "LinktoolsAcpAgent", session_id: str) -> None:
        self._agent = agent
        self._session_id = session_id

    async def publish(self, event: Any) -> None:
        update = self._agent.codec.encode_event(event)
        if update is not None:
            await self._agent.client.session_update(self._session_id, update)

    async def request_approval(
        self, request: ApprovalRequest, cancellation: Any
    ) -> "ApprovalDecision | None":
        return await self._agent.client.request_approval(request, cancellation)


class LinktoolsAcpAgent:
    def __init__(
        self,
        *,
        runtime: Runtime,
        codec: AcpCodec,
        client: AcpClient,
        protocol: AcpProtocol,
    ) -> None:
        self.runtime = runtime
        self.codec = codec
        self.client = client
        self.protocol = protocol
        self._initialized = False
        self._connection: Any = None

    def handler_registry(self) -> "dict[str, Any]":
        return {method: getattr(self, name) for method, name in STANDARD_AGENT_METHODS.items()}

    def on_connect(self, connection: Any) -> None:
        self._connection = connection
        self.client.set_connection(connection)

    @protocol_handler
    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any = None,
        client_info: Any = None,
        **kwargs: Any,
    ) -> Any:
        acp = require_sdk()
        if protocol_version != acp.PROTOCOL_VERSION:
            raise request_error("no_common_protocol_version")
        self.client.set_connection(self._connection, client_capabilities)
        self._initialized = True
        return acp.InitializeResponse(
            protocolVersion=protocol_version,
            agentCapabilities=self.protocol.capabilities.build(
                self.protocol.capability_input,
                client_capabilities=client_capabilities,
            ),
            authMethods=[],
            agentInfo=self.protocol.capabilities.agent_info(version=self.protocol.version),
        )

    @protocol_handler
    async def authenticate(self, method_id: str, **kwargs: Any) -> Any:
        self._require_initialized()
        raise request_error("unknown_auth_method")

    @protocol_handler
    async def new_session(
        self,
        cwd: str,
        additional_directories: "list[str] | None" = None,
        mcp_servers: "list[Any] | None" = None,
        **kwargs: Any,
    ) -> Any:
        self._require_initialized()
        import acp.schema as schema

        session_id = uuid4().hex
        tool_sources = self.codec.decode_mcp_servers(mcp_servers)
        active = await self.runtime.sessions.create(
            session_id=session_id,
            workspace=SessionWorkspace(cwd=cwd, additional_directories=tuple(additional_directories or ())),
            settings=SessionSettings(
                agent_id=self.protocol.modes[0].id,
                tool_source_fingerprints=tuple(
                    self.codec.mcp_server_fingerprint(item) for item in tool_sources
                ),
            ),
            principal=self.protocol.principal,
            tool_sources=tool_sources,
        )
        await self._register_client_resources(session_id)
        return schema.NewSessionResponse(
            sessionId=active.record.id,
            modes=self.protocol.mode_state(active.record.settings.agent_id),
            configOptions=list(self.protocol.config_registry.response_state()),
        )

    @protocol_handler
    async def list_sessions(
        self, cwd: "str | None" = None, cursor: "str | None" = None, **kwargs: Any
    ) -> Any:
        self._require_initialized()
        import acp.schema as schema

        sessions = await self.runtime.sessions.list(principal=self.protocol.principal)
        values = [session.record for session in sessions if cwd is None or session.record.workspace.cwd == cwd]
        return schema.ListSessionsResponse(
            sessions=[self.codec.encode_session(record) for record in values],
            nextCursor=None,
        )

    @protocol_handler
    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: "list[Any] | None" = None,
        additional_directories: "list[str] | None" = None,
        **kwargs: Any,
    ) -> Any:
        self._require_initialized()
        import acp.schema as schema

        active = await self.runtime.sessions.get(session_id, principal=self.protocol.principal)
        settings = active.record.settings
        tool_sources = self.codec.decode_mcp_servers(mcp_servers)
        settings = replace(
            settings,
            tool_source_fingerprints=tuple(
                self.codec.mcp_server_fingerprint(item) for item in tool_sources
            ),
        )
        async with await self.runtime.sessions.prepare_load(
            session_id=session_id,
            workspace=SessionWorkspace(cwd=cwd, additional_directories=tuple(additional_directories or ())),
            settings=settings,
            principal=self.protocol.principal,
            tool_sources=tool_sources,
        ) as load:
            updates = self.codec.encode_history(session_id, load.history)
            record = await load.commit()
            try:
                await self.runtime.sessions.register_owner(
                    session_id,
                    "acp.client",
                    self.client.resource_owner(session_id),
                    lease=load.lease,
                )
            except Exception as exc:
                await load.release()
                await self.runtime.sessions.close(
                    session_id, principal=self.protocol.principal, reason="owner_registration"
                )
                raise internal_error("client_owner_registration_failed", session_id=session_id) from exc
            try:
                await self.client.replay(session_id, updates)
            except Exception as exc:
                await load.release()
                await self.runtime.sessions.close(
                    session_id, principal=self.protocol.principal, reason="transport_error"
                )
                raise internal_error("history_replay_failed", session_id=session_id) from exc
        return schema.LoadSessionResponse(
            modes=self.protocol.mode_state(record.settings.agent_id),
            configOptions=list(self.protocol.config_registry.response_state()),
        )

    @protocol_handler
    async def resume_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: "list[str] | None" = None,
        mcp_servers: "list[Any] | None" = None,
        **kwargs: Any,
    ) -> Any:
        self._require_initialized()
        import acp.schema as schema

        source = await self.runtime.sessions.get(session_id, principal=self.protocol.principal)
        tool_sources = self.codec.decode_mcp_servers(mcp_servers)
        active = await self.runtime.sessions.resume(
            session_id,
            workspace=SessionWorkspace(cwd=cwd, additional_directories=tuple(additional_directories or ())),
            settings=replace(
                source.record.settings,
                tool_source_fingerprints=tuple(
                    self.codec.mcp_server_fingerprint(item) for item in tool_sources
                ),
            ),
            principal=self.protocol.principal,
            tool_sources=tool_sources,
        )
        await self._register_client_resources(session_id)
        return schema.ResumeSessionResponse(
            modes=self.protocol.mode_state(active.record.settings.agent_id),
            configOptions=list(self.protocol.config_registry.response_state()),
        )

    @protocol_handler
    async def fork_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: "list[str] | None" = None,
        mcp_servers: "list[Any] | None" = None,
        **kwargs: Any,
    ) -> Any:
        self._require_initialized()
        import acp.schema as schema

        source = await self.runtime.sessions.get(session_id, principal=self.protocol.principal)
        tool_sources = self.codec.decode_mcp_servers(mcp_servers)
        target_session_id = uuid4().hex
        active = await self.runtime.sessions.fork(
            session_id,
            target_session_id,
            workspace=SessionWorkspace(cwd=cwd, additional_directories=tuple(additional_directories or ())),
            settings=replace(
                source.record.settings,
                tool_source_fingerprints=tuple(
                    self.codec.mcp_server_fingerprint(item) for item in tool_sources
                ),
            ),
            principal=self.protocol.principal,
            tool_sources=tool_sources,
        )
        await self._register_client_resources(target_session_id)
        return schema.ForkSessionResponse(
            sessionId=active.record.id,
            modes=self.protocol.mode_state(active.record.settings.agent_id),
            configOptions=list(self.protocol.config_registry.response_state()),
        )

    @protocol_handler
    async def close_session(self, session_id: str, **kwargs: Any) -> Any:
        self._require_initialized()
        import acp.schema as schema

        result = await self.runtime.sessions.close(session_id, principal=self.protocol.principal)
        if not result.closed:
            raise internal_error("session_cleanup_failed", session_id=session_id, details={"errorIds": [item.error_id for item in result.failures]})
        return schema.CloseSessionResponse()

    @protocol_handler
    async def set_session_mode(self, session_id: str, mode_id: str, **kwargs: Any) -> Any:
        self._require_initialized()
        if mode_id not in {mode.id for mode in self.protocol.modes}:
            raise request_error("unknown_mode", session_id=session_id)
        active = await self.runtime.sessions.get(session_id, principal=self.protocol.principal)
        await self.runtime.sessions.update(session_id, settings=replace(active.record.settings, agent_id=mode_id), principal=self.protocol.principal)
        return None

    @protocol_handler
    async def set_config_option(self, config_id: str, session_id: str, value: "str | bool", **kwargs: Any) -> Any:
        self._require_initialized()
        active = await self.runtime.sessions.get(session_id, principal=self.protocol.principal)
        try:
            settings = self.protocol.config_registry.update(
                active.record.settings, config_id, value
            )
        except UnknownSessionConfigOptionError as exc:
            logger.info(
                "event=acp.config.unknown_option session_id=%s config_id=%s error_id=%s",
                session_id,
                config_id,
                type(exc).__name__,
            )
            raise
        except InvalidSessionConfigValueError:
            raise
        await self.runtime.sessions.update(
            session_id, settings=settings, principal=self.protocol.principal
        )
        return None

    @protocol_handler
    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        self._require_initialized()
        await self.runtime.interactions.cancel(session_id)

    @protocol_handler
    async def prompt(self, session_id: str, prompt: "list[Any]", **kwargs: Any) -> Any:
        self._require_initialized()
        import acp.schema as schema

        active = await self.runtime.sessions.get(session_id, principal=self.protocol.principal)
        mapped = self.codec.decode_prompt(prompt)
        spec = await self.protocol.resolve_spec(active.record.settings.agent_id)
        result = await self.runtime.interactions.execute(
            session_id=session_id,
            spec=spec,
            prompt=mapped,
            observer=_ClientObserver(self, session_id),
            principal=self.protocol.principal,
        )
        if result.status.value == "failed":
            raise internal_error(
                "execution_failed",
                session_id=session_id,
                execution_id=result.execution_id,
            )
        return schema.PromptResponse(stopReason=result.stop_reason.value)

    async def ext_method(self, method: str, params: "dict[str, Any]") -> Any:
        raise require_sdk().RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: "dict[str, Any]") -> None:
        return None

    async def _register_client_resources(self, session_id: str) -> None:
        try:
            await self.runtime.sessions.register_owner(
                session_id, "acp.client", self.client.resource_owner(session_id)
            )
        except BaseException:
            await self.runtime.sessions.close(
                session_id,
                principal=self.protocol.principal,
                reason="owner_registration",
            )
            raise

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise request_error("initialize_required")


__all__ = ["LinktoolsAcpAgent", "STANDARD_AGENT_METHODS"]
