#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ACP SDK boundary, method registry, capabilities and protocol errors."""

from dataclasses import dataclass
from typing import Any, Callable, Mapping
from uuid import uuid4

from linktools.errors import ConfigError
from ..errors import (
    SessionBusyError,
    SessionCleanupRequiredError,
    SessionConflictError,
    UnknownSessionError,
)
from ..governance.identity import PrincipalContext


class AcpDependencyError(ConfigError):
    """The optional ACP SDK is unavailable or has the wrong version."""


def require_sdk() -> object:
    try:
        import acp
    except ImportError as exc:
        raise AcpDependencyError(
            "ACP support requires agent-client-protocol==0.12.0"
        ) from exc
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("agent-client-protocol")
    except PackageNotFoundError as exc:
        raise AcpDependencyError("ACP SDK package metadata is missing") from exc
    if installed != "0.12.0":
        raise AcpDependencyError(f"ACP SDK version must be 0.12.0, found {installed}")
    return acp


def request_error(
    reason: str,
    *,
    session_id: "str | None" = None,
    execution_id: "str | None" = None,
    details: "Mapping[str, Any] | None" = None,
) -> Exception:
    acp = require_sdk()
    data: "dict[str, Any]" = {"errorId": uuid4().hex, "reason": reason}
    if session_id is not None:
        data["sessionId"] = session_id
    if execution_id is not None:
        data["executionId"] = execution_id
    if details:
        data.update(details)
    return acp.RequestError.invalid_params(data)


def internal_error(
    reason: str,
    *,
    session_id: "str | None" = None,
    execution_id: "str | None" = None,
    details: "Mapping[str, Any] | None" = None,
) -> Exception:
    acp = require_sdk()
    data: "dict[str, Any]" = {"errorId": uuid4().hex, "reason": reason}
    if session_id is not None:
        data["sessionId"] = session_id
    if execution_id is not None:
        data["executionId"] = execution_id
    if details:
        data.update(details)
    return acp.RequestError.internal_error(data)


@dataclass(frozen=True, slots=True)
class AcpMode:
    id: str
    name: str
    description: "str | None" = None


@dataclass(frozen=True, slots=True)
class CapabilityInput:
    modes: "tuple[AcpMode, ...]" = ()
    image: bool = False
    audio: bool = False
    embedded_context: bool = False
    supports_load: bool = True
    supports_list: bool = True
    supports_fork: bool = True
    supports_resume: bool = True
    supports_close: bool = True
    supports_mcp_http: bool = True
    supports_mcp_sse: bool = True
    supports_mcp_acp: bool = False


class CapabilityBuilder:
    def build(self, values: CapabilityInput, *, client_capabilities: Any = None) -> Any:
        import acp.schema as schema

        fs = getattr(client_capabilities, "fs", None)
        directories = bool(
            fs is not None
            and (getattr(fs, "read_text_file", False) or getattr(fs, "write_text_file", False))
        )
        return schema.AgentCapabilities(
            loadSession=values.supports_load,
            promptCapabilities=schema.PromptCapabilities(
                image=values.image,
                audio=values.audio,
                embeddedContext=values.embedded_context,
            ),
            mcpCapabilities=schema.McpCapabilities(
                http=values.supports_mcp_http,
                sse=values.supports_mcp_sse,
                acp=values.supports_mcp_acp,
            ),
            sessionCapabilities=schema.SessionCapabilities(
                list=schema.SessionListCapabilities() if values.supports_list else None,
                additionalDirectories=(
                    schema.SessionAdditionalDirectoriesCapabilities()
                    if directories else None
                ),
                fork=schema.SessionForkCapabilities() if values.supports_fork else None,
                resume=schema.SessionResumeCapabilities() if values.supports_resume else None,
                close=schema.SessionCloseCapabilities() if values.supports_close else None,
            ),
        )

    def modes(self, values: CapabilityInput, current_mode_id: str) -> Any:
        import acp.schema as schema

        return schema.SessionModeState(
            currentModeId=current_mode_id,
            availableModes=[
                schema.SessionMode(id=mode.id, name=mode.name, description=mode.description)
                for mode in values.modes
            ],
        )

    def agent_info(self, *, name: str = "linktools-ai", version: str = "0.0.0") -> Any:
        import acp.schema as schema

        return schema.Implementation(name=name, title="Linktools AI", version=version)


class AcpProtocol:
    """Immutable ACP configuration shared by handlers on one connection."""

    def __init__(
        self,
        *,
        principal: PrincipalContext,
        spec_resolver: Any,
        modes: "tuple[AcpMode, ...]" = (),
        capability_input: "CapabilityInput | None" = None,
        version: str = "0.0.0",
    ) -> None:
        self.principal = principal
        self.spec_resolver = spec_resolver
        self.capability_input = capability_input or CapabilityInput(modes=modes)
        self.modes = self.capability_input.modes or (AcpMode("default", "Default"),)
        self.version = version
        self.capabilities = CapabilityBuilder()

    async def resolve_spec(self, mode_id: str) -> Any:
        return await self.spec_resolver(mode_id)

    def mode_state(self, current_mode_id: str) -> Any:
        return self.capabilities.modes(self.capability_input, current_mode_id)

    def domain_error(self, error: BaseException, *, session_id: "str | None" = None) -> Exception:
        if isinstance(error, UnknownSessionError):
            reason = "unknown_session"
        elif isinstance(error, SessionBusyError):
            reason = "session_busy"
        elif isinstance(error, SessionConflictError):
            reason = "session_conflict"
        elif isinstance(error, SessionCleanupRequiredError):
            reason = "session_cleanup_required"
        else:
            return internal_error("internal_error", session_id=session_id)
        return request_error(reason, session_id=session_id)


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


def protocol_handler(function: Callable[..., Any]) -> Callable[..., Any]:
    """Convert unexpected handler failures at the protocol boundary."""
    import functools

    @functools.wraps(function)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return await function(*args, **kwargs)
        except Exception as exc:
            acp = require_sdk()
            if isinstance(exc, acp.RequestError):
                raise
            if isinstance(exc, UnknownSessionError):
                raise request_error("unknown_session") from exc
            if isinstance(exc, SessionBusyError):
                raise request_error("session_busy") from exc
            if isinstance(exc, SessionConflictError):
                raise request_error("session_conflict") from exc
            if isinstance(exc, SessionCleanupRequiredError):
                raise request_error("session_cleanup_required") from exc
            raise internal_error("internal_error") from exc

    return wrapped


__all__ = [
    "AcpDependencyError",
    "AcpMode",
    "CapabilityBuilder",
    "CapabilityInput",
    "AcpProtocol",
    "STANDARD_AGENT_METHODS",
    "internal_error",
    "protocol_handler",
    "request_error",
    "require_sdk",
]
