#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Conservative ACP capability construction."""

from dataclasses import dataclass
from typing import Any


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
    """The only source of advertised ACP capabilities."""

    def build(self, values: CapabilityInput, *, client_capabilities: Any = None) -> Any:
        import acp.schema as schema

        client = client_capabilities
        fs = getattr(client, "fs", None)
        additional_directories = bool(
            fs is not None
            and (getattr(fs, "read_text_file", False) or getattr(fs, "write_text_file", False))
        )
        prompt = schema.PromptCapabilities(
            image=values.image,
            audio=values.audio,
            embeddedContext=values.embedded_context,
        )
        session = schema.SessionCapabilities(
            list=schema.SessionListCapabilities() if values.supports_list else None,
            additionalDirectories=(
                schema.SessionAdditionalDirectoriesCapabilities()
                if additional_directories
                else None
            ),
            fork=schema.SessionForkCapabilities() if values.supports_fork else None,
            resume=schema.SessionResumeCapabilities() if values.supports_resume else None,
            close=schema.SessionCloseCapabilities() if values.supports_close else None,
        )
        agent_capabilities = schema.AgentCapabilities(
            loadSession=values.supports_load,
            promptCapabilities=prompt,
            mcpCapabilities=schema.McpCapabilities(
                http=values.supports_mcp_http,
                sse=values.supports_mcp_sse,
                acp=values.supports_mcp_acp,
            ),
            sessionCapabilities=session,
        )
        return agent_capabilities

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


__all__ = ["AcpMode", "CapabilityBuilder", "CapabilityInput"]
