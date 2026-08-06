#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Public Session DTOs."""

from pydantic import BaseModel, ConfigDict, Field

from ..domain.execution import ExecutionHandle, Page
from ..domain.session import SessionStatus


class SessionView(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    owner_id: str
    project_id: str
    agent_id: str
    agent_revision: int
    bundle_digest: "str | None" = None
    profile: str
    settings: "dict[str, object]"
    status: SessionStatus
    revision: int
    current_execution_id: "str | None" = None


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    owner_id: str
    project_id: str
    agent_id: str
    agent_revision: int = Field(ge=1)
    bundle_digest: "str | None" = None
    profile: str
    settings: "dict[str, object]" = Field(default_factory=dict)
    session_id: "str | None" = None


class ListSessionsRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    owner_id: str
    limit: int = Field(default=100, ge=1, le=200)
    cursor: "str | None" = None


class LoadSessionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    owner_id: str
    project_id: str
    bundle_digest: "str | None" = None


class ResumeSessionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    input: object


class ForkSessionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    owner_id: str
    project_id: "str | None" = None
    session_id: "str | None" = None


class UpdateSessionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    expected_revision: int = Field(ge=1)
    settings: "dict[str, object] | None" = None
    mcp_resource_refs: "tuple[str, ...] | None" = None


class CloseSessionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    expected_revision: int = Field(ge=1)
    force: bool = False


class LoadedSession(BaseModel):
    model_config = ConfigDict(frozen=True)

    session: SessionView
    resource_refs: "tuple[str, ...]"


__all__ = [
    "CloseSessionRequest", "CreateSessionRequest", "ForkSessionRequest", "ListSessionsRequest",
    "LoadedSession", "LoadSessionRequest", "ResumeSessionRequest", "SessionView",
    "UpdateSessionRequest", "ExecutionHandle", "Page",
]
