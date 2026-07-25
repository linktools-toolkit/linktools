#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SessionReader: read-only session-history access owned by RunCoordinator.
Converts persisted ``SessionMessage`` records into pydantic-ai
``ModelMessage`` instances so AgentEngine can consume prior turns natively
via ``AgentInput.message_history`` -- AgentEngine itself never imports or
calls SessionStore."""

from typing import TYPE_CHECKING

from pydantic_ai.messages import ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart

from .models import MessageRole
from .store import SessionStore

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

    from .models import SessionMessage


def _as_text(content: object) -> str:
    return content if isinstance(content, str) else repr(content)


def _to_model_message(message: "SessionMessage") -> "ModelMessage":
    text = _as_text(message.content)
    if message.role is MessageRole.ASSISTANT:
        return ModelResponse(parts=[TextPart(content=text)])
    if message.role is MessageRole.SYSTEM:
        return ModelRequest(parts=[SystemPromptPart(content=text)])
    # USER and TOOL (no tool-call/tool-return round-trip data survives a
    # SessionMessage today) both surface as a plain user-turn part.
    return ModelRequest(parts=[UserPromptPart(content=text)])


class SessionReader:
    """Thin read-only wrapper over :class:`SessionStore`. The sole seam
    RunCoordinator uses to load prior-turn history before building
    ``AgentInput`` -- no other component reads SessionStore for this
    purpose."""

    def __init__(self, session_store: SessionStore) -> None:
        self._session_store = session_store

    async def load_message_history(self, session_id: str) -> "tuple[ModelMessage, ...]":
        prior = await self._session_store.list_messages(session_id)
        return tuple(_to_model_message(message) for message in prior)


__all__: "list[str]" = ["SessionReader"]
