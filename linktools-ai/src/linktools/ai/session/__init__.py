#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session lifecycle and conversation membership contracts."""

from ._contracts import (
    SessionRepository,
    ConversationCursor,
    SessionRecord,
    SessionTurnRef,
    SessionTurnCommitRef,
    SessionHistoryItem,
    SessionTurnItem,
    SessionTurn,
    SessionHistoryReader,
    CreateSessionRequest,
    ListSessionRequest,
    ResumeSessionRequest,
    ForkSessionRequest,
    UpdateSessionRequest,
    CloseSessionRequest,
    SessionView,
    SessionDetail,
)

from ._service import (
    SessionManager,
    SessionExecutionState,
    SessionExecutionReader,
    SessionExecutionService,
    SessionRepositories,
    SessionOperations,
    SessionReleaseCallback,
    SessionPendingExecutionLookup,
)

__all__ = [
    "SessionManager",
    "SessionExecutionState",
    "SessionExecutionReader",
    "SessionExecutionService",
    "SessionRepositories",
    "SessionOperations",
    "SessionReleaseCallback",
    "SessionPendingExecutionLookup",
    "SessionRepository",
    "ConversationCursor",
    "SessionRecord",
    "SessionTurnRef",
    "SessionTurnCommitRef",
    "SessionHistoryItem",
    "SessionTurnItem",
    "SessionTurn",
    "SessionHistoryReader",
    "CreateSessionRequest",
    "ListSessionRequest",
    "ResumeSessionRequest",
    "ForkSessionRequest",
    "UpdateSessionRequest",
    "CloseSessionRequest",
    "SessionView",
    "SessionDetail",
]
