#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session identity, lifecycle requests, and retained conversation views."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from ..core import (
    CorrelationData,
    ExecutionMode,
    ExecutionStatus,
    JsonValue,
    Page,
    Principal,
    OperationLedgerInput,
    ResourceRef,
    RunBudget,
    SessionStatus,
    ThinkingValue,
    canonical_sha256,
    normalize_correlation,
    normalize_execution_mode,
    normalize_thinking,
    validate_agent_id,
    validate_idempotency_key,
    validate_memory_scope,
)
from ..errors import AIError, ErrorCode
from ..storage import StoredPayload
from ..model import StoredUserInput, UserPromptInput, normalize_input_files, validate_user_input
from ..workspace import validate_workspace_path


def _request_correlation(value: Mapping[str, object] | None) -> CorrelationData:
    try:
        return normalize_correlation(value)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error


@dataclass(frozen=True, slots=True)
class ConversationCursor:
    agent_run_id: str
    history_id: str | None = None
    message_count: int | None = None


@dataclass(frozen=True, slots=True)
class SessionTurnRef:
    session_id: str
    turn_seq: int
    execution_id: str

    def __post_init__(self) -> None:
        if not self.session_id or not self.execution_id or self.turn_seq < 1:
            raise ValueError("session turn reference is invalid")


@dataclass(frozen=True, slots=True)
class SessionTurnCommit:
    session_id: str
    turn_seq: int
    execution_id: str
    start_message_index: int
    end_message_index: int
    stored_user_input: StoredUserInput
    output: StoredPayload | None
    status: ExecutionStatus
    created_at: datetime
    updated_at: datetime
    error_code: str | None
    safe_error_details: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.session_id, str) or not self.session_id
            or not isinstance(self.execution_id, str) or not self.execution_id
            or isinstance(self.turn_seq, bool) or not isinstance(self.turn_seq, int) or self.turn_seq < 1
            or isinstance(self.start_message_index, bool) or not isinstance(self.start_message_index, int)
            or isinstance(self.end_message_index, bool) or not isinstance(self.end_message_index, int)
            or self.start_message_index < 0
            or self.end_message_index < self.start_message_index
        ):
            raise ValueError("session turn commit is invalid")
        if any(not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None
               for value in (self.created_at, self.updated_at)):
            raise ValueError("session turn timestamps must be aware")
        if not isinstance(self.stored_user_input, StoredUserInput):
            raise TypeError("session turn input is invalid")
        if self.output is not None and not isinstance(self.output, StoredPayload):
            raise TypeError("session turn output is invalid")
        if not isinstance(self.status, ExecutionStatus) or self.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            raise ValueError("session turn status must be terminal")
        if self.status is ExecutionStatus.SUCCEEDED and self.output is None:
            raise ValueError("successful session turn output is required")
        if self.status is not ExecutionStatus.SUCCEEDED and self.output is not None:
            raise ValueError("unsuccessful session turn cannot have output")
        if self.error_code is not None and not isinstance(self.error_code, str):
            raise TypeError("session turn error code is invalid")
        if not isinstance(self.safe_error_details, Mapping):
            raise TypeError("session turn error details are invalid")
        canonical_sha256(self.safe_error_details)
        object.__setattr__(self, "safe_error_details", dict(self.safe_error_details))



@dataclass(frozen=True, slots=True)
class SessionRecord:
    session_id: str
    owner_principal_id: str
    status: SessionStatus
    revision: int
    cwd: str | None
    metadata: Mapping[str, JsonValue]
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None
    active_execution_id: str | None
    agent_id: str
    continuation: ConversationCursor | None = None
    history_quality: str = "complete"
    history_id: str | None = None
    timeline_parent_session_id: str | None = None
    timeline_parent_turn_seq: int = 0

    def __post_init__(self) -> None:
        try:
            validate_agent_id(self.agent_id)
        except (AIError, TypeError) as error:
            raise ValueError("session agent id is invalid") from error
        if (
            self.active_execution_id is not None
            and not self.active_execution_id.strip()
        ):
            raise ValueError("active execution identifier cannot be empty")
        if self.status is SessionStatus.CLOSED and self.active_execution_id is not None:
            raise ValueError("closed session cannot have an active execution")
        if self.cwd is not None:
            try:
                normalized_cwd = validate_workspace_path(self.cwd)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "session cwd must be a canonical workspace path"
                ) from error
            if normalized_cwd != self.cwd:
                raise ValueError("session cwd must be canonical")
        if self.history_quality not in {"complete", "conservative", "unavailable"}:
            raise ValueError("session history quality summary is invalid")
        if self.timeline_parent_session_id is None:
            if self.timeline_parent_turn_seq != 0:
                raise ValueError("root session timeline cannot have an ancestor cutoff")
        elif (
            not self.timeline_parent_session_id
            or self.timeline_parent_session_id == self.session_id
            or self.timeline_parent_turn_seq < 1
        ):
            raise ValueError("session timeline ancestor is invalid")


@dataclass(frozen=True, slots=True)
class SessionHistoryItem:
    message_seq: int
    item_kind: str
    content: JsonValue
    tool_name: "str | None" = None
    tool_call_id: "str | None" = None

    def __post_init__(self) -> None:
        if (
            self.message_seq < 1
            or not isinstance(self.item_kind, str)
            or not self.item_kind
        ):
            raise ValueError("session history item is invalid")


@dataclass(frozen=True, slots=True)
class SessionTurnItem:
    ordinal: int
    item_kind: str
    content: JsonValue
    tool_name: "str | None" = None
    tool_call_id: "str | None" = None

    def __post_init__(self) -> None:
        if self.ordinal < 1 or not self.item_kind:
            raise ValueError("session timeline item is invalid")


@dataclass(frozen=True, slots=True)
class SessionTurn:
    execution_id: str
    status: ExecutionStatus
    created_at: datetime
    updated_at: datetime
    user_input: JsonValue
    conversation_committed: bool
    items: tuple[SessionTurnItem, ...]
    error_code: "str | None" = None
    safe_error_details: "Mapping[str, JsonValue]" = field(default_factory=dict)
    output: JsonValue = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "safe_error_details", dict(self.safe_error_details))


class SessionHistoryReader(Protocol):
    async def history(
        self,
        session_id: str,
        *,
        tenant_id: str,
        continuation_agent_run_id: "str | None",
        continuation_history_id: "str | None" = None,
        cursor: "str | None",
        limit: int,
    ) -> "Page[SessionHistoryItem]": ...


@dataclass(frozen=True, slots=True)
class CreateSessionRequest:
    principal: Principal
    session_id: str
    idempotency_key: str
    cwd: "str | None" = None
    metadata: "Mapping[str, JsonValue]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class ListSessionRequest:
    principal: Principal
    cursor: "str | None" = None
    limit: int = 100


@dataclass(frozen=True, slots=True)
class ResumeSessionRequest:
    principal: Principal
    user_prompt: "UserPromptInput"
    idempotency_key: str
    memory_scope: "str | None"
    mode: ExecutionMode
    planning: bool
    thinking: ThinkingValue
    correlation: CorrelationData = field(default_factory=dict)
    files: tuple[str, ...] = ()
    budget: RunBudget | None = None

    def __post_init__(self) -> None:
        if self.budget is not None and not isinstance(self.budget, RunBudget):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        object.__setattr__(self, "user_prompt", validate_user_input(self.user_prompt))
        files = normalize_input_files(self.files)
        validate_idempotency_key(self.idempotency_key)
        if self.memory_scope is not None:
            validate_memory_scope(self.memory_scope)
        mode = normalize_execution_mode(self.mode)
        thinking = normalize_thinking(self.thinking)
        if not isinstance(self.planning, bool) or mode == "plan" and not self.planning:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "thinking", thinking)
        object.__setattr__(self, "correlation", _request_correlation(self.correlation))
        object.__setattr__(self, "files", files)


@dataclass(frozen=True, slots=True)
class ForkSessionRequest:
    principal: Principal
    new_session_id: str
    idempotency_key: str = ""
    cwd: "str | None" = None

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class UpdateSessionRequest:
    principal: Principal
    expected_revision: int
    idempotency_key: str
    metadata: "Mapping[str, JsonValue]"
    cwd: "str | None" = None

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class CloseSessionRequest:
    principal: Principal
    idempotency_key: str
    force: bool = False
    wait_timeout_seconds: int = 30

    def __post_init__(self) -> None:
        validate_idempotency_key(self.idempotency_key)
        if not 1 <= self.wait_timeout_seconds <= 300:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


@dataclass(frozen=True, slots=True)
class SessionView:
    session_id: str
    agent_id: str
    status: SessionStatus
    revision: int = 0
    cwd: "str | None" = None
    active_execution_id: "str | None" = None
    metadata: "Mapping[str, JsonValue]" = field(default_factory=dict)
    history_quality: str = "complete"


@dataclass(frozen=True, slots=True)
class SessionDetail:
    session: SessionView
    timeline: Page[SessionTurn]


class SessionRepository(Protocol):
    @property
    def tenant_id(self) -> str: ...

    async def create(self, record: SessionRecord) -> SessionRecord: ...

    async def create_with_operation(
        self,
        record: SessionRecord,
        *,
        operation: OperationLedgerInput,
    ) -> tuple[SessionRecord, bool]: ...

    async def create_fork_with_operation(
        self,
        source_session_id: str,
        target: SessionRecord,
        *,
        expected_source_revision: int,
        operation: OperationLedgerInput,
    ) -> tuple[SessionRecord, bool]: ...

    async def list(
        self, *, tenant_id: str, owner_principal_id: str | None = None
    ) -> tuple[SessionRecord, ...]: ...

    async def list_page(
        self,
        *,
        tenant_id: str,
        owner_principal_id: str | None,
        cursor: str | None,
        limit: int,
        snapshot: int | None = None,
    ) -> tuple[int, Page[SessionRecord]]: ...

    async def get_header(
        self, session_id: str, *, tenant_id: str
    ) -> ResourceRef | None: ...

    async def get(self, session_id: str, *, tenant_id: str) -> SessionRecord | None: ...

    async def timeline_head(self, session_id: str, *, tenant_id: str) -> int: ...

    async def list_timeline_turns(
        self,
        session_id: str,
        *,
        tenant_id: str,
        start_turn_seq: int,
        end_turn_seq: int,
    ) -> tuple[SessionTurnRef, ...]: ...

    async def list_timeline_commits(
        self,
        session_id: str,
        *,
        tenant_id: str,
        start_turn_seq: int,
        end_turn_seq: int,
    ) -> tuple[SessionTurnCommit, ...]: ...

    async def compare_and_swap(
        self,
        session_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        next_record: SessionRecord,
    ) -> SessionRecord: ...

    async def compare_and_swap_with_operation(
        self,
        session_id: str,
        *,
        tenant_id: str,
        expected_revision: int,
        next_record: SessionRecord,
        operation: OperationLedgerInput,
    ) -> tuple[SessionRecord, bool]: ...

    async def admit_execution(
        self,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
        expected: ConversationCursor | None,
    ) -> SessionRecord: ...

    async def release_execution(
        self,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
        history_unavailable: bool = False,
        require_turn_commit: bool = False,
    ) -> SessionRecord: ...

    async def transition_status(
        self,
        session_id: str,
        *,
        tenant_id: str,
        expected: frozenset[SessionStatus],
        next_status: SessionStatus,
        closed_at: datetime | None = None,
        require_no_active: bool = False,
    ) -> SessionRecord: ...

    async def advance_continuation(
        self,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
        expected: ConversationCursor | None,
        next_cursor: ConversationCursor,
        history_quality: str | None = None,
    ) -> SessionRecord: ...


__all__ = [
    "SessionRepository",
    "ConversationCursor",
    "SessionRecord",
    "SessionTurnRef",
    "SessionTurnCommit",
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
