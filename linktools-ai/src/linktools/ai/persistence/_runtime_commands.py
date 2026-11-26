#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Named state commands for multi-record Runtime checkpoints."""

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, cast
from uuid import uuid4

from linktools.core import environ
from ..agent import AgentBindingContract

from ..core import (
    ApprovalStatus,
    PersistenceDomain,
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    ExternalCallStatus,
    IdempotencyStatus,
    OperationLedgerInput,
    OperationLedgerRecord,
    Principal,
    SessionStatus,
    ToolOperationStatus,
    canonical_sha256,
    canonical_json_bytes,
    agent_run_id,
)
from ..errors import AIError, ErrorCode
from ..storage import StoredPayload, TurnCoordination, TurnScopeIdentity, sql_observation_phase
from ..steps import native_store_key
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from ..model.messages import encode_model_messages
from ._execution_content import restore_native_text_payload
from ._native_tool_error import NativeToolErrorReference
from ._message_root import NativeMessageRoot, NativeTextReference, NativeUserInputReference
from .contracts import ArchivedContextBlock
from ._archive import ContentArchive, PreparedArchive, archive_content_scope
from ..session import SessionTurnRef
from ..execution import ExecutionControlRecord
from .contracts import (
    deferred_resource_id,
    AgentAttemptClaim,
    BudgetRepository,
    ApprovalRecord,
    ConversationCursor,
    ConversationHistoryRecord,
    ConversationHistoryRepository,
    EventRepository,
    ExecutionCancelRequestCommit,
    ExecutionEventAppend,
    ExecutionRecord,
    ExecutionHistorySealRecord,
    ResultRecord,
    ExecutionRepository,
    ExecutionStartClaim,
    ExecutionStartReservation,
    ExecutionStartReservationResult,
    ExecutionStartReceipt,
    ExecutionTerminalPublication,
    ExecutionTerminalWriteReceipt,
    IdempotencyRecord,
    IdempotencyRepository,
    RuntimePayloadRef,
    ExecutionTerminalCommit,
    ExecutionTerminalCommitResult,
    ExternalCallRecord,
    PendingToolContinuation,
    OperationLedgerRepository,
    ExecutionRecoveryDetail,
    CheckpointRepositories,
    ExecutionRecoveryState,
    RecoveryHandoffPhase,
    RecoveryTerminalHandoff,
    RecoveryTerminalOutcome,
    SessionRecord,
    SessionTurnAfterImage,
    SessionRepository,
    ToolOperationAdmission,
    ToolBoundaryReceipt,
    ToolOperationRecord,
    decode_tool_operation_failure,
)
from .durability import (
    CommitObservation,
    DurableCommitResult,
    DurableCommitState,
    run_durable_commit,
)
from ._execution_commands import _execution_history_seal
from .graph_turns import GraphDeferredWait as _GraphDeferredWait
from ._execution_repositories import ExecutionRepositoryImpl, IdempotencyRepositoryImpl
from .turns import SourceTurnWriteFence, StateTurnCoordination, _duration, _TurnGraphReadError
from ._model_interaction_store import ModelInteractionStateStepArchive
from ._conversation_repositories import (
    ConversationHistoryRepositoryImpl,
    SessionRepositoryImpl,
)
from ._checkpoint_repositories import (
    ToolRepositoryImpl,
    tool_admission_matches as _tool_admission_matches,
)
from ._step_contracts import (
    AgentRunCheckpoint,
    AgentRunRecord,
    StepEvent,
)
from ._step_archive import (
    ExecutionTerminalSealPlan,
    PreparedExecutionProjection,
    PreparedAgentRunCheckpoint,
    PreparedAgentRunCheckpointBatch,
    StateStepArchive,
)
from ._repository_common import operation_identity_matches, validate_recovery_operation
from .state import Observed, GroupRecordReader, RecordQuery, StateGroupTransaction, StateStore, StateTransaction, StoredRecord, state_store_key

_logger = environ.get_logger("ai.runtime.state.commands")


def _same_execution_except_hold_metadata(
    current: ExecutionRecord, expected: ExecutionRecord,
) -> bool:
    """Holds share the revision counter without changing execution semantics."""
    return (
        current.revision >= expected.revision
        and replace(
            current,
            revision=expected.revision,
            updated_at=expected.updated_at,
            dependency_hold_ids=expected.dependency_hold_ids,
        ) == expected
    )


def _copy_archive_union(values: Sequence[PreparedArchive]) -> PreparedArchive:
    """Combine values without consuming a caller's preparation receipts."""
    unique = {id(value): value for value in values}
    return PreparedArchive.combine(tuple(
        PreparedArchive(owner=value.owner, records=dict(value.records), nodes=dict(value.nodes),
                        native_messages=dict(value.native_messages), native_publishers=dict(value.native_publishers))
        for value in unique.values()
    ))


def _timeline_turn_message_range(
    history: ConversationHistoryRecord,
    prepared: PreparedAgentRunCheckpointBatch,
    expected_cursor: ConversationCursor | None,
) -> tuple[int, int]:
    if not prepared.checkpoints:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    local_start = min(
        (chunk.first_message_index for chunk in prepared.checkpoints[0].chunks),
        default=None,
    )
    end = history.inherited_message_count + prepared.target_transcript_message_count
    if local_start is not None:
        start = history.inherited_message_count + local_start
    elif expected_cursor is None:
        if history.inherited_message_count != 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        start = 0
    elif expected_cursor.message_count is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    else:
        start = expected_cursor.message_count
    if end <= start:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return start, end


if TYPE_CHECKING:
    from ._execution_authority import ExecutionAuthority, InitialExecutionAuthority
    from ..core import JsonValue
    from ._session_turn_sql import SessionTurnKey, SqlSessionTurns, TerminalState
    from ._step_store import SqlStepStore
    from .capabilities import RuntimeCapabilityWrites, PreparedCapabilityWrites
    from ._task_repository import TaskRepositoryImpl, _TaskGraphCancelOwner
    from ..task._graph import TaskLease

    _GraphLaunch = tuple[SqlSessionTurns, SessionTurnKey, int, TaskRepositoryImpl, TaskLease]


class _GroupedStartUnavailable(AIError):
    """The transaction body explicitly declined this route; nothing may commit."""

    def __init__(self) -> None:
        super().__init__(ErrorCode.STORAGE_CONFLICT)


class _NativeChildResume(Protocol):
    async def guard_deferred_resume(
        self, child_store: StateStore, write: Callable[[StateTransaction], Awaitable[ExecutionRecord]],
    ) -> ExecutionRecord: ...

    def require_deferred_child(self, child: ExecutionRecord) -> None: ...


class StateCommands:
    """Named semantic commands that may enlist several Runtime domains."""

    def __init__(
        self,
        execution: ExecutionRepository,
        *,
        namespace: str,
        events: EventRepository,
        operations: OperationLedgerRepository | None = None,
        conversation: SessionRepository | None = None,
        checkpoint: CheckpointRepositories | None = None,
        conversation_history: ConversationHistoryRepository | None = None,
        tools: object | None = None,
        budgets: BudgetRepository | None = None,
        conversation_run_store: StateStepArchive | None = None,
        execution_run_store: StateStepArchive | None = None,
        background_tasks: "set[asyncio.Task[object]]",
        observe_cancellation: bool = True,
        capability_writes: "RuntimeCapabilityWrites | None" = None,
    ) -> None:
        self._execution = execution
        self._namespace = namespace
        self._tenant_id = execution.tenant_id
        self._events = events
        self._operations = operations
        self._conversation = conversation
        self._checkpoint = checkpoint
        self._conversation_history = conversation_history
        self._tools = tools
        self._budgets = budgets
        self._conversation_run_store = conversation_run_store
        self._execution_run_store = execution_run_store
        self._background_tasks = background_tasks
        self._observe_cancellation = observe_cancellation
        self._capability_writes = capability_writes

    @staticmethod
    async def read_session_start(
        sessions: SessionRepository, identities: IdempotencyRepository,
        session_id: str, idempotency_key: str, *, tenant_id: str,
    ) -> tuple[SessionRecord | None, IdempotencyRecord | None]:
        """Capture admission hints in one short read; reservation remains the authority."""
        stores = _dedupe_stores((sessions.state_store, identities.state_store))
        async def read(group: StateGroupTransaction | None):
            if (isinstance(group, GroupRecordReader) and isinstance(sessions, SessionRepositoryImpl)
                    and isinstance(identities, IdempotencyRepositoryImpl)):
                keys = {sessions.state_store: (sessions.session_record_key(session_id),)}
                store = identities.state_store
                keys[store] = (*keys.get(store, ()), identities._key("idempotency",
                    identities._identity_key("session.resume", idempotency_key)))
                await group.get_records(keys)
            return (await sessions.get(session_id, tenant_id=tenant_id),
                    await identities.get("session.resume", idempotency_key, tenant_id=tenant_id))
        if _same_group(stores):
            return await stores[0].storage_group.read_many(stores, read)
        return await read(None)

    @staticmethod
    async def read_pending_admission(
        identities: IdempotencyRepository, scope: str, idempotency_key: str, *,
        tenant_id: str, sessions: SessionRepository | None = None,
        session_id: str | None = None,
    ) -> tuple[IdempotencyRecord | None, SessionRecord | None]:
        """Recheck durable owners after the ephemeral claim in one fresh transaction."""
        stores = _dedupe_stores((identities.state_store,) if sessions is None
                                else (identities.state_store, sessions.state_store))

        async def read(_group: StateGroupTransaction | None):
            existing = await identities.state_store.read(
                lambda _transaction: identities.get(scope, idempotency_key, tenant_id=tenant_id)
            )
            # Replay/conflict takes precedence; do not decode a session it will not use.
            if existing is not None or sessions is None or session_id is None:
                return existing, None
            current = await sessions.state_store.read(
                lambda _transaction: sessions.get(session_id, tenant_id=tenant_id)
            )
            return None, current

        if _same_group(stores):
            return await stores[0].storage_group.read_many(stores, read)
        return await read(None)

    async def _load_execution(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        return await self._execution.state_store.read(
            lambda _transaction: self._execution.get(execution_id, tenant_id=tenant_id)
        )

    async def _load_session(self, session_id: str, *, tenant_id: str) -> SessionRecord | None:
        repository = self._conversation
        if repository is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        return await repository.state_store.read(
            lambda _transaction: repository.get(session_id, tenant_id=tenant_id)
        )

    async def commit_deferred_checkpoint(
        self,
        *,
        execution_id: str,
        tenant_id: str,
        expected_execution_revision: int,
        expected_event_seq: int,
        expected_agent_run_seq: int,
        continuation: PendingToolContinuation,
        audit_events: Sequence[ExecutionEventAppend] = (),
        approval_records: Sequence[ApprovalRecord] = (),
        external_records: Sequence[ExternalCallRecord] = (),
        occurred_at: datetime,
        background_tasks: "set[asyncio.Task[object]] | None" = None,
        producer_generation: int | None = None,
        turn_fence: SourceTurnWriteFence | None = None,
        _graph_wait: _GraphDeferredWait | None = None,
    ) -> ExecutionRecord:
        self._require_checkpoint()
        approvals = self._checkpoint.approvals
        external_calls = self._checkpoint.external_calls
        if occurred_at.tzinfo is None:
            raise ValueError("deferred checkpoint timestamp must be timezone-aware")
        if continuation.source_agent_run_id == "":
            raise ValueError("deferred checkpoint source step is required")
        approval_values = tuple(approval_records)
        external_values = tuple(external_records)
        request_events = tuple(
            ExecutionEventAppend(
                ExecutionEventType.APPROVAL_REQUESTED,
                {
                    "tool_call_id": item.tool_call_id,
                    "tool_name": item.tool_name,
                    "arguments_digest": item.arguments_payload.digest,
                },
            )
            for item in continuation.approvals
        ) + tuple(
            ExecutionEventAppend(
                ExecutionEventType.EXTERNAL_REQUESTED,
                {
                    "tool_call_id": item.tool_call_id,
                    "tool_name": item.tool_name,
                    "arguments_digest": item.arguments_payload.digest,
                },
            )
            for item in continuation.calls
        )
        if not request_events:
            raise ValueError("deferred checkpoint requires requests")
        if len(approval_values) != len(continuation.approvals):
            raise ValueError("approval records do not match continuation")
        if len(external_values) != len(continuation.calls):
            raise ValueError("external records do not match continuation")
        if any(
            record.execution_id != execution_id
            or self._tenant_id != tenant_id
            or record.status is not ApprovalStatus.PENDING
            for record in approval_values
        ):
            raise ValueError("approval record identity is invalid")
        if any(
            record.execution_id != execution_id
            or self._tenant_id != tenant_id
            or record.status is not ExternalCallStatus.PENDING
            for record in external_values
        ):
            raise ValueError("external record identity is invalid")
        if tuple(record.approval_id for record in approval_values) != tuple(
            deferred_resource_id(
                "approval-v1",
                tenant_id,
                execution_id,
                continuation.source_agent_run_id,
                item.tool_call_id,
            )
            for item in continuation.approvals
        ):
            raise ValueError("approval record ids do not match continuation")
        if tuple(record.call_id for record in external_values) != tuple(
            deferred_resource_id(
                "external-call-v1",
                tenant_id,
                execution_id,
                continuation.source_agent_run_id,
                item.tool_call_id,
            )
            for item in continuation.calls
        ):
            raise ValueError("external record ids do not match continuation")
        graph_stores = ()
        if _graph_wait is not None:
            from sqlalchemy import select
            from ._session_turn_sql import CanonicalDeferredFrontier
            from ._step_store import SqlStepStore

            graph = _graph_wait
            if (turn_fence is not None or _graph_wait.key.turn_id != execution_id
                    or producer_generation is None or producer_generation < 1
                    or type(graph.step_store) is not SqlStepStore or not graph.checkpoint_id
                    or graph.task_lease is None):
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            graph.turns._require_checkpoint_store(graph.step_store)
            launch = (graph.turns, graph.key, graph.generation, graph.task_repository, graph.task_lease)
            graph_stores = self._canonical_graph_launch_stores(execution_id, launch)
            turn_store, predicate, occupied = self._canonical_graph_turn_owner(graph.turns, graph.key, graph.generation)
            row = graph.turns._turns
            frontier = CanonicalDeferredFrontier.from_json(graph.key, graph.generation, {
                "checkpoint_id": graph.checkpoint_id,
                "approval_operation_ids": [item.operation_id for item in continuation.approvals],
                "external_operation_ids": [item.operation_id for item in continuation.calls],
                "operation_metadata": {
                    item.operation_id: dict(item.metadata) for item in (*continuation.approvals, *continuation.calls)
                },
            }).to_json()
            if len(canonical_json_bytes(frontier)) > graph.max_bytes:
                raise AIError(ErrorCode.CONTEXT_PAYLOAD_TOO_LARGE)
            state_query = select(row.c.state, row.c.result_json, row.c.lease_until, occupied).where(*predicate)

            async def read_graph(group: StateGroupTransaction):
                async def read(session):
                    return (await session.execute(state_query)).one_or_none()
                return await group.transaction(turn_store).run_sql(graph.turns._context, read)

        stores = _dedupe_stores(
            (
                self._execution.state_store,
                *((turn_fence.store,) if turn_fence is not None else ()),
                *graph_stores,
                approvals.state_store,
                external_calls.state_store,
            )
        )
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

        async def operation() -> ExecutionRecord:
            async def mutate(
                group: StateGroupTransaction,
            ) -> ExecutionRecord:
                if turn_fence is not None:
                    await turn_fence.guard(execution_id)
                execution_tx = group.transaction(self._execution.state_store)
                recovery_tx = group.transaction(approvals.state_store)
                if producer_generation is not None:
                    await self._execution.require_execution_producer_in_transaction(
                        execution_tx, execution_id,
                        expected_producer_generation=producer_generation, allow_cancelling=False,
                    )
                current_execution = await self._execution.get_in_transaction(
                    execution_tx, execution_id, tenant_id=tenant_id,
                )
                current_checkpoint = current_execution
                if (
                    current_execution is None
                    or current_checkpoint is None
                    or current_checkpoint.recovery is None
                    or current_execution.status is not ExecutionStatus.STARTED
                    or current_execution.revision != expected_execution_revision
                    or current_execution.event_seq != expected_event_seq
                    or current_execution.agent_run_seq
                    != expected_agent_run_seq
                    or current_checkpoint.recovery_state is not ExecutionRecoveryState.ACTIVE
                    or current_checkpoint.recovery.agent_run_id
                    != continuation.source_agent_run_id
                ):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                if _graph_wait is not None:
                    if (current_execution.root_execution_id != execution_id
                            or current_execution.session_id is not None
                            or current_execution.parent_execution_id is not None
                            or current_execution.parent_invocation_id is not None
                            or not current_execution.requires_task_invocation_capture
                            or not current_execution.restart_checkpoints
                            or current_execution.native_step_source is None
                            or current_execution.native_step_source != graph.step_store.backup_source
                            or expected_agent_run_seq < 1
                            or continuation.source_agent_run_id != agent_run_id(
                                namespace=self._namespace, tenant_id=self._tenant_id, execution_id=execution_id,
                                agent_run_seq=expected_agent_run_seq,
                            )):
                        raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                for record in approval_values:
                    await approvals.create_in_transaction(recovery_tx, record)
                for record in external_values:
                    await external_calls.create_in_transaction(recovery_tx, record)
                updated_execution = (
                    await self._execution.enter_deferred_wait_in_transaction(
                        execution_tx,
                        execution_id,
                        tenant_id=tenant_id,
                        expected_revision=expected_execution_revision,
                        expected_event_seq=expected_event_seq,
                        expected_agent_run_seq=expected_agent_run_seq,
                        audit_events=audit_events,
                        deferred_events=request_events,
                        occurred_at=occurred_at,
                        recovery=replace(current_checkpoint.recovery, pending_tools=continuation),
                    )
                )
                if _graph_wait is not None:
                    async def park(session):
                        snapshot = await graph.step_store._capture_turn_checkpoint_row(
                            session, run_id=continuation.source_agent_run_id,
                            conversation_id=graph.step_conversation_id, replay_key=graph.checkpoint_id,
                        )
                        if snapshot is None or snapshot[0]["state"] != "interrupted" or not snapshot[1].count:
                            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                        changed = await session.execute(row.update().where(
                            *predicate, occupied, row.c.state == "accepted", row.c.lease_until > graph.turns._now,
                        ).values(state="waiting", lease_until=None, result_json=frontier))
                        if changed.rowcount != 1:
                            raise AIError(ErrorCode.STORAGE_CONFLICT)

                    await group.transaction(turn_store).run_sql(graph.turns._context, park)
                    await graph.task_repository._guard_bound_execution_in_transaction(
                        group.transaction(graph.task_repository.state_store), graph.task_lease, execution_id=execution_id,
                    )
                return updated_execution

            return await stores[0].storage_group.mutate(stores, mutate)

        async def readback() -> CommitObservation[ExecutionRecord]:
            async def read(group: StateGroupTransaction) -> CommitObservation[ExecutionRecord]:
                execution = await self._execution.get_in_transaction(
                    group.transaction(self._execution.state_store),
                    execution_id, tenant_id=tenant_id,
                )
                if execution is None or execution.recovery is None:
                    return _partial_integrity()
                if producer_generation is not None:
                    if execution.producer_generation != producer_generation:
                        return CommitObservation(
                            DurableCommitState.NOT_COMMITTED, error=AIError(ErrorCode.STORAGE_CONFLICT),
                        )
                    if execution.status is ExecutionStatus.CANCELLING:
                        return CommitObservation(
                            DurableCommitState.NOT_COMMITTED, error=AIError(ErrorCode.EXECUTION_CANCELLED),
                        )
                complete_members = True
                for expected in approval_values:
                    actual = await approvals.get_in_transaction(
                        group.transaction(approvals.state_store),
                        expected.approval_id, tenant_id=tenant_id,
                    )
                    if actual is not None and (
                        actual.approval_id != expected.approval_id
                        or actual.execution_id != execution_id
                    ):
                        return _partial_integrity()
                    complete_members = complete_members and actual is not None and (
                        actual.created_at == expected.created_at
                        and (actual.status is not ApprovalStatus.PENDING or actual == expected)
                    )
                for expected in external_values:
                    actual = await external_calls.get_in_transaction(
                        group.transaction(external_calls.state_store),
                        expected.call_id, tenant_id=tenant_id,
                    )
                    if actual is not None and (
                        actual.call_id != expected.call_id
                        or actual.execution_id != execution_id
                    ):
                        return _partial_integrity()
                    complete_members = complete_members and actual is not None and (
                        actual.created_at == expected.created_at
                        and (actual.status is not ExternalCallStatus.PENDING or actual == expected)
                    )
                if (
                    execution.status is ExecutionStatus.STARTED
                    and execution.revision == expected_execution_revision
                    and execution.event_seq == expected_event_seq
                    and execution.agent_run_seq == expected_agent_run_seq
                    and execution.recovery_state is ExecutionRecoveryState.ACTIVE
                    and execution.recovery.agent_run_id == continuation.source_agent_run_id
                ):
                    # Other owners may resolve or pre-create members. This command
                    # atomically publishes its members and execution frontier.
                    if _graph_wait is not None:
                        previous = await read_graph(group)
                        if previous is None or previous[0] != "accepted" or not previous[3]:
                            return _partial_integrity()
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                if (
                    execution.status is ExecutionStatus.WAITING_DEFERRED
                    and execution.agent_run_seq == expected_agent_run_seq
                    and execution.recovery_state is ExecutionRecoveryState.WAITING
                    and execution.recovery.pending_tools == continuation
                ):
                    if not complete_members:
                        return _partial_integrity()
                    if _graph_wait is not None:
                        parked = await read_graph(group)
                        if parked is None or tuple(parked) != ("waiting", frontier, None, True):
                            return _partial_integrity()
                    return CommitObservation(DurableCommitState.COMMITTED, value=execution)
                return CommitObservation(DurableCommitState.UNRESOLVED)

            return await stores[0].storage_group.read_many(stores, read)

        owner_tasks = self._background_tasks if background_tasks is None else background_tasks
        outcome = await run_durable_commit(
            operation,
            readback,
            background_tasks=owner_tasks,
        )
        return _require_durable_record(outcome)

    async def commit_deferred_cancel_checkpoint(
        self,
        commit: ExecutionCancelRequestCommit,
        *,
        expected_pending_tools: PendingToolContinuation,
        background_tasks: "set[asyncio.Task[object]] | None" = None,
        _graph_wait: "tuple[SqlSessionTurns, SessionTurnKey, int, str] | None" = None,
    ) -> ExecutionRecord:
        self._require_checkpoint()
        approvals = self._checkpoint.approvals
        external_calls = self._checkpoint.external_calls
        graph_store = None
        if _graph_wait is not None:
            from sqlalchemy import select
            from ._session_turn_sql import CanonicalDeferredFrontier

            turns, key, generation, checkpoint_id = _graph_wait
            if key.turn_id != commit.execution_id:
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            graph_store, predicate, occupied = self._canonical_graph_turn_owner(turns, key, generation)
            frontier = CanonicalDeferredFrontier.from_json(key, generation, {
                "checkpoint_id": checkpoint_id,
                "approval_operation_ids": [item.operation_id for item in expected_pending_tools.approvals],
                "external_operation_ids": [item.operation_id for item in expected_pending_tools.calls],
                "operation_metadata": {
                    item.operation_id: dict(item.metadata)
                    for item in (*expected_pending_tools.approvals, *expected_pending_tools.calls)
                },
            }).to_json()
            row = turns._turns
            graph_statement = select(row.c.state, row.c.result_json, row.c.lease_until, occupied).where(*predicate)

            async def graph_state(group: StateGroupTransaction):
                async def read(session):
                    return (await session.execute(graph_statement)).one_or_none()

                return await group.transaction(graph_store).run_sql(turns._context, read)

        stores = _dedupe_stores(
            (
                self._execution.state_store,
                *((graph_store,) if graph_store is not None else ()),
                approvals.state_store,
                external_calls.state_store,
            )
        )
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        approval_ids = tuple(
            deferred_resource_id(
                "approval-v1",
                self._tenant_id,
                commit.execution_id,
                expected_pending_tools.source_agent_run_id,
                item.tool_call_id,
            )
            for item in expected_pending_tools.approvals
        )
        call_ids = tuple(
            deferred_resource_id(
                "external-call-v1",
                self._tenant_id,
                commit.execution_id,
                expected_pending_tools.source_agent_run_id,
                item.tool_call_id,
            )
            for item in expected_pending_tools.calls
        )

        async def operation() -> ExecutionRecord:
            async def mutate(group: StateGroupTransaction) -> ExecutionRecord:
                execution_tx = group.transaction(self._execution.state_store)
                recovery_tx = group.transaction(approvals.state_store)
                execution = await self._execution.get_in_transaction(
                    execution_tx,
                    commit.execution_id,
                    tenant_id=self._tenant_id,
                )
                checkpoint = execution
                if (
                    execution is None
                    or checkpoint is None
                    or checkpoint.recovery is None
                    or execution.status is not ExecutionStatus.WAITING_DEFERRED
                    or execution.revision != commit.expected_revision
                    or execution.event_seq != commit.expected_event_seq
                    or checkpoint.recovery_state is not ExecutionRecoveryState.WAITING
                    or checkpoint.recovery.pending_tools != expected_pending_tools
                ):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                if _graph_wait is not None:
                    if (execution.root_execution_id != commit.execution_id
                            or execution.session_id is not None or execution.parent_execution_id is not None
                            or execution.parent_invocation_id is not None or not execution.requires_task_invocation_capture
                            or execution.agent_run_seq < 1
                            or expected_pending_tools.source_agent_run_id != agent_run_id(
                                namespace=self._namespace, tenant_id=self._tenant_id, execution_id=commit.execution_id,
                                agent_run_seq=execution.agent_run_seq,
                            )):
                        raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                    prior = await graph_state(group)
                    if prior is None or tuple(prior) != ("waiting", frontier, None, True):
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                await approvals.cancel_pending_in_transaction(
                    recovery_tx,
                    approval_ids,
                    execution_id=commit.execution_id,
                    tenant_id=self._tenant_id,
                    decided_at=commit.requested_at,
                )
                await external_calls.cancel_pending_in_transaction(
                    recovery_tx,
                    call_ids,
                    execution_id=commit.execution_id,
                    tenant_id=self._tenant_id,
                    cancelled_at=commit.requested_at,
                )
                updated = await self._execution.request_cancel_in_transaction(
                    execution_tx,
                    commit,
                    expected_status=ExecutionStatus.WAITING_DEFERRED,
                    recovery=replace(checkpoint.recovery, pending_tools=None),
                )
                if graph_store is not None:
                    async def stop_graph(session):
                        stopped = await session.execute(row.update().where(
                            *predicate, occupied, row.c.state == "waiting", row.c.lease_until.is_(None),
                            row.c.result_json["checkpoint_id"].as_string() == checkpoint_id,
                        ).values(state="cancelling", lease_until=turns._now))
                        if stopped.rowcount != 1:
                            raise AIError(ErrorCode.STORAGE_CONFLICT)

                    await group.transaction(graph_store).run_sql(turns._context, stop_graph)
                return updated

            return await stores[0].storage_group.mutate(stores, mutate)

        async def readback() -> CommitObservation[ExecutionRecord]:
            async def read(group: StateGroupTransaction) -> CommitObservation[ExecutionRecord]:
                execution = await self._execution.get_in_transaction(
                    group.transaction(self._execution.state_store),
                    commit.execution_id, tenant_id=self._tenant_id,
                )
                if execution is None or execution.recovery is None:
                    return _partial_integrity()
                page = await self._events.list(
                    commit.execution_id, tenant_id=self._tenant_id,
                    after_event_seq=commit.expected_event_seq, limit=2,
                )
                items = page.items
                revision_delta = execution.revision - commit.expected_revision
                sequence_delta = execution.event_seq - commit.expected_event_seq
                if (
                    any(
                        event.execution_id != commit.execution_id
                        or event.event_seq != commit.expected_event_seq + index
                        for index, event in enumerate(items, 1)
                    )
                    or revision_delta < 0
                    or sequence_delta < 0
                    or revision_delta < sequence_delta
                    or items and items[-1].event_seq > execution.event_seq
                    or sequence_delta > 0 and not items
                ):
                    return _partial_integrity()
                own_cancellation = (
                    bool(items)
                    and items[0].event_type == ExecutionEventType.CANCEL_REQUESTED
                    and items[0].payload == {"operation_id": commit.operation_id}
                )
                pending_members = False
                for approval_id in approval_ids:
                    record = await approvals.get_in_transaction(
                        group.transaction(approvals.state_store),
                        approval_id, tenant_id=self._tenant_id,
                    )
                    if (
                        record is None
                        or record.approval_id != approval_id
                        or record.execution_id != commit.execution_id
                    ):
                        return _partial_integrity()
                    pending_members = pending_members or record.status is ApprovalStatus.PENDING
                for call_id in call_ids:
                    record = await external_calls.get_in_transaction(
                        group.transaction(external_calls.state_store),
                        call_id, tenant_id=self._tenant_id,
                    )
                    if (
                        record is None
                        or record.call_id != call_id
                        or record.execution_id != commit.execution_id
                    ):
                        return _partial_integrity()
                    pending_members = pending_members or record.status is ExternalCallStatus.PENDING
                if (
                    execution.status is ExecutionStatus.WAITING_DEFERRED
                    and execution.revision == commit.expected_revision
                    and execution.event_seq == commit.expected_event_seq
                    and execution.recovery_state is ExecutionRecoveryState.WAITING
                    and execution.recovery.pending_tools == expected_pending_tools
                ):
                    # Member resolution has its own owner and does not prove
                    # that this atomic cancellation command was published.
                    if _graph_wait is not None:
                        prior = await graph_state(group)
                        if prior is None or tuple(prior) != ("waiting", frontier, None, True):
                            return _partial_integrity()
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                if not own_cancellation:
                    if items:
                        return CommitObservation(
                            DurableCommitState.NOT_COMMITTED,
                            error=AIError(ErrorCode.STORAGE_CONFLICT),
                        )
                    return CommitObservation(DurableCommitState.UNRESOLVED)
                if (
                    execution.status
                    in {
                        ExecutionStatus.CANCELLING,
                        ExecutionStatus.FINALIZING,
                        ExecutionStatus.SUCCEEDED,
                        ExecutionStatus.FAILED,
                        ExecutionStatus.CANCELLED,
                    }
                    and execution.event_seq >= commit.expected_event_seq + 1
                    and execution.recovery_state
                    in {ExecutionRecoveryState.ACTIVE, ExecutionRecoveryState.COMPLETED}
                    and execution.recovery.pending_tools is None
                    and execution.recovery.handoff_phase is RecoveryHandoffPhase.NONE
                    and execution.recovery.terminal_handoff is None
                ):
                    if pending_members:
                        return _partial_integrity()
                    if _graph_wait is not None:
                        observed = await graph_state(group)
                        if observed is None:
                            return CommitObservation(DurableCommitState.UNRESOLVED)
                        graph_phase, graph_payload, lease, active = observed
                        if (execution.status is not ExecutionStatus.CANCELLING or graph_phase != "cancelling"
                                or graph_payload != frontier or lease is None or not active):
                            return CommitObservation(DurableCommitState.UNRESOLVED)
                    return CommitObservation(DurableCommitState.COMMITTED, value=execution)
                return CommitObservation(DurableCommitState.UNRESOLVED)

            read_stores = _dedupe_stores((*stores, self._events.state_store))
            return await stores[0].storage_group.read_many(read_stores, read)

        owner_tasks = self._background_tasks if background_tasks is None else background_tasks
        outcome = await run_durable_commit(
            operation,
            readback,
            background_tasks=owner_tasks,
        )
        return _require_durable_record(outcome)

    async def _mark_canonical_graph_deferred_ready(
        self, expected_execution: ExecutionRecord, *, graph: _GraphDeferredWait,
        _queued_task: tuple[str, str, int] | None = None,
    ) -> bool:
        """Make only a fully settled parked Graph frontier eligible for reacquisition."""
        from sqlalchemy import select
        from ._session_turn_sql import CanonicalDeferredFrontier
        from ._step_store import SqlStepStore
        from ._task_repository import TaskRepositoryImpl

        self._require_checkpoint()
        repository, tools = self._execution, self._tools
        execution = expected_execution
        if (type(repository) is not ExecutionRepositoryImpl or type(tools) is not ToolRepositoryImpl
                or type(graph.step_store) is not SqlStepStore or type(graph.task_repository) is not TaskRepositoryImpl
                or (graph.task_lease is None) != (_queued_task is not None)
                or execution.execution_id != graph.key.turn_id or execution.root_execution_id != graph.key.turn_id
                or execution.session_id is not None or execution.parent_execution_id is not None
                or execution.parent_invocation_id is not None or not execution.requires_task_invocation_capture
                or execution.status is not ExecutionStatus.WAITING_DEFERRED
                or execution.recovery_state is not ExecutionRecoveryState.WAITING
                or execution.recovery is None or execution.recovery.pending_tools is None
                or execution.agent_run_seq < 1 or execution.producer_generation < 1
                or execution.native_step_source != graph.step_store.backup_source):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        pending = execution.recovery.pending_tools
        if pending.source_agent_run_id != agent_run_id(
            namespace=self._namespace, tenant_id=self._tenant_id, execution_id=execution.execution_id,
            agent_run_seq=execution.agent_run_seq,
        ):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        graph.turns._require_checkpoint_store(graph.step_store)
        approvals, calls = self._checkpoint.approvals, self._checkpoint.external_calls
        execution_store = repository.state_store
        turn_store, predicate, occupied = self._canonical_graph_turn_owner(graph.turns, graph.key, graph.generation)
        task_store = graph.task_repository.state_store
        if graph.task_lease is not None:
            launch = (graph.turns, graph.key, graph.generation, graph.task_repository, graph.task_lease)
            self._canonical_graph_launch_stores(execution.execution_id, launch)
        stores = _dedupe_stores((execution_store, turn_store, task_store,
                                 approvals.state_store, calls.state_store, tools.state_store))
        if not _same_group(stores):
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
        row = graph.turns._turns
        exact = CanonicalDeferredFrontier.from_json(graph.key, graph.generation, {
            "checkpoint_id": graph.checkpoint_id,
            "approval_operation_ids": [item.operation_id for item in pending.approvals],
            "external_operation_ids": [item.operation_id for item in pending.calls],
            "operation_metadata": {item.operation_id: dict(item.metadata) for item in (*pending.approvals, *pending.calls)},
        }).to_json()
        statement = select(row.c.result_json).where(
            *predicate, occupied, row.c.state == "waiting", row.c.lease_until.is_(None),
        )

        async def mark(group: StateGroupTransaction) -> bool:
            transaction = group.transaction(execution_store)
            observed = await repository.get_observed_in_transaction(transaction, execution.execution_id, tenant_id=self._tenant_id)
            if observed is None or observed.value != execution:
                raise AIError(ErrorCode.STORAGE_CONFLICT)

            tool_tx = group.transaction(tools.state_store)
            settlements: list[tuple[ToolOperationRecord, StoredPayload | None, str | None, StoredPayload | None]] = []

            async def original(item) -> ToolOperationRecord:
                effect = await tools.get_operation_in_transaction(tool_tx, item.operation_id, tenant_id=self._tenant_id)
                if (effect is None or effect.execution_id != execution.execution_id
                        or effect.agent_run_id != pending.source_agent_run_id or effect.tool_call_id != item.tool_call_id
                        or effect.tool_name != item.tool_name or effect.arguments_digest != item.arguments_payload.digest):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return effect

            def settle_known(effect, result_payload, error_code, error_payload) -> bool:
                target = ToolOperationStatus.FAILED if result_payload is None else ToolOperationStatus.COMPLETED
                if effect.status is ToolOperationStatus.PENDING and effect.owner is None:
                    settlements.append((effect, result_payload, error_code, error_payload))
                    return True
                if effect.status is not target:
                    return False

                def identity(payload):
                    return None if payload is None else (payload.digest, payload.size)

                if (identity(effect.result_payload) != identity(result_payload)
                        or effect.error_code != error_code or identity(effect.error_payload) != identity(error_payload)):
                    raise AIError(ErrorCode.TOOL_RESULT_CONFLICT)
                return True

            def failure(message: str) -> StoredPayload:
                return StoredPayload.inline_bytes(canonical_json_bytes({
                    "version": 1, "kind": "tool_call_failed", "message": message,
                }))

            for item in pending.approvals:
                approval_id = deferred_resource_id(
                    "approval-v1", self._tenant_id, execution.execution_id, pending.source_agent_run_id, item.tool_call_id,
                )
                record = await approvals.get_in_transaction(
                    group.transaction(approvals.state_store), approval_id, tenant_id=self._tenant_id,
                )
                if (record is None or record.execution_id != execution.execution_id or record.approval_id != approval_id):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if record.status not in {ApprovalStatus.APPROVED, ApprovalStatus.DENIED}:
                    return False
                effect = await original(item)
                if record.status is ApprovalStatus.APPROVED:
                    if not ((effect.status is ToolOperationStatus.PENDING and effect.owner is None)
                            or effect.status in {ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED}):
                        return False
                elif effect.status is not ToolOperationStatus.CANCELLED and not settle_known(
                    effect, None, ErrorCode.TOOL_EXECUTION_FAILED.value,
                    failure(record.decision_message or "The tool call was denied."),
                ):
                    return False

            for item in pending.calls:
                call_id = deferred_resource_id(
                    "external-call-v1", self._tenant_id, execution.execution_id, pending.source_agent_run_id, item.tool_call_id,
                )
                record = await calls.get_in_transaction(
                    group.transaction(calls.state_store), call_id, tenant_id=self._tenant_id,
                )
                if record is None or record.execution_id != execution.execution_id or record.call_id != call_id:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if record.status is not ExternalCallStatus.SUPPLIED or record.resolution_kind not in {"succeeded", "failed"}:
                    return False
                effect = await original(item)
                descriptor = record.result_payload
                if descriptor is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if descriptor.kind != "inline" or descriptor.encoding not in {"json", "base64", "utf-8"}:
                    # This transaction never opens ObjectStore to hydrate a
                    # decision. The existing external settlement route does.
                    terminal = (ToolOperationStatus.COMPLETED if record.resolution_kind == "succeeded"
                                else ToolOperationStatus.FAILED)
                    if effect.status is not terminal:
                        return False
                    continue
                try:
                    value = descriptor.decode()
                    if descriptor.encoding == "base64":
                        value = json.loads(value)
                except (TypeError, ValueError) as error:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                if record.resolution_kind == "succeeded":
                    if record.supplied_at is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    payload = StoredPayload.inline_bytes(encode_model_messages((ModelRequest(parts=[ToolReturnPart(
                        "runtime", value, tool_call_id=effect.tool_operation_id, timestamp=record.supplied_at,
                    )]),)))
                    if not settle_known(effect, payload, None, None):
                        return False
                elif not isinstance(value, str):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                elif not settle_known(effect, None, ErrorCode.TOOL_EXECUTION_FAILED.value, failure(value)):
                    return False

            execution_key = repository._key("execution", (execution.execution_id,))
            if await transaction.guard_record_revision(execution_key,
                    expected_storage_revision=observed.storage_revision) is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)

            async def ready(session):
                retained = await graph.step_store._capture_turn_checkpoint_row(
                    session, run_id=pending.source_agent_run_id,
                    conversation_id=graph.step_conversation_id, replay_key=graph.checkpoint_id,
                )
                if retained is None or retained[0]["state"] != "interrupted" or not retained[1].count:
                    raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                if await session.scalar(statement) != exact:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                changed = await session.execute(row.update().where(
                    *predicate, occupied, row.c.state == "waiting", row.c.lease_until.is_(None),
                ).values(state="ready"))
                if changed.rowcount != 1:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)

            # Only apply outcomes once all members are ready; a partially
            # resolved frontier does not commit standalone settlements.
            for effect, payload, code, error in settlements:
                await tools.settle_deferred_result_in_transaction(
                    tool_tx, effect, result_payload=payload, error_code=code, error_payload=error,
                )
            await group.transaction(turn_store).run_sql(graph.turns._context, ready)
            task_tx = group.transaction(task_store)
            if _queued_task is None:
                assert graph.task_lease is not None
                await graph.task_repository._guard_bound_execution_in_transaction(
                    task_tx, graph.task_lease, execution_id=execution.execution_id,
                )
            else:
                graph_id, node_id, fence = _queued_task
                await graph.task_repository._mark_bound_graph_deferred_ready_in_transaction(
                    task_tx, graph_id=graph_id, node_id=node_id, expected_fence=fence, execution_id=execution.execution_id,
                )
            return True

        return await execution_store.storage_group.mutate(stores, mark)

    async def claim_deferred_resume_checkpoint(
        self,
        *,
        execution_id: str,
        tenant_id: str,
        expected_execution_revision: int,
        expected_event_seq: int,
        expected_agent_run_seq: int,
        expected_pending_tools: PendingToolContinuation,
        background_tasks: "set[asyncio.Task[object]] | None" = None,
        recovery_operation: OperationLedgerInput | None = None,
        _graph_ready: "tuple[_GraphDeferredWait, int] | None" = None,
        _native_child: _NativeChildResume | None = None,
    ) -> ExecutionRecord:
        stores = [self._execution.state_store]
        operations = self._operations
        if _native_child is not None and (_graph_ready is not None or recovery_operation is not None):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if _graph_ready is not None:
            from sqlalchemy import select
            from ._session_turn_sql import CanonicalDeferredFrontier
            from ._step_store import SqlStepStore

            graph, lease_seconds = _graph_ready
            if (type(graph.step_store) is not SqlStepStore or graph.key.turn_id != execution_id
                    or type(lease_seconds) is not int or lease_seconds < 1
                    or tenant_id != self._tenant_id or graph.task_lease is None):
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            graph.turns._require_checkpoint_store(graph.step_store)
            launch = (graph.turns, graph.key, graph.generation, graph.task_repository, graph.task_lease)
            stores.extend(self._canonical_graph_launch_stores(execution_id, launch))
            turn_store, predicate, occupied = self._canonical_graph_turn_owner(graph.turns, graph.key, graph.generation)
            row = graph.turns._turns
            expected_run = agent_run_id(namespace=self._namespace, tenant_id=tenant_id, execution_id=execution_id,
                                        agent_run_seq=expected_agent_run_seq)
            if expected_pending_tools.source_agent_run_id != expected_run:
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            frontier = CanonicalDeferredFrontier.from_json(graph.key, graph.generation, {
                "checkpoint_id": graph.checkpoint_id,
                "approval_operation_ids": [item.operation_id for item in expected_pending_tools.approvals],
                "external_operation_ids": [item.operation_id for item in expected_pending_tools.calls],
                "operation_metadata": {item.operation_id: dict(item.metadata)
                    for item in (*expected_pending_tools.approvals, *expected_pending_tools.calls)},
            }).to_json()
            ready_slot = (*predicate, occupied, row.c.state == "ready", row.c.lease_until.is_(None))

        if recovery_operation is not None:
            if operations is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            stores.append(operations.state_store)
        stores = list(_dedupe_stores(stores))
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        next_sequence = expected_agent_run_seq + 1
        # Automatic Graph continuation already has the committed READY
        # frontier and live bound task lease. Its producer CAS creates the
        # unique claim; an explicit recovery operation still follows its ledger.
        producer_claim_id = recovery_operation.result_ref if recovery_operation is not None else uuid4().hex
        producer_generation = expected_execution_revision + 1
        next_agent_run_id = agent_run_id(
            namespace=self._namespace,
            tenant_id=tenant_id,
            execution_id=execution_id,
            agent_run_seq=next_sequence,
        )

        async def operation() -> ExecutionRecord:
            async def mutate(
                execution_tx: StateTransaction, group: StateGroupTransaction | None,
            ) -> ExecutionRecord:
                execution = await self._execution.get_in_transaction(
                    execution_tx,
                    execution_id,
                    tenant_id=tenant_id,
                )
                if recovery_operation is not None:
                    existing = await operations.get_in_transaction(
                        group.transaction(operations.state_store),
                        recovery_operation.operation_id, tenant_id=tenant_id,
                    )
                    if existing is not None:
                        if _graph_ready is not None:
                            # A previous producer receipt does not make this
                            # different invocation its dispatch owner.
                            raise AIError(ErrorCode.STORAGE_CONFLICT, retryable=False)
                        if not operation_identity_matches(existing, recovery_operation):
                            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                        if execution is None:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        return execution
                    validate_recovery_operation(
                        recovery_operation, execution_id=execution_id,
                        tenant_id=tenant_id, producer=True,
                    )
                checkpoint = execution
                if (
                    execution is None
                    or checkpoint is None
                    or checkpoint.recovery is None
                    or execution.status is not ExecutionStatus.WAITING_DEFERRED
                    or execution.revision != expected_execution_revision
                    or execution.event_seq != expected_event_seq
                    or execution.agent_run_seq != expected_agent_run_seq
                    or checkpoint.recovery_state is not ExecutionRecoveryState.WAITING
                    or checkpoint.recovery.pending_tools != expected_pending_tools
                ):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                if _native_child is not None:
                    _native_child.require_deferred_child(execution)
                if _graph_ready is not None and (
                    execution.root_execution_id != execution_id or execution.session_id is not None
                    or execution.parent_execution_id is not None or execution.parent_invocation_id is not None
                    or not execution.requires_task_invocation_capture or not execution.restart_checkpoints
                    or execution.native_step_source is None or execution.native_step_source != graph.step_store.backup_source
                ):
                    raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                updated_execution = (
                    await self._execution.claim_deferred_resume_in_transaction(
                        execution_tx,
                        execution_id,
                        tenant_id=tenant_id,
                        expected_revision=expected_execution_revision,
                        expected_event_seq=expected_event_seq,
                        expected_agent_run_seq=expected_agent_run_seq,
                        producer_claim_id=producer_claim_id,
                        recovery=replace(checkpoint.recovery, agent_run_id=next_agent_run_id,
                                         source_agent_run_id=expected_pending_tools.source_agent_run_id),
                    )
                )
                if recovery_operation is not None:
                    await operations.append_in_transaction(
                        group.transaction(operations.state_store), recovery_operation,
                    )
                if _graph_ready is not None:
                    async def claim_turn(session):
                        if await session.scalar(select(row.c.result_json).where(*ready_slot)) != frontier:
                            raise AIError(ErrorCode.STORAGE_CONFLICT)
                        retained = await graph.step_store._capture_turn_checkpoint_row(
                            session, run_id=expected_run, conversation_id=graph.step_conversation_id,
                            replay_key=graph.checkpoint_id,
                        )
                        if retained is None or retained[0]["state"] != "interrupted" or not retained[1].count:
                            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                        changed = await session.execute(row.update().where(*ready_slot).values(
                            state="accepted", generation=graph.generation + 1,
                            lease_until=graph.turns._now + lease_seconds, result_json=None,
                        ))
                        if changed.rowcount != 1:
                            raise AIError(ErrorCode.STORAGE_CONFLICT)

                    await group.transaction(turn_store).run_sql(graph.turns._context, claim_turn)
                    await graph.task_repository._guard_bound_execution_in_transaction(
                        group.transaction(graph.task_repository.state_store), graph.task_lease, execution_id=execution_id,
                    )
                return updated_execution

            if _native_child is not None:
                return await _native_child.guard_deferred_resume(
                    self._execution.state_store, lambda transaction: mutate(transaction, None),
                )
            return await stores[0].storage_group.mutate(
                stores, lambda group: mutate(group.transaction(self._execution.state_store), group),
            )

        async def readback() -> CommitObservation[ExecutionRecord]:
            if _graph_ready is not None or _native_child is not None:
                # A lost ACK must not launch a producer or replay any effect.
                # The caller can reconcile through the existing exact ledgers.
                return CommitObservation(DurableCommitState.UNRESOLVED)
            if recovery_operation is not None:
                existing = await operations.get(recovery_operation.operation_id, tenant_id=tenant_id)
                if existing is None:
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                if not operation_identity_matches(existing, recovery_operation):
                    return CommitObservation(
                        DurableCommitState.NOT_COMMITTED,
                        error=AIError(ErrorCode.IDEMPOTENCY_CONFLICT),
                    )
                current = await self._load_execution(execution_id, tenant_id=tenant_id)
                if current is None:
                    return _partial_integrity()
                return CommitObservation(DurableCommitState.COMMITTED, value=current)
            execution = await self._load_execution(execution_id, tenant_id=tenant_id)
            checkpoint = execution
            if execution is None or checkpoint is None or checkpoint.recovery is None:
                return _partial_integrity()
            own_claim = (
                execution is not None
                and execution.producer_generation == producer_generation
                and execution.producer_claim_id == producer_claim_id
            )
            if (
                own_claim
                and execution.status is ExecutionStatus.STARTED
                and execution.agent_run_seq == next_sequence
                and checkpoint.recovery_state is ExecutionRecoveryState.ACTIVE
                and checkpoint.recovery.agent_run_id == next_agent_run_id
                and checkpoint.recovery.source_agent_run_id == expected_pending_tools.source_agent_run_id
                and checkpoint.recovery.pending_tools == expected_pending_tools
            ):
                return CommitObservation(
                    DurableCommitState.COMMITTED,
                    value=execution,
                )
            if (
                execution.status is ExecutionStatus.WAITING_DEFERRED
                and execution.revision == expected_execution_revision
                and checkpoint.recovery.pending_tools == expected_pending_tools
            ):
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if not own_claim:
                return CommitObservation(
                    DurableCommitState.NOT_COMMITTED,
                    error=AIError(ErrorCode.STORAGE_CONFLICT, retryable=False),
                )
            return _partial_integrity()

        owner_tasks = self._background_tasks if background_tasks is None else background_tasks
        outcome = await run_durable_commit(
            operation,
            readback,
            background_tasks=owner_tasks,
        )
        return _require_durable_record(outcome)

    async def _commit_or_raise(
        self,
        operation: Callable[[], Awaitable[object]],
        readback: Callable[[], Awaitable[CommitObservation[object]]],
    ) -> None:
        result = await run_durable_commit(
            operation,
            readback,
            background_tasks=self._background_tasks,
        )
        if result.state is DurableCommitState.COMMITTED:
            if result.cancelled:
                raise asyncio.CancelledError
            return
        if result.state is DurableCommitState.NOT_COMMITTED:
            if result.error is not None:
                raise result.error
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if result.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                "durable command left partial state",
            ) from result.error
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from result.error

    async def _promote_history_in_transaction(
        self,
        transaction: StateTransaction,
        session: SessionRecord,
        prepared: PreparedAgentRunCheckpoint,
    ) -> ConversationHistoryRecord:
        if self._conversation_history is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        history_id = session.history_id or prepared.owner_id
        history = await self._conversation_history.get_in_transaction(
            transaction,
            history_id,
            tenant_id=self._tenant_id,
        )
        if history is None:
            history = ConversationHistoryRecord(
                history_id=history_id,
                session_id=session.session_id,
                parent_history_id=None,
                prefix_index_head_id=None,
                inherited_message_count=0,
            )
            await self._conversation_history.create_in_transaction(
                transaction,
                history,
            )
        return history



    async def commit_cancel_checkpoint(
        self,
        commit: ExecutionCancelRequestCommit,
        *,
        expected_status: ExecutionStatus,
        audit_events: Sequence[ExecutionEventAppend] = (),
        producer_generation: int | None = None,
        background_tasks: "set[asyncio.Task[object]] | None" = None,
    ) -> ExecutionRecord:
        expected_events = tuple(audit_events) + (
            ExecutionEventAppend(
                ExecutionEventType.CANCEL_REQUESTED,
                {"operation_id": commit.operation_id},
            ),
        )
        target_revision = commit.expected_revision + len(expected_events)
        target_sequence = commit.expected_event_seq + len(expected_events)

        async def operation() -> ExecutionRecord:
            if producer_generation is None:
                return await self._execution.request_cancel(commit, pending_events=audit_events)

            async def mutate(transaction: StateTransaction) -> ExecutionRecord:
                await self._execution.require_execution_producer_in_transaction(
                    transaction, commit.execution_id,
                    expected_producer_generation=producer_generation,
                )
                return await self._execution.request_cancel_in_transaction(
                    transaction, commit, expected_status=expected_status,
                    pending_events=audit_events,
                )

            return await self._execution.state_store.mutate(mutate)

        async def readback() -> CommitObservation[ExecutionRecord]:
            try:
                execution = await self._load_execution(
                    commit.execution_id,
                    tenant_id=self._tenant_id,
                )
                if execution is None:
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                    )
                page = await self._events.list(
                    commit.execution_id,
                    tenant_id=self._tenant_id,
                    after_event_seq=commit.expected_event_seq,
                    limit=len(expected_events) + 1,
                )
                items = page.items
                if any(
                    event.event_seq != commit.expected_event_seq + index
                    for index, event in enumerate(items, 1)
                ):
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                    )
                revision_delta = execution.revision - commit.expected_revision
                sequence_delta = execution.event_seq - commit.expected_event_seq
                if (
                    revision_delta < 0
                    or sequence_delta < 0
                    or revision_delta < sequence_delta
                    or items and items[-1].event_seq > execution.event_seq
                ):
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                    )

                prefix = items[:len(expected_events)]
                prefix_matches = (
                    len(prefix) == len(expected_events)
                    and all(
                        actual.event_type == expected.event_type
                        and actual.payload == expected.payload
                        for actual, expected in zip(prefix, expected_events)
                    )
                )
                if prefix_matches:
                    if (
                        execution.revision < target_revision
                        or execution.event_seq < target_sequence
                    ):
                        return CommitObservation(
                            DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                            error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                        )
                    if execution.status in {
                        ExecutionStatus.CANCELLING,
                        ExecutionStatus.FINALIZING,
                        ExecutionStatus.SUCCEEDED,
                        ExecutionStatus.FAILED,
                        ExecutionStatus.CANCELLED,
                    }:
                        return CommitObservation(DurableCommitState.COMMITTED, value=execution)
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                    )

                if revision_delta == 0 and sequence_delta == 0:
                    if execution.status is expected_status and not items:
                        return CommitObservation(DurableCommitState.NOT_COMMITTED)
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                    )

                if sequence_delta > 0 and not items:
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                    )
                return CommitObservation(
                    DurableCommitState.NOT_COMMITTED,
                    error=AIError(ErrorCode.STORAGE_CONFLICT),
                )
            except AIError as error:
                if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=error,
                    )
                return CommitObservation(DurableCommitState.UNRESOLVED, error=error)

        owner_tasks = (
            self._background_tasks
            if background_tasks is None
            else background_tasks
        )
        outcome = await run_durable_commit(
            operation,
            readback,
            background_tasks=owner_tasks,
        )
        if outcome.state is DurableCommitState.COMMITTED:
            if outcome.value is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if outcome.cancelled:
                raise asyncio.CancelledError
            return outcome.value
        if outcome.state is DurableCommitState.NOT_COMMITTED:
            if outcome.error is not None:
                raise outcome.error
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if outcome.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from outcome.error
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from outcome.error

    def can_claim_and_reserve_start(
        self, authority: StateTurnCoordination, reservation: ExecutionStartReservation, expected_session: SessionRecord,
    ) -> bool:
        return self._grouped_start_stores(authority, reservation, expected_session) is not None

    def _grouped_start_stores(
        self, authority: StateTurnCoordination, reservation: ExecutionStartReservation, expected_session: SessionRecord,
    ) -> tuple[StateStore, ...] | None:
        initial = reservation.execution
        if (type(authority) is not StateTurnCoordination or authority._coowned
                or type(self._conversation) is not SessionRepositoryImpl
                or type(self._execution) is not ExecutionRepositoryImpl
                or initial.session_id is None or initial.session_id != expected_session.session_id
                or not initial.conversation or initial.parent_execution_id is not None or initial.input_context is not None
                or initial.stored_user_input is None or initial.stored_user_input.codec != "text"
                or initial.budget_scope_id is not None or reservation.budget is not None):
            return None
        execution_store = self._execution.state_store
        session_store = self._conversation.state_store
        stores = _dedupe_stores((authority._store, execution_store, session_store,
                                self._execution._idempotency.state_store, self._events.state_store))
        return stores if _same_group(stores) else None

    async def claim_and_reserve_start(
        self, authority: StateTurnCoordination, scope: TurnScopeIdentity, reservation: ExecutionStartReservation, *,
        expected_session: SessionRecord, session_key: str, request_key: str, owner_id: str, lease_seconds: float,
    ) -> tuple[TurnCoordination | None, ExecutionStartReservationResult | None] | None:
        """Return coordinated local admission or a rollback-proven unsupported route.

        The original coordination and reservation APIs remain the paths for split
        storage and deferred publication. Only this command owns the combined
        commit and its readback.
        """
        initial, identity = reservation.execution, reservation.idempotency
        stores = self._grouped_start_stores(authority, reservation, expected_session)
        if stores is None:
            return None
        execution_store = self._execution.state_store
        session_store = self._conversation.state_store
        authority._writable()
        local_scope = authority._local_scope(scope)
        sql_coordination = authority._sql_source(local_scope)
        if (session_key != f"session:{initial.session_id}"
                or request_key != canonical_sha256([identity.scope, identity.idempotency_key])):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        duration = _duration(lease_seconds)
        token = uuid4().hex
        await self._execution.prepare_content(initial)
        claim = ExecutionStartClaim(initial.execution_id, initial.revision, initial.event_seq,
                                    identity.scope, identity.idempotency_key, identity.request_digest,
                                    initial.created_at)

        async def find_receipt() -> ExecutionStartReservationResult | None:
            existing = await self._execution.get_start_idempotency(claim)
            if existing is None:
                return None
            if (existing.request_digest != identity.request_digest
                    or existing.resource_kind is not identity.resource_kind):
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            record = await self._execution.get(existing.resource_id, tenant_id=self._tenant_id)
            if (record is None or record.session_id != initial.session_id
                    or record.parent_execution_id is not None):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return ExecutionStartReservationResult(record, existing, False)

        async def mutate(group: StateGroupTransaction) -> tuple[TurnCoordination | None, ExecutionStartReservationResult | None]:
            execution_tx = group.transaction(execution_store)
            session_tx = group.transaction(session_store)
            authority_tx = group.transaction(authority._store)
            if isinstance(group, GroupRecordReader):
                keys: dict[StateStore, tuple[bytes, ...]] = {}
                coordinates = () if sql_coordination is not None else (
                    (authority._store, authority._records.key(local_scope, "turn_session", session_key)),
                    (authority._store, authority._records.key(local_scope, "turn_request", request_key)),
                    (authority._store, authority._records.key(local_scope, "turn_execution", initial.execution_id)),
                )
                for store, key in (
                    (execution_store, self._execution.execution_record_key(initial.execution_id)),
                    (self._execution._idempotency.state_store, self._execution.start_idempotency_record_key(identity)),
                    (session_store, self._conversation.session_record_key(initial.session_id)),
                    *coordinates,
                ):
                    keys[store] = (*keys.get(store, ()), key)
                try:
                    await group.get_records(keys)
                except AIError as error:
                    if error.code is not ErrorCode.STORAGE_INTEGRITY_ERROR:
                        raise
                    # Preserve receipt priority if a companion failed eager decoding.
            session = await self._conversation.get_in_transaction(session_tx, initial.session_id, tenant_id=self._tenant_id)
            if session is None:
                raise AIError(ErrorCode.SESSION_NOT_FOUND)
            _require_session_capture(session, expected_session)
            if session.status is not SessionStatus.OPEN:
                raise AIError(ErrorCode.SESSION_CONFLICT)
            replay = await find_receipt()
            if replay is not None:
                if (replay.idempotency.status is IdempotencyStatus.START_UNKNOWN
                        or replay.execution.status is ExecutionStatus.START_UNKNOWN):
                    raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
                if (replay.idempotency.status is IdempotencyStatus.RESERVED
                        or (replay.execution.status is ExecutionStatus.STARTED and replay.execution.agent_run_seq == 0)):
                    # Original replay/launch owns these incomplete start paths.
                    raise _GroupedStartUnavailable()
                else:
                    return None, replay
            if session.continuation != expected_session.continuation:
                raise AIError(ErrorCode.SESSION_BUSY)
            if session.history_quality == "unavailable":
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
            stale = None
            if session.active_execution_id is not None:
                stale = await self._execution.get_in_transaction(execution_tx, session.active_execution_id, tenant_id=self._tenant_id)
                if (stale is None or stale.session_id != initial.session_id
                        or stale.parent_execution_id is not None):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if stale.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
                    # Leave guarded recovery/old actor replay to its established entry point.
                    raise AIError(ErrorCode.SESSION_BUSY)
            async def reserve() -> ExecutionStartReservationResult:
                if stale is not None:
                    await self._conversation.release_execution_in_transaction(
                        session_tx, initial.session_id, tenant_id=self._tenant_id, execution_id=stale.execution_id,
                        history_unavailable=stale.invalidates_session_history, require_turn_commit=stale.conversation,
                    )
                result = await self._reserve_start_in_transaction(
                    group, reservation, claim, expected_cursor=expected_session.continuation,
                    expected_session=expected_session,
                )
                if result is None:
                    raise _GroupedStartUnavailable()
                if not result.created or result.execution.execution_id != initial.execution_id:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                return result

            coordinated = await authority._claim_in_transaction(
                authority_tx, local_scope, session_key=session_key, request_key=request_key,
                request_digest=identity.request_digest, execution_id=initial.execution_id,
                duration=duration, owner_id=owner_id, token=token,
            )
            if coordinated is None:
                raise AIError(ErrorCode.SESSION_BUSY)
            if coordinated.lease is None:
                return coordinated, None
            return coordinated, await reserve()

        async def readback(group: StateGroupTransaction) -> tuple[TurnCoordination | None, ExecutionStartReservationResult | None] | None:
            replay = await find_receipt()
            if replay is None or replay.execution.execution_id != initial.execution_id:
                return None
            graph = None
            if sql_coordination is not None:
                coordinated = await authority._confirmed(
                    local_scope, initial.execution_id, receipt=token, kind="claim",
                    expected_claim=(session_key, request_key, identity.request_digest, owner_id),
                )
                if not isinstance(coordinated, TurnCoordination) or coordinated.lease is None:
                    return None
            else:
                graph = await authority._anchor(group.transaction(authority._store), local_scope, "turn_execution", initial.execution_id)
                if (graph is None or graph.execution.data["token"] != token
                        or graph.execution.data["owner_id"] != owner_id
                        or graph.execution.data["request_key"] != request_key
                        or graph.execution.data["request_digest"] != identity.request_digest
                        or graph.execution.data["session_key"] != session_key):
                    return None
                coordinated = authority._coordination(graph, owned=True)
            session = await self._conversation.get_in_transaction(
                group.transaction(session_store), initial.session_id, tenant_id=self._tenant_id,
            )
            if (session is None or session.active_execution_id != initial.execution_id
                    or session.continuation != expected_session.continuation
                    or replay.idempotency.status is not IdempotencyStatus.STARTED
                    or replay.execution.status is not ExecutionStatus.STARTED
                    or replay.execution.agent_run_seq != 0
                    or graph is not None and graph.session.lease_expires_at <= await group.transaction(authority._store).now()):
                return None
            reconciled = await self._reconcile_start_unknown(
                claim, session_id=initial.session_id, expected_cursor=expected_session.continuation,
                expected_session=expected_session,
            )
            if reconciled != replay.execution:
                return None
            return coordinated, replace(replay, session=session, created=True)

        for attempt in range(3):
            try:
                return await execution_store.storage_group.mutate(stores, mutate)
            except _GroupedStartUnavailable:
                return None
            except _TurnGraphReadError as error:
                if not await authority._anchor_advanced(error):
                    raise
                if attempt == 2:
                    raise AIError(ErrorCode.STORAGE_CONFLICT) from error
            except AIError as error:
                if error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
                    try:
                        confirmed = await execution_store.storage_group.read_many(stores, readback)
                    except Exception as read_error:
                        raise error from read_error
                    if confirmed is not None:
                        return confirmed
                    raise
                if error.code is not ErrorCode.STORAGE_CONFLICT or attempt == 2:
                    raise
            await asyncio.sleep(0)
        raise AssertionError("unreachable grouped admission attempt")

    async def _reserve_canonical_graph_start(
        self, reservation: ExecutionStartReservation, *, turns: "SqlSessionTurns", key: "SessionTurnKey",
        expected_revision: int, context_from: int, context_through: int,
        request_key: str, request: "dict[str, JsonValue]", lease_seconds: int,
        task_repository: "TaskRepositoryImpl", task_lease: "TaskLease",
        instruction_scopes: tuple[str, ...] | None = None,
    ) -> ExecutionStartReservationResult:
        """Reserve the root Graph owner and its exact canonical turn in one group."""
        from ._task_repository import TaskRepositoryImpl

        repository = self._execution
        execution = reservation.execution
        binding = execution.binding
        if (
            type(repository) is not ExecutionRepositoryImpl
            or type(task_repository) is not TaskRepositoryImpl or task_lease.tenant_id != self._tenant_id
            or turns._scope != state_store_key(self._namespace, self._tenant_id, "regenerable-session")
            or execution.execution_id != key.turn_id or execution.root_execution_id != key.turn_id
            or execution.session_id is not None or execution.parent_execution_id is not None
            or execution.parent_invocation_id is not None or not execution.requires_task_invocation_capture
            or not isinstance(binding, AgentBindingContract)
        ):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if execution.input_context is not None or execution.context_imported or "input" in request:
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
        from ..model import native_graph_input_content, native_graph_input_metadata

        prompt = native_graph_input_content(execution.stored_user_input)
        if (
            prompt is None or request.get("prompt") != (prompt if isinstance(prompt, str) else "")
            or request.get("principal") != {
                "principal_id": execution.principal_id, "tenant_id": self._tenant_id,
                "kind": execution.principal_kind,
            }
            or "graph_execution" in request
        ):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        transcript_store = turns._transcript_write_binding
        if transcript_store is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        store, archive_provider, native = transcript_store
        if not _same_group((repository.state_store, store, task_repository.state_store)):
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
        archive = await archive_provider(create=True)
        if not _same_group((store, archive.store)):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        metadata = native_graph_input_metadata(execution.stored_user_input)
        if metadata is not None and not archive.supports_attachment_objects:
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
        prepared = PreparedArchive()
        source = NativeUserInputReference(await native.prepare_archived_messages(
            (ModelRequest(
                parts=[UserPromptPart(content=prompt, timestamp=execution.created_at)],
                metadata=metadata,
            ),), prepared,
        ), 0)
        admission = await turns._prepare_accept(
            key, expected_revision=expected_revision, context_from=context_from, context_through=context_through,
            request_key=request_key, request={
                **request, "graph_execution": execution.execution_id, "prompt": "", "input": source.to_json(),
            },
            binding=binding.to_payload(), lease_seconds=lease_seconds, budget=None, instruction_scopes=instruction_scopes,
        )
        stores = tuple(dict.fromkeys((repository.state_store, archive.store, task_repository.state_store, *admission.stores)))

        async def mutate(group: StateGroupTransaction) -> ExecutionStartReservationResult:
            from ._native_content_index import stage_owner
            from ._session_transcript import SESSION_TURN_TABLE

            reserved = await repository.reserve_started(reservation, _admitted_input_source=source)
            if not reserved.created:
                return reserved
            if not await turns._accept_in_group(group, admission):
                raise AIError(ErrorCode.SESSION_CONFLICT)
            await task_repository.bind_execution(
                task_lease, tenant_id=self._tenant_id, execution_id=key.turn_id,
            )
            values = turns._turn_values(key)

            async def stage(session) -> None:
                stage_owner(session, SESSION_TURN_TABLE, turns._scope,
                            (values["session_key"], values["turn_id"]), source.root)

            await group.transaction(store).run_sql(turns._context, stage)
            await archive.publish(group.transaction(archive.store), prepared)
            return reserved

        # An unknown commit is deliberately not retried or promoted into a claim.
        result = await stores[0].storage_group.mutate(stores, mutate)
        if result.created:
            turns._note_accepted(admission)
        return result

    def _canonical_graph_turn_owner(self, turns: "SqlSessionTurns", key: "SessionTurnKey", generation: int):
        """The linked row and active Session slot are the Graph turn lease authority."""
        from sqlalchemy import select

        binding = turns._transcript_write_binding
        if (type(self._execution) is not ExecutionRepositoryImpl
                or turns._scope != state_store_key(self._namespace, self._tenant_id, "regenerable-session")
                or binding is None):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        store = binding[0]
        if not _same_group((self._execution.state_store, store)):
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
        if generation < 1:
            raise ValueError("Graph turn generation must be positive")
        sessions, rows = turns._sessions, turns._turns
        values = turns._turn_values(key)
        occupied = select(sessions.c.active_turn_id).filter_by(
            **turns._session_values(key.session_id), active_turn_id=values["turn_id"],
        ).exists()
        predicate = (
            *(rows.c[name] == value for name, value in values.items()),
            rows.c.generation == generation,
            rows.c.request_json["graph_execution"].as_string() == key.turn_id,
        )
        return store, predicate, occupied

    def _canonical_graph_launch_stores(
        self, execution_id: str, launch: "tuple[SqlSessionTurns, SessionTurnKey, int, TaskRepositoryImpl, TaskLease | _TaskGraphCancelOwner]",
    ) -> tuple[StateStore, StateStore]:
        from ._task_repository import TaskRepositoryImpl

        turns, key, generation, task_repository, lease = launch
        store, _, _ = self._canonical_graph_turn_owner(turns, key, generation)
        if (type(task_repository) is not TaskRepositoryImpl or key.turn_id != execution_id
                or lease.tenant_id != self._tenant_id):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if not _same_group((store, task_repository.state_store)):
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
        return store, task_repository.state_store

    def _canonical_graph_producer_boundary(
        self, *, turns: "SqlSessionTurns", key: "SessionTurnKey", generation: int, cancelling: bool = False,
    ) -> tuple[StateStore, Callable[[StateGroupTransaction], Awaitable[None]]]:
        """Return the exact linked live turn fence for a producer's grouped write."""
        store, predicate, occupied = self._canonical_graph_turn_owner(turns, key, generation)
        row = turns._turns

        async def guard(group: StateGroupTransaction) -> None:
            async def write(session):
                result = await session.execute(row.update().where(
                    *predicate, occupied, row.c.state == ("cancelling" if cancelling else "accepted"),
                    row.c.lease_until > turns._now,
                ).values(lease_until=row.c.lease_until))
                if result.rowcount != 1:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)

            await group.transaction(store).run_sql(turns._context, write)

        return store, guard

    async def _guard_canonical_graph_launch(
        self, group: StateGroupTransaction, execution_id: str, launch: "_GraphLaunch",
    ) -> None:
        turns, key, generation, task_repository, lease = launch
        _, guard = self._canonical_graph_producer_boundary(turns=turns, key=key, generation=generation)
        await guard(group)
        await task_repository._guard_bound_execution_in_transaction(
            group.transaction(task_repository.state_store), lease, execution_id=execution_id,
        )

    async def _renew_canonical_graph_turn(
        self, *, turns: "SqlSessionTurns", key: "SessionTurnKey", generation: int,
        lease_seconds: int, cancelling: bool = False,
    ) -> bool:
        from sqlalchemy import case

        if lease_seconds < 1:
            raise ValueError("the turn lease must be positive")
        _, predicate, occupied = self._canonical_graph_turn_owner(turns, key, generation)
        row = turns._turns
        expires = turns._now + lease_seconds

        async def renew(session):
            updated = await session.execute(row.update().where(
                *predicate, occupied, row.c.state == ("cancelling" if cancelling else "accepted"),
                row.c.lease_until > turns._now,
            ).values(lease_until=case((row.c.lease_until < expires, expires), else_=row.c.lease_until)))
            return updated.rowcount == 1

        return await turns._context.run_mutation(renew, domain="session.renew")

    async def _request_canonical_graph_cancel(
        self, commit: ExecutionCancelRequestCommit, *, turns: "SqlSessionTurns", key: "SessionTurnKey",
        generation: int, expected_status: ExecutionStatus, producer_generation: int | None,
        audit_events: Sequence[ExecutionEventAppend] = (),
        _group: StateGroupTransaction | None = None,
    ) -> ExecutionRecord:
        """Stop both authorities or neither, without making an expired producer live."""
        if commit.execution_id != key.turn_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if (expected_status not in {ExecutionStatus.PENDING_START, ExecutionStatus.STARTED}
                or (expected_status is ExecutionStatus.PENDING_START) != (producer_generation is None)
                or (producer_generation is not None
                    and (type(producer_generation) is not int or producer_generation < 0))):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        store, predicate, occupied = self._canonical_graph_turn_owner(turns, key, generation)
        repository = cast(ExecutionRepositoryImpl, self._execution)
        execution_store = repository.state_store
        row = turns._turns

        async def stop(group: StateGroupTransaction) -> ExecutionRecord:
            tx = group.transaction(execution_store)
            if producer_generation:
                await repository.require_execution_producer_in_transaction(
                    tx, key.turn_id, expected_producer_generation=producer_generation, allow_cancelling=False,
                )
            execution = await repository.request_cancel_in_transaction(
                tx, commit, expected_status=expected_status, pending_events=audit_events,
            )
            if execution.producer_generation != (producer_generation or 0):
                raise AIError(ErrorCode.STORAGE_CONFLICT)

            async def cancel(session):
                updated = await session.execute(row.update().where(
                    *predicate, occupied, row.c.state == "accepted", row.c.lease_until > turns._now,
                ).values(state="cancelling"))
                if updated.rowcount != 1:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)

            await group.transaction(store).run_sql(turns._context, cancel)
            return execution

        if _group is not None:
            return await stop(_group)
        return await execution_store.storage_group.mutate(_dedupe_stores((execution_store, store)), stop)

    async def _request_linked_graph_cancel(
        self, commit: ExecutionCancelRequestCommit, *, turns: "SqlSessionTurns",
        task_repository: "TaskRepositoryImpl", task_lease: "TaskLease",
        expected_status: ExecutionStatus, producer_generation: int | None,
        audit_events: Sequence[ExecutionEventAppend] = (),
    ) -> ExecutionRecord | None:
        """Resolve the real Execution by its existing unique turn key and stop both owners."""
        from ._task_repository import TaskRepositoryImpl

        binding = turns._transcript_write_binding
        if (type(self._execution) is not ExecutionRepositoryImpl or type(task_repository) is not TaskRepositoryImpl
                or turns._scope != state_store_key(self._namespace, self._tenant_id, "regenerable-session")
                or task_lease.tenant_id != self._tenant_id or binding is None):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        execution_store, store, task_store = self._execution.state_store, binding[0], task_repository.state_store
        stores = _dedupe_stores((execution_store, store, task_store))
        if not _same_group(stores):
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)

        execution_id = commit.execution_id

        async def stop(group: StateGroupTransaction) -> ExecutionRecord | None:
            linked = await turns._linked_graph_turn(execution_id, transaction=group.transaction(store))
            if linked is None:
                return None
            key, generation = linked
            execution = await self._request_canonical_graph_cancel(
                commit, turns=turns, key=key, generation=generation, expected_status=expected_status,
                producer_generation=producer_generation, audit_events=audit_events, _group=group,
            )
            # The two cancellation writes remain provisional until the exact task
            # lease and its real bound Execution are revision-fenced as well.
            await task_repository._guard_bound_execution_in_transaction(
                group.transaction(task_store), task_lease, execution_id=execution_id,
            )
            return execution

        return await execution_store.storage_group.mutate(stores, stop)

    async def _guard_cold_graph_task(
        self, group: StateGroupTransaction, repository: "TaskRepositoryImpl",
        authority: "TaskLease | _TaskGraphCancelOwner", *, execution_id: str,
    ) -> None:
        from ._task_repository import _TaskGraphCancelOwner

        task_tx = group.transaction(repository.state_store)
        if isinstance(authority, _TaskGraphCancelOwner):
            await repository._guard_cancelled_execution_in_transaction(task_tx, authority, execution_id=execution_id)
        else:
            await repository._guard_bound_execution_in_transaction(task_tx, authority, execution_id=execution_id)

    async def _claim_expired_undispatched_graph_turn(
        self, expected_execution: ExecutionRecord, *, turns: "SqlSessionTurns", key: "SessionTurnKey",
        generation: int, task_repository: "TaskRepositoryImpl", task_lease: "TaskLease",
        recovery_operation: OperationLedgerRecord, lease_seconds: int,
        _native_step: "SqlStepStore | None" = None,
    ) -> tuple[ExecutionRecord, int]:
        """Move the accepted turn to its recovering task without starting a model."""
        if lease_seconds < 1:
            raise ValueError("the turn lease must be positive")
        execution = expected_execution
        if (execution.execution_id != key.turn_id or execution.root_execution_id != key.turn_id
                or execution.session_id is not None or execution.parent_execution_id is not None
                or execution.parent_invocation_id is not None or not execution.requires_task_invocation_capture
                or execution.status is not ExecutionStatus.STARTED):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if (execution.producer_generation != 0 or execution.producer_claim_id is not None
                or execution.agent_run_seq != 0 or execution.recovery_state is not ExecutionRecoveryState.ADMITTED
                or execution.recovery is None or execution.recovery.pending_tools is not None
                or execution.recovery.source_agent_run_id is not None
                or execution.restart_checkpoints != (_native_step is not None)
                or execution.history_seal_digest is not None):
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
        launch = (turns, key, generation, task_repository, task_lease)
        repository = self._execution
        execution_store = repository.state_store
        tools = None
        if _native_step is not None:
            from ._step_store import SqlStepStore

            tools = self._tools
            if (type(_native_step) is not SqlStepStore or type(tools) is not ToolRepositoryImpl
                    or execution.native_step_source is None or execution.native_step_source != _native_step.backup_source):
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            turns._require_checkpoint_store(_native_step)
            if not _same_group((execution_store, tools.state_store)):
                raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
        stores = _dedupe_stores((execution_store, *self._canonical_graph_launch_stores(key.turn_id, launch),
                                 *((tools.state_store,) if tools is not None else ())))
        store, predicate, occupied = self._canonical_graph_turn_owner(turns, key, generation)
        row = turns._turns

        async def claim(group: StateGroupTransaction) -> tuple[ExecutionRecord, int]:
            tx = group.transaction(execution_store)
            current = await repository.get_in_transaction(tx, key.turn_id, tenant_id=self._tenant_id)
            if current is None or current != execution:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            # Invalidate the stale attempt CAS without fabricating an agent run
            # or treating this turn handover as model dispatch authority.
            now = await tx.now()
            renewed = await repository.compare_and_swap_in_transaction(
                tx, key.turn_id, tenant_id=self._tenant_id, expected_revision=current.revision,
                next_record=replace(current, revision=current.revision + 1, updated_at=max(current.updated_at, now)),
            )

            async def take_over(session):
                updated = await session.execute(row.update().where(
                    *predicate, occupied, row.c.state == "accepted", row.c.lease_until <= turns._now,
                ).values(generation=generation + 1, lease_until=turns._now + lease_seconds))
                if updated.rowcount != 1:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)

            await group.transaction(store).run_sql(turns._context, take_over)
            await task_repository._guard_recovering_bound_execution_in_transaction(
                group.transaction(task_repository.state_store), task_lease, recovery_operation,
                execution_id=key.turn_id,
            )
            if _native_step is not None:
                assert tools is not None
                first_run = agent_run_id(namespace=self._namespace, tenant_id=self._tenant_id,
                                         execution_id=key.turn_id, agent_run_seq=1)
                if await _native_step.get_run_in_transaction(tx, run_id=first_run) is not None:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                existing_effects = await group.transaction(tools.state_store).list_records(RecordQuery(
                    kind="tool_operation", parent_key=tools._parent("tool_operation", "execution", (key.turn_id,)), limit=1,
                ))
                if existing_effects:
                    raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
            return renewed, generation + 1

        # An unknown commit cannot grant launch or a second takeover.
        return await execution_store.storage_group.mutate(stores, claim)

    def _cold_graph_step_store(
        self, execution: ExecutionRecord, turns: "SqlSessionTurns",
        native: "SqlStepStore", run_id: str, authority: "_TaskGraphCancelOwner",
    ) -> ToolRepositoryImpl:
        from ._step_store import SqlStepStore
        from ._task_repository import _TaskGraphCancelOwner

        tools = self._tools
        if (type(native) is not SqlStepStore or not isinstance(authority, _TaskGraphCancelOwner)
                or type(tools) is not ToolRepositoryImpl
                or execution.native_step_source is None or execution.native_step_source != native.backup_source
                or execution.agent_run_seq < 1
                or run_id != agent_run_id(namespace=self._namespace, tenant_id=self._tenant_id,
                                           execution_id=execution.execution_id, agent_run_seq=execution.agent_run_seq)):
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
        turns._require_checkpoint_store(native)
        if not _same_group((self._execution.state_store, tools.state_store)):
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
        return tools

    async def _require_cold_cancelled_graph_step(
        self, group: StateGroupTransaction, execution_id: str, *, tools: ToolRepositoryImpl,
        native: "SqlStepStore", run_id: str, conversation_id: str, _write_if_missing: bool = False,
    ) -> None:
        from ..steps import NativeLifecycleEvent

        execution_tx = group.transaction(self._execution.state_store)
        key = (run_id, NativeLifecycleEvent.source_key())
        marker = (await native.read_fixed_events((key,), transaction=execution_tx)).get(key)
        unresolved = await group.transaction(tools.state_store).list_records(RecordQuery(
            kind="tool_operation", parent_key=tools._parent("tool_operation", "execution", (execution_id,)),
            states=frozenset({
                ToolOperationStatus.PENDING.value, ToolOperationStatus.CLAIMED.value,
                ToolOperationStatus.EFFECT_UNKNOWN.value,
            }), limit=1,
        ))
        if unresolved:
            raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
        if marker is None and _write_if_missing:
            await native._record_cold_cancelled_lifecycle(
                execution_tx, run_id=run_id, conversation_id=conversation_id, occurred_at=await execution_tx.now(),
            )
        elif (not isinstance(marker, NativeLifecycleEvent) or marker.outcome != "cancelled"
                or marker.conversation_id != conversation_id):
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)

    async def _claim_expired_linked_graph_cancellation(
        self, expected_execution: ExecutionRecord, *, turns: "SqlSessionTurns", key: "SessionTurnKey",
        generation: int, task_repository: "TaskRepositoryImpl", task_lease: "TaskLease | _TaskGraphCancelOwner",
        lease_seconds: int,
        _cold_step: "tuple[SqlStepStore, str, str] | None" = None,
    ) -> tuple[ExecutionRecord, int]:
        """Rotate both producers of an expired Stop, without creating a new model attempt."""
        if lease_seconds < 1:
            raise ValueError("the turn lease must be positive")
        if (expected_execution.execution_id != key.turn_id
                or expected_execution.root_execution_id != key.turn_id
                or expected_execution.session_id is not None
                or expected_execution.parent_execution_id is not None
                or expected_execution.parent_invocation_id is not None
                or not expected_execution.requires_task_invocation_capture
                or expected_execution.status is not ExecutionStatus.CANCELLING):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if (expected_execution.restart_checkpoints != (_cold_step is not None)
                or expected_execution.recovery_state in {ExecutionRecoveryState.HANDOFF, ExecutionRecoveryState.WAITING}
                or (expected_execution.recovery is not None and expected_execution.recovery.pending_tools is not None)):
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
        launch = (turns, key, generation, task_repository, task_lease)
        repository = self._execution
        execution_store = repository.state_store
        cold_tools = (None if _cold_step is None else self._cold_graph_step_store(
            expected_execution, turns, _cold_step[0], _cold_step[1], task_lease,
        ))
        stores = _dedupe_stores((execution_store, *self._canonical_graph_launch_stores(key.turn_id, launch),
                                 *((cold_tools.state_store,) if cold_tools is not None else ())))
        store, predicate, occupied = self._canonical_graph_turn_owner(turns, key, generation)
        row = turns._turns
        claim_id = uuid4().hex

        async def claim(group: StateGroupTransaction) -> tuple[ExecutionRecord, int]:
            tx = group.transaction(execution_store)
            current = await repository.get_in_transaction(tx, key.turn_id, tenant_id=self._tenant_id)
            if (current is None or current != expected_execution
                    or current.producer_generation >= current.revision + 1):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            now = await tx.now()
            rotated = await repository.compare_and_swap_in_transaction(
                tx, key.turn_id, tenant_id=self._tenant_id, expected_revision=current.revision,
                next_record=replace(
                    current, revision=current.revision + 1, producer_generation=current.revision + 1,
                    producer_claim_id=claim_id, updated_at=max(current.updated_at, now),
                ),
            )

            async def take_over(session):
                updated = await session.execute(row.update().where(
                    *predicate, occupied, row.c.state == "cancelling", row.c.lease_until <= turns._now,
                ).values(generation=generation + 1, lease_until=turns._now + lease_seconds))
                if updated.rowcount != 1:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)

            await group.transaction(store).run_sql(turns._context, take_over)
            await self._guard_cold_graph_task(group, task_repository, task_lease, execution_id=key.turn_id)
            if _cold_step is not None:
                assert cold_tools is not None
                native, run_id, conversation_id = _cold_step
                await self._require_cold_cancelled_graph_step(
                    group, key.turn_id, tools=cold_tools, native=native, run_id=run_id,
                    conversation_id=conversation_id, _write_if_missing=True,
                )
            return rotated, generation + 1

        # Unknown commit doesn't confer a permission to settle/dispatch.
        return await execution_store.storage_group.mutate(stores, claim)

    async def _observe_canonical_graph_turn(
        self, *, turns: "SqlSessionTurns", key: "SessionTurnKey", generation: int,
    ) -> tuple[ExecutionRecord, str] | None:
        """Read an exact grouped outcome; an unknown commit is not permission to retry."""
        from sqlalchemy import select

        store, predicate, occupied = self._canonical_graph_turn_owner(turns, key, generation)
        repository = self._execution
        execution_store = repository.state_store
        row = turns._turns
        statement = select(row.c.state, occupied).where(*predicate)

        async def observe(group: StateGroupTransaction) -> tuple[ExecutionRecord, str] | None:
            async def turn_state(session):
                return (await session.execute(statement)).one_or_none()

            turn = await group.transaction(store).run_sql(turns._context, turn_state)
            if turn is None:
                return None
            state, active = turn
            execution = await repository.get_in_transaction(
                group.transaction(execution_store), key.turn_id, tenant_id=self._tenant_id,
            )
            compatible = {
                "accepted": {ExecutionStatus.PENDING_START, ExecutionStatus.STARTED, ExecutionStatus.FINALIZING},
                "cancelling": {ExecutionStatus.CANCELLING},
                "succeeded": {ExecutionStatus.SUCCEEDED},
                "failed": {ExecutionStatus.FAILED},
                "cancelled": {ExecutionStatus.CANCELLED},
            }
            if (
                execution is None or execution.root_execution_id != key.turn_id
                or execution.session_id is not None or not execution.requires_task_invocation_capture
                or execution.status not in compatible.get(state, ())
                or active != (state in {"accepted", "cancelling"})
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return execution, state

        return await execution_store.storage_group.read_many(_dedupe_stores((execution_store, store)), observe)

    async def reserve_start_checkpoint(
        self, reservation: ExecutionStartReservation, *,
        expected_cursor: ConversationCursor | None,
        expected_session: SessionRecord | None = None,
        authority: "ExecutionAuthority | None" = None,
    ) -> ExecutionStartReservationResult | None:
        """Reserve and admit a new local execution without publishing a worker."""
        initial = reservation.execution
        if (type(self._execution) is not ExecutionRepositoryImpl
                or initial.parent_execution_id is not None):
            return None
        stores = [self._execution.state_store]
        if initial.session_id is not None:
            self._require_conversation()
            if type(self._conversation) is not SessionRepositoryImpl:
                return None
            stores.append(self._conversation.state_store)
        if not _same_group(stores):
            return None
        await self._execution.prepare_content(initial)
        identity = reservation.idempotency
        claim = ExecutionStartClaim(initial.execution_id, initial.revision, initial.event_seq,
                                    identity.scope, identity.idempotency_key,
                                    identity.request_digest, initial.created_at)

        async def mutate(group: StateGroupTransaction) -> ExecutionStartReservationResult | None:
            return await self._reserve_start_in_transaction(
                group, reservation, claim, expected_cursor=expected_cursor, expected_session=expected_session,
                authority=authority,
            )

        for attempt in range(2):
            try:
                return await stores[0].storage_group.mutate(stores, mutate)
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN:
                    raise
                observed_identity = await self._execution.get_start_idempotency(claim)
                own = await self._execution.get(initial.execution_id, tenant_id=self._tenant_id)
                if observed_identity is None:
                    if own is not None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                    if attempt == 0:
                        continue
                    raise
                if (observed_identity.request_digest != identity.request_digest
                        or observed_identity.resource_kind != identity.resource_kind):
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT) from error
                if observed_identity.resource_id != initial.execution_id:
                    winner = await self._execution.get(observed_identity.resource_id, tenant_id=self._tenant_id)
                    if winner is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                    return ExecutionStartReservationResult(winner, observed_identity, False)
                started = await self._reconcile_start_unknown(
                    claim, session_id=initial.session_id,
                    expected_cursor=expected_cursor, expected_session=expected_session,
                )
                if started is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                session = (None if initial.session_id is None else
                       await self._load_session(initial.session_id, tenant_id=self._tenant_id))
                if initial.session_id is not None and (session is None or session.active_execution_id != initial.execution_id):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                return ExecutionStartReservationResult(started, observed_identity, True, session=session)
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

    async def _reserve_start_in_transaction(
        self, group: StateGroupTransaction, reservation: ExecutionStartReservation, claim: ExecutionStartClaim, *,
        expected_cursor: ConversationCursor | None, expected_session: SessionRecord | None,
        authority: "ExecutionAuthority | InitialExecutionAuthority | None" = None,
    ) -> ExecutionStartReservationResult | None:
        initial, identity = reservation.execution, reservation.idempotency
        if initial.session_id is not None and isinstance(group, GroupRecordReader):
            keys = {self._execution.state_store: (
                self._execution.execution_record_key(initial.execution_id),
                self._execution.start_idempotency_record_key(identity),
            )}
            store = self._conversation.state_store
            keys[store] = (*keys.get(store, ()), self._conversation.session_record_key(initial.session_id))
            await group.get_records(keys)
        # Busy, stale-owner and changed-cursor admissions retain the existing
        # reservation/rejection path, including durable failed request keys.
        if initial.session_id is not None:
            session = await self._conversation.get_in_transaction(
                group.transaction(self._conversation.state_store), initial.session_id,
                tenant_id=self._tenant_id,
            )
            if session is not None and expected_session is not None:
                _require_session_capture(session, expected_session)
            if (session is None or session.status is not SessionStatus.OPEN
                    or session.active_execution_id is not None
                    or session.continuation != expected_cursor
                    or session.history_quality == "unavailable"):
                return None
        reserved = await self._execution.reserve_started(reservation, authority=authority)
        if not reserved.created:
            return reserved
        turns: list[SessionTurnRef] = []
        started = reserved.execution
        if initial.session_id is not None:
            await self._conversation.admit_execution_in_transaction(
                group.transaction(self._conversation.state_store), initial.session_id,
                tenant_id=self._tenant_id, execution_id=claim.execution_id,
                expected=expected_cursor, expected_session=expected_session,
                admitted_turn=turns.append,
            )
        started_identity = await self._execution.get_start_idempotency(claim)
        if started_identity is None or started_identity.status is not IdempotencyStatus.STARTED:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        admitted_session = (None if initial.session_id is None else
            await self._conversation.get_in_transaction(
                group.transaction(self._conversation.state_store), initial.session_id,
                tenant_id=self._tenant_id,
            ))
        observed = await self._execution.get_observed_in_transaction(
            group.transaction(self._execution.state_store), initial.execution_id, tenant_id=self._tenant_id,
        )
        if observed is None or observed.value != started:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return ExecutionStartReceipt(
            started, started_identity, True, session=admitted_session,
            observed=observed, turn=turns[0] if turns else None,
        )

    async def commit_start_checkpoint(
        self,
        claim: ExecutionStartClaim,
        *,
        session_id: str | None = None,
        expected_cursor: ConversationCursor | None = None,
        expected_session: SessionRecord | None = None,
        admitted_turn: Callable[[SessionTurnRef], None] | None = None,
    ) -> ExecutionRecord:
        _logger.debug(
            "start checkpoint requested: execution=%s session=%s",
            claim.execution_id,
            session_id,
        )
        stores = [self._execution.state_store]
        if session_id is not None:
            self._require_conversation()
            stores.append(self._conversation.state_store)
        if _same_group(stores):
            async def callback(group: StateGroupTransaction) -> ExecutionRecord:
                transaction = group.transaction(self._execution.state_store)
                if session_id is not None:
                    session_transaction = group.transaction(self._conversation.state_store)
                    session = await self._conversation.get_in_transaction(
                        session_transaction,
                        session_id,
                        tenant_id=self._tenant_id,
                    )
                    if session is not None and expected_session is not None:
                        _require_session_capture(session, expected_session)
                    if session is None:
                        raise AIError(ErrorCode.STORAGE_NOT_FOUND)
                    if session.active_execution_id not in {None, claim.execution_id}:
                        owner = await self._execution.get_in_transaction(
                            transaction,
                            session.active_execution_id,
                            tenant_id=self._tenant_id,
                        )
                        if (
                            owner is None
                            or owner.session_id != session_id
                            or owner.parent_execution_id is not None
                        ):
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        if owner.status not in {
                            ExecutionStatus.SUCCEEDED,
                            ExecutionStatus.FAILED,
                            ExecutionStatus.CANCELLED,
                        }:
                            if session.status is not SessionStatus.OPEN:
                                raise AIError(ErrorCode.SESSION_CONFLICT)
                            raise AIError(ErrorCode.SESSION_BUSY)
                        await self._conversation.release_execution_in_transaction(
                            session_transaction,
                            session_id,
                            tenant_id=self._tenant_id,
                            execution_id=owner.execution_id,
                            history_unavailable=owner.invalidates_session_history,
                            require_turn_commit=owner.conversation,
                        )
                    await self._conversation.admit_execution_in_transaction(
                        session_transaction,
                        session_id,
                        tenant_id=self._tenant_id,
                        execution_id=claim.execution_id,
                        expected=expected_cursor,
                        expected_session=expected_session,
                        **({"admitted_turn": admitted_turn} if type(self._conversation) is SessionRepositoryImpl else {}),
                    )
                return await self._execution.claim_start_in_transaction(transaction, claim)

            try:
                return await stores[0].storage_group.mutate(stores, callback)
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN:
                    raise
                observed = await self._reconcile_start_unknown(
                    claim,
                    session_id=session_id,
                    expected_cursor=expected_cursor, expected_session=expected_session,
                )
                if observed is not None:
                    return observed
                return await stores[0].storage_group.mutate(stores, callback)
        if session_id is not None:
            for attempt in range(2):
                try:
                    await self._admit_split_start(
                        session_id,
                        tenant_id=self._tenant_id,
                        execution_id=claim.execution_id,
                        expected=expected_cursor,
                        expected_session=expected_session,
                    )
                    break
                except AIError as error:
                    if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN or attempt == 1:
                        raise
                    session = await self._load_session(
                        session_id,
                        tenant_id=self._tenant_id,
                    )
                    if (
                        session is not None
                        and session.active_execution_id == claim.execution_id
                        and session.continuation == expected_cursor
                    ):
                        break
        return await self._execution.claim_start(claim)

    async def commit_agent_attempt_checkpoint(
        self,
        claim: AgentAttemptClaim,
        *,
        recovery_operation: OperationLedgerInput | None = None,
        source_agent_run_id: str | None = None,
        completed_tool_call_ids: frozenset[str] = frozenset(),
        turn_fence: SourceTurnWriteFence | None = None,
        _graph_launch: "_GraphLaunch | None" = None,
    ) -> Observed[ExecutionRecord]:
        return await self._commit_agent_attempt_checkpoint(
            claim, recovery_operation=recovery_operation, source_agent_run_id=source_agent_run_id,
            completed_tool_call_ids=completed_tool_call_ids, turn_fence=turn_fence, _graph_launch=_graph_launch,
        )

    async def commit_prepared_agent_attempt_checkpoint(
        self, claim: AgentAttemptClaim, input_context: RuntimePayloadRef, *,
        context_write: tuple[ContentArchive, PreparedArchive] | None = None,
        observed: Observed[ExecutionRecord] | None = None,
        turn_fence: SourceTurnWriteFence | None = None,
        _graph_launch: "_GraphLaunch | None" = None,
    ) -> Observed[ExecutionRecord]:
        return await self._commit_agent_attempt_checkpoint(
            claim, prepared_input_context=input_context, context_write=context_write, observed=observed,
            turn_fence=turn_fence, _graph_launch=_graph_launch,
        )

    async def _commit_agent_attempt_checkpoint(
        self, claim: AgentAttemptClaim, *,
        recovery_operation: OperationLedgerInput | None = None,
        source_agent_run_id: str | None = None,
        completed_tool_call_ids: frozenset[str] = frozenset(),
        prepared_input_context: RuntimePayloadRef | None = None,
        context_write: tuple[ContentArchive, PreparedArchive] | None = None,
        observed: Observed[ExecutionRecord] | None = None,
        producer_claim_id: str | None = None,
        turn_fence: SourceTurnWriteFence | None = None,
        _graph_launch: "_GraphLaunch | None" = None,
    ) -> Observed[ExecutionRecord]:
        """Atomically publish the execution sequence and active recovery run."""
        stores = [self._execution.state_store]
        if turn_fence is not None:
            stores.append(turn_fence.store)
        if _graph_launch is not None:
            if turn_fence is not None or recovery_operation is not None:
                raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
            stores.extend(self._canonical_graph_launch_stores(claim.execution_id, _graph_launch))
        operations = self._operations
        if recovery_operation is not None:
            if operations is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            stores.append(operations.state_store)
        stores = list(_dedupe_stores(stores))
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        next_sequence = claim.expected_agent_run_seq + 1
        producer_claim_id = producer_claim_id or (
            recovery_operation.result_ref if recovery_operation is not None else uuid4().hex
        )
        producer_generation = claim.expected_execution_revision + 1
        next_agent_run_id = agent_run_id(
            namespace=self._namespace,
            tenant_id=self._tenant_id,
            execution_id=claim.execution_id,
            agent_run_seq=next_sequence,
        )

        async def callback(
            group: StateGroupTransaction,
        ) -> Observed[ExecutionRecord]:
            if turn_fence is not None:
                await turn_fence.guard(claim.execution_id)
            execution_transaction = group.transaction(self._execution.state_store)
            existing_operation = None
            if recovery_operation is not None:
                if operations is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                existing_operation = await operations.get_in_transaction(
                    group.transaction(operations.state_store),
                    recovery_operation.operation_id, tenant_id=self._tenant_id,
                )
                if existing_operation is not None:
                    if (
                        existing_operation.execution_id != claim.execution_id
                        or not operation_identity_matches(existing_operation, recovery_operation)
                    ):
                        raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                else:
                    validate_recovery_operation(
                        recovery_operation, execution_id=claim.execution_id,
                        tenant_id=self._tenant_id, producer=True,
                    )
            current_observed = observed
            if (current_observed is None or recovery_operation is not None
                    or current_observed.value.execution_id != claim.execution_id
                    or current_observed.value.revision != claim.expected_execution_revision
                    or current_observed.value.agent_run_seq != claim.expected_agent_run_seq
                    or type(self._execution) is not ExecutionRepositoryImpl):
                current_observed = await self._execution.get_observed_in_transaction(
                    execution_transaction, claim.execution_id, tenant_id=self._tenant_id,
                )
            if current_observed is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            current_execution = current_observed.value
            current_recovery = current_execution
            if current_recovery.recovery is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            if existing_operation is not None:
                return current_observed
            if (
                current_execution.revision != claim.expected_execution_revision
                or current_execution.agent_run_seq
                != claim.expected_agent_run_seq
                or current_recovery.recovery_state is not claim.expected_recovery_state
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if current_recovery.recovery_state not in {
                ExecutionRecoveryState.ADMITTED, ExecutionRecoveryState.ACTIVE,
            }:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if current_execution.status not in (
                {ExecutionStatus.STARTED, ExecutionStatus.RECOVERY_REQUIRED}
                if current_recovery.recovery_state is ExecutionRecoveryState.ACTIVE
                else {ExecutionStatus.STARTED}
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if source_agent_run_id not in {
                current_recovery.recovery.agent_run_id, current_recovery.recovery.source_agent_run_id,
            }:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if context_write is not None:
                content_archive, archive_records = context_write
                await content_archive.store.mutate(lambda tx: content_archive.publish(tx, archive_records))
            pending_tools = current_recovery.recovery.pending_tools
            if pending_tools is not None:
                approvals = tuple(call for call in pending_tools.approvals
                                  if call.tool_call_id not in completed_tool_call_ids)
                calls = tuple(call for call in pending_tools.calls
                              if call.tool_call_id not in completed_tool_call_ids)
                pending_tools = (replace(pending_tools, approvals=approvals, calls=calls)
                                 if approvals or calls else None)
            updated_recovery = replace(
                current_recovery.recovery,
                agent_run_id=next_agent_run_id,
                source_agent_run_id=source_agent_run_id,
                pending_tools=pending_tools,
            )
            confirmed = None
            if current_recovery.recovery_state is ExecutionRecoveryState.ACTIVE:
                updated_execution = await self._execution.transition_recovery_in_transaction(
                    execution_transaction, current_execution,
                    next_status=ExecutionStatus.STARTED,
                    event_type=ExecutionEventType.EXECUTION_RESUMED,
                    event_payload={"producer_claim_id": producer_claim_id},
                    next_error_code=None, next_safe_error_details={},
                    agent_run_seq=next_sequence,
                    recovery=updated_recovery,
                )
            elif prepared_input_context is not None:
                if type(self._execution) is not ExecutionRepositoryImpl:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                confirmed = await self._execution.claim_prepared_agent_run_in_transaction(
                    execution_transaction, claim.execution_id, tenant_id=self._tenant_id,
                    expected_revision=claim.expected_execution_revision,
                    expected_agent_run_seq=claim.expected_agent_run_seq,
                    input_context=prepared_input_context, producer_claim_id=producer_claim_id,
                    recovery=updated_recovery, observed=current_observed,
                )
                updated_execution = confirmed.value
            else:
                updated_execution = await self._execution.claim_next_agent_run_in_transaction(
                    execution_transaction,
                    claim.execution_id,
                    tenant_id=self._tenant_id,
                    expected_revision=claim.expected_execution_revision,
                    expected_agent_run_seq=claim.expected_agent_run_seq,
                    producer_claim_id=producer_claim_id,
                    recovery=updated_recovery,
                )
            if recovery_operation is not None:
                if operations is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                await operations.append_in_transaction(
                    group.transaction(operations.state_store), recovery_operation,
                )
            if confirmed is None:
                confirmed = await self._execution.get_observed_in_transaction(
                    execution_transaction, claim.execution_id, tenant_id=self._tenant_id,
                )
            if confirmed is None or confirmed.value != updated_execution:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if _graph_launch is not None:
                # Match Graph admission's lock order: Execution -> Session -> Task.
                await self._guard_canonical_graph_launch(group, claim.execution_id, _graph_launch)
            return confirmed

        if recovery_operation is not None:
            async def readback() -> CommitObservation[Observed[ExecutionRecord]]:
                if operations is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                existing = await operations.get(
                    recovery_operation.operation_id, tenant_id=self._tenant_id,
                )
                if existing is None:
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                if (
                    existing.execution_id != claim.execution_id
                    or not operation_identity_matches(existing, recovery_operation)
                ):
                    return CommitObservation(
                        DurableCommitState.NOT_COMMITTED,
                        error=AIError(ErrorCode.IDEMPOTENCY_CONFLICT),
                    )
                current_observed = await self._execution.get_observed(
                    claim.execution_id, tenant_id=self._tenant_id,
                )
                if current_observed is None:
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                    )
                return CommitObservation(
                    DurableCommitState.COMMITTED,
                    value=current_observed,
                )

            outcome = await run_durable_commit(
                lambda: stores[0].storage_group.mutate(stores, callback),
                readback,
                background_tasks=self._background_tasks,
            )
            if outcome.state is DurableCommitState.COMMITTED:
                if outcome.value is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if outcome.cancelled:
                    raise asyncio.CancelledError
                return outcome.value
            if outcome.state is DurableCommitState.NOT_COMMITTED:
                if outcome.error is not None:
                    raise outcome.error
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if outcome.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from outcome.error
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from outcome.error

        _logger.debug(
            "agent attempt checkpoint requested: execution=%s sequence=%s",
            claim.execution_id,
            next_sequence,
        )
        try:
            return await stores[0].storage_group.mutate(stores, callback)
        except AIError as error:
            if _graph_launch is not None:
                # An unconfirmed grouped producer claim is not dispatch authority.
                raise
            if error.code is ErrorCode.STORAGE_CONFLICT and observed is not None:
                return await self._commit_agent_attempt_checkpoint(
                    claim, prepared_input_context=prepared_input_context, context_write=context_write,
                    producer_claim_id=producer_claim_id, turn_fence=turn_fence,
                )
            if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise
            observed = await self._execution.get_observed(
                claim.execution_id,
                tenant_id=self._tenant_id,
            )
            execution = None if observed is None else observed.value
            recovery = execution
            own_claim = (
                execution is not None
                and execution.producer_generation == producer_generation
                and execution.producer_claim_id == producer_claim_id
            )
            execution_target = (
                execution is not None
                and execution.revision >= claim.expected_execution_revision + (2 if prepared_input_context is not None else 1)
                and (prepared_input_context is None or execution.input_context == prepared_input_context)
                and execution.agent_run_seq == next_sequence
            )
            recovery_target = (
                recovery is not None
                and recovery.recovery is not None
                and recovery.recovery_state is ExecutionRecoveryState.ACTIVE
                and recovery.recovery.agent_run_id == next_agent_run_id
                and recovery.recovery.source_agent_run_id == source_agent_run_id
            )
            execution_predecessor = (
                execution is not None
                and execution.revision == claim.expected_execution_revision
                and execution.agent_run_seq == claim.expected_agent_run_seq
            )
            recovery_predecessor = (
                recovery is not None
                and recovery.recovery is not None
                and recovery.recovery_state is claim.expected_recovery_state
            )
            if own_claim and execution_target and recovery_target:
                _logger.warning(
                    "agent attempt commit outcome reconciled: execution=%s sequence=%s",
                    claim.execution_id,
                    next_sequence,
                )
                return observed
            if not (execution_predecessor and recovery_predecessor):
                if not own_claim:
                    raise AIError(ErrorCode.STORAGE_CONFLICT, retryable=False) from error
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            return await stores[0].storage_group.mutate(stores, callback)

    async def commit_native_subagent_start(
        self, claim: ExecutionStartClaim, *, turns: "SqlSessionTurns", key: "SessionTurnKey", generation: int,
        tools: ToolRepositoryImpl, operation_id: str, owner: str, fence: int, invocation_id: str,
        principal: Principal,
    ) -> ExecutionRecord:
        """Fence the child start CAS with the same native owner that reserved it."""
        if (tools is not self._tools or claim.scope != "execution.subagent"
                or claim.idempotency_key != canonical_sha256({
                    "kind": "subagent", "parent_execution_id": key.turn_id, "invocation_id": invocation_id,
                })):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        repository = self._execution

        async def start(transaction: StateTransaction) -> ExecutionRecord:
            child = await repository.claim_start_in_transaction(transaction, claim)
            if (child.lineage_kind is not ExecutionLineageKind.SUBAGENT or child.session_id is not None
                    or (child.parent_execution_id, child.root_execution_id, child.parent_invocation_id,
                        child.principal_id, child.principal_kind)
                    != (key.turn_id, key.turn_id, invocation_id, principal.principal_id, principal.kind)):
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            return child

        return await turns.with_claimed_subagent_parent(
            key, generation=generation, tools=tools, operation_id=operation_id, owner=owner, fence=fence,
            invocation_id=invocation_id, child_store=repository.state_store, principal=principal, write=start,
        )

    async def commit_recovered_native_subagent_start(
        self, turns: "SqlSessionTurns", key: "SessionTurnKey", *, generation: int, principal: Principal,
        original: ToolOperationRecord, claim: ExecutionStartClaim,
    ) -> ExecutionRecord:
        """First-start an existing child only under its successor's retained exact claim."""
        tools = self._tools
        repository = self._execution
        store = turns.transcript_state_store
        if type(tools) is not ToolRepositoryImpl or type(repository) is not ExecutionRepositoryImpl or store is None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        if (principal.tenant_id != self._tenant_id or original.execution_id != key.turn_id
                or original.agent_run_id != key.turn_id or original.tool_name != "delegate_task"
                or original.replay_safe or original.owner is None or original.fence < 1
                or original.status not in {ToolOperationStatus.CLAIMED, ToolOperationStatus.EFFECT_UNKNOWN}
                or original.producer_generation is None or original.producer_generation >= generation
                or claim.scope != "execution.subagent" or claim.idempotency_key != canonical_sha256({
                    "kind": "subagent", "parent_execution_id": key.turn_id, "invocation_id": original.tool_call_id,
                })):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        members = _dedupe_stores((store, tools.state_store, repository.state_store))
        if not _same_group(members):
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)

        async def start(group: StateGroupTransaction) -> ExecutionRecord:
            await turns.guard_step_in_transaction(
                group.transaction(store), key, generation=generation, principal=principal, _ordinary_only=True,
            )
            await tools.guard_recovered_subagent_start_in_transaction(group.transaction(tools.state_store), original)
            return await repository.claim_existing_native_subagent_in_transaction(
                group.transaction(repository.state_store), claim, parent_id=key.turn_id,
                invocation_id=original.tool_call_id, principal=principal,
            )

        # Unlike generic recovery, every exception (including an unknown ACK)
        # leaves the caller without launch authority. Never issue a second CAS.
        return await store.storage_group.mutate(members, start)

    async def settle_native_subagent_result(
        self, turns: "SqlSessionTurns", key: "SessionTurnKey", *, generation: int, principal: Principal,
        original: ToolOperationRecord, child: ExecutionRecord, result_payload: StoredPayload | None,
        error_code: str | None, error_payload: StoredPayload | None,
    ) -> ToolOperationRecord:
        """Settle the original native tool receipt from exact durable child evidence."""
        tools = self._tools
        execution_store = self._execution.state_store
        store = turns.transcript_state_store
        if tools is None or store is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if (principal.tenant_id != self._tenant_id or original.execution_id != key.turn_id
                or original.agent_run_id != key.turn_id or original.tool_name != "delegate_task"
                or original.status not in {ToolOperationStatus.CLAIMED, ToolOperationStatus.EFFECT_UNKNOWN}
                or original.owner is None or original.producer_generation is None
                or original.producer_generation >= generation):
            raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
        members = _dedupe_stores((store, tools.state_store, execution_store))
        if not _same_group(members):
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)

        async def settle(group: StateGroupTransaction) -> ToolOperationRecord:
            await turns.guard_step_in_transaction(
                group.transaction(store), key, generation=generation, principal=principal,
            )
            verified = await self._execution.require_native_subagent_terminal(
                group.transaction(execution_store), parent_id=key.turn_id,
                invocation_id=original.tool_call_id, principal=principal, expected=child,
            )
            terminal_at = verified.completed_at
            if terminal_at is None or (verified.status is ExecutionStatus.SUCCEEDED) != (result_payload is not None):
                raise AIError(ErrorCode.TOOL_RESULT_CONFLICT)
            return await tools.settle_recovered_subagent_in_transaction(
                group.transaction(tools.state_store), original, result_payload=result_payload,
                error_code=error_code, error_payload=error_payload, terminal_at=terminal_at,
            )

        return await store.storage_group.mutate(members, settle)

    async def commit_start_attempt_checkpoint(
        self,
        claim: ExecutionStartClaim,
        *,
        session_id: str | None = None,
        expected_cursor: ConversationCursor | None = None,
        expected_session: SessionRecord | None = None,
        admitted_turn: Callable[[SessionTurnRef], None] | None = None,
        _graph_launch: "_GraphLaunch | None" = None,
    ) -> ExecutionRecord:
        """Commit the initial execution start and optional recovery admission."""
        if _graph_launch is not None:
            if session_id is not None:
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            repository = self._execution
            store = repository.state_store
            members = _dedupe_stores((store, *self._canonical_graph_launch_stores(claim.execution_id, _graph_launch)))

            async def start(group: StateGroupTransaction) -> ExecutionRecord:
                started = await repository.claim_start_in_transaction(group.transaction(store), claim)
                await self._guard_canonical_graph_launch(group, claim.execution_id, _graph_launch)
                return started

            return await store.storage_group.mutate(members, start)
        return await self.commit_start_checkpoint(
            claim,
            session_id=session_id,
            expected_cursor=expected_cursor,
            expected_session=expected_session, admitted_turn=admitted_turn,
        )

    async def commit_tool_terminal(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        observed: Observed[ToolOperationRecord] | None = None,
        result_payload: StoredPayload | None = None,
        error_code: str | None = None,
        error_payload: StoredPayload | None = None,
    ) -> tuple[Observed[ToolOperationRecord], bool]:
        tools = self._tools
        if tools is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if result_payload is not None:
            if error_code is not None or error_payload is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        elif error_code is None or error_payload is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        else:
            decode_tool_operation_failure(error_code, error_payload)
        expected_status = (
            ToolOperationStatus.COMPLETED
            if result_payload is not None
            else ToolOperationStatus.FAILED
        )
        terminal_error_code: str | None = None
        terminal_error_payload: StoredPayload | None = None
        if result_payload is None:
            terminal_error_code = cast(str, error_code)
            terminal_error_payload = cast(StoredPayload, error_payload)
        stores = [tools.state_store]
        if self._observe_cancellation:
            stores.append(self._execution.state_store)
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        cancelled = False
        cancellation_requested = False

        async def callback(group: StateGroupTransaction) -> Observed[ToolOperationRecord]:
            nonlocal cancellation_requested
            transaction = group.transaction(tools.state_store)
            if result_payload is not None:
                record = await tools.complete_in_transaction(
                    transaction,
                    tool_operation_id,
                    tenant_id=tenant_id,
                    owner=owner,
                    fence=fence,
                    observed=observed,
                    result_payload=result_payload,
                )
            else:
                record = await tools.fail_in_transaction(
                    transaction,
                    tool_operation_id,
                    tenant_id=tenant_id,
                    owner=owner,
                    fence=fence,
                    observed=observed,
                    error_code=terminal_error_code,
                    error_payload=terminal_error_payload,
                )
            if self._observe_cancellation:
                execution = await self._execution.get_in_transaction(
                    group.transaction(self._execution.state_store), record.value.execution_id, tenant_id=tenant_id,
                )
                cancellation_requested = execution is None or execution.status in {
                    ExecutionStatus.CANCELLING, ExecutionStatus.CANCELLED,
                }
            return record

        async def readback() -> CommitObservation[Observed[ToolOperationRecord]]:
            nonlocal cancellation_requested
            receipt = await tools.observe_operation(
                tool_operation_id,
                tenant_id=tenant_id,
            )
            observed = None if receipt is None else receipt.value
            if observed is None:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if observed.status is expected_status:
                if observed.owner != owner or observed.fence != fence:
                    return CommitObservation(
                        DurableCommitState.NOT_COMMITTED,
                        error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT),
                    )
                if result_payload is not None:
                    if observed.result_payload != result_payload:
                        return CommitObservation(
                            DurableCommitState.NOT_COMMITTED,
                            error=AIError(ErrorCode.TOOL_RESULT_CONFLICT),
                        )
                elif (
                    observed.error_code != terminal_error_code
                    or observed.error_payload != terminal_error_payload
                ):
                    return CommitObservation(
                        DurableCommitState.NOT_COMMITTED,
                        error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT),
                    )
                if self._observe_cancellation:
                    execution = await self._load_execution(observed.execution_id, tenant_id=tenant_id)
                    cancellation_requested = execution is None or execution.status in {
                        ExecutionStatus.CANCELLING, ExecutionStatus.CANCELLED,
                    }
                return CommitObservation(
                    DurableCommitState.COMMITTED,
                    value=receipt,
                )
            if (
                observed.status in {ToolOperationStatus.CLAIMED, ToolOperationStatus.EFFECT_UNKNOWN}
                and observed.owner == owner
                and observed.fence == fence
            ):
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if observed.status is ToolOperationStatus.EFFECT_UNKNOWN:
                return CommitObservation(
                    DurableCommitState.NOT_COMMITTED,
                    error=AIError(ErrorCode.TOOL_EFFECT_UNKNOWN),
                )
            return CommitObservation(
                DurableCommitState.NOT_COMMITTED,
                error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT),
            )

        while True:
            result = await run_durable_commit(
                lambda: stores[0].storage_group.mutate(stores, callback),
                readback,
                background_tasks=self._background_tasks,
            )
            cancelled = cancelled or result.cancelled
            if result.state is DurableCommitState.COMMITTED:
                if result.value is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if cancelled:
                    raise asyncio.CancelledError
                return result.value, cancellation_requested
            if result.state is DurableCommitState.NOT_COMMITTED:
                if (
                    isinstance(result.error, AIError)
                    and result.error.code is ErrorCode.STORAGE_CONFLICT
                ):
                    observed = None
                    await asyncio.sleep(0)
                    continue
                if result.error is not None:
                    raise result.error
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if result.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from result.error
            if isinstance(result.error, AIError) and result.error.code in {
                ErrorCode.TOOL_OPERATION_CONFLICT,
                ErrorCode.TOOL_RESULT_CONFLICT,
                ErrorCode.TOOL_EFFECT_UNKNOWN,
            }:
                raise result.error
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from result.error

    async def commit_tool_deferred(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
    ) -> ToolOperationRecord:
        tools = self._tools
        if tools is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        stores = [tools.state_store]

        async def callback(group: StateGroupTransaction) -> ToolOperationRecord:
            return await tools.defer_in_transaction(
                group.transaction(tools.state_store),
                tool_operation_id,
                tenant_id=tenant_id,
                owner=owner,
                fence=fence,
            )

        async def readback() -> CommitObservation[ToolOperationRecord]:
            observed = await tools.get_operation(
                tool_operation_id,
                tenant_id=tenant_id,
            )
            if observed is None:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if (
                observed.status is ToolOperationStatus.PENDING
                and observed.owner is None
                and observed.fence == fence
            ):
                return CommitObservation(
                    DurableCommitState.COMMITTED,
                    value=observed,
                )
            if (
                observed.status is ToolOperationStatus.CLAIMED
                and observed.owner == owner
                and observed.fence == fence
            ):
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            return CommitObservation(
                DurableCommitState.NOT_COMMITTED,
                error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT),
            )

        result = await run_durable_commit(
            lambda: stores[0].storage_group.mutate(stores, callback),
            readback,
            background_tasks=self._background_tasks,
        )
        if result.state is DurableCommitState.COMMITTED and result.value is not None:
            if result.cancelled:
                raise asyncio.CancelledError
            return result.value
        if result.error is not None:
            raise result.error
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

    async def commit_external_tool_result(
        self, expected: ToolOperationRecord, supplied: ExternalCallRecord, *,
        source_agent_run_id: str,
        result_payload: StoredPayload | None = None, error_code: str | None = None,
        error_payload: StoredPayload | None = None,
    ) -> ToolOperationRecord:
        tools = self._tools
        if tools is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        external = self._checkpoint.external_calls
        stores = [tools.state_store, external.state_store]
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if (supplied.execution_id != expected.execution_id or supplied.call_id != deferred_resource_id(
                "external-call-v1", self._tenant_id, expected.execution_id,
                source_agent_run_id, expected.tool_call_id)):
            raise AIError(ErrorCode.EXTERNAL_RESULT_CONFLICT)
        async def mutate(group: StateGroupTransaction) -> ToolOperationRecord:
            source = await external.get_in_transaction(
                group.transaction(external.state_store), supplied.call_id, tenant_id=self._tenant_id,
            )
            if source != supplied or source.status is not ExternalCallStatus.SUPPLIED:
                raise AIError(ErrorCode.EXTERNAL_RESULT_CONFLICT)
            return await tools.settle_deferred_result_in_transaction(
                group.transaction(tools.state_store), expected, result_payload=result_payload,
                error_code=error_code, error_payload=error_payload,
            )
        while True:
            try:
                return await stores[0].storage_group.mutate(stores, mutate)
            except AIError as error:
                if error.code not in {ErrorCode.STORAGE_CONFLICT, ErrorCode.STORAGE_COMMIT_UNKNOWN}:
                    raise
                # Repeating this exact supplied result is an idempotent source-checked CAS.
                await asyncio.sleep(0)

    async def commit_denied_tool_result(
        self, expected: ToolOperationRecord, decision: ApprovalRecord, *,
        source_agent_run_id: str, error_payload: StoredPayload,
    ) -> ToolOperationRecord:
        tools = self._tools
        if tools is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        approvals = self._checkpoint.approvals
        stores = [tools.state_store, approvals.state_store]
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if (decision.execution_id != expected.execution_id or decision.approval_id != deferred_resource_id(
                "approval-v1", self._tenant_id, expected.execution_id,
                source_agent_run_id, expected.tool_call_id)):
            raise AIError(ErrorCode.APPROVAL_CONFLICT)
        async def mutate(group: StateGroupTransaction) -> ToolOperationRecord:
            source = await approvals.get_in_transaction(
                group.transaction(approvals.state_store), decision.approval_id, tenant_id=self._tenant_id,
            )
            if source != decision or source.status not in {ApprovalStatus.DENIED, ApprovalStatus.CANCELLED}:
                raise AIError(ErrorCode.APPROVAL_CONFLICT)
            return await tools.settle_deferred_result_in_transaction(
                group.transaction(tools.state_store), expected, result_payload=None,
                error_code=ErrorCode.TOOL_EXECUTION_FAILED.value, error_payload=error_payload,
            )
        while True:
            try:
                return await stores[0].storage_group.mutate(stores, mutate)
            except AIError as error:
                if error.code not in {ErrorCode.STORAGE_CONFLICT, ErrorCode.STORAGE_COMMIT_UNKNOWN}:
                    raise
                await asyncio.sleep(0)

    async def reconcile_tool_turn_takeover(
        self, tool_operation_id: str, *, execution_id: str,
        producer_generation: int, producer_claim_id: str,
    ) -> ToolOperationRecord:
        tools = self._tools
        if tools is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        stores = [self._execution.state_store, tools.state_store]
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        async def mutate(group: StateGroupTransaction) -> ToolOperationRecord:
            execution, _ = await self._execution.require_execution_producer_in_transaction(
                group.transaction(self._execution.state_store), execution_id,
                expected_producer_generation=producer_generation, allow_cancelling=False,
            )
            if execution.producer_claim_id != producer_claim_id:
                raise AIError(ErrorCode.STORAGE_CONFLICT, retryable=False)
            return await tools.reconcile_turn_claim_in_transaction(
                group.transaction(tools.state_store), tool_operation_id,
                execution_id=execution_id, producer_generation=producer_generation,
                producer_claim_id=producer_claim_id, turn_stopping=False,
            )
        while True:
            try:
                return await stores[0].storage_group.mutate(stores, mutate)
            except AIError as error:
                if error.code not in {ErrorCode.STORAGE_CONFLICT, ErrorCode.STORAGE_COMMIT_UNKNOWN}:
                    raise
                execution = await self._load_execution(execution_id, tenant_id=self._tenant_id)
                if (execution is None or execution.status is not ExecutionStatus.STARTED
                        or execution.producer_generation != producer_generation
                        or execution.producer_claim_id != producer_claim_id):
                    raise AIError(ErrorCode.STORAGE_CONFLICT, retryable=False) from error
                if error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
                    observed = await tools.get_operation(tool_operation_id, tenant_id=self._tenant_id)
                    if observed is not None and observed.status is not ToolOperationStatus.CLAIMED:
                        return observed
                await asyncio.sleep(0)


    async def reconcile_tool_turn_cancellation(
        self, tool_operation_id: str, *, execution_id: str,
        producer_generation: int, producer_claim_id: str,
    ) -> ToolOperationRecord:
        tools = self._tools
        if tools is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        stores = [self._execution.state_store, tools.state_store]
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        async def mutate(group: StateGroupTransaction) -> ToolOperationRecord:
            execution, _ = await self._execution.require_execution_producer_in_transaction(
                group.transaction(self._execution.state_store), execution_id,
                expected_producer_generation=producer_generation, allow_cancelling=True,
            )
            if (execution.status is not ExecutionStatus.CANCELLING
                    or execution.producer_claim_id != producer_claim_id):
                raise AIError(ErrorCode.STORAGE_CONFLICT, retryable=False)
            return await tools.reconcile_turn_claim_in_transaction(
                group.transaction(tools.state_store), tool_operation_id,
                execution_id=execution_id, producer_generation=producer_generation,
                producer_claim_id=producer_claim_id, turn_stopping=True,
            )
        while True:
            try:
                return await stores[0].storage_group.mutate(stores, mutate)
            except AIError as error:
                if error.code not in {ErrorCode.STORAGE_CONFLICT, ErrorCode.STORAGE_COMMIT_UNKNOWN}:
                    raise
                execution = await self._load_execution(execution_id, tenant_id=self._tenant_id)
                if (execution is None or execution.status is not ExecutionStatus.CANCELLING
                        or execution.producer_generation != producer_generation
                        or execution.producer_claim_id != producer_claim_id):
                    raise AIError(ErrorCode.STORAGE_CONFLICT, retryable=False) from error
                if error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
                    observed = await tools.get_operation(tool_operation_id, tenant_id=self._tenant_id)
                    if observed is not None and observed.status is not ToolOperationStatus.CLAIMED:
                        return observed
                await asyncio.sleep(0)

    def _tool_budget_repository(self, execution: ExecutionControlRecord | ExecutionRecord) -> BudgetRepository | None:
        if execution.budget_scope_id is None:
            return None
        if self._budgets is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                          safe_details={"reason": "run_budget_owner_missing"})
        return self._budgets

    def _tool_budget_call_id(self, operation: ToolOperationRecord) -> str:
        return canonical_sha256({
            "execution_id": operation.execution_id,
            "tool_operation_id": operation.tool_operation_id,
            "fence": operation.fence,
        })

    async def commit_tool_admission_boundary(
        self, request: ToolOperationAdmission, *, execution: Observed[ExecutionRecord],
        observed: Observed[ToolOperationRecord] | None = None,
        next_execution: ExecutionRecord | None = None,
        known_new: bool = False,
        write: Callable[[StateGroupTransaction], Awaitable[None]] | None = None,
        readback_write: Callable[[], Awaitable[bool]] | None = None,
        stores: Sequence[StateStore] = (),
        turn_fence: SourceTurnWriteFence | None = None,
        _graph_approval: bool = False,
    ) -> ToolBoundaryReceipt:
        """Fence final leaf admission and publish its prepared native event once."""
        tools = self._tools
        if tools is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        current = execution.value
        if (request.execution_id != current.execution_id
                or request.producer_generation != current.producer_generation
                or request.owner != current.producer_claim_id):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        graph_approvals = None
        if _graph_approval:
            self._require_checkpoint()
            continuation = None if current.recovery is None else current.recovery.pending_tools
            approved = () if continuation is None else tuple(
                item for item in continuation.approvals if item.tool_call_id == request.tool_call_id
            )
            if (type(tools) is not ToolRepositoryImpl or len(approved) != 1 or known_new or observed is not None
                    or not current.requires_task_invocation_capture or current.session_id is not None
                    or current.root_execution_id != current.execution_id or write is None or not stores
                    or current.recovery is None or current.recovery.agent_run_id != request.agent_run_id
                    or current.recovery.source_agent_run_id != continuation.source_agent_run_id
                    or request.recovery_agent_run_id != continuation.source_agent_run_id
                    or request.tool_operation_id != approved[0].operation_id
                    or request.tool_name != approved[0].tool_name
                    or request.arguments_digest != approved[0].arguments_payload.digest):
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            graph_approvals = self._checkpoint.approvals
            graph_approval_id = deferred_resource_id(
                "approval-v1", self._tenant_id, current.execution_id, continuation.source_agent_run_id, request.tool_call_id,
            )

        target = current if next_execution is None else next_execution
        budget = self._tool_budget_repository(current)
        participants = _dedupe_stores((self._execution.state_store, tools.state_store,
                                      *((budget.state_store,) if budget is not None else ()),
                                      *((turn_fence.store,) if turn_fence is not None else ()),
                                      *((graph_approvals.state_store,) if graph_approvals is not None else ()), *stores))
        if not _same_group(participants):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

        control_conflict = False
        retry_execution: Observed[ExecutionRecord] | None = None
        cancelled = False
        admission_id = uuid4().hex
        budget_call_id: str | None = None

        async def operation() -> ToolBoundaryReceipt:
            async def mutate(group: StateGroupTransaction) -> ToolBoundaryReceipt:
                nonlocal control_conflict, budget_call_id
                try:
                    control = await self._execution.commit_boundary_in_transaction(
                        group.transaction(self._execution.state_store), execution,
                        next_record=target, expected_producer_generation=current.producer_generation,
                        expected_producer_claim_id=request.owner,
                        **({} if turn_fence is None else {"turn_fence": turn_fence}),
                    )
                except AIError as error:
                    control_conflict = error.code is ErrorCode.STORAGE_CONFLICT
                    raise
                tool_tx = group.transaction(tools.state_store)
                expected_pending = None
                if graph_approvals is not None:
                    source = await graph_approvals.get_in_transaction(
                        group.transaction(graph_approvals.state_store), graph_approval_id, tenant_id=self._tenant_id,
                    )
                    expected_pending = await tools.get_operation_in_transaction(
                        tool_tx, request.tool_operation_id, tenant_id=self._tenant_id,
                    )
                    if (source is None or source.execution_id != request.execution_id
                            or source.approval_id != graph_approval_id or source.status is not ApprovalStatus.APPROVED
                            or expected_pending is None or expected_pending.status is not ToolOperationStatus.PENDING
                            or expected_pending.owner is not None
                            or expected_pending.agent_run_id != request.recovery_agent_run_id):
                        raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
                effect = await tools.admit_in_transaction(
                    tool_tx, request, observed=observed, known_new=known_new, expected_pending=expected_pending,
                )
                usage = None
                if budget is not None and effect.value.status is ToolOperationStatus.CLAIMED:
                    budget_call_id = self._tool_budget_call_id(effect.value)
                    usage = await budget.admit_tool_in_transaction(
                        group.transaction(budget.state_store), current.budget_scope_id, budget_call_id,
                        admission_id=admission_id,
                    )
                if write is not None:
                    await write(group)
                return ToolBoundaryReceipt(control, effect, usage)
            async def commit() -> ToolBoundaryReceipt:
                return await participants[0].storage_group.mutate(participants, mutate)
            return await commit() if budget is None else await budget.retry_admission(commit)

        async def readback() -> CommitObservation[ToolBoundaryReceipt]:
            nonlocal retry_execution
            if _graph_approval:
                # Without a confirmed same-group approval+effect receipt the
                # original side effect cannot be run or reclaimed speculatively.
                return CommitObservation(DurableCommitState.UNRESOLVED)
            control = await self._execution.get_observed(current.execution_id, tenant_id=self._tenant_id)
            if control is None:
                return _partial_integrity()
            if control_conflict:
                if (control.value == current and control.value.status is ExecutionStatus.STARTED
                        and control.storage_revision > execution.storage_revision):
                    retry_execution = control
                return CommitObservation(DurableCommitState.NOT_COMMITTED,
                                         error=AIError(ErrorCode.STORAGE_CONFLICT))
            effect = await tools.observe_operation(request.tool_operation_id, tenant_id=self._tenant_id)
            if effect is not None and not _tool_admission_matches(effect.value, request):
                return CommitObservation(DurableCommitState.NOT_COMMITTED,
                                         error=AIError(ErrorCode.IDEMPOTENCY_CONFLICT))
            if (control.value == target and control.storage_revision >= execution.storage_revision + 1
                    and effect is not None):
                usage = None
                if (budget is not None and effect.value.status is ToolOperationStatus.CLAIMED
                        and budget_call_id is None):
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                if budget is not None and budget_call_id is not None:
                    if self._tool_budget_call_id(effect.value) != budget_call_id:
                        return CommitObservation(DurableCommitState.NOT_COMMITTED,
                                                 error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT))
                    usage = await budget.read_tool_admission(
                        current.budget_scope_id, budget_call_id, admission_id=admission_id,
                    )
                    if usage is None:
                        return CommitObservation(DurableCommitState.NOT_COMMITTED)
                if write is not None and (readback_write is None or not await readback_write()):
                    return CommitObservation(DurableCommitState.UNRESOLVED,
                                             error=AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN))
                return CommitObservation(DurableCommitState.COMMITTED, value=ToolBoundaryReceipt(control, effect, usage))
            if control.value == current and control.storage_revision == execution.storage_revision:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            return CommitObservation(DurableCommitState.NOT_COMMITTED,
                                     error=AIError(ErrorCode.STORAGE_CONFLICT))

        while True:
            control_conflict = False
            retry_execution = None
            result = await run_durable_commit(operation, readback, background_tasks=self._background_tasks)
            cancelled = cancelled or result.cancelled
            if result.state is DurableCommitState.COMMITTED and result.value is not None:
                if cancelled:
                    raise asyncio.CancelledError
                return result.value
            if (result.state is DurableCommitState.NOT_COMMITTED and retry_execution is not None
                    and isinstance(result.error, AIError) and result.error.code is ErrorCode.STORAGE_CONFLICT):
                if cancelled:
                    raise asyncio.CancelledError
                execution = retry_execution
                continue
            if result.error is not None:
                raise result.error
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

    async def commit_tool_terminal_boundary(
        self, tool_operation_id: str, *, tenant_id: str, owner: str, fence: int,
        execution: Observed[ExecutionRecord], observed: Observed[ToolOperationRecord] | None = None,
        result_payload: StoredPayload | None = None, error_code: str | None = None,
        error_payload: StoredPayload | None = None,
        error_source: NativeToolErrorReference | None = None,
        write: Callable[[StateGroupTransaction], Awaitable[None]] | None = None,
        readback_write: Callable[[], Awaitable[bool]] | None = None,
        stores: Sequence[StateStore] = (),
    ) -> ToolBoundaryReceipt:
        """Settle a permitted certain effect independently of the live producer."""
        tools = self._tools
        if tools is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if (result_payload is None) == (error_code is None or error_payload is None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if result_payload is not None and (error_code is not None or error_payload is not None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if error_source is not None and (result_payload is not None or write is None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if result_payload is None:
            decode_tool_operation_failure(error_code, error_payload)
        if observed is not None and observed.value.execution_id != execution.value.execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        participants = _dedupe_stores((tools.state_store, *stores))
        if not _same_group(participants):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        write_error: BaseException | None = None
        retry_observed: Observed[ToolOperationRecord] | None = None
        cancelled = False

        async def settle(transaction: StateTransaction) -> Observed[ToolOperationRecord]:
            if result_payload is not None:
                return await tools.complete_in_transaction(
                    transaction, tool_operation_id, tenant_id=tenant_id, owner=owner, fence=fence,
                    observed=observed, result_payload=result_payload,
                )
            return await tools.fail_in_transaction(
                transaction, tool_operation_id, tenant_id=tenant_id, owner=owner, fence=fence,
                observed=observed, error_code=error_code, error_payload=error_payload, error_source=error_source,
            )

        async def operation() -> ToolBoundaryReceipt:
            async def mutate(group: StateGroupTransaction) -> ToolBoundaryReceipt:
                nonlocal write_error
                if write is not None and error_source is not None:
                    try:
                        await write(group)
                    except BaseException as error:
                        write_error = error
                        raise
                effect = await settle(group.transaction(tools.state_store))
                if effect.value.execution_id != execution.value.execution_id:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if write is not None and error_source is None:
                    try:
                        await write(group)
                    except BaseException as error:
                        write_error = error
                        raise
                return ToolBoundaryReceipt(execution, effect)
            return await participants[0].storage_group.mutate(participants, mutate)

        async def readback() -> CommitObservation[ToolBoundaryReceipt]:
            nonlocal retry_observed
            effect = await tools.observe_operation(tool_operation_id, tenant_id=tenant_id)
            if effect is None:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            value = effect.value
            if value.owner != owner or value.fence != fence or value.execution_id != execution.value.execution_id:
                return CommitObservation(DurableCommitState.NOT_COMMITTED,
                                         error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT))
            expected_status = ToolOperationStatus.COMPLETED if result_payload is not None else ToolOperationStatus.FAILED
            if value.status is expected_status:
                if (value.result_payload != result_payload or value.error_code != error_code
                        or value.error_payload != error_payload):
                    return CommitObservation(DurableCommitState.NOT_COMMITTED,
                                             error=AIError(ErrorCode.TOOL_RESULT_CONFLICT))
                if write is not None and (readback_write is None or not await readback_write()):
                    return CommitObservation(DurableCommitState.UNRESOLVED,
                                             error=AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN))
                return CommitObservation(DurableCommitState.COMMITTED, value=ToolBoundaryReceipt(execution, effect))
            if value.status in {ToolOperationStatus.CLAIMED, ToolOperationStatus.EFFECT_UNKNOWN}:
                retry_observed = effect
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            return CommitObservation(DurableCommitState.NOT_COMMITTED,
                                     error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT))

        while True:
            result = await run_durable_commit(operation, readback, background_tasks=self._background_tasks)
            cancelled = cancelled or result.cancelled
            if result.state is DurableCommitState.COMMITTED and result.value is not None:
                if cancelled:
                    raise asyncio.CancelledError
                return result.value
            if write_error is not None:
                # Failure to publish an SDK event cannot erase a permitted external outcome.
                # This is a new short transaction after the failed group has rolled back.
                await self.commit_tool_terminal(
                    tool_operation_id, tenant_id=tenant_id, owner=owner, fence=fence,
                    observed=None, result_payload=result_payload, error_code=error_code, error_payload=error_payload,
                )
                raise write_error
            if (result.state is DurableCommitState.NOT_COMMITTED
                    and isinstance(result.error, AIError) and result.error.code is ErrorCode.STORAGE_CONFLICT
                    and retry_observed is not None):
                observed = retry_observed
                retry_observed = None
                continue
            if result.error is not None:
                raise result.error
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

    async def commit_tool_admission(
        self,
        request: ToolOperationAdmission,
        *, observed: Observed[ToolOperationRecord] | None = None,
    ) -> Observed[ToolOperationRecord]:
        admission_id = uuid4().hex
        budget_scope_id: str | None = None
        budget_call_id: str | None = None

        async def attempt() -> Observed[ToolOperationRecord]:
            if self._tools is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            _logger.debug(
                "tool admission checkpoint requested: operation=%s",
                request.tool_operation_id,
            )
            stores = [self._tools.state_store]
            if request.producer_generation is not None:
                stores.append(self._execution.state_store)
                if self._budgets is not None:
                    stores.append(self._budgets.state_store)
                if not _same_group(stores):
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

            async def callback(group: StateGroupTransaction) -> Observed[ToolOperationRecord]:
                nonlocal budget_scope_id, budget_call_id
                execution = None
                if request.producer_generation is not None:
                    execution_transaction = group.transaction(self._execution.state_store)
                    execution, _ = await self._execution.require_execution_producer_in_transaction(
                        execution_transaction,
                        request.execution_id,
                        expected_producer_generation=request.producer_generation,
                        allow_cancelling=False,
                    )
                    if execution.producer_claim_id != request.owner:
                        raise AIError(ErrorCode.STORAGE_CONFLICT, retryable=False)
                operation = await self._tools.admit_in_transaction(
                    group.transaction(self._tools.state_store),
                    request, observed=observed,
                )
                budget = None if execution is None else self._tool_budget_repository(execution)
                if budget is not None and operation.value.status is ToolOperationStatus.CLAIMED:
                    budget_scope_id = execution.budget_scope_id
                    budget_call_id = self._tool_budget_call_id(operation.value)
                    await budget.admit_tool_in_transaction(
                        group.transaction(budget.state_store), budget_scope_id, budget_call_id,
                        admission_id=admission_id,
                    )
                return operation

            async def commit() -> Observed[ToolOperationRecord]:
                return await stores[0].storage_group.mutate(stores, callback)

            try:
                return await commit() if self._budgets is None else await self._budgets.retry_admission(commit)
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN:
                    raise
                receipt = await self._tools.observe_operation(
                    request.tool_operation_id,
                    tenant_id=self._tools.tenant_id,
                )
                source = None if receipt is None else receipt.value
                if source is None:
                    return await stores[0].storage_group.mutate(stores, callback)
                if not _tool_admission_matches(source, request):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                if budget_call_id is not None:
                    if self._tool_budget_call_id(source) != budget_call_id:
                        raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT) from error
                    usage = await self._budgets.read_tool_admission(
                        budget_scope_id, budget_call_id, admission_id=admission_id,
                    )
                    if usage is None:
                        return await commit()
                if source.status is ToolOperationStatus.EFFECT_UNKNOWN:
                    raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN) from error
                if source.status is ToolOperationStatus.CANCELLED:
                    raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT) from error
                if source.status in {
                    ToolOperationStatus.CLAIMED,
                    ToolOperationStatus.COMPLETED,
                    ToolOperationStatus.FAILED,
                }:
                    return receipt
                raise AIError(ErrorCode.STORAGE_CONFLICT) from error

        while True:
            try:
                result = await attempt()
                if result.value.status is ToolOperationStatus.EFFECT_UNKNOWN:
                    raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
                return result
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                if request.producer_generation is not None:
                    execution = await self._load_execution(request.execution_id, tenant_id=self._tenant_id)
                    if (
                        execution is None
                        or execution.producer_generation != request.producer_generation
                        or execution.producer_claim_id != request.owner
                        or execution.status is not ExecutionStatus.STARTED
                    ):
                        raise AIError(ErrorCode.STORAGE_CONFLICT, retryable=False) from error
                tools = self._tools
                if tools is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY) from error
                receipt = await tools.observe_operation(
                    request.tool_operation_id,
                    tenant_id=tools.tenant_id,
                )
                source = None if receipt is None else receipt.value
                if source is None:
                    await asyncio.sleep(0)
                    continue
                if not _tool_admission_matches(source, request):
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT) from error
                if source.status is ToolOperationStatus.EFFECT_UNKNOWN:
                    raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN) from error
                if source.status is ToolOperationStatus.CANCELLED:
                    raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT) from error
                if source.status in {
                    ToolOperationStatus.COMPLETED,
                    ToolOperationStatus.FAILED,
                }:
                    return receipt
                observed = None
                await asyncio.sleep(0)

    async def commit_recovery_handoff(
        self,
        checkpoint: ExecutionRecord,
        *,
        expected_revision: int,
        producer_generation: int,
    ) -> ExecutionRecord:
        """Publish a worker's terminal handoff while its producer fence is held."""
        if (
            checkpoint.recovery_state is not ExecutionRecoveryState.HANDOFF
            or checkpoint.recovery.handoff_phase is not RecoveryHandoffPhase.PREPARED
            or checkpoint.recovery.terminal_handoff is None
            or checkpoint.revision != expected_revision + 1
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if isinstance(self._execution, ExecutionRepositoryImpl):
            await self._execution.prepare_content(checkpoint)
        stores = (self._execution.state_store,)

        async def operation() -> ExecutionRecord:
            async def mutate(group: StateGroupTransaction) -> ExecutionRecord:
                await self._execution.require_execution_producer_in_transaction(
                    group.transaction(self._execution.state_store),
                    checkpoint.execution_id,
                    expected_producer_generation=producer_generation,
                )
                return await self._execution.compare_and_swap_in_transaction(
                    group.transaction(self._execution.state_store),
                    checkpoint.execution_id,
                    tenant_id=self._tenant_id,
                    expected_revision=expected_revision,
                    next_record=checkpoint,
                )
            return await stores[0].storage_group.mutate(stores, mutate)

        async def readback() -> CommitObservation[object]:
            observed = await self._execution.get_recovery(checkpoint.execution_id, tenant_id=self._tenant_id)
            if observed == checkpoint:
                return CommitObservation(DurableCommitState.COMMITTED)
            return CommitObservation(DurableCommitState.NOT_COMMITTED)

        await self._commit_or_raise(operation, readback)
        return checkpoint

    def _pending_tool_store(self, operations: Sequence[ToolOperationRecord]) -> tuple[StateStore, ...]:
        if not operations:
            return ()
        if self._tools is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return (self._tools.state_store,)

    async def _materialize_pending_tools(
        self, execution: ExecutionRecord, operations: Sequence[ToolOperationRecord], *,
        read_only: bool, terminal: bool, transaction: StateTransaction | None = None,
    ) -> None:
        if not operations:
            return
        if self._tools is None or execution.agent_run_seq != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        expected_run = agent_run_id(namespace=self._namespace, tenant_id=self._tenant_id,
                                    execution_id=execution.execution_id, agent_run_seq=1)
        if len({value.tool_operation_id for value in operations}) != len(operations):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        batch_tools = (self._tools if transaction is not None and not read_only and terminal
                       and type(self._tools) is ToolRepositoryImpl else None)
        for value in operations:
            if (value.execution_id != execution.execution_id or value.agent_run_id != expected_run
                    or value.binding_digest != execution.binding_digest or not value.replay_safe
                    or value.producer_generation != execution.producer_generation
                    or value.owner != (None if value.status is ToolOperationStatus.PENDING else execution.producer_claim_id)
                    or isinstance(value.fence, bool) or not isinstance(value.fence, int) or value.fence < 1
                    or terminal and value.status not in {ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED}):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if batch_tools is not None:
                continue
            actual = (await self._tools.get_operation(value.tool_operation_id, tenant_id=self._tenant_id)
                      if read_only else await self._tools.reserve(value))
            if actual != value:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        if batch_tools is not None:
            assert transaction is not None
            actual = await batch_tools.reserve_many_in_transaction(transaction, operations)
            if actual != tuple(operations):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    @sql_observation_phase("persistence.terminal.receipt_lookup")
    async def read_pending_turn(
        self, reservation: ExecutionStartReservation, *, expected_session: SessionRecord | None,
        tool_operations: Sequence[ToolOperationRecord] = (),
    ) -> ExecutionTerminalCommitResult | None:
        """Confirm a terminal turn from source without retrying its write."""
        if (not reservation.execution.conversation
                or reservation.execution.session_id != (None if expected_session is None else expected_session.session_id)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        stores = [self._execution.state_store, *self._pending_tool_store(tool_operations)]
        if expected_session is not None:
            stores.append(self._conversation.state_store)
        elif not reservation.execution.durable_trace:
            if self._conversation_run_store is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            stores.extend(self._conversation_run_store.transaction_stores)
        if reservation.execution.durable_trace:
            if self._execution_run_store is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            stores.extend(self._execution_run_store.transaction_stores)
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

        async def read(group: StateGroupTransaction) -> ExecutionTerminalCommitResult | None:
            return await self._read_pending_turn_in_group(
                group, reservation, expected_session, tool_operations=tool_operations,
            )

        return await stores[0].storage_group.read_many(stores, read)

    async def _read_pending_turn_in_group(
        self, group: StateGroupTransaction, reservation: ExecutionStartReservation,
        expected_session: SessionRecord | None, *, tool_operations: Sequence[ToolOperationRecord] = (),
    ) -> ExecutionTerminalCommitResult | None:
        result = await self._execution.pending_turn_result_in_transaction(
            group.transaction(self._execution.state_store), reservation,
        )
        if result is None:
            return None
        execution = result.execution
        if (not execution.conversation or not execution.retention_closed
                or execution.session_id != reservation.execution.session_id):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        recovery = execution.recovery
        expected_run_id = (None if execution.agent_run_seq == 0 else agent_run_id(
            namespace=self._namespace, tenant_id=self._tenant_id,
            execution_id=execution.execution_id, agent_run_seq=execution.agent_run_seq,
        ))
        if (recovery is None or execution.recovery_state is not ExecutionRecoveryState.COMPLETED
                or recovery.agent_run_id != expected_run_id):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution.durable_trace:
            seal = await self._execution.get_history_seal(execution.execution_id, tenant_id=self._tenant_id)
            run_ids = () if expected_run_id is None else (expected_run_id,)
            heads = await self._execution_run_store.execution_history_heads(run_ids)
            if (seal is None or execution.history_seal_digest != seal.seal_digest
                    or seal.execution_event_high_water != execution.event_seq):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            empty_failure = not seal.run_heads and execution.status in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}
            if not empty_failure and seal.run_heads != tuple(heads[run_id] for run_id in run_ids):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for run_id in run_ids:
                run = await self._execution_run_store.get_agent_run(agent_run_id=run_id)
                if (run is None) != empty_failure:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (expected_session is None and not execution.durable_trace
                and execution.status is ExecutionStatus.SUCCEEDED):
            archive = self._conversation_run_store
            if archive is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            run = await archive.get_agent_run(agent_run_id=expected_run_id)
            checkpoint = await archive.latest_checkpoint(agent_run_id=expected_run_id)
            if (run is None or checkpoint is None or checkpoint.state != "complete"
                    or checkpoint.pending_request_index is not None or checkpoint.agent_run_id != run.agent_run_id
                    or run.parent_agent_run_id is not None or run.agent_id != execution.agent_id
                    or not await archive.verify_checkpoint_projection(
                        agent_run_id=run.agent_run_id, checkpoint=checkpoint,
                    )
                    or await archive.transcript_message_count_for_run(run) != len(checkpoint.messages)):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if expected_session is not None:
            transaction = group.transaction(self._conversation.state_store)
            session = await self._conversation.get_in_transaction(
                transaction, expected_session.session_id, tenant_id=self._tenant_id,
            )
            committed = await self._conversation.pending_turn_commit_in_transaction(
                transaction, session_id=expected_session.session_id, execution_id=execution.execution_id,
            )
            successful = execution.status is ExecutionStatus.SUCCEEDED
            if (session is None or committed is None or session.history_id != expected_session.history_id
                    or (result.result.output is not None) != successful):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if successful:
                if (execution.conversation_agent_run_id is None
                        or committed.end_message_index <= committed.start_message_index):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            elif committed.start_message_index != 0 or committed.end_message_index != 0:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution.execution_id == reservation.execution.execution_id:
            await self._materialize_pending_tools(execution, tool_operations, read_only=True, terminal=True)
        return result

    async def read_pending_turn_promotion(
        self, reservation: ExecutionStartReservation, *, expected_session: SessionRecord | None,
        captured_input_context: RuntimePayloadRef | None, producer_claim_id: str,
        tool_operations: Sequence[ToolOperationRecord] = (),
    ) -> ExecutionRecord | None:
        """Confirm only the caller's first durable promotion receipt."""
        if not isinstance(producer_claim_id, str) or not producer_claim_id.strip():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        initial, identity = reservation.execution, reservation.idempotency
        if initial.session_id != (None if expected_session is None else expected_session.session_id):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        stores = [self._execution.state_store, *self._pending_tool_store(tool_operations)]
        if expected_session is not None:
            stores.append(self._conversation.state_store)
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        claim = ExecutionStartClaim(initial.execution_id, 0, 0, identity.scope,
                                    identity.idempotency_key, identity.request_digest, initial.created_at)

        async def read(group: StateGroupTransaction) -> ExecutionRecord | None:
            idempotency = await self._execution.get_start_idempotency(claim)
            if idempotency is None:
                return None
            if (idempotency.request_digest != identity.request_digest
                    or idempotency.resource_kind != identity.resource_kind):
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if (idempotency.resource_id != initial.execution_id
                    or idempotency.status is not IdempotencyStatus.STARTED):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            execution = await self._execution.get_in_transaction(
                group.transaction(self._execution.state_store), initial.execution_id, tenant_id=self._tenant_id,
            )
            recovery = None if execution is None else execution.recovery
            prepared = captured_input_context is not None
            if (execution is None or recovery is None
                    or execution.status is not ExecutionStatus.STARTED
                    or execution.session_id != initial.session_id or execution.event_seq != 1
                    or execution.revision != (3 if prepared else 1)
                    or execution.agent_run_seq != (1 if prepared else 0)
                    or execution.input_context != (captured_input_context if prepared else initial.input_context)
                    or execution.recovery_state is not (ExecutionRecoveryState.ACTIVE if prepared else ExecutionRecoveryState.ADMITTED)
                    or recovery.agent_run_id != (agent_run_id(namespace=self._namespace, tenant_id=self._tenant_id,
                                                              execution_id=initial.execution_id, agent_run_seq=1)
                                                 if prepared else None)
                    or (prepared and (execution.producer_generation != 2 or execution.producer_claim_id != producer_claim_id))
                    or (not prepared and (execution.producer_claim_id is not None or execution.producer_generation != 0))):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if expected_session is not None:
                transaction = group.transaction(self._conversation.state_store)
                session = await self._conversation.get_in_transaction(
                    transaction, expected_session.session_id, tenant_id=self._tenant_id,
                )
                if (session.active_execution_id != initial.execution_id
                        or session.continuation != expected_session.continuation):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
            await self._materialize_pending_tools(execution, tool_operations, read_only=True, terminal=False)
            return execution

        return await stores[0].storage_group.read_many(stores, read)

    async def promote_pending_turn(
        self, reservation: ExecutionStartReservation, *,
        expected_session: SessionRecord | None, captured_input_context: RuntimePayloadRef | None,
        producer_claim_id: str | None = None,
        context_write: tuple[ContentArchive, PreparedArchive] | None = None,
        tool_operations: Sequence[ToolOperationRecord] = (),
        authority: "ExecutionAuthority | None" = None,
    ) -> ExecutionRecord:
        """Confirm durable producer ownership before a local turn performs effects."""
        producer_claim_id = uuid4().hex if producer_claim_id is None else producer_claim_id
        if not isinstance(producer_claim_id, str) or not producer_claim_id.strip():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if expected_session is not None:
            self._require_conversation()
        initial = reservation.execution
        if (type(self._execution) is not ExecutionRepositoryImpl
                or (expected_session is not None and type(self._conversation) is not SessionRepositoryImpl)):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if (initial.status is not ExecutionStatus.PENDING_START or initial.revision != 0
                or initial.event_seq != 0 or initial.agent_run_seq != 0
                or initial.binding_kind != "agent" or initial.parent_execution_id is not None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if expected_session is None:
            if (initial.session_id is not None or initial.conversation_agent_run_id is not None
                    or initial.previous_execution_id is not None or initial.fork_base_execution_id is not None
                    or initial.lineage_kind is not ExecutionLineageKind.RUN
                    or initial.root_execution_id != initial.execution_id):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        elif (initial.session_id != expected_session.session_id
              or expected_session.active_execution_id is not None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        stores = [self._execution.state_store, *self._pending_tool_store(tool_operations)]
        if expected_session is not None:
            stores.append(self._conversation.state_store)
        stores = tuple(_dedupe_stores(stores))
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        await self._execution.prepare_content(initial)
        identity = reservation.idempotency
        claim = ExecutionStartClaim(initial.execution_id, 0, 0, identity.scope,
                                    identity.idempotency_key, identity.request_digest,
                                    initial.created_at)
        resolved: ExecutionRecord | None = None
        rejected = False

        async def mutate(group: StateGroupTransaction) -> ExecutionRecord:
            nonlocal resolved, rejected
            if expected_session is not None:
                session_transaction = group.transaction(self._conversation.state_store)
                session = await self._conversation.get_in_transaction(
                    session_transaction, expected_session.session_id, tenant_id=self._tenant_id,
                )
                if session != expected_session:
                    rejected = True
                    raise AIError(ErrorCode.SESSION_BUSY)
            try:
                reserved = await self._execution.reserve_start(reservation, authority=authority)
            except AIError as error:
                if error.code is ErrorCode.IDEMPOTENCY_CONFLICT:
                    rejected = True
                raise
            if not reserved.created:
                rejected = True
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            started = await self.commit_start_attempt_checkpoint(
                claim,
                session_id=None if expected_session is None else expected_session.session_id,
                expected_cursor=None if expected_session is None else expected_session.continuation,
            )
            if captured_input_context is None:
                resolved = started
            else:
                activated = await self._commit_agent_attempt_checkpoint(
                    AgentAttemptClaim(initial.execution_id, started.revision, 0,
                                      ExecutionRecoveryState.ADMITTED),
                    prepared_input_context=captured_input_context, producer_claim_id=producer_claim_id,
                    context_write=context_write,
                )
                resolved = activated.value
            confirmed_execution = resolved
            if (captured_input_context is not None
                    and (confirmed_execution.producer_claim_id != producer_claim_id
                         or confirmed_execution.producer_generation != 2)):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._materialize_pending_tools(confirmed_execution, tool_operations, read_only=False, terminal=False)
            return resolved

        async def operation() -> object:
            return await stores[0].storage_group.mutate(stores, mutate)

        async def readback() -> CommitObservation[object]:
            nonlocal resolved
            if rejected:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            try:
                observed = await self.read_pending_turn_promotion(
                    reservation, expected_session=expected_session,
                    captured_input_context=captured_input_context, producer_claim_id=producer_claim_id,
                    tool_operations=tool_operations,
                )
            except AIError as error:
                if error.code is ErrorCode.IDEMPOTENCY_CONFLICT:
                    return CommitObservation(DurableCommitState.NOT_COMMITTED, error=error)
                if error.code is ErrorCode.STORAGE_CONFLICT:
                    own_execution = await self._execution.get(initial.execution_id, tenant_id=self._tenant_id)
                    return CommitObservation(
                        DurableCommitState.NOT_COMMITTED if own_execution is None else DurableCommitState.UNRESOLVED,
                        error=error,
                    )
                raise
            if observed is None:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            resolved = observed
            return CommitObservation(DurableCommitState.COMMITTED, value=observed)

        await self._commit_or_raise(operation, readback)
        assert resolved is not None
        return resolved

    @sql_observation_phase("persistence.terminal.pending_commit")
    async def commit_pending_turn(
        self, reservation: ExecutionStartReservation, commit: ExecutionTerminalCommit, *,
        expected_session: SessionRecord | None = None, next_cursor: ConversationCursor | None = None,
        conversation_agent_run: AgentRunRecord | None = None,
        conversation_checkpoint: AgentRunCheckpoint | None = None,
        execution_projections: Sequence[PreparedExecutionProjection] = (),
        audit_events: Sequence[ExecutionEventAppend] = (),
        capability_run_id: str | None = None,
        context_write: tuple[ContentArchive, PreparedArchive] | None = None,
        tool_operations: Sequence[ToolOperationRecord] = (),
        include_publication: bool = False,
        authority: "ExecutionAuthority | None" = None,
    ) -> ExecutionTerminalCommitResult:
        """Materialize one volatile turn and its replay-safe outcomes atomically.

        The committing owner can request evidence for its own confirmed write.
        A prior winner or an uncertain commit readback is still a source result,
        not evidence that this caller published that write.
        """
        tool_operations = tuple(tool_operations)
        audit_events = tuple(audit_events)
        execution_projections = tuple(execution_projections)
        archive = self._conversation_run_store
        if type(self._execution) is not ExecutionRepositoryImpl:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        execution = commit.execution
        await self._execution.prepare_content(replace(execution, result=commit.result))
        successful = execution.status is ExecutionStatus.SUCCEEDED
        for candidate in (reservation.execution, execution):
            if (not candidate.conversation or candidate.binding_kind != "agent"
                    or candidate.parent_execution_id is not None
                    or candidate.parent_invocation_id is not None or candidate.requires_task_invocation_capture
                    or candidate.budget_scope_id is not None or candidate.restart_checkpoints
                    or candidate.dependency_hold_ids):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (execution.status not in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}
                or reservation.budget is not None
                or execution.agent_run_seq not in ({1} if successful else {0, 1})
                or reservation.execution.status is not ExecutionStatus.PENDING_START
                or reservation.execution.agent_run_seq != 0 or reservation.execution.event_seq != 0
                or commit.expected_event_seq != 1 or commit.expected_revision < (2 if successful else 1)
                or commit.operation is not None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution.durable_trace != reservation.execution.durable_trace:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not execution.durable_trace and execution_projections:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution.durable_trace:
            if self._execution_run_store is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            expected_runs = (() if execution.agent_run_seq == 0 else (agent_run_id(
                namespace=self._namespace, tenant_id=self._tenant_id,
                execution_id=execution.execution_id, agent_run_seq=1,
            ),))
            if ((successful or execution_projections)
                    and tuple(projection.run.agent_run_id for projection in execution_projections) != expected_runs):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for projection in execution_projections:
                if (projection.run.agent_id != execution.agent_id or projection.run.parent_agent_run_id is not None
                        or projection.base_event_offset != 0 or projection.base_checkpoint_offset != 0
                        or projection.base_interaction_offset != 0
                        or not projection.checkpoints and projection.durable_projection_digest != "empty"
                        or projection.producer_generation != commit.producer_generation):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        seal = (_execution_history_seal(
            commit, audit_events=audit_events, projections=execution_projections,
            current_run=None, current_events=(), current_batch=None,
        ) if execution.durable_trace else None)
        stores = [self._execution.state_store, *self._pending_tool_store(tool_operations)]
        if execution.durable_trace:
            stores.extend(self._execution_run_store.transaction_stores)
        if expected_session is None:
            if next_cursor is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for candidate in (reservation.execution, execution):
                if (candidate.session_id is not None or candidate.conversation_agent_run_id is not None
                        or candidate.previous_execution_id is not None or candidate.fork_base_execution_id is not None
                        or candidate.lineage_kind is not ExecutionLineageKind.RUN
                        or candidate.root_execution_id != candidate.execution_id):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if successful and not execution.durable_trace:
                if (conversation_agent_run is None or conversation_checkpoint is None
                        or conversation_checkpoint.state != "complete"
                        or conversation_checkpoint.pending_request_index is not None
                        or conversation_agent_run.parent_agent_run_id is not None
                        or conversation_agent_run.agent_id != execution.agent_id
                        or conversation_agent_run.agent_run_id != agent_run_id(
                            namespace=self._namespace, tenant_id=self._tenant_id,
                            execution_id=execution.execution_id, agent_run_seq=1,
                        )
                        or any(getattr(conversation_agent_run, name) != getattr(conversation_checkpoint, name)
                               for name in ("agent_run_id", "agent_conversation_id", "agent_id", "parent_agent_run_id"))):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if archive is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                stores.extend(archive.transaction_stores)
            elif conversation_agent_run is not None or conversation_checkpoint is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        else:
            if type(self._conversation) is not SessionRepositoryImpl:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            if execution.session_id != expected_session.session_id or expected_session.active_execution_id is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if successful:
                if (next_cursor is None or conversation_agent_run is None or conversation_checkpoint is None):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if (type(archive) not in {StateStepArchive, ModelInteractionStateStepArchive}
                        or self._conversation_history is None):
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                if (conversation_checkpoint.state != "complete" or conversation_checkpoint.pending_request_index is not None
                        or conversation_agent_run.parent_agent_run_id is not None
                        or any(getattr(conversation_agent_run, name) != getattr(conversation_checkpoint, name)
                               for name in ("agent_conversation_id", "agent_id", "parent_agent_run_id"))
                        or execution.conversation_agent_run_id != conversation_agent_run.agent_run_id
                        or execution.session_id != expected_session.session_id
                        or expected_session.active_execution_id is not None
                        or next_cursor.history_id != expected_session.history_id
                        or next_cursor.agent_run_id != conversation_agent_run.agent_run_id
                        or conversation_checkpoint.agent_run_id != conversation_agent_run.agent_run_id):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                stores.extend((self._conversation.state_store, archive.state_store))
            else:
                if any(value is not None for value in (
                    next_cursor, conversation_agent_run, conversation_checkpoint,
                )) or execution.conversation_agent_run_id is not None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                stores.append(self._conversation.state_store)
        if successful and execution_projections and expected_session is not None:
            if execution_projections[0].run != conversation_agent_run:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        stores = tuple(_dedupe_stores(stores))
        if not _same_group(stores):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        repository = self._execution

        existing = await self.read_pending_turn(
            reservation, expected_session=expected_session, tool_operations=tool_operations,
        )
        if existing is not None:
            if self._capability_writes is not None:
                prefix = await self._capability_writes.prepare_terminal(
                    execution.execution_id, capability_run_id, stores[0].storage_group,
                )
                try:
                    if not await prefix.readback():
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    await prefix.settle()
                finally:
                    await prefix.abort()
            return existing
        prepared = None
        history = None
        final_cursor = None
        start = end = 0
        if expected_session is not None and successful:
            assert archive is not None and conversation_agent_run is not None and conversation_checkpoint is not None
            assert next_cursor is not None and self._conversation_history is not None
            _checkpoint, prepared, history = await self._prepare_conversation_checkpoint(
                archive, conversation_agent_run, conversation_checkpoint, next_cursor.history_id,
            )
            if prepared[0].owner_id != next_cursor.history_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if history is None or history.session_id != expected_session.session_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            start, end = _timeline_turn_message_range(history, prepared, expected_session.continuation)
            final_cursor = replace(next_cursor, message_count=end)
        elif successful and conversation_agent_run is not None and conversation_checkpoint is not None:
            assert archive is not None
            await archive.prepare_run_registration(conversation_agent_run)
            _checkpoint, prepared = await archive.prepare_conversation_checkpoint(
                conversation_agent_run, conversation_checkpoint,
            )
        execution = replace(execution, recovery=ExecutionRecoveryDetail(
            agent_run_id=(None if execution.agent_run_seq == 0 else agent_run_id(
                namespace=self._namespace, tenant_id=self._tenant_id,
                execution_id=execution.execution_id, agent_run_seq=execution.agent_run_seq,
            )),
        ), history_seal_digest=None if seal is None else seal.seal_digest)
        commit = replace(commit, execution=execution)
        resolved: ExecutionTerminalCommitResult | None = None
        publication_confirmed = False
        candidate_published = False
        capability_prefix = (await self._capability_writes.prepare_terminal(
            execution.execution_id, capability_run_id, stores[0].storage_group,
        ) if self._capability_writes is not None else None)
        if capability_prefix is not None:
            stores = tuple(_dedupe_stores((*stores, *capability_prefix.stores)))

        fresh_record_keys: dict[StateStore, tuple[bytes, ...]] = {}
        if (isinstance(stores[0].storage_group, GroupRecordReader)
                and expected_session is not None and prepared is not None and archive is not None
                and conversation_agent_run is not None
                and isinstance(self._conversation_history, ConversationHistoryRepositoryImpl)):
            entries = [
                (repository.state_store, repository.pending_turn_record_keys(reservation)),
                (self._conversation.state_store, (self._conversation.session_record_key(expected_session.session_id),
                 self._conversation_history.history_record_key(expected_session.history_id))),
                *(await archive.checkpoint_publication_record_keys(conversation_agent_run, prepared[0])).items(),
            ]
            if all(store.storage_group is stores[0].storage_group for store, _keys in entries):
                for store, keys in entries:
                    fresh_record_keys[store] = tuple(dict.fromkeys((*fresh_record_keys.get(store, ()), *keys)))
                stores = tuple(_dedupe_stores((*stores, *fresh_record_keys)))

        async def mutate(group: StateGroupTransaction) -> ExecutionTerminalCommitResult:
            nonlocal final_cursor, candidate_published
            after_image = None
            candidate_published = False
            if context_write is not None:
                content_archive, archive_records = context_write
                await content_archive.store.mutate(lambda tx: content_archive.publish(tx, archive_records))
            if fresh_record_keys and isinstance(group, GroupRecordReader):
                # Mutable owners still use their CAS; immutable records still compare complete bytes.
                await group.get_records(fresh_record_keys)
            execution_transaction = group.transaction(repository.state_store)
            winner = await self._read_pending_turn_in_group(
                group, reservation, expected_session, tool_operations=tool_operations,
            )
            if winner is not None:
                if capability_prefix is not None and not await capability_prefix.readback():
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return winner
            if expected_session is not None:
                conversation_transaction = group.transaction(self._conversation.state_store)
                current = await self._conversation.get_in_transaction(
                    conversation_transaction, expected_session.session_id, tenant_id=self._tenant_id,
                )
                if current != expected_session:
                    raise AIError(ErrorCode.SESSION_BUSY)
                if successful:
                    assert history is not None and self._conversation_history is not None
                    current_history = await self._conversation_history.get_in_transaction(
                        conversation_transaction, history.history_id, tenant_id=self._tenant_id,
                    )
                    if current_history != history:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
            final, insertion = await repository.insert_pending_turn_in_transaction(
                execution_transaction, reservation, commit, audit_events=tuple(audit_events), authority=authority,
            )
            if seal is not None:
                await self._execution_run_store.execution_history_heads(expected_runs)
                for run_id in expected_runs:
                    if await self._execution_run_store.get_agent_run(agent_run_id=run_id) is not None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                for projection in execution_projections:
                    await self._execution_run_store.sync_projection_in_transaction(
                        group.transaction(self._execution_run_store.state_store), projection.run,
                        events=projection.events, checkpoints=projection.checkpoints,
                        interactions=projection.interactions, observation=projection.observation,
                        producer_generation=projection.producer_generation,
                        archive=projection.archive, execution_id=execution.execution_id,
                        execution_guard=(insertion.execution, insertion.record),
                        inserted_terminal=insertion,
                    )
                    if not await self._execution_run_store.verify_execution_projection_head(projection):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await repository.put_history_seal_in_transaction(execution_transaction, seal)
            if expected_session is not None:
                if successful:
                    assert archive is not None and conversation_agent_run is not None and prepared is not None
                    assert final_cursor is not None
                    sequence = await archive.materialize_checkpoint_in_transaction(
                        group.transaction(archive.state_store), conversation_agent_run, prepared[0],
                    )
                    final_cursor = await self._pin_conversation_cursor(archive, final_cursor, prepared[0], sequence)
                final_session, turn_commit = await self._conversation.commit_pending_turn_in_transaction(
                    conversation_transaction, expected=expected_session, execution_id=execution.execution_id,
                    next_cursor=final_cursor, start_message_index=start, end_message_index=end,
                    successful=successful,
                )
                if include_publication:
                    messages = (prepared.session_turn_messages(turn_commit,
                                inherited_message_count=history.inherited_message_count)
                                if prepared is not None and history is not None else
                                () if turn_commit.start_message_index == turn_commit.end_message_index else None)
                    if messages is not None:
                        after_image = SessionTurnAfterImage(
                            turn_commit, final_session.owner_principal_id, final_session.revision,
                            messages, commit.result.output,
                        )
            elif prepared is not None:
                assert archive is not None and conversation_agent_run is not None
                await archive.materialize_checkpoint_in_transaction(
                    group.transaction(archive.state_store), conversation_agent_run, prepared[0],
                )
            await self._materialize_pending_tools(
                final.execution, tool_operations, read_only=False, terminal=True,
                transaction=group.transaction(self._tools.state_store) if tool_operations else None,
            )
            if capability_prefix is not None:
                await capability_prefix.apply(group)
            candidate_published = True
            if not include_publication:
                return final
            return ExecutionTerminalPublication(
                final.execution, final.result,
                observed=Observed(final.execution, insertion.record.storage_revision),
                session_released=expected_session is not None, history_seal=seal, turn_after_image=after_image,
            )

        async def operation() -> object:
            nonlocal resolved, publication_confirmed
            resolved = await stores[0].storage_group.mutate(stores, mutate)
            publication_confirmed = candidate_published
            return resolved

        async def readback() -> CommitObservation[object]:
            nonlocal resolved
            try:
                resolved = await self.read_pending_turn(
                    reservation, expected_session=expected_session, tool_operations=tool_operations,
                )
            except AIError as error:
                if error.code is ErrorCode.IDEMPOTENCY_CONFLICT:
                    return CommitObservation(DurableCommitState.NOT_COMMITTED, error=error)
                raise
            if resolved is not None and capability_prefix is not None and not await capability_prefix.readback():
                return CommitObservation(
                    DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                    error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                )
            return CommitObservation(
                DurableCommitState.COMMITTED if resolved is not None else DurableCommitState.NOT_COMMITTED,
                value=resolved,
            )

        try:
            outcome = await run_durable_commit(operation, readback, background_tasks=self._background_tasks)
            if outcome.state is DurableCommitState.COMMITTED:
                assert resolved is not None
                if capability_prefix is not None:
                    await capability_prefix.settle()
                if publication_confirmed and final_cursor is not None and prepared is not None:
                    await self._confirm_conversation_context(archive, final_cursor, prepared[0])
                if outcome.cancelled:
                    raise asyncio.CancelledError
                return resolved
            if outcome.state is DurableCommitState.NOT_COMMITTED:
                if outcome.error is not None:
                    raise outcome.error
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if outcome.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from outcome.error
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from outcome.error
        finally:
            if capability_prefix is not None:
                await capability_prefix.abort()

    async def _pin_conversation_cursor(
        self, archive: StateStepArchive, cursor: ConversationCursor,
        prepared: PreparedAgentRunCheckpoint, sequence: int | None,
    ) -> ConversationCursor:
        if sequence is None:
            sequence = await archive.latest_checkpoint_sequence(agent_run_id=cursor.agent_run_id)
            stored = await archive.checkpoint_at_sequence(
                agent_run_id=cursor.agent_run_id, checkpoint_sequence=sequence,
            )
            if stored != prepared.stored:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return replace(cursor, checkpoint_sequence=sequence, projection_digest=prepared.stored.projection_digest)

    async def _confirm_conversation_context(
        self, archive: StateStepArchive, cursor: ConversationCursor, prepared: PreparedAgentRunCheckpoint,
        *, archive_records: PreparedArchive | None = None,
    ) -> None:
        if cursor.history_id is None or cursor.message_count is None or cursor.checkpoint_sequence is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await archive.confirm_session_context(
            checkpoint_sequence=cursor.checkpoint_sequence, prepared=prepared, archive_records=archive_records,
        )

    @sql_observation_phase("persistence.terminal.checkpoint_prepare")
    async def _prepare_conversation_checkpoint(
        self,
        owner: StateStepArchive,
        run: AgentRunRecord,
        checkpoint: AgentRunCheckpoint,
        history_id: str,
        *,
        archive: PreparedArchive | None = None,
    ) -> tuple[AgentRunCheckpoint, PreparedAgentRunCheckpointBatch, ConversationHistoryRecord | None]:
        await owner.prepare_run_registration(run)
        history_owner = self._conversation_history
        if history_owner is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        preparation_keys = None
        if (type(history_owner) is ConversationHistoryRepositoryImpl
                and history_owner.namespace == self._namespace and history_owner.tenant_id == self._tenant_id
                and history_owner.state_store is self._conversation.state_store
                and history_owner.state_store is owner.state_store):
            preparation_keys = await owner.conversation_preparation_record_keys(run, checkpoint)
        if preparation_keys is None:
            relocated, prepared = await owner.prepare_conversation_checkpoint(run, checkpoint, archive=archive)
            history = await history_owner.get(history_id, tenant_id=self._tenant_id)
        else:
            requests = dict(preparation_keys)
            requests[history_owner.state_store] = tuple(dict.fromkeys((
                *requests.get(history_owner.state_store, ()), history_owner.history_record_key(history_id),
            )))

            async def capture_preparation(group: StateGroupTransaction):
                if not isinstance(group, GroupRecordReader):
                    raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
                captured = await group.get_records(requests)
                descriptor = await history_owner.get_in_transaction(
                    group.transaction(history_owner.state_store), history_id, tenant_id=self._tenant_id,
                )
                observed = {store: {key: captured[store].get(key) for key in keys}
                            for store, keys in preparation_keys.items()}
                return descriptor, observed

            history, observed_records = await owner.state_store.storage_group.read_many(
                tuple(requests), capture_preparation,
            )
            relocated, prepared = await owner.prepare_conversation_checkpoint(
                run, checkpoint, archive=archive, observed_records=observed_records,
            )
        return relocated, prepared, history

    def _native_terminal_text(
        self, output: StoredPayload | None, checkpoints: Sequence[PreparedAgentRunCheckpoint],
    ) -> NativeTextReference | None:
        """Select one exact output part from this terminal publication's final response."""
        if (output is None or output.kind != "inline" or output.encoding != "json" or not checkpoints):
            return None
        prepared = checkpoints[-1]
        messages = prepared.context_messages
        if not messages or len(prepared.projection.items) != 1:
            return None
        block = prepared.projection.items[0]
        if not isinstance(block, ArchivedContextBlock) or not isinstance(block.root, NativeMessageRoot):
            return None
        return self._native_terminal_output_reference(output, block.root, messages)

    def _native_terminal_output_reference(
        self, output: StoredPayload, root: NativeMessageRoot, messages: Sequence[ModelMessage],
    ) -> NativeTextReference | None:
        opened = self._execution.state_store.storage_group.opened_archive
        if (output.kind != "inline" or output.encoding != "json" or root.count != len(messages)
                or opened is None or root.group_id != opened[0]
                or root.store_key != native_store_key(self._namespace, self._tenant_id)):
            return None
        response_index = next((index for index in range(len(messages) - 1, -1, -1)
                               if isinstance(messages[index], ModelResponse)), None)
        if response_index is None:
            return None
        message = messages[response_index]
        trailing = messages[response_index + 1:]
        root = root.slice(response_index, response_index + 1)
        raw = encode_model_messages((message,))
        expected = canonical_json_bytes(output.to_json())
        matches = []
        for index, part in enumerate(message.parts):
            if isinstance(part, TextPart):
                if trailing:
                    continue
                projections = tuple(NativeTextReference(root, projection, index)
                                    for projection in ("text", "assistant-text-output"))
            elif isinstance(part, ToolCallPart):
                # Output-tool completion may append its acknowledgement. Never
                # walk back across a retry, a new response or unrelated effects.
                if any(not isinstance(item, ModelRequest) or not item.parts
                       or any(not isinstance(reply, ToolReturnPart)
                              or reply.tool_call_id != part.tool_call_id or reply.tool_name != part.tool_name
                              for reply in item.parts) for item in trailing):
                    continue
                projections = (NativeTextReference(root, "tool-call-output", index, part.tool_name, part.tool_call_id),)
            else:
                continue
            for reference in projections:
                try:
                    value = restore_native_text_payload(reference, raw)
                except AIError:
                    continue
                if canonical_json_bytes(value.to_json()) == expected:
                    matches.append(reference)
        return matches[0] if len(matches) == 1 else None

    async def _commit_canonical_graph_terminal(
        self, commit: ExecutionTerminalCommit, *, expected_execution: ExecutionRecord,
        turns: "SqlSessionTurns", key: "SessionTurnKey", generation: int, turn_no: int,
        transcript: Sequence[ModelMessage] | None = None,
        binding_refs: tuple[tuple[int, int | None], ...] = (),
        output_ref: tuple[int, int] | None = None,
        pending_events: Sequence[ExecutionEventAppend] = (),
        start_identity: IdempotencyRecord | None = None,
        _graph_trace_plan: ExecutionTerminalSealPlan | None = None,
        _graph_trace_suffix: NativeMessageRoot | None = None,
        _graph_trace_messages: Sequence[ModelMessage] = (),
        _native_step_suffix: "tuple[NativeMessageRoot, Sequence[ModelMessage]] | None" = None,
        _graph_step_checkpoint: "tuple[SqlStepStore, str, str, str | None, int] | None" = None,
        _cold_graph_task: "tuple[TaskRepositoryImpl, TaskLease | _TaskGraphCancelOwner] | None" = None,
    ) -> ExecutionTerminalCommitResult:
        """Finalize the exact attached Graph and session owner together."""
        from sqlalchemy import select

        initial, terminal = expected_execution, commit.execution
        if (type(self._execution) is not ExecutionRepositoryImpl
                or turns._scope != state_store_key(self._namespace, self._tenant_id, "regenerable-session")
                or any(value.execution_id != key.turn_id or value.root_execution_id != key.turn_id
                       or value.session_id is not None or value.parent_execution_id is not None
                       or value.parent_invocation_id is not None or value.binding_kind != "agent"
                       or not value.requires_task_invocation_capture for value in (initial, terminal))
                or commit.producer_generation != initial.producer_generation
                or commit.expected_revision != initial.revision
                or commit.expected_event_seq != initial.event_seq):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if (initial.restart_checkpoints != terminal.restart_checkpoints
                or initial.restart_checkpoints != (_graph_step_checkpoint is not None)
                or terminal.durable_trace != initial.durable_trace
                or (terminal.durable_trace != (_graph_trace_plan is not None) and _cold_graph_task is None)):
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
        state: "TerminalState"
        result: "JsonValue"
        if terminal.status is ExecutionStatus.SUCCEEDED:
            state, result = "succeeded", None
            if ((transcript is None and _graph_trace_suffix is None and _native_step_suffix is None)
                    or commit.result.output is None):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        elif terminal.status is ExecutionStatus.CANCELLED:
            if initial.status is not ExecutionStatus.CANCELLING:
                raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
            state, result = "cancelled", {"error_code": ErrorCode.EXECUTION_CANCELLED.value}
        elif terminal.status is ExecutionStatus.FAILED and terminal.error_code is not None:
            state, result = "failed", {"error_code": terminal.error_code}
        else:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if state != "succeeded" and commit.result.output is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        if _native_step_suffix is not None and (
            state != "succeeded" or initial.durable_trace or _graph_trace_plan is not None
            or _graph_trace_suffix is not None or _graph_trace_messages
            or _graph_step_checkpoint is None or _cold_graph_task is not None
            or transcript is not None or binding_refs or commit.producer_generation is None
        ):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        cold_stores = ()
        if _cold_graph_task is not None:
            if (state != "cancelled" or generation < 2 or _graph_trace_plan is not None
                    or _graph_trace_suffix is not None or _graph_trace_messages or pending_events
                    or initial.producer_generation < 1 or not initial.producer_claim_id
                    or terminal.producer_generation != initial.producer_generation
                    or terminal.producer_claim_id != initial.producer_claim_id
                    or initial.recovery_state in {ExecutionRecoveryState.HANDOFF, ExecutionRecoveryState.WAITING}
                    or (initial.recovery is not None and initial.recovery.pending_tools is not None)):
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            task_repository, task_lease = _cold_graph_task
            cold_stores = self._canonical_graph_launch_stores(
                key.turn_id, (turns, key, generation, task_repository, task_lease),
            )

        cold_tools = None
        if _graph_step_checkpoint is not None:
            from ._step_store import SqlStepStore

            step_store, step_run, step_conversation, replay_key, prefix_count = _graph_step_checkpoint
            if (type(step_store) is not SqlStepStore
                    or (_cold_graph_task is not None and (state != "cancelled" or replay_key is not None or prefix_count != 0))
                    or (replay_key is None and state == "succeeded")
                    or (replay_key is not None and not replay_key)
                    or (state == "succeeded") != (_graph_trace_suffix is not None or _native_step_suffix is not None)
                    or (state != "succeeded" and initial.recovery is not None and (
                        initial.recovery.pending_tools is not None
                        or initial.recovery_state in {ExecutionRecoveryState.WAITING, ExecutionRecoveryState.HANDOFF}
                    ))
                    or type(prefix_count) is not int or prefix_count < 0
                    or initial.native_step_source is None or initial.native_step_source != step_store.backup_source
                    or terminal.native_step_source != initial.native_step_source or initial.agent_run_seq < 1
                    or step_run != agent_run_id(namespace=self._namespace, tenant_id=self._tenant_id,
                                                execution_id=key.turn_id, agent_run_seq=initial.agent_run_seq)):
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            turns._require_checkpoint_store(step_store)
            if _cold_graph_task is not None:
                cold_tools = self._cold_graph_step_store(initial, turns, step_store, step_run, _cold_graph_task[1])
                cold_stores = (*cold_stores, cold_tools.state_store)

        binding = turns._transcript_write_binding
        execution_store = self._execution.state_store
        if binding is None or not _same_group((execution_store, binding[0])):
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
        run_store = self._execution_run_store
        trace_archive = trace_content = trace_suffix = seal = None
        projections = ()
        if _cold_graph_task is not None and initial.durable_trace:
            if run_store is None or not _same_group((execution_store, *run_store.transaction_stores)):
                raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
            cold_stores = (*cold_stores, *run_store.transaction_stores)
        if _graph_trace_plan is not None:
            plan = _graph_trace_plan
            allowed_runs = {
                agent_run_id(namespace=self._namespace, tenant_id=self._tenant_id,
                             execution_id=key.turn_id, agent_run_seq=sequence)
                for sequence in range(1, initial.agent_run_seq + 1)
            }
            tokens = dict(plan.seal_tokens)
            projections = plan.projections
            projection_ids = {value.run.agent_run_id for value in projections}
            if (plan.execution_id != key.turn_id or plan.binding_digest != initial.binding_digest
                    or initial.binding_digest != terminal.binding_digest
                    or len(tokens) != len(plan.seal_tokens) or not tokens.keys() <= allowed_runs
                    or any(not token or token != plan.terminal_attempt_token for token in tokens.values())
                    or len(projection_ids) != len(projections) or not projection_ids <= tokens.keys()
                    or any(value.producer_generation != commit.producer_generation for value in projections)
                    or (projections and commit.producer_generation is None)):
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            if run_store is None or not _same_group((execution_store, *run_store.transaction_stores)):
                raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
            trace_archive = await run_store.transcript_repository.content_archive(create=True)
            if (trace_archive.store.storage_group is not execution_store.storage_group
                    or trace_archive.content_scope != archive_content_scope(self._namespace, self._tenant_id)):
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            trace_content = _copy_archive_union(tuple(
                content for projection in projections for content in (
                    projection.archive,
                    *(checkpoint.archive for checkpoint in projection.checkpoints),
                    *((projection.observation.archive,) if projection.observation is not None else ()),
                )
            ))
            if state == "succeeded":
                if (_graph_trace_suffix is None or not _graph_trace_messages
                        or transcript is not None or binding_refs or not projections):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                trace_suffix = (trace_archive, trace_content, _graph_trace_suffix)
            seal = _execution_history_seal(
                commit, audit_events=pending_events, projections=projections,
                current_run=None, current_events=(), current_batch=None,
            )
            commit = replace(commit, execution=replace(terminal, history_seal_digest=seal.seal_digest))
            projections = tuple(replace(
                projection, archive=PreparedArchive(),
                checkpoints=tuple(replace(item, archive=PreparedArchive()) for item in projection.checkpoints),
                observation=(None if projection.observation is None else
                             replace(projection.observation, archive=PreparedArchive())),
            ) for projection in projections)
        if trace_suffix is None and (_graph_trace_suffix is not None or _graph_trace_messages):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        native_suffix = None
        native_messages: Sequence[ModelMessage] = ()
        if _native_step_suffix is not None:
            root, native_messages = _native_step_suffix
            archive = await binding[1](create=True)
            native_suffix = (archive, PreparedArchive(), root)
        raw_source = trace_suffix or native_suffix
        prepared = await turns._prepare_finish(
            key, generation=generation, turn_no=turn_no, state=state, result=result,
            transcript=transcript, binding_refs=binding_refs, output_ref=output_ref,
            _graph_trace_suffix=raw_source, _graph_step_source=native_suffix is not None,
        )
        result_source = None
        if prepared.transcript is not None:
            source_messages = (
                (_graph_trace_messages if native_suffix is None else native_messages)
                if raw_source is not None else transcript
            )
            assert source_messages is not None and commit.result.output is not None
            archive, content, root = prepared.transcript
            if archive.content_scope != archive_content_scope(self._namespace, self._tenant_id):
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            result_source = self._native_terminal_output_reference(commit.result.output, root, source_messages)
            if result_source is None:
                raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
            await self._execution.prepare_terminal_content(commit, content, result_source=result_source)
        elif trace_content is not None:
            await self._execution.prepare_terminal_content(commit, trace_content)
        else:
            await self._execution.prepare_content(terminal)

        row = turns._turns
        relationship = select(row.c.turn_id).filter_by(
            **turns._turn_values(key), generation=generation, turn_no=turn_no,
            state="cancelling" if state == "cancelled" else "accepted",
        ).where(row.c.request_json["graph_execution"].as_string() == key.turn_id)

        attempted: ExecutionTerminalWriteReceipt | None = None

        async def mutate(group: StateGroupTransaction) -> ExecutionTerminalCommitResult:
            nonlocal attempted, seal
            async def require_attached(session):
                if await session.scalar(relationship) is None:
                    raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

            await group.transaction(binding[0]).run_sql(turns._context, require_attached)
            execution_tx = group.transaction(execution_store)
            guard = None
            if (seal is not None or _cold_graph_task is not None or _graph_step_checkpoint is not None
                ) and commit.producer_generation is not None:
                guard = await self._execution.require_execution_producer_in_transaction(
                    execution_tx, key.turn_id, expected_producer_generation=commit.producer_generation,
                )
            effective = await self._effective_terminal_commit(
                execution_tx, commit, expected_execution=initial, start_identity=start_identity,
            )
            if _graph_step_checkpoint is not None:
                async def check_step(session):
                    assert replay_key is not None
                    return await step_store._capture_complete_turn_checkpoint_root(
                        session, run_id=step_run, conversation_id=step_conversation, replay_key=replay_key,
                    )

                checkpoint_root = (None if replay_key is None else
                    await group.transaction(binding[0]).run_sql(turns._context, check_step))
                if state == "succeeded":
                    assert checkpoint_root is not None
                    suffix = _graph_trace_suffix if native_suffix is None else native_suffix[2]
                    if suffix is None or checkpoint_root.count != prefix_count + suffix.count:
                        raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
                    exact_suffix = checkpoint_root.slice(prefix_count, checkpoint_root.count)
                    if native_suffix is not None and exact_suffix != suffix:
                        raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                    if native_suffix is None:
                        await step_store._require_same_native_message_occurrences(execution_tx, exact_suffix, suffix)
                    # Its official checkpoint already owns these exact physical
                    # roots. Trace and Session may stage owner edges, not resave
                    # the body; leave other interaction/payload roots untouched.
                    if trace_content is not None:
                        backing = {(part.first, part.count) for part in checkpoint_root.slices}
                        published = tuple(sequence for sequence in trace_content.native_messages if sequence.slices
                            and all((part.first, part.count) in backing for part in sequence.slices))
                        for sequence in published:
                            trace_content.native_messages.pop(sequence)
                            trace_content.native_publishers.pop(sequence, None)
                if cold_tools is not None:
                    await self._require_cold_cancelled_graph_step(
                        group, key.turn_id, tools=cold_tools, native=step_store, run_id=step_run,
                        conversation_id=step_conversation,
                    )
                elif state != "succeeded":
                    from ..steps import NativeLifecycleEvent

                    # A complete checkpoint alone can predate a failed run. Match
                    # the actual owned terminal marker, never an interrupted draft.
                    event_key = (step_run, NativeLifecycleEvent.source_key())
                    sources = await step_store.read_fixed_events((event_key,), transaction=execution_tx)
                    marker = sources.get(event_key)
                    if (not isinstance(marker, NativeLifecycleEvent) or marker.outcome != state
                            or marker.conversation_id != step_conversation):
                        raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
            if not await turns._finish_in_group(group, prepared):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if _cold_graph_task is not None:
                task_repository, task_lease = _cold_graph_task
                await self._guard_cold_graph_task(group, task_repository, task_lease, execution_id=key.turn_id)
                if initial.durable_trace:
                    assert run_store is not None
                    # The original producer generation was revoked by the takeover.
                    # Read existing durable heads with that Execution fenced; no
                    # staged/provider messages or new run registration is needed.
                    candidates = tuple(sorted(
                        agent_run_id(namespace=self._namespace, tenant_id=self._tenant_id,
                                     execution_id=key.turn_id, agent_run_seq=sequence)
                        for sequence in range(1, initial.agent_run_seq + 1)
                    ))
                    existing = tuple([
                        run_id for run_id in candidates if await run_store.get_agent_run(agent_run_id=run_id) is not None
                    ])
                    heads = await run_store.execution_history_heads(candidates)
                    for run_id in set(candidates).difference(existing):
                        head = heads[run_id]
                        if (head.event_count or head.checkpoint_count or head.transcript_message_count
                                or head.interaction_count or head.projection_digest != "empty"):
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    seal = ExecutionHistorySealRecord(
                        key.turn_id, tuple(heads[run_id] for run_id in existing), effective.expected_event_seq + 1,
                    )
                    effective = replace(effective, execution=replace(effective.execution, history_seal_digest=seal.seal_digest))
            elif seal is not None:
                assert run_store is not None and trace_archive is not None and trace_content is not None
                await run_store.transcript_repository.publish_archive(trace_content)
                trace_tx = group.transaction(run_store.state_store)
                for projection in projections:
                    await run_store.sync_projection_in_transaction(
                        trace_tx, projection.run, events=projection.events, checkpoints=projection.checkpoints,
                        interactions=projection.interactions, observation=projection.observation,
                        producer_generation=projection.producer_generation, execution_id=key.turn_id,
                        execution_guard=guard,
                    )
                await run_store.transcript_repository.publish_native_archive(trace_tx, trace_content, owner_written=True)
                for projection in projections:
                    await run_store.share_published_content(
                        projection.run, execution_id=key.turn_id, checkpoints=projection.checkpoints,
                        observation=projection.observation, archive=trace_content,
                    )
            if seal is not None:
                await self._execution.put_history_seal_in_transaction(execution_tx, seal)
            committed = await self._execution.commit_terminal_in_transaction(
                execution_tx, effective, pending_events=pending_events, result_source=result_source,
            )

            if not isinstance(committed, ExecutionTerminalWriteReceipt):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if seal is None:
                attempted = committed
                return committed
            observed = await self._execution.confirm_terminal_write_in_transaction(execution_tx, committed)
            if observed is None or observed.value != committed.execution:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            attempted = ExecutionTerminalPublication(
                committed.execution, committed.result, observed=observed, session_released=True, history_seal=seal,
            )
            return attempted

        trace_stores = () if trace_archive is None or run_store is None else (*run_store.transaction_stores, trace_archive.store)
        stores = _dedupe_stores((execution_store, *prepared.stores, *trace_stores, *cold_stores))
        try:
            return await execution_store.storage_group.mutate(stores, mutate)
        except AIError as error:
            # Only a completed transaction body can supply the exact attempted
            # owner receipt. An unknown ACK is never permission to retry.
            if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN or attempted is None:
                raise

        receipt = attempted
        _, predicate, occupied = self._canonical_graph_turn_owner(turns, key, generation)
        outcome = select(row.c.state, row.c.result_json, row.c.lease_until, occupied).where(
            *predicate, row.c.turn_no == turn_no,
        )
        expected_result = prepared.result

        async def observe(group: StateGroupTransaction) -> ExecutionTerminalCommitResult | None:
            async def read_turn(session):
                return (await session.execute(outcome)).one_or_none()

            actual = await group.transaction(binding[0]).run_sql(turns._context, read_turn)
            if actual is None:
                return None
            actual_state, result_json, lease, active = actual
            if actual_state != state or lease is not None or active:
                return None
            if isinstance(result_json, dict) and "usage" in result_json and isinstance(expected_result, dict):
                expected = {**expected_result, "usage": asdict(receipt.result.usage)}
            else:
                expected = expected_result
            if result_json != expected:
                return None
            execution_tx = group.transaction(execution_store)
            observed = await self._execution.get_observed_in_transaction(
                execution_tx, key.turn_id, tenant_id=self._tenant_id,
            )
            if observed is None or not _same_execution_except_hold_metadata(observed.value, receipt.execution):
                return None
            if seal is not None and await self._execution._get_history_seal_in_transaction(
                execution_tx, key.turn_id,
            ) != seal:
                return None
            return replace(receipt, execution=observed.value, observed=observed)

        try:
            confirmed = await execution_store.storage_group.read_many(stores, observe)
        except Exception as readback_error:
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from readback_error
        if confirmed is None:
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        return confirmed

    @sql_observation_phase("persistence.terminal.durable_commit")
    async def commit_terminal_checkpoint(
        self,
        commit: ExecutionTerminalCommit,
        *,
        expected_execution: ExecutionRecord | None = None,
        start_identity: IdempotencyRecord | None = None,
        session_turn: SessionTurnRef | None = None,
        capability_run_id: str | None = None,
        session_id: str | None = None,
        expected_cursor: ConversationCursor | None = None,
        next_cursor: ConversationCursor | None = None,
        conversation_agent_run: AgentRunRecord | None = None,
        conversation_checkpoint: AgentRunCheckpoint | None = None,
        conversation_message_range: tuple[int, int] | None = None,
        execution_run: AgentRunRecord | None = None,
        execution_events: Sequence[StepEvent] = (),
        execution_checkpoints: Sequence[AgentRunCheckpoint] = (),
        execution_projections: Sequence[PreparedExecutionProjection] = (),
        audit_events: Sequence[ExecutionEventAppend] = (),
        background_tasks: "set[asyncio.Task[object]] | None" = None,
        turn_fence: SourceTurnWriteFence | None = None,
    ) -> ExecutionTerminalCommitResult:
        if expected_execution is not None and (
            expected_execution.execution_id != commit.execution.execution_id
            or expected_execution.revision != commit.expected_revision
            or expected_execution.event_seq != commit.expected_event_seq
        ):
            raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
        if isinstance(self._execution, ExecutionRepositoryImpl):
            await self._execution.prepare_content(replace(commit.execution, result=commit.result))
        publish_turn = session_id is not None and commit.execution.conversation
        effective_commit = commit
        if not commit.execution.durable_trace and (
            execution_run is not None or execution_events or execution_checkpoints or execution_projections
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution_projections and (
            self._execution_run_store is None or execution_run is None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (conversation_agent_run is None) != (conversation_checkpoint is None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if conversation_agent_run is not None and not commit.execution.conversation:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _logger.debug(
            "terminal checkpoint requested: execution=%s session=%s recovery=%s",
            commit.execution.execution_id,
            session_id,
            commit.execution.recovery is not None,
        )
        owner_tasks = (
            self._background_tasks
            if background_tasks is None
            else background_tasks
        )
        stores = [self._execution.state_store]
        if session_id is not None:
            self._require_conversation()
            stores.append(self._conversation.state_store)
        if execution_run is not None:
            if self._execution_run_store is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            stores.extend(self._execution_run_store.transaction_stores)
        if conversation_agent_run is not None or conversation_checkpoint is not None:
            if self._conversation_run_store is None or conversation_agent_run is None or conversation_checkpoint is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            stores.extend(self._conversation_run_store.transaction_stores)
        if turn_fence is not None:
            stores = list(_dedupe_stores((*stores, turn_fence.store)))
            if not _same_group(stores):
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH,
                              "A source-owned terminal commit cannot publish across SQL groups")
        terminal_archives = tuple(
            archive for archive, participates in (
                (self._execution_run_store, execution_run is not None and bool(execution_projections or execution_checkpoints)),
                (self._conversation_run_store, conversation_agent_run is not None),
            ) if participates and archive is not None
        )
        shared_archive = None
        execution_archive_scope = archive_content_scope(self._namespace, self._tenant_id)
        shared_execution_scope = False
        if _same_group(stores) and terminal_archives:
            roots = tuple([
                (await archive.transcript_repository.content_archive(create=True)).empty_root()
                for archive in terminal_archives
            ])
            same_content_scope = all(root == roots[0] for root in roots)
            shared_execution_scope = same_content_scope and roots[0].content_scope == execution_archive_scope
            shared_archive = _copy_archive_union(tuple(
                archive for projection in execution_projections
                for archive in (
                    projection.archive,
                    *(checkpoint.archive for checkpoint in projection.checkpoints),
                    *((projection.observation.archive,) if projection.observation is not None else ()),
                )
            )) if same_content_scope else None
        prepared_conversation = ()
        timeline_history = None
        if (
            self._conversation_run_store is not None
            and conversation_agent_run is not None
            and conversation_checkpoint is not None
        ):
            await self._conversation_run_store.prepare_run_registration(conversation_agent_run)
            if session_id is None:
                conversation_checkpoint, prepared_conversation = await self._conversation_run_store.prepare_conversation_checkpoint(
                    conversation_agent_run, conversation_checkpoint, archive=shared_archive,
                )
            else:
                if next_cursor is None or next_cursor.history_id is None or self._conversation_history is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                conversation_checkpoint, prepared_conversation, timeline_history = await self._prepare_conversation_checkpoint(
                    self._conversation_run_store, conversation_agent_run, conversation_checkpoint,
                    next_cursor.history_id, archive=shared_archive,
                )
        timeline_range: tuple[int, int] | None = conversation_message_range
        if prepared_conversation and session_id is not None:
            if (session_id is None or next_cursor is None or timeline_history is None
                    or timeline_history.session_id != session_id):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            timeline_range = _timeline_turn_message_range(
                timeline_history,
                prepared_conversation,
                expected_cursor,
            )
            next_cursor = replace(
                next_cursor,
                message_count=timeline_range[1],
            )

        prepared_execution: PreparedAgentRunCheckpointBatch | tuple[()] = ()
        if self._execution_run_store is not None and execution_run is not None:
            await self._execution_run_store.prepare_run_registration(execution_run)
        if (
            not execution_projections
            and self._execution_run_store is not None
            and execution_run is not None
            and execution_checkpoints
        ):
            prepared_execution = await self._execution_run_store.prepare_checkpoints(
                execution_run,
                execution_checkpoints,
                archive=shared_archive,
            )
        seal = _execution_history_seal(
            commit,
            audit_events=audit_events,
            projections=execution_projections,
            current_run=execution_run,
            current_events=execution_events,
            current_batch=(
                prepared_execution
                if isinstance(prepared_execution, PreparedAgentRunCheckpointBatch)
                else None
            ),
        ) if commit.execution.durable_trace else None
        commit = replace(commit, execution=replace(
            commit.execution, history_seal_digest=None if seal is None else seal.seal_digest,
        ))
        effective_commit = commit
        if _same_group(stores):
            terminal_projections = execution_projections
            if shared_archive is not None:
                shared_archive = _copy_archive_union((
                    shared_archive,
                    *(checkpoint.archive for batch in (
                        prepared_conversation, prepared_execution,
                    ) for checkpoint in batch),
                ))
                if prepared_conversation:
                    prepared_conversation = replace(prepared_conversation, checkpoints=tuple(
                        replace(checkpoint, archive=PreparedArchive()) for checkpoint in prepared_conversation
                    ))
                if prepared_execution:
                    prepared_execution = replace(prepared_execution, checkpoints=tuple(
                        replace(checkpoint, archive=PreparedArchive()) for checkpoint in prepared_execution
                    ))
                terminal_projections = tuple(replace(
                    projection,
                    archive=PreparedArchive(),
                    checkpoints=tuple(replace(checkpoint, archive=PreparedArchive())
                                      for checkpoint in projection.checkpoints),
                    observation=(None if projection.observation is None else
                                 replace(projection.observation, archive=PreparedArchive())),
                ) for projection in execution_projections)
            result_source = None
            if type(self._execution) is ExecutionRepositoryImpl:
                for checkpoints in (tuple(prepared_conversation), tuple(prepared_execution),
                        tuple(checkpoint for projection in terminal_projections for checkpoint in projection.checkpoints)):
                    result_source = self._native_terminal_text(commit.result.output, checkpoints)
                    if result_source is not None:
                        break
                if shared_archive is not None and shared_execution_scope:
                    await self._execution.prepare_terminal_content(commit, shared_archive, result_source=result_source)
                elif self._execution_run_store is not None and (prepared_execution or terminal_projections):
                    content = await self._execution_run_store.transcript_repository.content_archive(create=True)
                    if content.content_scope == execution_archive_scope:
                        if prepared_execution:
                            checkpoint = prepared_execution[0]
                            bundle = _copy_archive_union((checkpoint.archive,))
                            await self._execution.prepare_terminal_content(commit, bundle, result_source=result_source)
                            prepared_execution = replace(prepared_execution, checkpoints=(
                                replace(checkpoint, archive=bundle), *prepared_execution.checkpoints[1:],
                            ))
                        else:
                            projection = terminal_projections[0]
                            bundle = _copy_archive_union((projection.archive,))
                            await self._execution.prepare_terminal_content(commit, bundle, result_source=result_source)
                            terminal_projections = (replace(projection, archive=bundle), *terminal_projections[1:])
            prepared = (await self._capability_writes.prepare_terminal(
                commit.execution.execution_id, capability_run_id, stores[0].storage_group,
            ) if self._capability_writes is not None else None)
            if prepared is not None:
                stores.extend(prepared.stores)
                if turn_fence is not None and not _same_group(stores):
                    raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH,
                                  "A source-owned terminal commit cannot publish across SQL groups")
            fresh_record_keys: dict[StateStore, tuple[bytes, ...]] = {}
            if (shared_archive is not None
                    and isinstance(stores[0].storage_group, GroupRecordReader)
                    and type(self._execution) is ExecutionRepositoryImpl
                    and type(self._conversation) is SessionRepositoryImpl
                    and type(self._conversation_history) is ConversationHistoryRepositoryImpl
                    and type(self._conversation_run_store) in {StateStepArchive, ModelInteractionStateStepArchive}
                    and conversation_agent_run is not None and prepared_conversation
                    and session_id is not None and next_cursor is not None):
                content = await terminal_archives[0].transcript_repository.content_archive(create=True)
                entries = [
                    (self._execution.state_store, self._execution.terminal_record_keys(
                        commit.execution, include_history_seal=seal is not None,
                    )),
                    (content.store, content.publication_record_keys(shared_archive)),
                    (self._conversation.state_store, (
                        self._conversation.session_record_key(session_id),
                        self._conversation_history.history_record_key(next_cursor.history_id),
                    )),
                    *(await self._conversation_run_store.checkpoint_publication_record_keys(
                        conversation_agent_run, prepared_conversation[0],
                    )).items(),
                ]
                if all(store.storage_group is stores[0].storage_group for store, _keys in entries):
                    for store, keys in entries:
                        fresh_record_keys[store] = tuple(dict.fromkeys((*fresh_record_keys.get(store, ()), *keys)))
                    stores = list(_dedupe_stores((*stores, *fresh_record_keys)))
            if fresh_record_keys:
                if start_identity is not None:
                    store = self._execution.state_store
                    fresh_record_keys[store] = (*fresh_record_keys.get(store, ()), self._execution.start_idempotency_record_key(start_identity))
                if session_turn is not None and session_id is not None:
                    store = self._conversation.state_store
                    fresh_record_keys[store] = (*fresh_record_keys.get(store, ()), *self._conversation.timeline_commit_record_keys(
                        session_id, start_turn_seq=session_turn.turn_seq, end_turn_seq=session_turn.turn_seq + 1,
                    ))
            try:
                async def callback(group: StateGroupTransaction) -> ExecutionTerminalCommitResult:
                    nonlocal effective_commit, next_cursor
                    if turn_fence is not None:
                        await turn_fence.guard(commit.execution.execution_id)
                    if isinstance(self._execution, ExecutionRepositoryImpl):
                        await self._execution.prefetch_terminal_records(
                            commit.execution.execution_id, additional_records=fresh_record_keys,
                            **({"terminal_write_sources": True} if type(self._execution) is ExecutionRepositoryImpl else {}),
                        )
                    if fresh_record_keys and isinstance(group, GroupRecordReader):
                        await group.get_records(fresh_record_keys)
                    execution_transaction = group.transaction(self._execution.state_store)
                    closed_result = None
                    final_session = turn_commit = None
                    if (type(self._execution) is ExecutionRepositoryImpl
                            and not commit.execution.durable_trace and execution_run is None):
                        effective_commit = await self._effective_terminal_commit(
                            execution_transaction, commit, expected_execution=expected_execution, start_identity=start_identity,
                        )
                        # Establish the execution-first fence and final control CAS
                        # before dependent writes in this same atomic transaction.
                        closed_result = await self._execution.commit_closed_terminal_in_transaction(
                            execution_transaction, effective_commit, pending_events=audit_events,
                        )
                    else:
                        if commit.producer_generation is not None:
                            execution_guard = await self._execution.require_execution_producer_in_transaction(
                                execution_transaction, commit.execution.execution_id,
                                expected_producer_generation=commit.producer_generation,
                            )
                        else:
                            execution_guard = await self._execution.require_execution_history_write_in_transaction(
                                execution_transaction, commit.execution.execution_id,
                            )
                        effective_commit = await self._effective_terminal_commit(
                            execution_transaction,
                            commit,
                            expected_execution=expected_execution, start_identity=start_identity,
                        )
                    if shared_archive is not None:
                        await terminal_archives[0].transcript_repository.publish_archive(shared_archive)
                    if session_id is not None:
                        conversation_transaction = group.transaction(self._conversation.state_store)
                        if conversation_agent_run is None or conversation_checkpoint is None:
                            if publish_turn:
                                final_session, turn_commit = await self._conversation.publish_and_release_turn_in_transaction(
                                    conversation_transaction, session_id, tenant_id=self._tenant_id,
                                    execution_id=commit.execution.execution_id,
                                    start_message_index=0 if timeline_range is None else timeline_range[0],
                                    end_message_index=0 if timeline_range is None else timeline_range[1],
                                    history_unavailable=commit.execution.invalidates_session_history,
                                    allow_unadmitted=commit.execution.started_at is None,
                                )
                            else:
                                final_session = await self._conversation.release_execution_in_transaction(
                                    conversation_transaction,
                                    session_id,
                                    tenant_id=self._tenant_id,
                                    execution_id=commit.execution.execution_id,
                                    history_unavailable=commit.execution.invalidates_session_history,
                                )
                        else:
                            if next_cursor is None or self._conversation_run_store is None:
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                            sequence = await self._conversation_run_store.materialize_checkpoint_in_transaction(
                                group.transaction(self._conversation_run_store.state_store),
                                conversation_agent_run,
                                prepared_conversation[0],
                            )
                            next_cursor = await self._pin_conversation_cursor(
                                self._conversation_run_store, next_cursor, prepared_conversation[0], sequence,
                            )
                            session = await self._conversation.get_in_transaction(
                                conversation_transaction,
                                session_id,
                                tenant_id=self._tenant_id,
                            )
                            await self._promote_history_in_transaction(
                                conversation_transaction,
                                session,
                                prepared_conversation[0],
                            )
                            final_session = await self._conversation.complete_execution_in_transaction(
                                conversation_transaction,
                                session_id,
                                tenant_id=self._tenant_id,
                                execution_id=commit.execution.execution_id,
                                expected=expected_cursor,
                                next_cursor=next_cursor,
                                history_quality="complete",
                            )
                            if timeline_range is None:
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                            turn_commit = await self._conversation.commit_timeline_turn_in_transaction(
                                conversation_transaction,
                                session_id,
                                tenant_id=self._tenant_id,
                                execution_id=commit.execution.execution_id,
                                start_message_index=timeline_range[0],
                                end_message_index=timeline_range[1],
                                admitted_turn=session_turn,
                            )
                    elif conversation_agent_run is not None:
                        await self._conversation_run_store.materialize_checkpoint_in_transaction(
                            group.transaction(self._conversation_run_store.state_store),
                            conversation_agent_run, prepared_conversation[0],
                        )
                    if execution_run is not None:
                        execution_transaction_for_steps = group.transaction(
                            self._execution_run_store.state_store
                        )
                        if execution_projections:
                            for projection in terminal_projections:
                                await self._execution_run_store.sync_projection_in_transaction(
                                    execution_transaction_for_steps,
                                    projection.run,
                                    events=projection.events,
                                    checkpoints=projection.checkpoints,
                                    interactions=projection.interactions,
                                    observation=projection.observation,
                                    producer_generation=projection.producer_generation,
                                    archive=projection.archive,
                                    execution_id=commit.execution.execution_id,
                                    execution_guard=execution_guard,
                                )
                        else:
                            await self._execution_run_store.sync_projection_in_transaction(
                                execution_transaction_for_steps,
                                execution_run,
                                events=execution_events,
                                checkpoints=(
                                    prepared_execution.checkpoints
                                    if isinstance(
                                        prepared_execution,
                                        PreparedAgentRunCheckpointBatch,
                                    )
                                    else ()
                                ),
                                execution_id=commit.execution.execution_id,
                                execution_guard=execution_guard,
                            )
                    if prepared is not None:
                        await prepared.apply(group)
                    if shared_archive is not None:
                        await terminal_archives[0].transcript_repository.publish_native_archive(
                            group.transaction(terminal_archives[0].state_store), shared_archive, owner_written=True,
                        )
                        if self._execution_run_store is not None:
                            for projection in terminal_projections:
                                await self._execution_run_store.share_published_content(
                                    projection.run, execution_id=commit.execution.execution_id,
                                    checkpoints=projection.checkpoints, observation=projection.observation,
                                    archive=shared_archive,
                                )
                            if not terminal_projections and execution_run is not None:
                                await self._execution_run_store.share_published_content(
                                    execution_run, execution_id=commit.execution.execution_id,
                                    checkpoints=tuple(prepared_execution), observation=None, archive=shared_archive,
                                )
                    if closed_result is None:
                        if seal is not None:
                            await self._execution.put_history_seal_in_transaction(execution_transaction, seal)
                        if result_source is None:
                            closed_result = await self._execution.commit_terminal_in_transaction(
                                execution_transaction, effective_commit, pending_events=audit_events,
                            )
                        else:
                            closed_result = await self._execution.commit_terminal_in_transaction(
                                execution_transaction, effective_commit, pending_events=audit_events, result_source=result_source,
                            )
                    if closed_result.execution.recovery_state is ExecutionRecoveryState.HANDOFF:
                        return closed_result
                    observed = (await self._execution.confirm_terminal_write_in_transaction(
                            execution_transaction, closed_result)
                        if isinstance(self._execution, ExecutionRepositoryImpl)
                        and isinstance(closed_result, ExecutionTerminalWriteReceipt)
                        else await self._execution.get_observed_in_transaction(
                            execution_transaction, closed_result.execution.execution_id, tenant_id=self._tenant_id,
                        ))
                    if observed is None or observed.value != closed_result.execution:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    after_image = None
                    if (isinstance(closed_result, ExecutionTerminalWriteReceipt)
                            and type(self._conversation) is SessionRepositoryImpl and final_session is not None
                            and turn_commit is not None and final_session.active_execution_id is None):
                        messages = (() if turn_commit.start_message_index == turn_commit.end_message_index else
                            prepared_conversation.session_turn_messages(
                                turn_commit, inherited_message_count=timeline_history.inherited_message_count,
                            ) if isinstance(prepared_conversation, PreparedAgentRunCheckpointBatch)
                            and timeline_history is not None else None)
                        if messages is not None:
                            after_image = SessionTurnAfterImage(
                                turn_commit, final_session.owner_principal_id, final_session.revision,
                                messages, commit.result.output,
                            )
                    return ExecutionTerminalPublication(
                        closed_result.execution, closed_result.result, observed=observed,
                        session_released=session_id is not None, history_seal=seal, turn_after_image=after_image,
                    )

                async def readback() -> CommitObservation[ExecutionTerminalCommitResult]:
                    try:
                        visible = await self._terminal_targets_visible(
                            effective_commit,
                            expected_execution=expected_execution,
                            pending_event_count=len(audit_events),
                            session_id=session_id,
                            next_cursor=next_cursor,
                            conversation_agent_run=conversation_agent_run,
                            conversation_checkpoint=conversation_checkpoint,
                            prepared_conversation_checkpoint=prepared_conversation[0] if prepared_conversation else None,
                            execution_run=execution_run,
                            execution_events=execution_events,
                            execution_checkpoints=execution_checkpoints,
                            execution_projections=execution_projections,
                            audit_events=audit_events,
                        )
                        if not visible:
                            return CommitObservation(DurableCommitState.NOT_COMMITTED)
                        if prepared is not None and not await prepared.readback():
                            return CommitObservation(
                                DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                                error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                            )
                        execution = await self._load_execution(
                            commit.execution.execution_id,
                            tenant_id=self._tenant_id,
                        )
                        if execution is None:
                            return CommitObservation(
                                DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                                error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                            )
                        return CommitObservation(
                            DurableCommitState.COMMITTED,
                            value=ExecutionTerminalCommitResult(
                                execution,
                                commit.result,
                            ),
                        )
                    except AIError as error:
                        if error.code is ErrorCode.EXECUTION_RESULT_CONFLICT:
                            return CommitObservation(DurableCommitState.NOT_COMMITTED, error=error)
                        if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                            return CommitObservation(
                                DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                                error=error,
                            )
                        return CommitObservation(
                            DurableCommitState.UNRESOLVED,
                            error=error,
                        )

                publication_confirmed = False

                async def publish_terminal() -> ExecutionTerminalCommitResult:
                    nonlocal publication_confirmed
                    value = await stores[0].storage_group.mutate(stores, callback)
                    publication_confirmed = True
                    return value

                result = await run_durable_commit(
                    publish_terminal,
                    readback,
                    background_tasks=owner_tasks,
                )
                if result.state is DurableCommitState.COMMITTED:
                    if result.value is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if prepared is not None:
                        await prepared.settle()
                    if publication_confirmed and publish_turn and next_cursor is not None and prepared_conversation:
                        await self._confirm_conversation_context(self._conversation_run_store, next_cursor, prepared_conversation[0], archive_records=shared_archive)
                    if result.cancelled:
                        raise asyncio.CancelledError
                    return result.value
                if result.state is DurableCommitState.NOT_COMMITTED:
                    if result.error is not None:
                        raise result.error
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                if result.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
                    raise AIError(
                        ErrorCode.STORAGE_INTEGRITY_ERROR,
                        "terminal checkpoint left partial durable state",
                    ) from result.error
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from result.error
            finally:
                if prepared is not None:
                    await prepared.abort()
        if (session_id is not None and commit.execution.started_at is None
                and (commit.execution.recovery is None
                     or commit.execution.recovery.handoff_phase is RecoveryHandoffPhase.NONE)):
            # A rejected split-store admission can still own the session. Keep
            # its release obligation on the terminal row until it is confirmed.
            handoff = RecoveryTerminalHandoff(
                outcome=RecoveryTerminalOutcome(
                    terminal_status=commit.execution.status,
                    error_code=commit.execution.error_code,
                    safe_error_details=commit.execution.safe_error_details,
                    stop_reason=commit.result.stop_reason,
                    output=commit.result.output,
                    object_source_domain=None,
                    usage=commit.result.usage,
                    terminal_event_type=commit.terminal_event_type,
                    terminal_event_payload=commit.terminal_event_payload,
                    result_created_at=commit.result.created_at,
                    error_diagnostics=commit.execution.error_diagnostics,
                ),
                source_agent_run_id=None,
                conversation=None,
            )
            commit = replace(commit, execution=replace(
                commit.execution,
                recovery=replace(
                    commit.execution.recovery or ExecutionRecoveryDetail(),
                    handoff_phase=RecoveryHandoffPhase.PREPARED,
                    terminal_handoff=handoff,
                    pending_tools=None,
                ),
            ))
            effective_commit = commit
        if session_id is None and conversation_agent_run is not None:
            await self._materialize_prepared_checkpoint_with_reconciliation(
                self._conversation_run_store, conversation_agent_run, prepared_conversation,
            )
        if session_id is not None and conversation_agent_run is not None and conversation_checkpoint is not None:
            if next_cursor is None or self._conversation_run_store is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

            conversation_stores = [
                self._conversation.state_store,
                self._conversation_run_store.state_store,
            ]

            async def commit_conversation(group: StateGroupTransaction) -> None:
                nonlocal next_cursor
                conversation_transaction = group.transaction(self._conversation.state_store)
                sequence = await self._conversation_run_store.materialize_checkpoint_in_transaction(
                    group.transaction(self._conversation_run_store.state_store),
                    conversation_agent_run,
                    prepared_conversation[0],
                )
                next_cursor = await self._pin_conversation_cursor(
                    self._conversation_run_store, next_cursor, prepared_conversation[0], sequence,
                )
                session = await self._conversation.get_in_transaction(
                    conversation_transaction,
                    session_id,
                    tenant_id=self._tenant_id,
                )
                await self._promote_history_in_transaction(
                    conversation_transaction,
                    session,
                    prepared_conversation[0],
                )
                await self._conversation.advance_continuation_in_transaction(
                    group.transaction(self._conversation.state_store),
                    session_id,
                    tenant_id=self._tenant_id,
                    execution_id=commit.execution.execution_id,
                    expected=expected_cursor,
                    next_cursor=next_cursor,
                    release_execution=False,
                    history_quality="complete",
                )
                if timeline_range is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

            if _same_group(conversation_stores):
                for attempt in range(2):
                    try:
                        await self._conversation.state_store.storage_group.mutate(
                            conversation_stores,
                            commit_conversation,
                        )
                        break
                    except AIError as error:
                        if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN or attempt == 1:
                            raise
                        if await self._conversation_target_visible(
                            session_id,
                            tenant_id=self._tenant_id,
                            execution_id=commit.execution.execution_id,
                            next_cursor=next_cursor,
                            run=conversation_agent_run,
                            checkpoint=conversation_checkpoint,
                        ):
                            break
            else:
                await self._materialize_prepared_checkpoint_with_reconciliation(
                    self._conversation_run_store,
                    conversation_agent_run,
                    prepared_conversation,
                )
                next_cursor = await self._pin_conversation_cursor(
                    self._conversation_run_store, next_cursor, prepared_conversation[0], None,
                )

                async def commit_conversation_state(
                    transaction: StateTransaction,
                ) -> None:
                    session = await self._conversation.get_in_transaction(
                        transaction,
                        session_id,
                        tenant_id=self._tenant_id,
                    )
                    await self._promote_history_in_transaction(
                        transaction,
                        session,
                        prepared_conversation[0],
                    )
                    await self._conversation.advance_continuation_in_transaction(
                        transaction,
                        session_id,
                        tenant_id=self._tenant_id,
                        execution_id=commit.execution.execution_id,
                        expected=expected_cursor,
                        next_cursor=next_cursor,
                        release_execution=False,
                        history_quality="complete",
                    )
                    if timeline_range is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

                for attempt in range(2):
                    try:
                        await self._conversation.state_store.mutate(
                            commit_conversation_state
                        )
                        break
                    except AIError as error:
                        if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN or attempt == 1:
                            raise
                        session = await self._load_session(
                            session_id,
                            tenant_id=self._tenant_id,
                        )
                        if session is not None and session.continuation == next_cursor:
                            break

        execution_stores = [self._execution.state_store]
        if execution_run is not None:
            if self._execution_run_store is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            execution_stores.extend(self._execution_run_store.transaction_stores)
        execution_run_store_in_transaction = False
        execution_prefix: "PreparedCapabilityWrites | None" = None

        async def commit_execution(group: StateGroupTransaction) -> ExecutionTerminalCommitResult:
            nonlocal effective_commit
            execution_transaction = group.transaction(self._execution.state_store)
            if commit.producer_generation is not None:
                execution_guard = await self._execution.require_execution_producer_in_transaction(
                    execution_transaction, commit.execution.execution_id,
                    expected_producer_generation=commit.producer_generation,
                )
            else:
                execution_guard = await self._execution.require_execution_history_write_in_transaction(
                    execution_transaction, commit.execution.execution_id,
                )
            effective_commit = await self._effective_terminal_commit(
                execution_transaction, commit, expected_execution=expected_execution, start_identity=start_identity,
            )
            if execution_run is not None and execution_run_store_in_transaction:
                execution_transaction_for_steps = group.transaction(
                    self._execution_run_store.state_store
                )
                if execution_projections:
                    for projection in execution_projections:
                        await self._execution_run_store.sync_projection_in_transaction(
                            execution_transaction_for_steps,
                            projection.run,
                            events=projection.events,
                            checkpoints=projection.checkpoints,
                            interactions=projection.interactions,
                            observation=projection.observation,
                            producer_generation=projection.producer_generation,
                            archive=projection.archive,
                            execution_id=commit.execution.execution_id,
                            execution_guard=execution_guard,
                        )
                else:
                    await self._execution_run_store.sync_projection_in_transaction(
                        execution_transaction_for_steps,
                        execution_run,
                        events=execution_events,
                        checkpoints=(
                            prepared_execution.checkpoints
                            if isinstance(
                                prepared_execution,
                                PreparedAgentRunCheckpointBatch,
                            )
                            else ()
                        ),
                        execution_id=commit.execution.execution_id,
                        execution_guard=execution_guard,
                    )
            if seal is not None:
                await self._execution.put_history_seal_in_transaction(execution_transaction, seal)
            if execution_prefix is not None:
                await execution_prefix.apply(group)
            return await self._execution.commit_terminal_in_transaction(
                execution_transaction,
                effective_commit,
                pending_events=audit_events,
            )

        async def execution_target_visible() -> bool:
            return await self._terminal_targets_visible(
                effective_commit,
                expected_execution=expected_execution,
                pending_event_count=len(audit_events),
                session_id=session_id,
                next_cursor=next_cursor,
                conversation_agent_run=conversation_agent_run,
                conversation_checkpoint=conversation_checkpoint,
                prepared_conversation_checkpoint=prepared_conversation[0] if prepared_conversation else None,
                execution_run=execution_run,
                execution_events=execution_events,
                execution_checkpoints=execution_checkpoints,
                execution_projections=execution_projections,
                audit_events=audit_events,
                require_session_release=False,
            )

        async def commit_execution_with_reconciliation(
            stores: Sequence[StateStore],
        ) -> ExecutionTerminalCommitResult:
            nonlocal execution_prefix
            execution_prefix = (await self._capability_writes.prepare_terminal(
                commit.execution.execution_id, capability_run_id, self._execution.state_store.storage_group,
            ) if self._capability_writes is not None else None)
            if execution_prefix is not None:
                stores = tuple(_dedupe_stores((*stores, *execution_prefix.stores)))
            try:
                async def readback() -> CommitObservation[ExecutionTerminalCommitResult]:
                    try:
                        if not await execution_target_visible():
                            return CommitObservation(
                                DurableCommitState.NOT_COMMITTED
                            )
                        if execution_prefix is not None and not await execution_prefix.readback():
                            return CommitObservation(
                                DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                                error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                            )
                        execution = await self._load_execution(
                            commit.execution.execution_id,
                            tenant_id=self._tenant_id,
                        )
                        if execution is None:
                            return CommitObservation(
                                DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                                error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                            )
                        return CommitObservation(
                            DurableCommitState.COMMITTED,
                            value=ExecutionTerminalCommitResult(
                                execution,
                                commit.result,
                            ),
                        )
                    except AIError as error:
                        if error.code is ErrorCode.EXECUTION_RESULT_CONFLICT:
                            return CommitObservation(DurableCommitState.NOT_COMMITTED, error=error)
                        if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                            return CommitObservation(
                                DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                                error=error,
                            )
                        return CommitObservation(
                            DurableCommitState.UNRESOLVED,
                            error=error,
                        )

                outcome = await run_durable_commit(
                    lambda: self._execution.state_store.storage_group.mutate(
                        stores,
                        commit_execution,
                    ),
                    readback,
                    background_tasks=owner_tasks,
                )
                if outcome.state is DurableCommitState.COMMITTED:
                    if outcome.value is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if execution_prefix is not None:
                        await execution_prefix.settle()
                    if outcome.cancelled:
                        raise asyncio.CancelledError
                    return outcome.value
                if outcome.state is DurableCommitState.NOT_COMMITTED:
                    if outcome.error is not None:
                        raise outcome.error
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                if outcome.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
                    raise AIError(
                        ErrorCode.STORAGE_INTEGRITY_ERROR,
                        "terminal checkpoint left partial durable state",
                    ) from outcome.error
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from outcome.error

            finally:
                if execution_prefix is not None:
                    await execution_prefix.abort()

        if await execution_target_visible():
            if self._capability_writes is not None:
                prefix = await self._capability_writes.prepare_terminal(
                    commit.execution.execution_id, capability_run_id, self._execution.state_store.storage_group,
                )
                try:
                    if not await prefix.readback():
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    await prefix.settle()
                finally:
                    await prefix.abort()
            execution = await self._load_execution(
                commit.execution.execution_id,
                tenant_id=self._tenant_id,
            )
            if execution is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result = ExecutionTerminalCommitResult(execution, commit.result)
        elif _same_group(execution_stores):
            execution_run_store_in_transaction = execution_run is not None
            result = await commit_execution_with_reconciliation(execution_stores)
        else:
            if execution_projections:
                for projection in execution_projections:
                    await self._sync_prepared_projection_with_reconciliation(
                        self._execution_run_store,
                        projection,
                        execution_id=commit.execution.execution_id,
                    )
            elif execution_run is not None:
                await self._sync_projection_with_reconciliation(
                    execution_run,
                    execution_id=commit.execution.execution_id,
                    events=execution_events,
                    checkpoints=execution_checkpoints,
                )
            terminal_stores = [self._execution.state_store]
            result = await commit_execution_with_reconciliation(terminal_stores)
        if session_id is not None:
            async def release_conversation(transaction: StateTransaction) -> None:
                if publish_turn:
                    await self._conversation.publish_and_release_turn_in_transaction(
                        transaction, session_id, tenant_id=self._tenant_id,
                        execution_id=commit.execution.execution_id,
                        start_message_index=0 if timeline_range is None else timeline_range[0],
                        end_message_index=0 if timeline_range is None else timeline_range[1],
                        history_unavailable=commit.execution.invalidates_session_history,
                        allow_unadmitted=commit.execution.started_at is None,
                    )
                else:
                    await self._conversation.release_execution_in_transaction(
                        transaction, session_id, tenant_id=self._tenant_id,
                        execution_id=commit.execution.execution_id,
                        history_unavailable=commit.execution.invalidates_session_history,
                    )

            for attempt in range(2):
                try:
                    await self._conversation.state_store.mutate(release_conversation)
                    break
                except AIError as error:
                    if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN or attempt == 1:
                        raise
                    session = await self._load_session(session_id, tenant_id=self._tenant_id)
                    if session is not None and session.active_execution_id is None:
                        if publish_turn:
                            await self.verify_conversation_turn(result.execution, result.result)
                        break
        return result

    async def finish_conversation_turn(
        self, execution: ExecutionRecord, result: ResultRecord | None,
        *, message_range: tuple[int, int] | None = None,
    ) -> None:
        """Finish a retained turn from its durable terminal outcome before releasing admission."""
        if execution.session_id is None or not execution.conversation:
            return
        if execution.status is ExecutionStatus.SUCCEEDED and message_range is None:
            await self.verify_conversation_turn(execution, result)
            return
        if result is None or execution.completed_at is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        start, end = (0, 0) if message_range is None else message_range

        async def finish(transaction: StateTransaction) -> None:
            await self._conversation.publish_and_release_turn_in_transaction(
                transaction, execution.session_id, tenant_id=self._tenant_id,
                execution_id=execution.execution_id, start_message_index=start, end_message_index=end,
                history_unavailable=execution.invalidates_session_history,
                allow_unadmitted=execution.started_at is None,
            )

        try:
            await self._conversation.state_store.mutate(finish)
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise
            await self.verify_conversation_turn(execution, result)

    async def verify_conversation_turn(
        self, execution: ExecutionRecord, result: ResultRecord | None,
    ) -> None:
        if execution.session_id is None or not execution.conversation:
            return
        if result is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        value = await self._conversation.state_store.read(
            lambda transaction: self._conversation.pending_turn_commit_in_transaction(
                transaction, session_id=execution.session_id, execution_id=execution.execution_id,
                allow_unadmitted=execution.started_at is None,
            )
        )
        if value is None:
            return
        if (execution.completed_at is None or execution.result != result
                or value.execution_id != execution.execution_id or value.session_id != execution.session_id):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _materialize_prepared_checkpoint_with_reconciliation(
        self,
        archive: StateStepArchive,
        run: AgentRunRecord,
        batch: PreparedAgentRunCheckpointBatch,
        *,
        execution_id: str | None = None,
    ) -> None:
        if len(batch) != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        prepared = batch[0]
        if await self._prepared_checkpoint_visible(archive, run, prepared):
            return

        await archive.prepare_run_registration(run)

        async def materialize(transaction: StateTransaction) -> None:
            await archive.materialize_checkpoint_in_transaction(
                transaction,
                run,
                prepared,
                execution_id=execution_id,
            )

        async def readback() -> CommitObservation[None]:
            try:
                visible = await self._prepared_checkpoint_visible(archive, run, prepared)
            except AIError as error:
                if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=error,
                    )
                return CommitObservation(DurableCommitState.UNRESOLVED, error=error)
            return CommitObservation(
                DurableCommitState.COMMITTED
                if visible
                else DurableCommitState.NOT_COMMITTED
            )

        async def operation() -> None:
            await archive.state_store.storage_group.mutate(
                archive.transaction_stores,
                lambda group: materialize(group.transaction(archive.state_store)),
            )

        await self._commit_or_raise(operation, readback)

    async def _sync_prepared_projection_with_reconciliation(
        self,
        archive: StateStepArchive | None,
        projection: PreparedExecutionProjection,
        *,
        execution_id: str,
    ) -> None:
        if archive is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        async def sync() -> None:
            await archive.sync_prepared_projection(
                projection.run,
                events=projection.events,
                checkpoints=projection.checkpoints,
                interactions=projection.interactions,
                observation=projection.observation,
                producer_generation=projection.producer_generation,
                archive=projection.archive,
                execution_id=execution_id,
            )

        async def readback() -> CommitObservation[None]:
            try:
                visible = await archive.verify_execution_projection_head(projection)
            except AIError as error:
                if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=error,
                    )
                return CommitObservation(DurableCommitState.UNRESOLVED, error=error)
            return CommitObservation(
                DurableCommitState.COMMITTED
                if visible
                else DurableCommitState.NOT_COMMITTED
            )

        await self._commit_or_raise(sync, readback)

    async def _prepared_checkpoint_visible(
        self,
        archive: StateStepArchive,
        run: AgentRunRecord,
        prepared: PreparedAgentRunCheckpoint,
    ) -> bool:
        if await archive.get_agent_run(agent_run_id=run.agent_run_id) != run:
            return False
        checkpoint = await archive.latest_checkpoint(
            agent_run_id=run.agent_run_id,
            include_interrupted=True,
        )
        if checkpoint is None:
            return False
        sequence = await archive.latest_checkpoint_sequence(agent_run_id=run.agent_run_id)
        stored = await archive.checkpoint_at_sequence(agent_run_id=run.agent_run_id, checkpoint_sequence=sequence)
        return (
            checkpoint.step_index == prepared.stored.step_index
            and checkpoint.timestamp == prepared.stored.timestamp
            and checkpoint.state == prepared.stored.state
            and stored == prepared.stored
        )

    async def _step_checkpoint_visible(
        self,
        archive: StateStepArchive,
        run: AgentRunRecord,
        checkpoint: AgentRunCheckpoint,
    ) -> bool:
        return (
            await archive.get_agent_run(agent_run_id=run.agent_run_id) == run
            and await archive.verify_checkpoint_projection(
                agent_run_id=run.agent_run_id,
                checkpoint=checkpoint,
            )
        )

    async def _conversation_target_visible(
        self,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
        next_cursor: ConversationCursor,
        run: AgentRunRecord,
        checkpoint: AgentRunCheckpoint,
    ) -> bool:
        session = await self._load_session(session_id, tenant_id=tenant_id)
        return (
            session is not None
            and session.active_execution_id == execution_id
            and session.continuation == next_cursor
            and self._conversation_run_store is not None
            and await self._step_checkpoint_visible(
                self._conversation_run_store,
                run,
                checkpoint,
            )
        )

    async def _sync_projection_with_reconciliation(
        self,
        run: AgentRunRecord,
        *,
        execution_id: str,
        events: Sequence[StepEvent],
        checkpoints: Sequence[AgentRunCheckpoint],
    ) -> None:
        if self._execution_run_store is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        async def operation() -> None:
            await self._execution_run_store.sync_projection(
                run,
                events=events,
                checkpoints=checkpoints,
                execution_id=execution_id,
            )

        async def readback() -> CommitObservation[None]:
            try:
                stored_run = await self._execution_run_store.get_agent_run(agent_run_id=run.agent_run_id)
                stored_events = await self._execution_run_store.list_events(
                    agent_run_id=run.agent_run_id
                )
                stored_checkpoint = await self._execution_run_store.latest_checkpoint(
                    agent_run_id=run.agent_run_id,
                    include_interrupted=True,
                )
            except AIError as error:
                if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=error,
                    )
                return CommitObservation(
                    DurableCommitState.UNRESOLVED,
                    error=error,
                )
            if stored_run is not None and stored_run != run:
                return CommitObservation(
                    DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                    error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                )
            visible = (
                stored_run == run
                and (
                    not events
                    or tuple(stored_events[-len(events) :]) == tuple(events)
                )
                and (not checkpoints or stored_checkpoint == checkpoints[-1])
            )
            return CommitObservation(
                DurableCommitState.COMMITTED
                if visible
                else DurableCommitState.NOT_COMMITTED
            )

        result = await run_durable_commit(
            operation,
            readback,
            background_tasks=self._background_tasks,
        )
        if result.state is DurableCommitState.COMMITTED:
            if result.cancelled:
                raise asyncio.CancelledError
            return
        if result.state is DurableCommitState.NOT_COMMITTED:
            if result.error is not None:
                raise result.error
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if result.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                "execution projection left partial durable state",
            ) from result.error
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from result.error

    def _require_conversation(self) -> None:
        if self._conversation is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    def _require_checkpoint(self) -> None:
        if self._checkpoint is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    async def _effective_terminal_commit(
        self,
        transaction: StateTransaction,
        commit: ExecutionTerminalCommit,
        *,
        expected_execution: ExecutionRecord | None = None,
        start_identity: IdempotencyRecord | None = None,
    ) -> ExecutionTerminalCommit:
        commit = replace(commit, execution=commit.execution.finalize_terminal(commit.result))
        if expected_execution is not None:
            current = await self._execution.get_in_transaction(
                transaction, commit.execution.execution_id, tenant_id=self._tenant_id,
            )
            if current is None or not _same_execution_except_hold_metadata(current, expected_execution):
                raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
            commit = replace(
                commit,
                expected_revision=current.revision,
                execution=replace(
                    commit.execution,
                    dependency_hold_ids=current.dependency_hold_ids,
                    updated_at=max(commit.execution.updated_at, current.updated_at),
                ),
            )
        if commit.idempotency is not None:
            return commit
        return replace(
            commit,
            idempotency=await self._execution.terminal_idempotency_in_transaction(
                transaction,
                commit, **({"identity": start_identity} if type(self._execution) is ExecutionRepositoryImpl else {}),
            ),
        )

    async def _reconcile_start_unknown(
        self,
        claim: ExecutionStartClaim,
        *,
        session_id: str | None,
        expected_cursor: ConversationCursor | None,
        expected_session: SessionRecord | None = None,
    ) -> ExecutionRecord | None:
        execution = await self._load_execution(
            claim.execution_id,
            tenant_id=self._tenant_id,
        )
        if execution is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            execution.status is ExecutionStatus.PENDING_START
            and execution.revision == claim.expected_revision
            and execution.event_seq == claim.expected_event_seq
        ):
            return None
        if (
            execution.status is not ExecutionStatus.STARTED
            or execution.revision != claim.expected_revision + 1
            or execution.event_seq != claim.expected_event_seq + 1
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        idempotency = await self._execution.get_start_idempotency(claim)
        if idempotency is None or idempotency.status is not IdempotencyStatus.STARTED:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        started_events = await self._events.list(
            claim.execution_id,
            tenant_id=self._tenant_id,
            after_event_seq=claim.expected_event_seq,
            limit=1,
        )
        if (
            len(started_events.items) != 1
            or started_events.items[0].event_seq != claim.expected_event_seq + 1
            or started_events.items[0].event_type
            != ExecutionEventType.EXECUTION_STARTED.value
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if session_id is not None:
            session = await self._load_session(session_id, tenant_id=self._tenant_id)
            if (session is None or session.session_id != session_id or session.active_execution_id != claim.execution_id
                    or session.continuation != expected_cursor):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if expected_session is not None:
                _require_session_capture(session, expected_session)
        return execution

    async def _terminal_targets_visible(
        self,
        commit: ExecutionTerminalCommit,
        *,
        expected_execution: ExecutionRecord | None = None,
        pending_event_count: int,
        session_id: str | None,
        next_cursor: ConversationCursor | None,
        conversation_agent_run: AgentRunRecord | None,
        conversation_checkpoint: AgentRunCheckpoint | None,
        prepared_conversation_checkpoint: PreparedAgentRunCheckpoint | None,
        execution_run: AgentRunRecord | None,
        execution_events: Sequence[StepEvent],
        execution_checkpoints: Sequence[AgentRunCheckpoint],
        execution_projections: Sequence[PreparedExecutionProjection],
        audit_events: Sequence[ExecutionEventAppend],
        require_session_release: bool = True,
    ) -> bool:
        execution = await self._load_execution(
            commit.execution.execution_id,
            tenant_id=self._tenant_id,
        )
        if execution is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            execution.status not in {
                ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
            }
            if expected_execution is not None
            else execution.revision == commit.expected_revision
            and execution.event_seq == commit.expected_event_seq
        ):
            seal = await self._execution.get_history_seal(execution.execution_id, tenant_id=self._tenant_id)
            if execution.history_seal_digest is not None or seal is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if expected_execution is not None and not _same_execution_except_hold_metadata(execution, expected_execution):
                raise AIError(ErrorCode.EXECUTION_RESULT_CONFLICT)
            return False
        expected_revision = commit.expected_revision + pending_event_count + 1
        expected_event_seq = commit.expected_event_seq + pending_event_count + 1
        expected_terminal = replace(
            commit.execution,
            revision=expected_revision,
            event_seq=expected_event_seq,
            result=commit.result,
        ).finalize_terminal(commit.result)
        if not _same_execution_except_hold_metadata(execution, expected_terminal):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        history_seal = await self._execution.get_history_seal(
            commit.execution.execution_id,
            tenant_id=self._tenant_id,
        ) if execution.durable_trace else None
        if execution.durable_trace and history_seal is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution.history_seal_digest != (None if history_seal is None else history_seal.seal_digest):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution.durable_trace and (execution_projections or execution_run is None):
            expected_seal = _execution_history_seal(
                commit,
                audit_events=audit_events,
                projections=execution_projections,
                current_run=execution_run,
                current_events=execution_events,
                current_batch=None,
            )
            if history_seal != expected_seal:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution_projections:
            if self._execution_run_store is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for projection in execution_projections:
                if not await self._execution_run_store.verify_execution_projection_head(
                    projection
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        idempotency = await self._execution.get_terminal_idempotency(
            commit.execution.execution_id,
            tenant_id=self._tenant_id,
        )
        if idempotency is not None:
            expected_status = (
                IdempotencyStatus.COMPLETED
                if commit.execution.status is ExecutionStatus.SUCCEEDED
                else IdempotencyStatus.CANCELLED
                if commit.execution.status is ExecutionStatus.CANCELLED
                else IdempotencyStatus.FAILED
            )
            expected_digest = None if commit.result.output is None else commit.result.output.digest
            if (
                idempotency.status is not expected_status
                or idempotency.result_digest != expected_digest
                or idempotency.error_code != commit.execution.error_code
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if commit.operation is not None:
            if self._operations is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            operation = await self._operations.get(
                commit.operation.operation_id,
                tenant_id=self._tenant_id,
            )
            if (
                operation is None
                or operation.status is not commit.operation.next_status
                or operation.result_ref != commit.operation.result_ref
                or operation.result_digest != commit.operation.result_digest
                or operation.error_code != commit.operation.error_code
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        events = await self._events.list(
            commit.execution.execution_id,
            tenant_id=self._tenant_id,
            after_event_seq=commit.expected_event_seq,
            limit=pending_event_count + 1,
        )
        expected_events = tuple(audit_events) + (
            ExecutionEventAppend(commit.terminal_event_type, commit.terminal_event_payload),
        )
        if len(events.items) != len(expected_events) or any(
            actual.event_seq != commit.expected_event_seq + index + 1
            or actual.event_type != expected.event_type
            or actual.payload != expected.payload
            for index, (actual, expected) in enumerate(
                zip(events.items, expected_events, strict=True)
            )
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if session_id is not None:
            session = await self._load_session(
                session_id,
                tenant_id=self._tenant_id,
            )
            if session is None or require_session_release and session.active_execution_id is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if (
                require_session_release
                and commit.execution.invalidates_session_history
                and session.history_quality != "unavailable"
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if (conversation_agent_run is not None and next_cursor is not None
                    and next_cursor.checkpoint_sequence is None and next_cursor.projection_digest is None
                    and prepared_conversation_checkpoint is not None):
                archive = self._conversation_run_store
                if archive is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                next_cursor = await self._pin_conversation_cursor(
                    archive, next_cursor, prepared_conversation_checkpoint, None,
                )
            if conversation_agent_run is not None and (
                next_cursor is None or session.continuation != next_cursor
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if session_id is not None and commit.execution.conversation and require_session_release:
            await self.verify_conversation_turn(execution, commit.result)
        if conversation_agent_run is not None or conversation_checkpoint is not None:
            archive = self._conversation_run_store
            run, checkpoint = conversation_agent_run, conversation_checkpoint
            if archive is None or run is None or checkpoint is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            stored_run = await archive.get_agent_run(agent_run_id=run.agent_run_id)
            if not await archive.verify_checkpoint_projection(
                agent_run_id=run.agent_run_id,
                checkpoint=checkpoint,
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if stored_run != run:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if execution_run is not None:
            if self._execution_run_store is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            stored_run = await self._execution_run_store.get_agent_run(agent_run_id=execution_run.agent_run_id)
            stored_events = await self._execution_run_store.list_events(agent_run_id=execution_run.agent_run_id)
            if stored_run != execution_run:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if execution_events and tuple(stored_events[-len(execution_events) :]) != tuple(execution_events):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            stored_checkpoint = await self._execution_run_store.latest_checkpoint(
                agent_run_id=execution_run.agent_run_id,
                include_interrupted=True,
            )
            if execution_checkpoints and stored_checkpoint != execution_checkpoints[-1]:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if execution_checkpoints and not await self._execution_run_store.verify_checkpoint_projection(
                agent_run_id=execution_run.agent_run_id,
                checkpoint=execution_checkpoints[-1],
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return True

    async def _admit_split_start(
        self,
        session_id: str,
        *,
        tenant_id: str,
        execution_id: str,
        expected: ConversationCursor | None,
        expected_session: SessionRecord | None = None,
    ) -> None:
        session = await self._conversation.get(session_id, tenant_id=tenant_id)
        if session is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if expected_session is not None:
            _require_session_capture(session, expected_session)
        owner_id = session.active_execution_id
        if owner_id not in {None, execution_id}:
            owner = await self._execution.get(owner_id, tenant_id=tenant_id)
            if (
                owner is None
                or owner.session_id != session_id
                or owner.parent_execution_id is not None
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if owner.status not in {
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }:
                if session.status is not SessionStatus.OPEN:
                    raise AIError(ErrorCode.SESSION_CONFLICT)
                raise AIError(ErrorCode.SESSION_BUSY)
            await self._conversation.release_execution(
                session_id,
                tenant_id=tenant_id,
                execution_id=owner.execution_id,
                history_unavailable=owner.invalidates_session_history,
                require_turn_commit=owner.conversation,
            )
        await self._conversation.admit_execution(
            session_id,
            tenant_id=tenant_id,
            execution_id=execution_id,
            expected=expected,
            expected_session=expected_session,
        )



def _require_session_capture(current: SessionRecord, expected: SessionRecord) -> None:
    if current.session_id != expected.session_id:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if current.owner_principal_id != expected.owner_principal_id:
        raise AIError(ErrorCode.AUTHORIZATION_DENIED)
    if current.agent_id != expected.agent_id:
        raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)
    if current.cwd != expected.cwd:
        raise AIError(ErrorCode.SESSION_CONFLICT)


def _dedupe_stores(stores: Sequence[StateStore]) -> tuple[StateStore, ...]:
    result: list[StateStore] = []
    seen: set[int] = set()
    for store in stores:
        identity = id(store)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(store)
    return tuple(result)


def _partial_integrity(
    error: AIError | None = None,
) -> CommitObservation[object]:
    return CommitObservation(
        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
        error=error or AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
    )


def _require_durable_record(
    outcome: DurableCommitResult[ExecutionRecord],
) -> ExecutionRecord:
    if outcome.state is DurableCommitState.COMMITTED:
        if outcome.value is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if outcome.cancelled:
            raise asyncio.CancelledError
        return outcome.value
    if outcome.state is DurableCommitState.NOT_COMMITTED:
        if outcome.error is not None:
            raise outcome.error
        raise AIError(ErrorCode.STORAGE_CONFLICT)
    if outcome.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from outcome.error
    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from outcome.error


def _same_group(stores: Sequence[StateStore]) -> bool:
    return bool(stores) and all(store.storage_group is stores[0].storage_group for store in stores[1:])






__all__ = ["StateCommands"]
