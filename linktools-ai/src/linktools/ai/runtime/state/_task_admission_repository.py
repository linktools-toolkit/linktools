#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable TaskGraph admission repository."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import TypeVar

from linktools.core import environ

from ...core import (
    OperationKind,
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationStatus,
    Page,
    ResourceKind,
    TaskStatus,
    canonical_sha256,
)
from ...errors import AIError, ErrorCode
from ...task import (
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphSubmission,
    TaskSubmissionRef,
    TaskGraphLaunch,
    TaskGraphView,
    TaskNode,
    TaskNodeView,
)
from ._durability import CommitObservation, DurableCommitState, run_durable_commit
from ._plan import RuntimeDomain
from ._repositories import (
    RepositoryBase,
    append_operation,
    decode_operation,
    decode_record_cursor,
    projected_record,
    record_cursor,
    replace_checked,
)
from ._store import (
    RecordQuery,
    StateStore,
    StateTransaction,
    StoredOperation,
    StoredRecord,
    operation_key,
    sortable_identity,
)
from ._task_events import (
    _TaskEventState,
    _append_task_events,
    _append_task_state_events,
    _guard_task_event_owner,
    _task_graph_event_drafts,
)
from ._task_state import (
    _effective_graph_status,
    _is_sha256,
    _isolated_graph_status,
    _require_canonical_graph_status,
)

_ValueT = TypeVar("_ValueT")

_logger = environ.get_logger("ai.runtime.state.task_repository")
_RECOVERABLE_GRAPH_STATES = frozenset(
    {
        TaskStatus.PENDING.value,
        TaskStatus.RUNNING.value,
    }
)


def _task_submit_result_digest(graph: TaskGraph) -> str:
    states = tuple(
        TaskNodeView(
            graph.graph_id,
            node.node_id,
            node.dependencies,
            (
                node.dependency_status({})
                if not node.dependencies
                else TaskStatus.PENDING
            ),
            None,
            0,
            None,
            None,
            ErrorCode.TASK_DEPENDENCY_FAILED.value
            if not node.dependencies
            and node.dependency_status({}) is TaskStatus.BLOCKED
            else None,
            None,
        )
        for node in graph.nodes
    )
    status = _isolated_graph_status(states, graph.nodes)
    return canonical_sha256({"graph_id": graph.graph_id, "status": status.value})


def _same_task_admission_contract(
    left: TaskGraphAdmission,
    right: TaskGraphAdmission,
) -> bool:
    return (
        left.version == right.version
        and left.graph_id == right.graph_id
        and left.principal == right.principal
        and left.limits == right.limits
        and left.operation_id == right.operation_id
        and left.initial_request_digest == right.initial_request_digest
    )


class _SubmissionConflict(AIError):
    def __init__(self) -> None:
        super().__init__(ErrorCode.STORAGE_CONFLICT)


class TaskAdmissionRepositoryImpl(RepositoryBase):
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=RuntimeDomain.TASK,
        )
        self._background_tasks: set[asyncio.Task[object]] = set()

    def _graph_key(self, graph_id: str) -> bytes:
        return self._key("task_graph", graph_id)

    def _definition_key(self, graph_id: str, node_id: str) -> bytes:
        return self._key("task_node_definition", [graph_id, node_id])

    def _state_key(self, graph_id: str, node_id: str) -> bytes:
        return self._key("task_node_state", [graph_id, node_id])

    def _definition_parent(self, graph_id: str) -> bytes:
        return self._parent("task_node_definition", "graph", graph_id)

    def _state_parent(self, graph_id: str) -> bytes:
        return self._parent("task_node_state", "graph", graph_id)

    def _admission_key(self, graph_id: str) -> bytes:
        return self._key("task_admission", graph_id)

    def _recovery_scope(self) -> bytes:
        return self._scope("task_admission", "recoverable", "graphs")

    def _submission_key(self, graph_id: str) -> bytes:
        return self._key("task_submission", graph_id)

    def _prepared_key(self, graph_id: str) -> bytes:
        return self._key("task_submission_payload", graph_id)

    async def _replace_submission(
        self, transaction: StateTransaction, record: StoredRecord, state: str,
    ) -> None:
        if not await transaction.replace_record(
            replace(record, state=state, storage_version=record.storage_version + 1),
            expected_storage_version=record.storage_version,
        ):
            raise _SubmissionConflict()

    async def _submission_in_transaction(
        self,
        transaction: StateTransaction,
        submission: TaskSubmissionRef,
    ) -> StoredRecord | None:
        if (
            submission.namespace != self._namespace
            or submission.tenant_id != self._tenant_id
        ):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        record = await transaction.get_record(self._submission_key(submission.graph_id))
        if record is None:
            return None
        stored = await self._decode(record, TaskSubmissionRef)
        if (
            record.kind != "task_submission"
            or record.scope_digest is not None
            or record.parent_digest is not None
            or record.sort_key != sortable_identity(submission.graph_id)
            or record.state not in {"prepared", "admitted", "cancelled"}
            or stored.namespace != self._namespace
            or stored.tenant_id != self._tenant_id
            or stored.graph_id != submission.graph_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if stored != submission:
            if stored.operation_id != submission.operation_id:
                graph = await transaction.get_record(self._graph_key(submission.graph_id))
                if graph is not None:
                    admission = await transaction.get_record(self._admission_key(submission.graph_id))
                    if admission is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    await self._require_committed_admission(transaction, graph, admission)
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        return record

    async def submission_status(self, submission: TaskSubmissionRef) -> str | None:
        async def read(transaction: StateTransaction) -> str | None:
            record = await self._submission_in_transaction(transaction, submission)
            return None if record is None else record.state
        return await self._store.read(read)

    async def prepare(
        self, submission: TaskGraphSubmission
    ) -> TaskGraphSubmission:
        async def mutate(transaction: StateTransaction) -> TaskGraphSubmission:
            record = await self._submission_in_transaction(transaction, submission.ref)
            if record is None:
                await transaction.insert_records((
                    self._stored(
                        "task_submission", submission.graph.graph_id,
                        submission.ref, state="prepared",
                    ),
                    self._stored(
                        "task_submission_payload", submission.graph.graph_id, submission,
                    ),
                ))
                return submission
            return await self._prepared_submission(transaction, record, submission)

        async def readback() -> CommitObservation[TaskGraphSubmission]:
            async def read(transaction: StateTransaction) -> CommitObservation[TaskGraphSubmission]:
                record = await self._submission_in_transaction(transaction, submission.ref)
                if record is None:
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                stored = await self._prepared_submission(transaction, record, submission)
                return CommitObservation(DurableCommitState.COMMITTED, stored)
            return await self._store.read(read)

        return await self._commit(lambda: self._store.mutate(mutate), readback)

    async def _prepared_submission(
        self,
        transaction: StateTransaction,
        head: StoredRecord,
        candidate: TaskGraphSubmission,
    ) -> TaskGraphSubmission:
        if head.state != "prepared":
            return candidate
        payload = await transaction.get_record(self._prepared_key(candidate.graph.graph_id))
        if payload is None or payload.kind != "task_submission_payload":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        stored = await self._decode(payload, TaskGraphSubmission)
        if stored.ref != candidate.ref:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return stored

    async def cancel_submission(
        self,
        submission: TaskSubmissionRef,
        operation: OperationLedgerInput,
    ) -> bool:
        if (
            operation.tenant_id != self._tenant_id
            or operation.resource_kind is not ResourceKind.TASK_GRAPH
            or operation.resource_id != submission.graph_id
            or operation.execution_id is not None
            or operation.operation_kind is not OperationKind.TASK_CANCEL
            or operation.status is not OperationStatus.PENDING
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        async def mutate(transaction: StateTransaction) -> bool:
            record = await self._submission_in_transaction(transaction, submission)
            if record is None:
                record = self._stored("task_submission", submission.graph_id, submission, state="cancelled")
                await transaction.insert_records((record,))
            graph_record = await transaction.get_record(self._graph_key(submission.graph_id))
            admitted = graph_record is not None
            if (record.state == "admitted" and not admitted) or (
                record.state == "prepared" and admitted
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if record.state == "prepared":
                payload = await transaction.get_record(self._prepared_key(submission.graph_id))
                if payload is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                prepared = await self._decode(payload, TaskGraphSubmission)
                if prepared.ref != submission:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await transaction.delete_record(payload.key_digest)
            if record.state != "cancelled":
                await self._replace_submission(transaction, record, "cancelled")
            if admitted:
                self._validate_graph_record(graph_record, submission.graph_id)
                if await transaction.guard_record(
                    graph_record.key_digest,
                    expected_storage_version=graph_record.storage_version,
                ) is None:
                    raise _SubmissionConflict()
                pending = operation
            else:
                pending = replace(
                    operation,
                    status=OperationStatus.SUCCEEDED,
                    result_ref=submission.graph_id,
                    result_digest=canonical_sha256({
                        "graph_id": submission.graph_id,
                        "status": TaskStatus.CANCELLED.value,
                    }),
                )
            await append_operation(transaction, self, pending)
            return admitted

        async def readback() -> CommitObservation[bool]:
            async def read(transaction: StateTransaction) -> CommitObservation[bool]:
                record = await self._submission_in_transaction(transaction, submission)
                if record is None or record.state != "cancelled":
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                stored = await transaction.get_operation(operation_key(
                    self._namespace, self._tenant_id, self._domain.value,
                    operation.operation_id,
                ))
                if stored is None:
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                if decode_operation(stored).request_digest != operation.request_digest:
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                graph = await transaction.get_record(self._graph_key(submission.graph_id))
                return CommitObservation(DurableCommitState.COMMITTED, graph is not None)
            return await self._store.read(read)

        return await self._commit(lambda: self._store.mutate(mutate), readback)

    async def _commit(
        self,
        operation: Callable[[], Awaitable[_ValueT]],
        readback: Callable[[], Awaitable[CommitObservation[_ValueT]]],
    ) -> _ValueT:
        for attempt in range(8):
            result = await run_durable_commit(
                operation, readback, background_tasks=self._background_tasks,
            )
            if (
                result.state is DurableCommitState.NOT_COMMITTED
                and isinstance(result.error, _SubmissionConflict)
                and not result.cancelled
                and attempt < 7
            ):
                await asyncio.sleep(0)
                continue
            break
        if result.state is DurableCommitState.COMMITTED and result.value is not None:
            if result.cancelled:
                raise asyncio.CancelledError
            return result.value
        if result.cancelled:
            raise asyncio.CancelledError
        if isinstance(result.error, AIError) and result.error.code in {
            ErrorCode.IDEMPOTENCY_CONFLICT,
            ErrorCode.STORAGE_OWNER_MISMATCH,
            ErrorCode.STORAGE_NOT_FOUND,
            ErrorCode.STORAGE_CONFLICT,
        }:
            raise result.error
        if result.state in {
            DurableCommitState.NOT_COMMITTED,
            DurableCommitState.PARTIAL_INTEGRITY_ERROR,
        } and result.error is not None:
            raise result.error
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from result.error

    async def _current_graph_in_transaction(
        self,
        transaction: StateTransaction,
        graph_id: str,
    ) -> tuple[TaskGraphView, tuple[TaskNodeView, ...]]:
        graph_record = await transaction.get_record(self._graph_key(graph_id))
        if graph_record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._validate_graph_record(graph_record, graph_id)
        header = await self._decode(graph_record, TaskGraphView)
        if header.graph_id != graph_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        definition_records = await transaction.list_records(
            RecordQuery(
                parent_digest=self._definition_parent(graph_id),
                kind="task_node_definition",
            )
        )
        state_records = await transaction.list_records(
            RecordQuery(
                parent_digest=self._state_parent(graph_id),
                kind="task_node_state",
            )
        )
        definitions: dict[str, TaskNode] = {}
        for record in definition_records:
            value = await self._decode(record, TaskNode)
            self._validate_definition_record(record, graph_id, value.node_id)
            if value.node_id in definitions:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            definitions[value.node_id] = value
        states: dict[str, TaskNodeView] = {}
        for record in state_records:
            value = await self._decode(record, TaskNodeView)
            self._validate_state_record(record, graph_id, value.node_id)
            if value.node_id in states:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            states[value.node_id] = value
        if set(definitions) != set(states):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        nodes = tuple(definitions[node_id] for node_id in sorted(definitions))
        ordered_states = tuple(
            replace(states[node.node_id], dependencies=node.dependencies)
            for node in nodes
        )
        TaskGraph(graph_id, nodes)
        return (
            TaskGraphView(
                graph_id,
                _effective_graph_status(header, ordered_states, nodes),
                nodes,
            ),
            ordered_states,
        )

    def _validate_graph_record(self, record: StoredRecord, graph_id: str) -> None:
        if (
            record.kind != "task_graph"
            or record.key_digest != self._graph_key(graph_id)
            or record.scope_digest is not None
            or record.parent_digest is not None
            or record.sort_key != sortable_identity(graph_id)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _validate_admission_record(
        self,
        record: StoredRecord,
        graph_id: str,
    ) -> None:
        if (
            record.kind != "task_admission"
            or record.key_digest != self._admission_key(graph_id)
            or record.scope_digest != self._recovery_scope()
            or record.parent_digest is not None
            or record.sort_key != sortable_identity(graph_id)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _validate_definition_record(
        self,
        record: StoredRecord,
        graph_id: str,
        node_id: str,
    ) -> None:
        if (
            record.kind != "task_node_definition"
            or record.key_digest != self._definition_key(graph_id, node_id)
            or record.scope_digest is not None
            or record.parent_digest != self._definition_parent(graph_id)
            or record.sort_key != sortable_identity([graph_id, node_id])
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _validate_state_record(
        self,
        record: StoredRecord,
        graph_id: str,
        node_id: str,
    ) -> None:
        if (
            record.kind != "task_node_state"
            or record.key_digest != self._state_key(graph_id, node_id)
            or record.scope_digest is not None
            or record.parent_digest != self._state_parent(graph_id)
            or record.sort_key != sortable_identity([graph_id, node_id])
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def admit(
        self,
        admission: TaskGraphAdmission,
        graph: TaskGraph,
    ) -> TaskGraphView:
        return await self._admit(admission, graph, require_prepared=False)

    async def admit_prepared(
        self, submission: TaskGraphSubmission,
    ) -> TaskGraphView:
        if submission.namespace != self._namespace:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        return await self._admit(
            submission.admission, submission.graph, require_prepared=True,
        )

    async def _admit(
        self,
        admission: TaskGraphAdmission,
        graph: TaskGraph,
        *,
        require_prepared: bool,
    ) -> TaskGraphView:
        admission.validate_graph(graph)
        launch = admission.launch()
        if launch.principal.tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def operation() -> TaskGraphView:
            return await self._store.mutate(
                lambda transaction: self._admit_in_transaction(
                    transaction,
                    admission,
                    graph,
                    require_prepared=require_prepared,
                )
            )

        async def readback() -> CommitObservation[TaskGraphView]:
            return await self._store.read(
                lambda transaction: self._read_admission(
                    transaction,
                    admission,
                    graph,
                )
            )

        return await self._commit(operation, readback)

    async def get(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphAdmission | None:
        if tenant_id != self._tenant_id:
            return None

        async def read(transaction: StateTransaction) -> TaskGraphAdmission | None:
            graph_key = self._graph_key(graph_id)
            admission_key = self._admission_key(graph_id)
            records = await transaction.get_records((graph_key, admission_key))
            graph_record = records.get(graph_key)
            admission_record = records.get(admission_key)
            if graph_record is None and admission_record is None:
                return None
            if graph_record is None or admission_record is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._validate_graph_record(graph_record, graph_id)
            self._validate_admission_record(admission_record, graph_id)
            admission = await self._decode(admission_record, TaskGraphAdmission)
            if (
                admission.graph_id != graph_id
                or admission.principal.tenant_id != tenant_id
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            stored_operation = await transaction.get_operation(
                operation_key(
                    self._namespace,
                    self._tenant_id,
                    self._domain.value,
                    admission.operation_id,
                )
            )
            if stored_operation is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            existing, _ = await self._require_committed_admission(
                transaction,
                graph_record,
                admission_record,
                stored_operation=stored_operation,
            )
            if existing != admission:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return admission

        return await self.state_store.read(read)

    async def list_recoverable_page(
        self,
        *,
        cursor: str | None,
        limit: int,
    ) -> Page[TaskGraphLaunch]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 1000
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        after_sort_key, after_key_digest = decode_record_cursor(cursor)

        async def read(transaction: StateTransaction) -> Page[TaskGraphLaunch]:
            records = await transaction.list_records(
                RecordQuery(
                    kind="task_graph",
                    states=_RECOVERABLE_GRAPH_STATES,
                    after_sort_key=after_sort_key,
                    after_key_digest=after_key_digest,
                    limit=limit + 1,
                )
            )
            selected = records[:limit]
            headers: list[TaskGraphView] = []
            for record in selected:
                header = await self._decode(record, TaskGraphView)
                self._validate_graph_record(record, header.graph_id)
                _require_canonical_graph_status(header.status)
                if (
                    record.state != header.status.value
                    or header.status.value not in _RECOVERABLE_GRAPH_STATES
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                headers.append(header)
            admission_keys = tuple(
                self._admission_key(header.graph_id) for header in headers
            )
            admission_records = (
                await transaction.get_records(admission_keys)
                if admission_keys
                else {}
            )
            launches: list[TaskGraphLaunch] = []
            for header, admission_key in zip(headers, admission_keys, strict=True):
                admission_record = admission_records.get(admission_key)
                if admission_record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                self._validate_admission_record(admission_record, header.graph_id)
                admission = await self._decode(admission_record, TaskGraphAdmission)
                if (
                    admission.graph_id != header.graph_id
                    or admission.principal.tenant_id != self._tenant_id
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                stored_operation = await transaction.get_operation(
                    operation_key(
                        self._namespace,
                        self._tenant_id,
                        self._domain.value,
                        admission.operation_id,
                    )
                )
                if stored_operation is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                operation = decode_operation(stored_operation)
                self._validate_operation_identity(operation, admission)
                if (
                    operation.request_digest != admission.initial_request_digest
                    or operation.status is not OperationStatus.SUCCEEDED
                    or operation.result_ref != admission.graph_id
                    or not _is_sha256(operation.result_digest)
                    or operation.error_code is not None
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                launches.append(admission.launch())
            next_cursor = (
                record_cursor(selected[-1])
                if len(records) > limit and selected
                else None
            )
            return Page(tuple(launches), next_cursor)

        return await self.state_store.read(read)

    async def _admit_in_transaction(
        self,
        transaction: StateTransaction,
        admission: TaskGraphAdmission,
        graph: TaskGraph,
        *,
        require_prepared: bool,
    ) -> TaskGraphView:
        submission = TaskGraphSubmission(self._namespace, admission, graph)
        head = await self._submission_in_transaction(transaction, submission.ref)
        if head is None:
            if require_prepared:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            await transaction.insert_record(self._stored(
                "task_submission", graph.graph_id, submission.ref, state="admitted",
            ))
        elif head.state == "prepared":
            submission = await self._prepared_submission(transaction, head, submission)
            admission, graph = submission.admission, submission.graph
            await self._replace_submission(transaction, head, "admitted")
            await transaction.delete_record(self._prepared_key(graph.graph_id))
        elif head.state == "cancelled" and await transaction.get_record(
            self._graph_key(graph.graph_id)
        ) is None:
            return TaskGraphView(graph.graph_id, TaskStatus.CANCELLED, ())
        graph_key = self._graph_key(graph.graph_id)
        admission_key = self._admission_key(graph.graph_id)
        operation_key_value = operation_key(
            self._namespace,
            self._tenant_id,
            self._domain.value,
            admission.operation_id,
        )
        records = await transaction.get_records((graph_key, admission_key))
        graph_record = records.get(graph_key)
        admission_record = records.get(admission_key)
        stored_operation = await transaction.get_operation(operation_key_value)
        definition_records = await transaction.list_records(
            RecordQuery(
                parent_digest=self._definition_parent(graph.graph_id),
                kind="task_node_definition",
            )
        )
        state_records = await transaction.list_records(
            RecordQuery(
                parent_digest=self._state_parent(graph.graph_id),
                kind="task_node_state",
            )
        )
        if (
            graph_record is None
            and admission_record is None
            and stored_operation is None
        ):
            if definition_records or state_records or (
                head is not None and head.state == "admitted"
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return await self._create_admission(transaction, admission, graph)

        if (
            graph_record is not None
            and admission_record is not None
            and stored_operation is None
        ):
            if await self._is_canonical_occupied_admission(
                transaction,
                graph_record,
                admission_record,
                admission,
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        if graph_record is None or admission_record is None or stored_operation is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        operation = decode_operation(stored_operation)
        if operation.request_digest != admission.initial_request_digest:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        self._validate_operation_identity(operation, admission)
        existing, view = await self._require_committed_admission(
            transaction,
            graph_record,
            admission_record,
            stored_operation=stored_operation,
        )
        if existing != admission and not _same_task_admission_contract(existing, admission):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await self._repair_aggregate_projection(
            transaction,
            graph_record,
            view,
        )

    async def _create_admission(
        self,
        transaction: StateTransaction,
        admission: TaskGraphAdmission,
        graph: TaskGraph,
    ) -> TaskGraphView:
        view = await self._insert_admission_records(transaction, admission, graph)
        now = await transaction.now()
        operation_input = OperationLedgerInput(
            admission.operation_id,
            self._tenant_id,
            ResourceKind.TASK_GRAPH,
            graph.graph_id,
            None,
            OperationKind.TASK_NODE,
            OperationStatus.SUCCEEDED,
            admission.initial_request_digest,
            graph.graph_id,
            _task_submit_result_digest(graph),
            None,
            False,
            now,
            now,
        )
        await append_operation(transaction, self, operation_input)
        _logger.info(
            "task graph durably admitted: tenant=%s graph=%s",
            self._tenant_id,
            graph.graph_id,
        )
        return view

    async def _insert_admission_records(
        self,
        transaction: StateTransaction,
        admission: TaskGraphAdmission,
        graph: TaskGraph,
    ) -> TaskGraphView:
        node_views: list[TaskNodeView] = []
        for node in graph.nodes:
            node_status = (
                node.dependency_status({})
                if not node.dependencies
                else TaskStatus.PENDING
            )
            node_view = TaskNodeView(
                graph.graph_id,
                node.node_id,
                node.dependencies,
                node_status,
                None,
                0,
                None,
                None,
                ErrorCode.TASK_DEPENDENCY_FAILED.value
                if node_status is TaskStatus.BLOCKED
                else None,
                None,
            )
            node_views.append(node_view)
        status = _isolated_graph_status(tuple(node_views), graph.nodes)
        header = TaskGraphView(graph.graph_id, status, ())
        view = TaskGraphView(graph.graph_id, status, graph.nodes)
        records = [
            self._stored(
                "task_graph",
                graph.graph_id,
                header,
                state=status.value,
            ),
            self._stored(
                "task_admission",
                graph.graph_id,
                admission,
                scope=self._recovery_scope(),
            ),
        ]
        node_view_by_id = {value.node_id: value for value in node_views}
        for node in graph.nodes:
            node_view = node_view_by_id[node.node_id]
            records.append(
                self._stored(
                    "task_node_definition",
                    [graph.graph_id, node.node_id],
                    node,
                    parent=self._definition_parent(graph.graph_id),
                )
            )
            records.append(
                self._stored(
                    "task_node_state",
                    [graph.graph_id, node.node_id],
                    node_view,
                    parent=self._state_parent(graph.graph_id),
                    state=node_view.status.value,
                )
            )
        await transaction.insert_records(tuple(records))
        await _append_task_state_events(
            transaction,
            namespace=self._namespace,
            tenant_id=self._tenant_id,
            domain=self._domain.value,
            graph_key=self._graph_key(graph.graph_id),
            before=None,
            after=_TaskEventState(view, tuple(node_views)),
        )
        return view

    async def _read_admission(
        self,
        transaction: StateTransaction,
        admission: TaskGraphAdmission,
        graph: TaskGraph,
    ) -> CommitObservation[TaskGraphView]:
        head = await self._submission_in_transaction(
            transaction, TaskGraphSubmission(self._namespace, admission, graph).ref,
        )
        if head is not None and head.state == "cancelled" and await transaction.get_record(
            self._graph_key(graph.graph_id)
        ) is None:
            return CommitObservation(
                DurableCommitState.COMMITTED,
                TaskGraphView(graph.graph_id, TaskStatus.CANCELLED, ()),
            )
        graph_key = self._graph_key(graph.graph_id)
        admission_key = self._admission_key(graph.graph_id)
        records = await transaction.get_records((graph_key, admission_key))
        graph_record = records.get(graph_key)
        admission_record = records.get(admission_key)
        stored_operation = await transaction.get_operation(
            operation_key(
                self._namespace,
                self._tenant_id,
                self._domain.value,
                admission.operation_id,
            )
        )
        definition_records = await transaction.list_records(
            RecordQuery(
                parent_digest=self._definition_parent(graph.graph_id),
                kind="task_node_definition",
            )
        )
        state_records = await transaction.list_records(
            RecordQuery(
                parent_digest=self._state_parent(graph.graph_id),
                kind="task_node_state",
            )
        )
        if (
            graph_record is None
            and admission_record is None
            and stored_operation is None
            and not definition_records
            and not state_records
        ):
            return CommitObservation(DurableCommitState.NOT_COMMITTED)
        try:
            if (
                graph_record is None
                or admission_record is None
                or stored_operation is None
            ):
                if (
                    graph_record is not None
                    and admission_record is not None
                    and await self._is_canonical_occupied_admission(
                        transaction,
                        graph_record,
                        admission_record,
                        admission,
                    )
                ):
                    return CommitObservation(
                        DurableCommitState.NOT_COMMITTED,
                        error=AIError(ErrorCode.STORAGE_CONFLICT),
                    )
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            operation = decode_operation(stored_operation)
            if operation.request_digest != admission.initial_request_digest:
                return CommitObservation(
                    DurableCommitState.NOT_COMMITTED,
                    error=AIError(ErrorCode.IDEMPOTENCY_CONFLICT),
                )
            self._validate_operation_identity(operation, admission)
            existing, view = await self._require_committed_admission(
                transaction,
                graph_record,
                admission_record,
                stored_operation=stored_operation,
            )
            if existing != admission and not _same_task_admission_contract(existing, admission):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return CommitObservation(DurableCommitState.COMMITTED, view)
        except (KeyError, TypeError, ValueError):
            return CommitObservation(
                DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
            )
        except AIError as error:
            return CommitObservation(
                DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                error=error,
            )

    async def _is_canonical_occupied_admission(
        self,
        transaction: StateTransaction,
        graph_record: StoredRecord,
        admission_record: StoredRecord,
        candidate: TaskGraphAdmission,
    ) -> bool:
        existing = await self._decode(admission_record, TaskGraphAdmission)
        if existing.operation_id == candidate.operation_id:
            return False
        stored_operation = await transaction.get_operation(
            operation_key(
                self._namespace,
                self._tenant_id,
                self._domain.value,
                existing.operation_id,
            )
        )
        if stored_operation is None:
            return False
        await self._require_committed_admission(
            transaction,
            graph_record,
            admission_record,
            stored_operation=stored_operation,
        )
        return True

    async def _require_committed_admission(
        self,
        transaction: StateTransaction,
        graph_record: StoredRecord,
        admission_record: StoredRecord,
        *,
        stored_operation: StoredOperation | None = None,
    ) -> tuple[TaskGraphAdmission, TaskGraphView]:
        existing = await self._decode(admission_record, TaskGraphAdmission)
        graph_header = await self._decode(graph_record, TaskGraphView)
        self._validate_graph_record(graph_record, graph_header.graph_id)
        self._validate_admission_record(admission_record, graph_header.graph_id)
        current, _states = await self._current_graph_in_transaction(
            transaction,
            graph_header.graph_id,
        )
        persisted_graph = TaskGraph(current.graph_id, current.nodes)
        if stored_operation is None:
            stored_operation = await transaction.get_operation(
                operation_key(
                    self._namespace,
                    self._tenant_id,
                    self._domain.value,
                    existing.operation_id,
                )
            )
        if stored_operation is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        operation = decode_operation(stored_operation)
        self._validate_operation_identity(operation, existing)
        if operation.request_digest != existing.initial_request_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._validate_succeeded_operation(operation, persisted_graph)
        return existing, current

    async def _repair_aggregate_projection(
        self,
        transaction: StateTransaction,
        graph_record: StoredRecord,
        view: TaskGraphView,
    ) -> TaskGraphView:
        header = await self._decode(graph_record, TaskGraphView)
        graph_view = TaskGraphView(header.graph_id, header.status, view.nodes)
        if (
            graph_view.status is view.status
            and graph_record.state == view.status.value
        ):
            return view
        graph_record = await _guard_task_event_owner(
            transaction,
            self._graph_key(view.graph_id),
        )
        if (
            graph_view.status is not view.status
            or graph_record.state != view.status.value
        ):
            await replace_checked(
                transaction,
                projected_record(
                    self,
                    graph_record,
                    replace(graph_view, status=view.status),
                ),
                graph_record.storage_version,
            )
        await _append_task_events(
            transaction,
            namespace=self._namespace,
            tenant_id=self._tenant_id,
            domain=self._domain.value,
            graph_id=view.graph_id,
            graph_key=self._graph_key(view.graph_id),
            drafts=_task_graph_event_drafts(graph_view, view),
            owner_guarded=True,
        )
        return view

    def _validate_succeeded_operation(
        self,
        operation: OperationLedgerRecord,
        graph: TaskGraph,
    ) -> None:
        if (
            operation.status is not OperationStatus.SUCCEEDED
            or operation.result_ref != graph.graph_id
            or not _is_sha256(operation.result_digest)
            or operation.error_code is not None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _validate_operation_identity(
        self,
        operation: OperationLedgerRecord,
        admission: TaskGraphAdmission,
    ) -> None:
        if (
            operation.operation_id != admission.operation_id
            or operation.tenant_id != self._tenant_id
            or operation.resource_kind is not ResourceKind.TASK_GRAPH
            or operation.resource_id != admission.graph_id
            or operation.execution_id is not None
            or operation.operation_kind is not OperationKind.TASK_NODE
            or operation.compactable
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

__all__ = ["TaskAdmissionRepositoryImpl"]
