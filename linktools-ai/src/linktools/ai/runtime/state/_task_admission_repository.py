#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable TaskGraph admission repository."""

import asyncio
from dataclasses import replace

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
    _require_canonical_graph_status,
)

_logger = environ.get_logger("ai.runtime.state.task_repository")
_RECOVERABLE_GRAPH_STATES = frozenset(
    {
        TaskStatus.PENDING.value,
        TaskStatus.RUNNING.value,
        TaskStatus.RECOVERY_REQUIRED.value,
    }
)


def _task_submit_result_digest(graph: TaskGraph) -> str:
    status = TaskStatus.SUCCEEDED if not graph.nodes else TaskStatus.PENDING
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
                _effective_graph_status(header, ordered_states),
                nodes,
            ),
            ordered_states,
        )

    def _validate_graph_record(self, record: StoredRecord, graph_id: str) -> None:
        if (
            record.kind != "task_graph"
            or record.key_digest != self._graph_key(graph_id)
            or record.partition_digest != self._partition("task_graph")
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
            or record.partition_digest != self._partition("task_admission")
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
            or record.partition_digest != self._partition("task_node_definition")
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
            or record.partition_digest != self._partition("task_node_state")
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

        result = await run_durable_commit(
            operation,
            readback,
            background_tasks=self._background_tasks,
        )
        if result.state is DurableCommitState.COMMITTED and result.value is not None:
            if result.cancelled:
                raise asyncio.CancelledError
            return result.value
        if result.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
            if isinstance(result.error, AIError):
                raise result.error
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from result.error
        if result.state is DurableCommitState.NOT_COMMITTED:
            if result.cancelled:
                raise asyncio.CancelledError
            if result.error is not None:
                raise result.error
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if result.cancelled:
            raise asyncio.CancelledError
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from result.error

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
        if limit != 128:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        after_sort_key, after_key_digest = decode_record_cursor(cursor)

        async def read(transaction: StateTransaction) -> Page[TaskGraphLaunch]:
            records = await transaction.list_records(
                RecordQuery(
                    partition_digest=self._partition("task_graph"),
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
    ) -> TaskGraphView:
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
            if definition_records or state_records:
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
        if existing.correlation != admission.correlation:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
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
        status = TaskStatus.SUCCEEDED if not graph.nodes else TaskStatus.PENDING
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
        node_views: list[TaskNodeView] = []
        for node in graph.nodes:
            node_status = (
                TaskStatus.READY if not node.dependencies else TaskStatus.PENDING
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
                None,
                None,
            )
            node_views.append(node_view)
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
                    state=node_status.value,
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
            if existing.correlation != admission.correlation:
                return CommitObservation(
                    DurableCommitState.NOT_COMMITTED,
                    error=AIError(ErrorCode.IDEMPOTENCY_CONFLICT),
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
