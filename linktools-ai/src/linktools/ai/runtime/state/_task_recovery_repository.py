#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task repositories with durable recovery indexing and legacy admission repair."""

from collections.abc import Mapping
from dataclasses import replace

from ...core import OperationStatus, Page, TaskStatus
from ...errors import AIError, ErrorCode
from ...storage import StoredPayload
from ...task import (
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphLaunch,
    TaskGraphSnapshot,
    TaskGraphView,
    TaskLease,
    TaskNodeView,
    TaskResultRecord,
    TaskTerminalRecord,
)
from ._repositories import (
    TaskAdmissionRepositoryImpl,
    TaskRepositoryImpl,
    _decode_operation,
    _decode_record_cursor,
    _graph_status,
    _record_cursor,
    _replace_checked,
    _require_live_task_lease,
    _stored_operation_error,
    _task_graph_record,
    _validate_task_lease_scope,
)
from ._store import (
    RecordQuery,
    StateTransaction,
    StoredOperation,
    StoredRecord,
    operation_key,
)

_CURRENT_CURSOR_PREFIX = "a:"
_LEGACY_CURSOR_PREFIX = "g:"
_COMMIT_READBACK_CODES = frozenset(
    {ErrorCode.STORAGE_CONFLICT, ErrorCode.STORAGE_COMMIT_UNKNOWN}
)


class DurableTaskRepositoryImpl(TaskRepositoryImpl):
    """Keep Task authority and recovery projections convergent under CAS races."""

    def _admission_key(self, graph_id: str) -> bytes:
        return self._key("task_admission", graph_id)

    def _recovery_scope(self) -> bytes:
        return self._scope("task_admission", "recoverable", "graphs")

    def _legacy_recovery_scope(self) -> bytes:
        return self._scope("task_graph", "recoverable", "graphs")

    async def bind_execution(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str,
    ) -> TaskNodeView:
        try:
            return await super().bind_execution(
                lease,
                tenant_id=tenant_id,
                execution_id=execution_id,
            )
        except AIError as error:
            if error.code not in _COMMIT_READBACK_CODES:
                raise
            current = await self._node(lease.graph_id, lease.node_id, tenant_id)
            if current.fence != lease.fence:
                raise AIError(ErrorCode.TASK_FENCE_STALE) from error
            if current.execution_id == execution_id:
                return current
            if current.execution_id is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if current.owner != lease.owner:
                raise AIError(ErrorCode.TASK_OWNER_CONFLICT) from error
            if error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise AIError(
                    ErrorCode.STORAGE_RECOVERY_REQUIRED,
                    safe_details={
                        "phase": "task_execution_bind",
                        "graph_id": lease.graph_id,
                        "node_id": lease.node_id,
                    },
                ) from error
            raise AIError(ErrorCode.STORAGE_CONFLICT) from error

    async def reconcile_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        async def mutate(transaction: StateTransaction) -> TaskGraphView:
            view = await super(DurableTaskRepositoryImpl, self).reconcile_graph(
                graph_id,
                tenant_id=tenant_id,
            )
            await self._sync_recovery_projection(transaction, view)
            return view

        try:
            return await self.state_store.mutate(mutate)
        except AIError as error:
            if error.code not in _COMMIT_READBACK_CODES:
                raise
            view, converged = await self._projection_readback(
                graph_id,
                tenant_id=tenant_id,
            )
            if converged:
                return view
            if error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise AIError(
                    ErrorCode.STORAGE_RECOVERY_REQUIRED,
                    safe_details={
                        "phase": "task_reconcile",
                        "graph_id": graph_id,
                    },
                ) from error
            raise AIError(ErrorCode.STORAGE_CONFLICT) from error

    async def cancel_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        async def mutate(transaction: StateTransaction) -> TaskGraphView:
            view = await super(DurableTaskRepositoryImpl, self).cancel_graph(
                graph_id,
                tenant_id=tenant_id,
            )
            await self._sync_recovery_projection(transaction, view)
            return view

        try:
            return await self.state_store.mutate(mutate)
        except AIError as error:
            if error.code not in _COMMIT_READBACK_CODES:
                raise
            view, _ = await self._projection_readback(
                graph_id,
                tenant_id=tenant_id,
            )
            if view.status in {
                TaskStatus.SUCCEEDED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.BLOCKED,
            }:
                return view
            if error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise AIError(
                    ErrorCode.STORAGE_RECOVERY_REQUIRED,
                    safe_details={
                        "phase": "task_cancel",
                        "graph_id": graph_id,
                    },
                ) from error
            raise AIError(ErrorCode.STORAGE_CONFLICT) from error

    async def claim(
        self,
        graph_id: str,
        node_id: str,
        *,
        tenant_id: str,
        owner: str,
        lease_seconds: int,
    ) -> TaskLease:
        before = await self._node(graph_id, node_id, tenant_id)
        expected_fence = before.fence + 1
        try:
            return await super().claim(
                graph_id,
                node_id,
                tenant_id=tenant_id,
                owner=owner,
                lease_seconds=lease_seconds,
            )
        except AIError as error:
            if error.code not in _COMMIT_READBACK_CODES:
                raise
            current = await self._node(graph_id, node_id, tenant_id)
            if current.status is TaskStatus.RUNNING:
                if current.owner != owner:
                    raise AIError(ErrorCode.TASK_OWNER_CONFLICT) from error
                if current.fence != expected_fence:
                    raise AIError(ErrorCode.TASK_FENCE_STALE) from error
                if current.lease_expires_at is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                return TaskLease(
                    graph_id, node_id, tenant_id, owner, current.fence, current.lease_expires_at
                )
            if current.status in {
                TaskStatus.SUCCEEDED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.BLOCKED,
            }:
                raise AIError(ErrorCode.TASK_NOT_READY) from error
            if error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise AIError(
                    ErrorCode.STORAGE_RECOVERY_REQUIRED,
                    safe_details={
                        "phase": "task_claim",
                        "graph_id": graph_id,
                        "node_id": node_id,
                    },
                ) from error
            raise AIError(ErrorCode.STORAGE_CONFLICT) from error

    async def complete(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str | None,
        result_digest: str,
        result_payload: StoredPayload | None = None,
    ) -> TaskTerminalRecord:
        try:
            return await super().complete(
                lease,
                tenant_id=tenant_id,
                execution_id=execution_id,
                result_digest=result_digest,
                result_payload=result_payload,
            )
        except AIError as error:
            if error.code not in _COMMIT_READBACK_CODES:
                raise
            return await self._classify_terminal_readback(
                lease,
                tenant_id=tenant_id,
                status=TaskStatus.SUCCEEDED,
                execution_id=execution_id,
                result_digest=result_digest,
                result_payload=result_payload,
                error_code=None,
                error_digest=None,
                conflict=error,
            )

    async def fail(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        error_code: str,
        error_digest: str,
        execution_id: str | None = None,
    ) -> TaskTerminalRecord:
        try:
            return await super().fail(
                lease,
                tenant_id=tenant_id,
                error_code=error_code,
                error_digest=error_digest,
                execution_id=execution_id,
            )
        except AIError as error:
            if error.code not in _COMMIT_READBACK_CODES:
                raise
            return await self._classify_terminal_readback(
                lease,
                tenant_id=tenant_id,
                status=TaskStatus.FAILED,
                execution_id=execution_id,
                result_digest=None,
                result_payload=None,
                error_code=error_code,
                error_digest=error_digest,
                conflict=error,
            )

    async def _classify_terminal_readback(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        status: TaskStatus,
        execution_id: str | None,
        result_digest: str | None,
        result_payload: StoredPayload | None,
        error_code: str | None,
        error_digest: str | None,
        conflict: AIError,
    ) -> TaskTerminalRecord:
        current = await self._node(lease.graph_id, lease.node_id, tenant_id)
        if current.fence != lease.fence:
            raise AIError(ErrorCode.TASK_FENCE_STALE) from conflict
        if current.status is status:
            if (
                current.result_digest != result_digest
                or current.error_code != error_code
                or current.error_digest != error_digest
                or (
                    execution_id is not None
                    and current.execution_id != execution_id
                )
            ):
                raise AIError(ErrorCode.TASK_RESULT_CONFLICT) from conflict
            if status is TaskStatus.SUCCEEDED and result_payload is not None:
                results = await self.get_results(
                    lease.graph_id,
                    (lease.node_id,),
                    tenant_id=tenant_id,
                )
                result = results.get(lease.node_id)
                if result is None or result.result_digest != result_digest:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from conflict
            return TaskTerminalRecord(
                lease.node_id,
                lease.owner,
                lease.fence,
                status,
                result_digest,
                error_code,
                error_digest,
                execution_id=current.execution_id,
            )
        if current.status in {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.BLOCKED,
        }:
            raise AIError(ErrorCode.TASK_TERMINAL_CONFLICT) from conflict
        if current.owner != lease.owner:
            raise AIError(ErrorCode.TASK_FENCE_STALE) from conflict
        if conflict.code is ErrorCode.STORAGE_COMMIT_UNKNOWN:
            raise AIError(
                ErrorCode.STORAGE_RECOVERY_REQUIRED,
                safe_details={
                    "phase": "task_terminal_commit",
                    "graph_id": lease.graph_id,
                    "node_id": lease.node_id,
                },
            ) from conflict
        raise AIError(ErrorCode.STORAGE_CONFLICT) from conflict

    async def _projection_readback(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> tuple[TaskGraphView, bool]:
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def read(transaction: StateTransaction) -> tuple[TaskGraphView, bool]:
            graph_key = self._graph_key(graph_id)
            admission_key = self._admission_key(graph_id)
            records = await transaction.get_records((graph_key, admission_key))
            graph_record = records.get(graph_key)
            if graph_record is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            graph = await self._decode(graph_record, TaskGraphView)
            node_records = await transaction.list_records(
                RecordQuery(
                    parent_digest=self._parent("task_node", "graph", graph_id),
                    kind="task_node",
                )
            )
            states = await self._decode_many(node_records)
            snapshot = TaskGraphSnapshot(
                graph.graph_id,
                _graph_status(states),
                graph.nodes,
                tuple(
                    {state.node_id: state for state in states}[node.node_id]
                    for node in graph.nodes
                ),
            )
            view = TaskGraphView(graph.graph_id, snapshot.status, graph.nodes)
            graph_converged = (
                graph.status is snapshot.status
                and graph_record.state == snapshot.status.value
            )
            admission_record = records.get(admission_key)
            if admission_record is None:
                return view, graph_converged
            if admission_record.scope_digest == self._recovery_scope():
                if (
                    admission_record.kind != "task_admission"
                    or admission_record.key_digest != admission_key
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return view, graph_converged and admission_record.state == snapshot.status.value
            if (
                admission_record.scope_digest is None
                and graph_record.scope_digest == self._legacy_recovery_scope()
            ):
                return view, graph_converged
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        try:
            return await self.state_store.read(read)
        except (KeyError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    async def _sync_recovery_projection(
        self,
        transaction: StateTransaction,
        view: TaskGraphView,
    ) -> None:
        admission_key = self._admission_key(view.graph_id)
        graph_key = self._graph_key(view.graph_id)
        records = await transaction.get_records((admission_key, graph_key))
        admission_record = records.get(admission_key)
        if admission_record is None:
            return
        graph_record = records.get(graph_key)
        if graph_record is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if admission_record.key_digest != admission_key or admission_record.kind != "task_admission":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if admission_record.scope_digest == self._recovery_scope():
            if admission_record.state == view.status.value:
                return
            candidate = replace(
                admission_record,
                state=view.status.value,
                storage_version=admission_record.storage_version + 1,
            )
            if not await transaction.replace_record(
                candidate,
                expected_storage_version=admission_record.storage_version,
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return
        if (
            admission_record.scope_digest is None
            and graph_record.scope_digest == self._legacy_recovery_scope()
        ):
            return
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


class DurableTaskAdmissionRepositoryImpl(TaskAdmissionRepositoryImpl):
    """Index recoverable graphs by admission without rewriting graph identity."""

    def _recovery_scope(self) -> bytes:
        return self._scope("task_admission", "recoverable", "graphs")

    def _legacy_recovery_scope(self) -> bytes:
        return self._scope("task_graph", "recoverable", "graphs")

    async def list_recoverable_page(
        self,
        *,
        cursor: str | None,
        limit: int,
    ) -> Page[TaskGraphLaunch]:
        if limit != 128:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        phase, inner_cursor = _decode_recovery_cursor(cursor)
        if phase == "admission":
            current = await self._list_current_recoverable_page(
                cursor=inner_cursor,
                limit=limit,
            )
            if current.items or current.next_cursor is not None:
                if current.next_cursor is not None:
                    next_cursor = _CURRENT_CURSOR_PREFIX + current.next_cursor
                else:
                    legacy_probe = await self._list_legacy_recoverable_page(
                        cursor=None,
                        limit=1,
                    )
                    next_cursor = (
                        _LEGACY_CURSOR_PREFIX
                        if legacy_probe.items or legacy_probe.next_cursor is not None
                        else None
                    )
                return Page(current.items, next_cursor)
            inner_cursor = None
        legacy = await self._list_legacy_recoverable_page(
            cursor=inner_cursor,
            limit=limit,
        )
        return Page(
            legacy.items,
            (
                None
                if legacy.next_cursor is None
                else _LEGACY_CURSOR_PREFIX + legacy.next_cursor
            ),
        )

    async def _list_current_recoverable_page(
        self,
        *,
        cursor: str | None,
        limit: int,
    ) -> Page[TaskGraphLaunch]:
        after_sort_key, after_key_digest = _decode_record_cursor(cursor)

        async def read(transaction: StateTransaction) -> Page[TaskGraphLaunch]:
            records = await transaction.list_records(
                RecordQuery(
                    scope_digest=self._recovery_scope(),
                    kind="task_admission",
                    states=_RECOVERABLE_STATES,
                    after_sort_key=after_sort_key,
                    after_key_digest=after_key_digest,
                    limit=limit + 1,
                )
            )
            selected = records[:limit]
            admissions = tuple(
                [await self._decode(record, TaskGraphAdmission) for record in selected]
            )
            graph_records = await transaction.get_records(
                tuple(self._graph_key(admission.graph_id) for admission in admissions)
            )
            launches: list[TaskGraphLaunch] = []
            for record, admission in zip(selected, admissions, strict=True):
                graph_record = graph_records.get(self._graph_key(admission.graph_id))
                if graph_record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                graph_view = await self._decode(graph_record, TaskGraphView)
                if (
                    record.key_digest != self._admission_key(admission.graph_id)
                    or record.scope_digest != self._recovery_scope()
                    or record.state != graph_view.status.value
                    or graph_record.key_digest != self._graph_key(admission.graph_id)
                    or graph_record.state != graph_view.status.value
                    or graph_view.graph_id != admission.graph_id
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                launches.append(
                    admission.bind(TaskGraph(graph_view.graph_id, graph_view.nodes))
                )
            next_cursor = (
                _record_cursor(selected[-1])
                if len(records) > limit and selected
                else None
            )
            return Page(tuple(launches), next_cursor)

        return await self.state_store.read(read)

    async def _list_legacy_recoverable_page(
        self,
        *,
        cursor: str | None,
        limit: int,
    ) -> Page[TaskGraphLaunch]:
        after_sort_key, after_key_digest = _decode_record_cursor(cursor)

        async def read(transaction: StateTransaction) -> Page[TaskGraphLaunch]:
            records = await transaction.list_records(
                RecordQuery(
                    scope_digest=self._legacy_recovery_scope(),
                    kind="task_graph",
                    states=_RECOVERABLE_STATES,
                    after_sort_key=after_sort_key,
                    after_key_digest=after_key_digest,
                    limit=limit + 1,
                )
            )
            selected = records[:limit]
            graph_views = tuple(
                [await self._decode(record, TaskGraphView) for record in selected]
            )
            admission_records = await transaction.get_records(
                tuple(
                    self._admission_key(graph_view.graph_id)
                    for graph_view in graph_views
                )
            )
            launches: list[TaskGraphLaunch] = []
            for graph_record, graph_view in zip(selected, graph_views, strict=True):
                admission_record = admission_records.get(
                    self._admission_key(graph_view.graph_id)
                )
                if admission_record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if admission_record.scope_digest == self._recovery_scope():
                    continue
                if admission_record.scope_digest is not None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                admission = await self._decode(admission_record, TaskGraphAdmission)
                if (
                    graph_record.key_digest != self._graph_key(graph_view.graph_id)
                    or graph_record.scope_digest != self._legacy_recovery_scope()
                    or graph_record.state != graph_view.status.value
                    or admission.graph_id != graph_view.graph_id
                    or admission_record.key_digest != self._admission_key(graph_view.graph_id)
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                launches.append(
                    admission.bind(TaskGraph(graph_view.graph_id, graph_view.nodes))
                )
            next_cursor = (
                _record_cursor(selected[-1])
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
        if graph_record is None or stored_operation is None:
            return await super()._admit_in_transaction(transaction, admission, graph)

        operation = _decode_operation(stored_operation)
        if operation.request_digest != admission.request_digest:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        self._validate_operation_identity(operation, admission)
        if operation.status in {OperationStatus.FAILED, OperationStatus.CANCELLED}:
            raise _stored_operation_error(operation)
        if operation.status not in {
            OperationStatus.PENDING,
            OperationStatus.RUNNING,
            OperationStatus.EFFECT_UNKNOWN,
            OperationStatus.SUCCEEDED,
        }:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        node_records = await transaction.list_records(
            RecordQuery(
                parent_digest=self._parent("task_node", "graph", graph.graph_id),
                kind="task_node",
            )
        )
        if admission_record is not None:
            existing, view = await self._require_committed_admission(
                transaction,
                graph_record,
                admission_record,
                node_records,
                stored_operation=stored_operation,
            )
            if existing != admission:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._repair_aggregate_projection(
                transaction,
                graph_record,
                admission_record,
                view,
            )
            return view

        graph_view = await self._decode(graph_record, TaskGraphView)
        nodes = await self._decode_task_nodes(node_records)
        self._validate_graph(graph_view, graph, nodes)
        if operation.status is OperationStatus.SUCCEEDED:
            self._validate_succeeded_operation(operation, graph)

        status = _graph_status(nodes)
        next_graph = TaskGraphView(graph.graph_id, status, graph.nodes)
        if graph_view.status is not status or graph_record.state != status.value:
            await _replace_checked(
                transaction,
                _task_graph_record(self, graph_record, next_graph),
                graph_record.storage_version,
            )
        await transaction.insert_record(
            self._stored(
                "task_admission",
                graph.graph_id,
                admission,
                scope=self._recovery_scope(),
                state=status.value,
            )
        )
        if operation.status is not OperationStatus.SUCCEEDED:
            await self._settle_operation(transaction, stored_operation, operation, graph)
        return next_graph

    async def _insert_admission_records(
        self,
        transaction: StateTransaction,
        admission: TaskGraphAdmission,
        graph: TaskGraph,
    ) -> TaskGraphView:
        status = TaskStatus.SUCCEEDED if not graph.nodes else TaskStatus.PENDING
        view = TaskGraphView(graph.graph_id, status, graph.nodes)
        records = [
            self._stored(
                "task_graph",
                graph.graph_id,
                view,
                state=status.value,
            ),
            self._stored(
                "task_admission",
                graph.graph_id,
                admission,
                scope=self._recovery_scope(),
                state=status.value,
            ),
        ]
        for node in graph.nodes:
            node_status = TaskStatus.READY if not node.dependencies else TaskStatus.PENDING
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
            records.append(
                self._stored(
                    "task_node",
                    [graph.graph_id, node.node_id],
                    node_view,
                    parent=self._parent("task_node", "graph", graph.graph_id),
                    state=node_status.value,
                )
            )
        await transaction.insert_records(records)
        return view

    async def _require_committed_admission(
        self,
        transaction: StateTransaction,
        graph_record: StoredRecord,
        admission_record: StoredRecord,
        node_records: tuple[StoredRecord, ...],
        *,
        stored_operation: StoredOperation | None = None,
    ) -> tuple[TaskGraphAdmission, TaskGraphView]:
        existing = await self._decode(admission_record, TaskGraphAdmission)
        graph_view = await self._decode(graph_record, TaskGraphView)
        nodes = await self._decode_task_nodes(node_records)
        persisted_graph = TaskGraph(graph_view.graph_id, graph_view.nodes)
        existing.bind(persisted_graph)
        self._validate_graph(graph_view, persisted_graph, nodes)
        status = _graph_status(nodes)
        if graph_record.key_digest != self._graph_key(graph_view.graph_id):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if admission_record.key_digest != self._admission_key(graph_view.graph_id):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not self._recognized_layout(graph_record, admission_record):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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
        operation = _decode_operation(stored_operation)
        self._validate_operation_identity(operation, existing)
        if operation.request_digest != existing.request_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._validate_succeeded_operation(operation, persisted_graph)
        return existing, TaskGraphView(graph_view.graph_id, status, graph_view.nodes)

    async def _repair_aggregate_projection(
        self,
        transaction: StateTransaction,
        graph_record: StoredRecord,
        admission_record: StoredRecord,
        view: TaskGraphView,
    ) -> None:
        graph_view = await self._decode(graph_record, TaskGraphView)
        if graph_view.status is not view.status or graph_record.state != view.status.value:
            await _replace_checked(
                transaction,
                _task_graph_record(self, graph_record, replace(graph_view, status=view.status)),
                graph_record.storage_version,
            )
        if (
            admission_record.scope_digest == self._recovery_scope()
            and admission_record.state != view.status.value
        ):
            candidate = replace(
                admission_record,
                state=view.status.value,
                storage_version=admission_record.storage_version + 1,
            )
            if not await transaction.replace_record(
                candidate,
                expected_storage_version=admission_record.storage_version,
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)

    def _recognized_layout(
        self,
        graph_record: StoredRecord,
        admission_record: StoredRecord,
    ) -> bool:
        return admission_record.scope_digest == self._recovery_scope() or (
            admission_record.scope_digest is None
            and graph_record.scope_digest == self._legacy_recovery_scope()
        )


_RECOVERABLE_STATES = frozenset(
    {
        TaskStatus.PENDING.value,
        TaskStatus.READY.value,
        TaskStatus.RUNNING.value,
    }
)


def _decode_recovery_cursor(cursor: str | None) -> tuple[str, str | None]:
    if cursor is None:
        return "admission", None
    if cursor.startswith(_CURRENT_CURSOR_PREFIX):
        inner = cursor[len(_CURRENT_CURSOR_PREFIX) :]
        return "admission", inner or None
    if cursor.startswith(_LEGACY_CURSOR_PREFIX):
        inner = cursor[len(_LEGACY_CURSOR_PREFIX) :]
        return "graph", inner or None
    return "graph", cursor


__all__ = ["DurableTaskAdmissionRepositoryImpl", "DurableTaskRepositoryImpl"]
