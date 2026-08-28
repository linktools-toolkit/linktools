#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task repositories with durable recovery indexing and legacy admission repair."""

from dataclasses import replace

from ...core import OperationStatus, Page, TaskStatus
from ...errors import AIError, ErrorCode
from ...task import TaskGraph, TaskGraphAdmission, TaskGraphLaunch, TaskGraphView, TaskNodeView
from ._repositories import (
    TaskAdmissionRepositoryImpl,
    TaskRepositoryImpl,
    _decode_operation,
    _decode_record_cursor,
    _graph_status,
    _record_cursor,
    _replace_checked,
    _stored_operation_error,
    _task_graph_record,
)
from ._store import (
    RecordQuery,
    StateTransaction,
    StoredOperation,
    StoredRecord,
    operation_key,
)


class DurableTaskRepositoryImpl(TaskRepositoryImpl):
    """Keep the Task admission recovery index aligned with graph aggregate state."""

    def _admission_key(self, graph_id: str) -> bytes:
        return self._key("task_admission", graph_id)

    def _recovery_scope(self) -> bytes:
        return self._scope("task_admission", "recoverable", "graphs")

    async def reconcile_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        async def mutate(transaction: StateTransaction) -> TaskGraphView:
            view = await super(DurableTaskRepositoryImpl, self).reconcile_graph(
                graph_id,
                tenant_id=tenant_id,
            )
            await self._sync_admission_state(transaction, view)
            return view

        return await self.state_store.mutate(mutate)

    async def cancel_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        async def mutate(transaction: StateTransaction) -> TaskGraphView:
            view = await super(DurableTaskRepositoryImpl, self).cancel_graph(
                graph_id,
                tenant_id=tenant_id,
            )
            await self._sync_admission_state(transaction, view)
            return view

        return await self.state_store.mutate(mutate)

    async def _sync_admission_state(
        self,
        transaction: StateTransaction,
        view: TaskGraphView,
    ) -> None:
        key = self._admission_key(view.graph_id)
        record = await transaction.get_record(key)
        if record is None:
            return
        if (
            record.key_digest != key
            or record.kind != "task_admission"
            or record.scope_digest != self._recovery_scope()
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if record.state == view.status.value:
            return
        candidate = replace(
            record,
            state=view.status.value,
            storage_version=record.storage_version + 1,
        )
        if not await transaction.replace_record(
            candidate,
            expected_storage_version=record.storage_version,
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)


class DurableTaskAdmissionRepositoryImpl(TaskAdmissionRepositoryImpl):
    """Index recoverable graphs by admission without rewriting graph identity."""

    def _recovery_scope(self) -> bytes:
        return self._scope("task_admission", "recoverable", "graphs")

    async def list_recoverable_page(
        self,
        *,
        cursor: str | None,
        limit: int,
    ) -> Page[TaskGraphLaunch]:
        if limit != 128:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        after_sort_key, after_key_digest = _decode_record_cursor(cursor)

        async def read(transaction: StateTransaction) -> Page[TaskGraphLaunch]:
            records = await transaction.list_records(
                RecordQuery(
                    scope_digest=self._recovery_scope(),
                    kind="task_admission",
                    states=frozenset(
                        {
                            TaskStatus.PENDING.value,
                            TaskStatus.READY.value,
                            TaskStatus.RUNNING.value,
                        }
                    ),
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
        if (
            graph_record.key_digest != self._graph_key(graph_view.graph_id)
            or admission_record.key_digest != self._admission_key(graph_view.graph_id)
            or admission_record.scope_digest != self._recovery_scope()
        ):
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
        if graph_record.state != view.status.value:
            graph_view = await self._decode(graph_record, TaskGraphView)
            await _replace_checked(
                transaction,
                _task_graph_record(self, graph_record, replace(graph_view, status=view.status)),
                graph_record.storage_version,
            )
        else:
            graph_view = await self._decode(graph_record, TaskGraphView)
            if graph_view.status is not view.status:
                await _replace_checked(
                    transaction,
                    _task_graph_record(self, graph_record, replace(graph_view, status=view.status)),
                    graph_record.storage_version,
                )
        if admission_record.state != view.status.value:
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


__all__ = ["DurableTaskAdmissionRepositoryImpl", "DurableTaskRepositoryImpl"]
