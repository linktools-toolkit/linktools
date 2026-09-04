#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Best-effort Task metrics projected from durable repository truth."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Protocol

from ..core import JsonValue, TaskStatus, canonical_sha256
from ..errors import AIError
from ..observe import MetricRecorder, Observation
from ..storage import StoredPayload
from ._graph import (
    TaskGraphView,
    TaskLease,
    TaskNode,
    TaskNodeView,
    TaskResultRecord,
)

_TERMINAL = frozenset(
    {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.BLOCKED,
        TaskStatus.CANCELLED,
    }
)
_ATTEMPT_TERMINAL = frozenset(
    {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }
)


class _TaskRepository(Protocol):
    async def reconcile_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView: ...

    async def get_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphView | None: ...

    async def list_nodes(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> tuple[TaskNodeView, ...]: ...

    async def get_results(
        self,
        graph_id: str,
        node_ids: tuple[str, ...],
        *,
        tenant_id: str,
    ) -> Mapping[str, TaskResultRecord]: ...

    async def claim(
        self,
        graph_id: str,
        node_id: str,
        *,
        tenant_id: str,
        owner: str,
        lease_seconds: int,
    ) -> TaskLease: ...

    async def renew(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        lease_seconds: int,
    ) -> TaskLease: ...

    async def bind_execution(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str,
    ) -> TaskNodeView: ...

    async def complete(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str | None,
        result_digest: str,
        result_payload: StoredPayload | None = None,
    ) -> object: ...

    async def fail(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        error_code: str,
        error_digest: str,
        execution_id: str | None = None,
    ) -> object: ...

    async def cancel_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView: ...


class _MetricTaskRepository:
    """Decorate Task repository operations without changing durable semantics."""

    def __init__(
        self,
        delegate: _TaskRepository,
        recorder: MetricRecorder,
        *,
        source_namespace: str,
    ) -> None:
        self._delegate = delegate
        self._recorder = recorder
        self._source_namespace = source_namespace
        self._task_types: dict[tuple[str, str, str], str] = {}

    async def reconcile_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        view = await self._delegate.reconcile_graph(graph_id, tenant_id=tenant_id)
        self._remember_graph(view, tenant_id=tenant_id)
        self._record_graph(view, tenant_id=tenant_id)
        return view

    async def get_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphView | None:
        view = await self._delegate.get_graph(graph_id, tenant_id=tenant_id)
        if view is not None:
            self._remember_graph(view, tenant_id=tenant_id)
            self._record_graph(view, tenant_id=tenant_id)
        return view

    async def list_nodes(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> tuple[TaskNodeView, ...]:
        states = await self._delegate.list_nodes(graph_id, tenant_id=tenant_id)
        for state in states:
            self._record_attempt_state(state, tenant_id=tenant_id)
        return states

    async def get_results(
        self,
        graph_id: str,
        node_ids: tuple[str, ...],
        *,
        tenant_id: str,
    ) -> Mapping[str, TaskResultRecord]:
        return await self._delegate.get_results(
            graph_id,
            node_ids,
            tenant_id=tenant_id,
        )

    async def claim(
        self,
        graph_id: str,
        node_id: str,
        *,
        tenant_id: str,
        owner: str,
        lease_seconds: int,
    ) -> TaskLease:
        return await self._delegate.claim(
            graph_id,
            node_id,
            tenant_id=tenant_id,
            owner=owner,
            lease_seconds=lease_seconds,
        )

    async def renew(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        lease_seconds: int,
    ) -> TaskLease:
        return await self._delegate.renew(
            lease,
            tenant_id=tenant_id,
            lease_seconds=lease_seconds,
        )

    async def bind_execution(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str,
    ) -> TaskNodeView:
        return await self._delegate.bind_execution(
            lease,
            tenant_id=tenant_id,
            execution_id=execution_id,
        )

    async def complete(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str | None,
        result_digest: str,
        result_payload: StoredPayload | None = None,
    ) -> object:
        result = await self._delegate.complete(
            lease,
            tenant_id=tenant_id,
            execution_id=execution_id,
            result_digest=result_digest,
            result_payload=result_payload,
        )
        self._record_attempt(
            lease,
            tenant_id=tenant_id,
            status=TaskStatus.SUCCEEDED,
            error_code=None,
            execution_id=execution_id,
        )
        return result

    async def fail(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        error_code: str,
        error_digest: str,
        execution_id: str | None = None,
    ) -> object:
        result = await self._delegate.fail(
            lease,
            tenant_id=tenant_id,
            error_code=error_code,
            error_digest=error_digest,
            execution_id=execution_id,
        )
        self._record_attempt(
            lease,
            tenant_id=tenant_id,
            status=TaskStatus.FAILED,
            error_code=error_code,
            execution_id=execution_id,
        )
        return result

    async def cancel_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
        view = await self._delegate.cancel_graph(graph_id, tenant_id=tenant_id)
        self._remember_graph(view, tenant_id=tenant_id)
        self._record_graph(view, tenant_id=tenant_id)
        return view

    def _remember_graph(self, view: TaskGraphView, *, tenant_id: str) -> None:
        for node in view.nodes:
            task_type = _task_type(node)
            if task_type is not None:
                self._task_types[(tenant_id, view.graph_id, node.node_id)] = task_type

    def _record_graph(self, view: TaskGraphView, *, tenant_id: str) -> None:
        if view.status not in _TERMINAL:
            return
        self._try_record(
            Observation(
                version=1,
                observation_id=canonical_sha256(
                    {
                        "source_namespace": self._source_namespace,
                        "tenant_id": tenant_id,
                        "graph_id": view.graph_id,
                        "terminal": True,
                    }
                ),
                kind="linktools.task.graph.terminal",
                occurred_at=datetime.now(timezone.utc),
                source_namespace=self._source_namespace,
                tenant_id=tenant_id,
                status=view.status.value,
                error_code=None,
                correlation={"graph_id": view.graph_id},
                dimensions={},
                measurements=(),
            )
        )

    def _record_attempt_state(self, state: TaskNodeView, *, tenant_id: str) -> None:
        if state.status not in _ATTEMPT_TERMINAL or state.fence < 1:
            return
        self._record_attempt_values(
            graph_id=state.graph_id,
            node_id=state.node_id,
            fence=state.fence,
            tenant_id=tenant_id,
            status=state.status,
            error_code=state.error_code,
            execution_id=state.execution_id,
        )

    def _record_attempt(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        status: TaskStatus,
        error_code: str | None,
        execution_id: str | None,
    ) -> None:
        self._record_attempt_values(
            graph_id=lease.graph_id,
            node_id=lease.node_id,
            fence=lease.fence,
            tenant_id=tenant_id,
            status=status,
            error_code=error_code,
            execution_id=execution_id,
        )

    def _record_attempt_values(
        self,
        *,
        graph_id: str,
        node_id: str,
        fence: int,
        tenant_id: str,
        status: TaskStatus,
        error_code: str | None,
        execution_id: str | None,
    ) -> None:
        correlation: dict[str, str | int] = {
            "graph_id": graph_id,
            "node_id": node_id,
            "fence": fence,
        }
        if execution_id is not None:
            correlation["execution_id"] = execution_id
        task_type = self._task_types.get((tenant_id, graph_id, node_id))
        dimensions = {} if task_type is None else {"task_type": task_type}
        self._try_record(
            Observation(
                version=1,
                observation_id=canonical_sha256(
                    {
                        "source_namespace": self._source_namespace,
                        "tenant_id": tenant_id,
                        "graph_id": graph_id,
                        "node_id": node_id,
                        "fence": fence,
                        "attempt": True,
                    }
                ),
                kind="linktools.task.node.attempt",
                occurred_at=datetime.now(timezone.utc),
                source_namespace=self._source_namespace,
                tenant_id=tenant_id,
                status=status.value,
                error_code=error_code,
                correlation=correlation,
                dimensions=dimensions,
                measurements=(),
            )
        )

    def _try_record(self, observation: Observation) -> None:
        try:
            self._recorder.try_record(observation)
        except (AIError, TypeError, ValueError):
            return


def _task_type(node: TaskNode) -> str | None:
    value: JsonValue | None = node.input.get("type")
    if not isinstance(value, str) or not value or value != value.strip():
        return None
    return value


__all__: list[str] = []
