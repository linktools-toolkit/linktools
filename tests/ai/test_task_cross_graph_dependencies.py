#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-graph Task dependency authorization, retention, and lazy reads."""

from types import SimpleNamespace

import pytest

from linktools.ai.core import (
    AuthorizationAction,
    ExecutionStatus,
    Principal,
    ResourceKind,
    ResourceRef,
    TaskStatus,
    canonical_sha256,
)
from linktools.ai.runtime._planner import RuntimeTaskNodeRunner
from linktools.ai.task import (
    TaskDependency,
    TaskNode,
    TaskNodeContext,
    TaskResultRecord,
    TaskResultRef,
)


@pytest.mark.asyncio
async def test_task_context_reads_dependency_body_only_on_demand() -> None:
    value = {"answer": 42}
    digest = canonical_sha256(value)
    dependency = TaskDependency("source", digest, "source-execution")
    calls: list[str] = []

    async def read(current: TaskDependency):
        calls.append(current.execution_id)
        return value

    context = TaskNodeContext(
        app=None,
        principal=Principal("caller", "tenant", "service"),
        graph_id="target",
        node_id="node",
        execution_id="target-execution",
        input={},
        dependencies={"source": dependency},
        idempotency_key="task-context-lazy-read",
        _dependency_reader=read,
    )

    assert calls == []
    assert context.dependencies["source"] == dependency
    assert await context.read_dependency("source") == value
    assert calls == ["source-execution"]


@pytest.mark.asyncio
async def test_prepare_node_authorizes_source_header_and_deduplicates_hold() -> None:
    value = {"answer": 42}
    digest = canonical_sha256(value)
    reference = TaskResultRef(
        "runtime",
        "tenant",
        "source-graph",
        "source-node",
        digest,
    )
    node = TaskNode(
        "target-node",
        input={"type": "example.task", "version": 1},
        input_refs={"first": reference, "second": reference},
    )
    source_header = ResourceRef(
        ResourceKind.TASK_GRAPH,
        "source-graph",
        "tenant",
        "source-owner",
    )

    class TaskState:
        async def get_header(self, graph_id: str, *, tenant_id: str):
            assert (graph_id, tenant_id) == ("source-graph", "tenant")
            return source_header

        async def get_results(
            self,
            graph_id: str,
            node_ids: tuple[str, ...],
            *,
            tenant_id: str,
        ):
            assert (graph_id, tenant_id) == ("source-graph", "tenant")
            assert node_ids == ("source-node",)
            return {
                "source-node": TaskResultRecord(
                    "source-graph",
                    "source-node",
                    digest,
                    execution_id="source-execution",
                )
            }

        async def snapshot_graph(self, graph_id: str, *, tenant_id: str):
            assert (graph_id, tenant_id) == ("source-graph", "tenant")
            return SimpleNamespace(
                node_states=(
                    SimpleNamespace(
                        node_id="source-node",
                        status=TaskStatus.SUCCEEDED,
                        result_digest=digest,
                        execution_id="source-execution",
                    ),
                )
            )

    class Authorization:
        def __init__(self) -> None:
            self.calls: list[tuple[AuthorizationAction, ResourceRef]] = []

        async def authorize(
            self,
            principal: Principal,
            action: AuthorizationAction,
            resource: ResourceRef,
        ) -> None:
            assert principal.principal_id == "caller"
            self.calls.append((action, resource))

    class Execution:
        def __init__(self) -> None:
            self.holds: list[tuple[str, str, str]] = []

        async def inspect(self, execution_id: str, *, principal: Principal):
            assert execution_id == "source-execution"
            assert principal.principal_id == "caller"
            return SimpleNamespace(status=ExecutionStatus.SUCCEEDED)

        async def acquire_dependency_hold(
            self,
            execution_id: str,
            *,
            tenant_id: str,
            hold_id: str,
        ) -> bool:
            self.holds.append((execution_id, tenant_id, hold_id))
            return True

        async def release_dependency_hold(self, *args, **kwargs):
            raise AssertionError("successful preparation must retain its hold")

    authorization = Authorization()
    execution = Execution()
    runner = object.__new__(RuntimeTaskNodeRunner)
    runner._namespace = "runtime"
    runner._authorization = authorization
    runner._task_state = TaskState()
    runner._execution = execution
    runner._task_durable = True
    runner._execution_durable = True
    runner._recovery_durable = True
    runner._agent = SimpleNamespace(type="linktools.ai.agent")

    await runner.prepare_node(
        node,
        graph_id="target-graph",
        principal=Principal("caller", "tenant", "service"),
    )

    assert authorization.calls == [
        (AuthorizationAction.TASK_READ, source_header),
    ]
    assert len(execution.holds) == 1
    assert execution.holds[0][0:2] == ("source-execution", "tenant")
    assert execution.holds[0][2].startswith("task-ref:")
