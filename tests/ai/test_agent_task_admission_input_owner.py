#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Admission input routing keeps Agent conversion behind its runner owner."""
from __future__ import annotations

import asyncio
from dataclasses import fields
from types import SimpleNamespace

import pytest

from linktools.ai.core import canonical_sha256, service_principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.inputs import AgentTaskInput
from linktools.ai.model import StoredUserInput
from linktools.ai.persistence.codec import encode_domain
from linktools.ai.storage import StoredPayload
from linktools.ai.task import TaskGraph, TaskNode, TaskRef, TaskInvocationInputRef, TaskExpanderRef, TaskNodeResultRef
from linktools.ai.task._agent_node import AgentTaskNode
from linktools.ai.task.engine._agent import AgentTaskRunner
from linktools.ai.task.engine._runner import TaskNodeExecutor


class _Port:
    def __init__(self, calls: list[object], failure: BaseException | None = None) -> None:
        self.calls = calls
        self.failure = failure

    async def canonicalize_files(self, files: tuple[str, ...]) -> tuple[str, ...]:
        self.calls.append(("canonicalize", files))
        if self.failure is not None:
            raise self.failure
        return ("canonical/file.txt",)

    async def materialize_input(self, prompt: object, files: tuple[str, ...]) -> str:
        self.calls.append(("materialize", prompt, files))
        if self.failure is not None:
            raise self.failure
        return "frozen content"


class _Materializer:
    def __init__(self, calls: list[object]) -> None:
        self.calls = calls
        self.stored = StoredUserInput("text", StoredPayload.inline_text("frozen content"))

    async def store(self, content: object, *, tenant_id: str) -> StoredUserInput:
        self.calls.append(("store", content, tenant_id))
        return self.stored


def _setup(
    mode: str = "literal", *, port: bool = True, materializer: bool = True,
) -> tuple[TaskNodeExecutor, AgentTaskRunner, dict[tuple[str, int], SimpleNamespace], list[object]]:
    calls: list[object] = []
    runner = object.__new__(AgentTaskRunner)
    runner.execution_port = _Port(calls) if port else None
    # Deliberately opposite: captured task contract, not mutable runner input_mode, wins.
    runner.input_mode = "projected" if mode == "literal" else "literal"

    async def builder(context: object) -> str:
        pytest.fail("admission must not execute the input builder")
    runner._build_input = builder
    task = SimpleNamespace(runner=runner, contract={"type": "agent", "config": {"input_mode": mode}})
    executor = object.__new__(TaskNodeExecutor)
    executor._input_materializer = _Materializer(calls) if materializer else None
    return executor, runner, {("agent", 1): task}, calls


def _node(*, files: tuple[str, ...] = (), capture: bool = False, name: str = "node", rich: bool = False) -> AgentTaskNode:
    return AgentTaskNode.from_resolved(
        name, ("parent",) if rich else (), task=TaskRef("agent", 1),
        expander=TaskExpanderRef("expand", 2) if rich else None,
        input_refs={"prior": TaskNodeResultRef("parent")} if rich else None,
        output_contract={"kind": "json"} if rich else None, input=None if capture else AgentTaskInput("question", files=files, planning=False),
        input_capture=(TaskInvocationInputRef("ns", "tenant", "capture", "a" * 64, "source") if capture else None),
        original_input={"source": "original"}, budget_cost=3, timeout_seconds=12,
        max_attempts=2, retry_delay_seconds=0.25, effect_policy="replay_safe" if rich else "none", reconcile=rich,
        dependency_policy="all_succeeded", failure_policy="isolate" if rich else "propagate",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ("capture", "nonagent", "missing", "plain", "projected", "stored", "no-task"))
async def test_admission_short_circuits_do_not_touch_services(case: str) -> None:
    executor, runner, tasks, calls = _setup("projected" if case == "projected" else "literal", materializer=False)
    node = _node(capture=case == "capture")
    if case == "capture":
        runner.execution_port = None
    elif case == "nonagent":
        tasks[("agent", 1)].contract = {"type": "function"}
    elif case == "missing":
        tasks = {}
    elif case == "no-task":
        node = TaskNode("expander-only")
    elif case == "stored":
        body = dict(node.input)
        body["prompt"] = {"kind": "stored-user-content-v1", "intent": "task-admission",
                          "source_intent_digest": canonical_sha256(body["prompt"]),
                          "value": encode_domain(StoredUserInput("text", StoredPayload.inline_text("old")))}
        node = AgentTaskNode.from_resolved("node", task=node.task, input=body)
    assert await executor._materialize_node_input(node, "tenant", tasks) is node
    assert calls == []


@pytest.mark.asyncio
async def test_missing_port_precedes_input_decode_and_materializer() -> None:
    executor, runner, tasks, calls = _setup(port=False, materializer=False)
    node = AgentTaskNode.from_resolved("node", task=TaskRef("agent", 1), input={"invalid": True})
    with pytest.raises(AIError) as caught:
        await executor._materialize_node_input(node, "tenant", tasks)
    assert caught.value.code is ErrorCode.BINDING_NOT_REGISTERED
    tasks[("agent", 1)].runner = object()
    with pytest.raises(AIError) as caught:
        await executor._materialize_node_input(node, "tenant", tasks)
    assert caught.value.code is ErrorCode.BINDING_NOT_REGISTERED
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("literal", "projected"))
async def test_conversion_keeps_all_node_fields_and_captured_mode(mode: str) -> None:
    executor, runner, tasks, calls = _setup(mode)
    node = _node(files=("draft/file.txt",), rich=True)
    converted = await executor._materialize_node_input(node, "target-tenant", tasks)
    assert isinstance(converted, AgentTaskNode) and converted is not node
    for field in fields(node):
        if field.name != "_input":
            assert getattr(converted, field.name) == getattr(node, field.name), field.name
    if mode == "projected":
        assert calls == [("canonicalize", ("draft/file.txt",))]
        assert converted.input["files"] == ["canonical/file.txt"]
        assert converted.input["prompt"] == node.input["prompt"]
    else:
        assert calls == [("materialize", "question", ("draft/file.txt",)), ("store", "frozen content", "target-tenant")]
        prompt = converted.input["prompt"]
        assert prompt["intent"] == "task-admission"
        assert prompt["source_intent_digest"] == canonical_sha256(node.input["prompt"])
        assert prompt["value"] == encode_domain(executor._input_materializer.stored)


@pytest.mark.asyncio
async def test_projected_files_require_materializer_before_canonicalization() -> None:
    executor, runner, tasks, calls = _setup("projected", materializer=False)
    with pytest.raises(AIError) as caught:
        await executor._materialize_node_input(_node(files=("file",)), "tenant", tasks)
    assert caught.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", (False, True))
async def test_capture_admission_stops_on_failure_before_later_nodes(cancel: bool) -> None:
    executor, runner, tasks, calls = _setup()
    failure = asyncio.CancelledError() if cancel else RuntimeError("input failure")
    runner.execution_port.failure = failure
    executor._active_definitions = {}
    executor._default_definitions = (tasks, {})
    executor._admitted_binding_captures = {}
    executor._input_captures = None

    class Capture:
        async def capture(self, admission: object, graph: TaskGraph, **kwargs: object) -> object:
            calls.append("binding-capture")
            return object()
    executor._binding_capture_store = Capture()
    admission = SimpleNamespace(graph_id="graph", principal=service_principal("tenant", "caller"), budget_scope_id=None)
    graph = TaskGraph("graph", (_node(files=("first",), name="first"), _node(files=("second",), name="second")))
    with pytest.raises(type(failure)) as caught:
        await executor.capture_admission(admission, graph)
    assert caught.value is failure
    assert calls == ["binding-capture", ("materialize", "question", ("first",))]
