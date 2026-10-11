#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed JSON result references and submission-time graph diagnostics."""

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
import pickle
from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel
from typing_extensions import assert_type

from linktools.ai.core import JsonValue, Principal, TaskStatus, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime.state._codec import decode_domain, encode_domain
from linktools.ai.task import (
    Task, TaskDependency, TaskDependencyState, TaskGraph, TaskNode, TaskNodeContext,
    TaskNodeResultRef, TaskRef, TaskResultRef,
)


def test_explicit_typed_reference_construction_preserves_wire_values() -> None:
    definition = TaskRef[str]("authoring.prepare", 1)
    node = TaskNode[str]("prepare", task=definition)
    symbolic = TaskNodeResultRef[str](node)
    retained = TaskResultRef[str]("namespace", "tenant", "graph", "prepare", "a" * 64)
    assert symbolic == TaskNodeResultRef("prepare")
    for value, kind in ((definition, TaskRef), (node, TaskNode),
                        (symbolic, TaskNodeResultRef), (retained, TaskResultRef)):
        assert decode_domain(encode_domain(value), kind) == value
    with pytest.raises(FrozenInstanceError):
        node.node_id = "changed"  # type: ignore[misc]
    for reference in (definition, symbolic, retained):
        assert replace(reference) == reference
        assert pickle.loads(pickle.dumps(reference)) == reference


@pytest.mark.asyncio
async def test_typed_result_references_keep_json_and_alias_reads() -> None:
    async def prepare(context: TaskNodeContext[None]) -> str:
        assert_type(context.input, Mapping[str, JsonValue])
        return "prepared"

    preparation = Task("authoring.prepare", prepare, effect_policy="none")
    assert_type(Task[None]("authoring.default", prepare), Task[None, JsonValue])
    assert_type(Task[None, str]("authoring.explicit", prepare), Task[None, str])
    node = TaskNode("prepare", task=preparation)
    reference = TaskNodeResultRef(node)
    assert_type(preparation, Task[None, str])
    assert_type(preparation.ref, TaskRef[str])
    assert_type(node, TaskNode[str])
    assert_type(reference, TaskNodeResultRef[str])
    assert reference == TaskNodeResultRef("prepare")
    assert decode_domain(encode_domain(reference), TaskNodeResultRef) == reference

    async def finish(context: TaskNodeContext[None]) -> str:
        value = await context.read_dependency(reference)
        assert_type(value, str)
        if TYPE_CHECKING:
            _not_a_number: int = value  # type: ignore[assignment]
        alias = await context.read_dependency("text")
        assert_type(alias, JsonValue)
        assert alias == value
        return value.upper()

    finishing = Task("authoring.finish", finish, effect_policy="none")
    graph = TaskGraph("typed-results", (
        node,
        TaskNode("finish", (node.node_id,), task=finishing,
                 input_refs={"text": reference}),
    ))
    async with Runtime.open("typed-results", models=ModelRegistry(),
                            storage=RuntimeStorage.in_memory()) as runtime:
        run = await runtime.tasks.bind(preparation, finishing).start(
            graph, idempotency_key="typed-results",
        )
        assert (await run.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
        value = await run.result(reference)
        assert_type(value, str)
        retained = await run.result_ref(reference)
        assert_type(retained, TaskResultRef[str])
        assert value == await run.result("prepare") == "prepared"
        assert retained == await run.result_ref("prepare")
        assert decode_domain(encode_domain(retained), TaskResultRef) == retained
        assert await run.result("finish") == "PREPARED"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("failed", "missing", "denied", "corrupt"))
async def test_typed_dependency_reads_preserve_failure_and_integrity_checks(failure: str) -> None:
    calls: list[str] = []

    async def prepare(context: TaskNodeContext[None]) -> str:
        return "prepared"

    async def read(dependency: TaskDependency) -> JsonValue:
        calls.append(dependency.node_id)
        if failure == "denied":
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return "wrong result"

    reference = TaskNodeResultRef(TaskNode("prepare", task=Task("authoring.prepare", prepare)))
    digest = canonical_sha256("prepared")
    context = TaskNodeContext(
        app=None, principal=Principal("reader", "tenant"), graph_id="graph",
        node_id="finish", execution_id="execution", input={}, idempotency_key="read",
        dependencies={} if failure == "missing" else {
            "prepare": TaskDependency("prepare", digest, "source-execution"),
        },
        dependency_states={"prepare": TaskDependencyState(TaskStatus.FAILED)} if failure == "failed" else {},
        _dependency_reader=read,
    )
    if failure == "missing":
        with pytest.raises(KeyError):
            await context.read_dependency(reference)
    else:
        with pytest.raises(AIError) as caught:
            await context.read_dependency(reference)
        assert caught.value.code is {
            "failed": ErrorCode.TASK_DEPENDENCY_FAILED,
            "denied": ErrorCode.AUTHORIZATION_DENIED,
            "corrupt": ErrorCode.STORAGE_INTEGRITY_ERROR,
        }[failure]
    assert calls == ([] if failure in {"missing", "failed"} else ["prepare"])


@pytest.mark.asyncio
@pytest.mark.parametrize("texts, uppercase, expected", (
    (["Hello", "World"], True, ["HELLO", "WORLD"]),
    (["Hello", "World"], False, ["hello", "world"]),
    ([], True, []),
))
async def test_batch_example_expands_selected_branch_and_collects(
    texts: list[str], uppercase: bool, expected: list[str],
) -> None:
    from examples.task_graph import run

    assert await run(texts, uppercase=uppercase) == expected


class _Answer(BaseModel):
    answer: str


class _DifferentAnswer(BaseModel):
    different: int


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ("input", "output_type", "resolved_output", "unsupported_output"))
async def test_describe_submission_locates_invalid_node_without_execution(invalid: str) -> None:
    calls: list[str] = []

    async def execute(context: TaskNodeContext[None]) -> JsonValue:
        calls.append(context.node_id)
        return {"answer": "yes"}

    def normalize(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        if invalid == "input":
            raise ValueError("private input must not appear in diagnostics")
        return value

    definition = Task("authoring.validate", execute, output_type=_Answer,
                      normalize=normalize, effect_policy="none")
    node = TaskNode("invalid", task=definition)
    if invalid == "output_type":
        node = TaskNode("invalid", task=definition, output_type=_DifferentAnswer)
    elif invalid == "resolved_output":
        node = TaskNode.from_resolved("invalid", task=definition.ref,
            output_contract={"mode": "structured", "schema": _DifferentAnswer.model_json_schema()})
    elif invalid == "unsupported_output":
        node = TaskNode("invalid", task=definition, output_type=int)
    graph = TaskGraph("planning", (node,))
    async with Runtime.open("authoring-errors", models=ModelRegistry(),
                            storage=RuntimeStorage.in_memory()) as runtime:
        engine = runtime.tasks.bind(definition)
        with pytest.raises(AIError) as caught:
            await engine.describe_submission(graph, idempotency_key="plan")
        assert caught.value.code is (
            ErrorCode.REQUEST_FIELD_INVALID if invalid == "input" else ErrorCode.OUTPUT_CONTRACT_INVALID
        )
        assert caught.value.safe_details == {
            "graph_id": "planning", "node_id": "invalid",
            "reason": "task_input_invalid" if invalid == "input" else "task_output_contract_invalid",
        }
        assert "private input" not in str(caught.value)
        assert calls == []
        with pytest.raises(AIError) as missing:
            await engine.get("planning")
        assert missing.value.code is ErrorCode.AUTHORIZATION_DENIED
