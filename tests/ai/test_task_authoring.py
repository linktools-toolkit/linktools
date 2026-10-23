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

from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime.state._codec import decode_domain, encode_domain
from linktools.ai.task import (
    Task, TaskDependency, TaskDependencyState, TaskExpander, TaskExpansionContext,
    TaskGraph, TaskGraphTemplate, TaskNode, TaskNodeContext,
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
    async with Runtime.open("typed-results",
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
async def test_collection_result_types_survive_expansion_and_json_round_trip() -> None:
    async def words(context: TaskNodeContext[None]) -> list[str]:
        return ["one", "two"]

    async def counts(context: TaskNodeContext[None]) -> dict[str, int]:
        return {"one": 1, "two": 2}

    word_task = Task("authoring.words", words, effect_policy="none")
    count_task = Task("authoring.counts", counts, effect_policy="none")
    word_node = TaskNode("words", task=word_task)
    count_node = TaskNode("counts", task=count_task)
    word_ref = TaskNodeResultRef(word_node)
    count_ref = TaskNodeResultRef(count_node)
    assert_type(word_task, Task[None, list[str]])
    assert_type(count_task.ref, TaskRef[dict[str, int]])
    assert_type(word_ref, TaskNodeResultRef[list[str]])
    assert_type(count_node, TaskNode[dict[str, int]])
    if TYPE_CHECKING:
        TaskNode[list[int]]("wrong", task=word_task)  # type: ignore[arg-type]
        TaskNodeResultRef[list[int]](word_node)  # type: ignore[arg-type]

    async def collect(context: TaskNodeContext[None]) -> dict[str, list[dict[str, int]]]:
        names = await context.read_dependency(word_ref)
        totals = await context.read_dependency(count_ref)
        assert_type(names, list[str])
        assert_type(totals, dict[str, int])
        if TYPE_CHECKING:
            _not_numbers: list[int] = names  # type: ignore[assignment]
            _not_text: dict[str, str] = totals  # type: ignore[assignment]
        assert await context.read_dependency("names") == names
        return {"groups": [{name: totals[name]} for name in names]}

    collect_task = Task("authoring.collect", collect, effect_policy="none")
    collect_node = TaskNode("collect", ("words", "counts"), task=collect_task,
                            input_refs={"names": word_ref})
    collected = TaskNodeResultRef(collect_node)

    def expand(context: TaskExpansionContext) -> tuple[TaskNode[object], ...]:
        return (collect_node,)

    async def dispatch(context: TaskNodeContext[None]) -> JsonValue:
        return None

    dispatch_task = Task[None]("authoring.dispatch", dispatch, effect_policy="none")
    assert_type(dispatch_task, Task[None, JsonValue])
    expander = TaskExpander("authoring.expand", expand)
    template = TaskGraphTemplate((word_node, count_node, TaskNode(
        "dispatch", task=dispatch_task, expander=expander,
    )))
    assert decode_domain(encode_domain(template), TaskGraphTemplate) == template
    graph = TaskGraph("typed-collections", template.nodes)
    async with Runtime.open("typed-collections",
                            storage=RuntimeStorage.in_memory()) as runtime:
        engine = runtime.tasks.bind(word_task, count_task, expander).with_definitions(
            collect_task, dispatch_task,
        )
        assert_type(engine.definition(word_task.ref), Task[None, list[str]])
        assert_type(engine.definition(TaskRef("authoring.dispatch", 1)), Task[None, JsonValue])
        run = await engine.start(graph, idempotency_key="typed-collections")
        assert (await run.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
        value = await run.result(collected)
        assert_type(value, dict[str, list[dict[str, int]]])
        assert value == {"groups": [{"one": 1}, {"two": 2}]}
        assert await run.result(word_ref) == ["one", "two"]
        retained = await run.result_ref(collected)
        assert_type(retained, TaskResultRef[dict[str, list[dict[str, int]]]])
        assert decode_domain(encode_domain(retained), TaskResultRef) == retained
        reuse = TaskNode("reuse", task=collect_task, input_refs={"collected": retained})
        assert decode_domain(encode_domain(reuse), TaskNode) == reuse


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
    async with Runtime.open("authoring-errors",
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


@pytest.mark.asyncio
@pytest.mark.parametrize("model_output", (False, True))
async def test_result_type_annotations_do_not_bypass_json_validation(model_output: bool) -> None:
    async def invalid(context: TaskNodeContext[None]) -> object:
        return _Answer(answer="not a JSON object") if model_output else object()

    definition = Task("authoring.invalid", invalid, effect_policy="none",
                      output_type=_Answer if model_output else None)
    node = TaskNode("invalid", task=definition)
    reference = TaskNodeResultRef(node)
    async with Runtime.open("invalid-json-result",
                            storage=RuntimeStorage.in_memory()) as runtime:
        run = await runtime.tasks.bind(definition).start(
            TaskGraph("invalid-json-result", (node,)), idempotency_key="invalid-json-result",
        )
        assert (await run.wait(timeout_seconds=10)).result.status is TaskStatus.FAILED
        with pytest.raises(AIError) as caught:
            await run.result(reference)
        assert caught.value.code is ErrorCode.TASK_NODE_FAILED
        assert caught.value.safe_details["error_code"] == ErrorCode.OUTPUT_CONTRACT_INVALID.value
