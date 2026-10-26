#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conditional batch expansion and aggregation using public Task APIs."""

from linktools.ai.core import JsonValue
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.task_engine import TaskEngine
from linktools.ai.task import (
    Task, TaskExpander, TaskExpansionContext, TaskGraph, TaskNode, TaskNodeContext,
)


async def plan(context: TaskNodeContext[None]) -> dict[str, JsonValue]:
    return dict(context.input)


async def uppercase(context: TaskNodeContext[None]) -> str:
    text = context.input["text"]
    assert isinstance(text, str)
    return text.upper()


async def lowercase(context: TaskNodeContext[None]) -> str:
    text = context.input["text"]
    assert isinstance(text, str)
    return text.lower()


async def collect(context: TaskNodeContext[None]) -> list[JsonValue]:
    return [await context.read_dependency(name) for name in sorted(context.dependencies)]


plan_task = Task("example.batch.plan", plan, effect_policy="none")
uppercase_task = Task("example.batch.uppercase", uppercase, effect_policy="none")
lowercase_task = Task("example.batch.lowercase", lowercase, effect_policy="none")
collect_task = Task("example.batch.collect", collect, effect_policy="none")


def expand(context: TaskExpansionContext) -> tuple[TaskNode, ...]:
    value = context.output
    assert isinstance(value, dict)
    texts = value["texts"]
    assert isinstance(texts, list)
    worker = uppercase_task if value["uppercase"] else lowercase_task
    nodes = tuple(
        TaskNode(f"item-{index:04d}", task=worker, input={"text": text})
        for index, text in enumerate(texts)
    )
    return (*nodes, TaskNode("collect", tuple(node.node_id for node in nodes), task=collect_task))


batch_expander = TaskExpander("example.batch.expand", expand)


async def run(texts: list[str], *, uppercase: bool = True) -> JsonValue:
    """Run the chosen branch for each item, then collect in input order."""
    graph = TaskGraph("batch", (TaskNode(
        "plan", task=plan_task, input={"texts": list(texts), "uppercase": uppercase},
        expander=batch_expander,
    ),))
    async with Persistence.open(
        "batch-example", storage=PersistenceStorage.in_memory(),
    ) as persistence, TaskEngine.open(
        plan_task, uppercase_task, lowercase_task, collect_task, batch_expander,
        persistence=persistence,
    ) as engine:
        await engine.describe_submission(graph, idempotency_key="batch")
        execution = await engine.start(graph, idempotency_key="batch")
        await execution.wait(timeout_seconds=10)
        return await execution.result("collect")


if __name__ == "__main__":
    import asyncio

    print(asyncio.run(run(["Hello", "World"])))
