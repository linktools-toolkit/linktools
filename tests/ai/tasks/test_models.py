#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TaskPlan/TaskNode construction-time validation regression tests.

Covers the mandatory test scenarios: empty IDs, duplicate node/agent,
missing dependency, self-dependency, duplicate edge, and cycle. Each must
raise InvalidSpecError at construction time."""
import pytest

from linktools.ai.errors import InvalidSpecError
from linktools.ai.tasks.models import (
    DependencyFailurePolicy,
    TaskDependency,
    TaskGraphNodePayload,
    TaskNode,
    TaskPlan,
)


def _node(
    nid: str,
    agent: str = "agent-a",
    deps: "tuple[TaskDependency, ...]" = (),
) -> TaskNode:
    return TaskNode(nid, TaskGraphNodePayload(agent_id=agent, prompt="x"), dependencies=deps)


def test_empty_plan_id_rejected():
    with pytest.raises(InvalidSpecError, match="plan id"):
        TaskPlan("", (_node("a"),))


def test_empty_node_id_rejected():
    with pytest.raises(InvalidSpecError, match="node id"):
        TaskNode("", TaskGraphNodePayload("agent-a", "x"))


def test_empty_agent_id_rejected():
    with pytest.raises(InvalidSpecError, match="agent_id"):
        TaskNode("a", TaskGraphNodePayload("", "x"))


def test_duplicate_node_id_rejected():
    with pytest.raises(InvalidSpecError, match="duplicate node"):
        TaskPlan("p", (_node("a"), _node("a")))


def test_duplicate_agent_id_rejected():
    with pytest.raises(InvalidSpecError, match="agent .* appears twice"):
        TaskPlan("p", (_node("a", agent="dup"), _node("b", agent="dup")))


def test_missing_dependency_rejected():
    with pytest.raises(InvalidSpecError, match="missing node"):
        TaskPlan("p", (_node("a", deps=(TaskDependency("ghost"),)),))


def test_self_dependency_rejected():
    with pytest.raises(InvalidSpecError, match="self dependency"):
        TaskPlan("p", (_node("a", deps=(TaskDependency("a"),)),))


def test_duplicate_edge_rejected():
    with pytest.raises(InvalidSpecError, match="duplicate dependency"):
        TaskPlan(
            "p",
            (
                _node("a"),
                _node("b", deps=(TaskDependency("a"), TaskDependency("a"))),
            ),
        )


def test_cycle_rejected():
    with pytest.raises(InvalidSpecError, match="cycle"):
        TaskPlan(
            "p",
            (
                _node("a", agent="x", deps=(TaskDependency("c"),)),
                _node("b", agent="y", deps=(TaskDependency("a"),)),
                _node("c", agent="z", deps=(TaskDependency("b"),)),
            ),
        )


def test_empty_dependency_node_id_rejected():
    with pytest.raises(InvalidSpecError, match="dependency node_id"):
        TaskNode("a", TaskGraphNodePayload("agent-a", "x"), dependencies=(TaskDependency(""),))
