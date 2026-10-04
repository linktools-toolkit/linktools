#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task graph identity uses Task-owned semantic projections."""

from dataclasses import replace

import pytest

from linktools.ai.core import Principal, canonical_sha256, principal_identity_payload
from linktools.ai.runtime.state._codec import decode_domain, encode_domain
from linktools.ai.task import (
    TaskExpanderRef,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphLimits,
    TaskGraphRequest,
    TaskGraphTemplate,
    TaskInvocationInputRef,
    TaskNode,
    TaskNodeResultRef,
    TaskRef,
    TaskResultRef,
)


def _resolved_node() -> TaskNode:
    return TaskNode.from_resolved(
        "worker", ("upstream",), task=TaskRef("worker.task", 2),
        input={"value": 2}, original_input={"value": 1}, budget_cost=3,
        expander=TaskExpanderRef("worker.expander", 4),
        input_refs={
            "local": TaskNodeResultRef("upstream"),
            "external": TaskResultRef("ns", "tenant", "source", "result", "d" * 64),
        },
        timeout_seconds=5, max_attempts=2, retry_delay_seconds=1,
        output_contract={"type": "integer"}, effect_policy="replay_safe",
        reconcile=True, dependency_policy="all_terminal", failure_policy="isolate",
    )


def test_task_node_projection_preserves_resolved_execution_semantics() -> None:
    node = _resolved_node()
    expected = {
        "node_id": "worker", "dependencies": ["upstream"],
        "input": {"value": 2}, "original_input": {"value": 1}, "budget_cost": 3,
        "task": {"id": "worker.task", "revision": 2},
        "expander": {"id": "worker.expander", "revision": 4},
        "input_refs": {
            "local": {"node_id": "upstream"},
            "external": {"namespace": "ns", "tenant_id": "tenant", "graph_id": "source",
                         "node_id": "result", "result_digest": "d" * 64},
        },
        "timeout_seconds": 5.0, "max_attempts": 2, "retry_delay_seconds": 1.0,
        "output_contract": {"type": "integer"}, "effect_policy": "replay_safe",
        "reconcile": True, "dependency_policy": "all_terminal", "failure_policy": "isolate",
    }
    assert node.to_mapping() == expected
    restored = decode_domain(encode_domain(node), TaskNode)
    assert restored.to_mapping() == expected
    request = TaskGraphRequest(
        TaskGraph("graph", (node, TaskNode("upstream"))), Principal("user", "tenant"), "submit-graph",
    )
    assert TaskGraphAdmission.from_request(request).initial_request_digest == canonical_sha256({
        "principal": principal_identity_payload(request.principal),
        "graph_id": "graph", "nodes": [TaskNode("upstream").to_mapping(), expected],
        "limits": {"max_nodes": 128, "max_depth": 8, "max_budget": 1000, "max_concurrency": 8},
    })


def test_task_node_projection_preserves_capture_scope_and_provenance() -> None:
    capture = TaskInvocationInputRef("ns", "tenant", "capture", "c" * 64, "execution")
    node = TaskNode("worker", task=TaskRef("worker.task", 1), input_capture=capture)
    expected = {"namespace": "ns", "tenant_id": "tenant", "capture_id": "capture",
                "digest": "c" * 64, "source_execution_id": "execution"}
    assert node.to_mapping()["input_capture"] == expected
    assert decode_domain(encode_domain(node), TaskNode).to_mapping() == node.to_mapping()


def test_task_node_projection_keeps_prompt_intent_independent_of_storage() -> None:
    prompt = {"kind": "text", "text": "hello"}
    direct = TaskNode("worker", task=TaskRef("worker.task", 1),
                      input={"kind": "agent-task-input", "prompt": prompt})
    stored = TaskNode("worker", task=TaskRef("worker.task", 1), input={
        "kind": "agent-task-input", "prompt": {
            "kind": "stored-user-content-v1", "source_intent_digest": canonical_sha256(prompt),
            "physical_object_id": "one",
        },
    })
    assert direct.to_mapping() == stored.to_mapping()


def test_task_graph_template_projection_round_trips_all_semantics() -> None:
    template = TaskGraphTemplate(
        (_resolved_node(), TaskNode("upstream")), TaskGraphLimits(2, 3, 4, 5),
        task_contracts=({"kind": "task", "id": "worker.task", "revision": 2},),
        expander_contracts=({"kind": "expander", "id": "worker.expander", "revision": 4},),
        context_policy="captured",
    )
    mapping = template.to_mapping()
    assert mapping == {
        "contract": "task-graph-template-v1",
        "nodes": [TaskNode("upstream").to_mapping(), _resolved_node().to_mapping()],
        "limits": {"max_concurrency": 2, "max_depth": 3, "max_nodes": 4, "max_budget": 5},
        "task_contracts": [{"kind": "task", "id": "worker.task", "revision": 2}],
        "expander_contracts": [{"kind": "expander", "id": "worker.expander", "revision": 4}],
        "context_policy": "captured",
    }
    assert decode_domain(encode_domain(template), TaskGraphTemplate).to_mapping() == mapping
    assert replace(template, nodes=tuple(reversed(template.nodes))).to_mapping() == mapping
    assert canonical_sha256(replace(template, context_policy="clean").to_mapping()) != canonical_sha256(mapping)
    assert canonical_sha256(replace(template, limits=None).to_mapping()) != canonical_sha256(mapping)


def test_task_graph_template_projection_rejects_unresolved_output_type() -> None:
    template = TaskGraphTemplate((TaskNode("worker", output_type=int),))
    with pytest.raises(ValueError, match="resolved output contracts"):
        template.to_mapping()
