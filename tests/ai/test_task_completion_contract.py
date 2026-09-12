#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task node result identity contract."""

from linktools.ai.core import canonical_sha256
from linktools.ai.task import TaskNode, TaskNodeRunResult


def test_task_node_result_keeps_expansion_outside_execution_identity() -> None:
    digest = canonical_sha256({"value": 1})
    result = TaskNodeRunResult(
        digest,
        execution_id="execution",
        expanded_nodes=(TaskNode("child", input={"type": "app", "version": 1}),),
    )

    assert result.result_digest == digest
    assert result.execution_id == "execution"
    assert result.expanded_nodes[0].node_id == "child"
