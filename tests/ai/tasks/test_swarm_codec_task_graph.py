#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YAML task-graph defaults and strict strategy validation."""

import pytest

from linktools.ai.errors import InvalidSpecError
from linktools.ai.tasks.swarm.aggregation import AggregationMode
from linktools.ai.tasks.swarm.codec import SwarmSpecCodec


def test_minimal_task_graph_yaml_keeps_required_defaults():
    spec = SwarmSpecCodec().decode(
        "audit-workers",
        "name: audit-workers\nagents:\n  - agent_id: worker-a\nstrategy:\n  kind: task_graph\n",
    )

    assert spec.strategy.kind == "task_graph"
    assert spec.limits.max_rounds == 1
    assert spec.limits.max_delegations == 0
    assert spec.limits.max_depth == 0
    assert spec.aggregation.mode is AggregationMode.COLLECT


def test_partial_task_graph_limits_are_completed_from_defaults():
    spec = SwarmSpecCodec().decode(
        "audit-workers",
        "name: audit-workers\nagents: [worker-a]\nstrategy:\n  kind: task_graph\nlimits:\n  max_tasks: 3\n",
    )

    assert spec.limits.max_tasks == 3
    assert spec.limits.max_concurrency == 4
    assert spec.limits.max_rounds == 1


@pytest.mark.parametrize(
    "body",
    (
        "aggregation:\n  mode: merge\n",
        "limits:\n  max_rounds: 0\n",
        "limits:\n  max_delegations: -1\n",
    ),
)
def test_task_graph_yaml_rejects_invalid_policy_or_limit(body):
    with pytest.raises(InvalidSpecError):
        SwarmSpecCodec().decode(
            "audit-workers",
            "name: audit-workers\nagents: [worker-a]\nstrategy:\n  kind: task_graph\n"
            + body,
        )
