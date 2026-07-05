#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/events/test_payloads_approval_swarm_resource.py"""
from linktools.ai.events.payloads import (
    ApprovalApproved, ApprovalRejected, ApprovalRequested,
    ResourceChanged,
    SwarmCompleted, SwarmRoundCompleted, SwarmRoundStarted, SwarmStarted,
    SwarmTaskClaimed, SwarmTaskCompleted, SwarmTaskCreated, SwarmTaskFailed,
)


def test_approval_payloads():
    assert ApprovalRequested(approval_id="a1", tool_name="terminal", reason="risky").reason == "risky"
    assert ApprovalApproved(approval_id="a1").resolved_by is None
    assert ApprovalRejected(approval_id="a1").reason is None


def test_swarm_payloads():
    assert SwarmStarted(swarm_run_id="sr1", swarm_id="s1").swarm_id == "s1"
    assert SwarmRoundStarted(swarm_run_id="sr1", round=1).round == 1
    assert SwarmRoundCompleted(swarm_run_id="sr1", round=1).round == 1
    assert SwarmTaskCreated(swarm_run_id="sr1", task_id="t1", description="do x").description == "do x"
    assert SwarmTaskClaimed(swarm_run_id="sr1", task_id="t1", assigned_agent_id="agent-1").assigned_agent_id == "agent-1"
    assert SwarmTaskCompleted(swarm_run_id="sr1", task_id="t1").task_id == "t1"
    assert SwarmTaskFailed(swarm_run_id="sr1", task_id="t1", error_message="boom").error_message == "boom"
    assert SwarmCompleted(swarm_run_id="sr1").swarm_run_id == "sr1"


def test_resource_changed_payload():
    assert ResourceChanged(path="/a.txt", revision=5).revision == 5
