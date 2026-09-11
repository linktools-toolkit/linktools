#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime lifecycle naming contract coverage."""

import inspect

import linktools.ai.runtime as runtime_api
import linktools.ai.task as task_api
from linktools.ai.runtime import Runtime, StartEvaluationRequest
from linktools.ai.runtime._agent import Agent
from linktools.ai.runtime._evaluation import DefaultEvaluationService
from linktools.ai.runtime.service_api import EvaluationService
from linktools.ai.task import DefaultTaskGraphService, TaskGraphService


def test_evaluation_submission_uses_start_naming() -> None:
    assert hasattr(Agent, "start_evaluation")
    assert not hasattr(Agent, "run_evaluation")
    assert hasattr(DefaultEvaluationService, "start")
    assert not hasattr(DefaultEvaluationService, "run")
    assert hasattr(EvaluationService, "start")
    assert not hasattr(EvaluationService, "run")
    assert runtime_api.StartEvaluationRequest is StartEvaluationRequest
    assert not hasattr(runtime_api, "RunEvaluationRequest")


def test_task_graph_service_uses_canonical_lifecycle_naming() -> None:
    expected = {
        "start",
        "run",
        "inspect",
        "snapshot",
        "wait",
        "cancel",
        "recover",
        "list_events",
        "stream_events",
    }
    legacy = {
        "start_graph",
        "run_graph",
        "inspect_graph",
        "snapshot_graph",
        "wait_graph",
        "cancel_graph",
        "recover_graph",
        "list_graph_events",
        "stream_graph_events",
    }

    for name in expected:
        assert hasattr(TaskGraphService, name)
        assert hasattr(DefaultTaskGraphService, name)
    for name in legacy:
        assert not hasattr(TaskGraphService, name)
        assert not hasattr(DefaultTaskGraphService, name)

    assert runtime_api.TaskGraphService is TaskGraphService
    assert not hasattr(runtime_api, "TaskService")
    assert not hasattr(task_api, "TaskApi")
    assert not hasattr(task_api, "TaskQueryApi")
    assert not hasattr(task_api, "DefaultTaskService")
    assert not hasattr(task_api, "open_local_task_api")
    assert hasattr(task_api, "open_local_task_graph_service")

    parameters = inspect.signature(Runtime.__init__).parameters
    assert "graph" in parameters
    assert "task" not in parameters
