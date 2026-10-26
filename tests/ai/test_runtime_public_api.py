#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime composition and independent domain public import contracts."""

import subprocess
import sys

import pytest


def test_runtime_exports_only_composition_and_domain_entry_points() -> None:
    from linktools.ai import runtime

    expected = {
        "Agent", "Session", "Runtime", "RuntimeStatus", "RuntimeFeatures",
        "RuntimeFeaturesCapture", "RuntimeAgents", "RuntimeExecutions",
        "RuntimeMetrics", "RuntimeSessions", "RuntimeTasks", "RuntimeEvaluations",
        "RuntimeMetricsStatus", "RuntimeMetricsFlushResult",
    }
    assert set(runtime.__all__) == expected
    namespace: dict[str, object] = {}
    exec("from linktools.ai.runtime import *", namespace)
    assert {name for name in namespace if not name.startswith("_")} == expected
    for name in expected:
        assert namespace[name] is getattr(runtime, name)


@pytest.mark.parametrize("first", ["runtime", "execution", "inputs", "task", "observe"])
def test_domain_contracts_resolve_in_fresh_import_orders(first: str) -> None:
    code = f"""
import importlib
importlib.import_module('linktools.ai.{first}')
from linktools.ai import Execution as RootExecution
from linktools.ai.runtime import Agent, Runtime, RuntimeFeatures, Session
from linktools.ai.execution import Execution, ExecutionInputContext, ExecutionRequest, ToolEffectApplied
from linktools.ai.inputs import AgentInputCapture, AgentTaskInput, CaptureInputRequest, CaptureGraphRequest
from linktools.ai.task import CancelGraphRequest, TaskEvent, TaskGraphService
from linktools.ai.core import PromptLimits
from linktools.ai.errors import ObservationError
from linktools.ai.observe import WaitResult
assert RootExecution is Execution
for symbol in (Agent, Runtime, RuntimeFeatures, Session, Execution, ExecutionInputContext,
               ExecutionRequest, ToolEffectApplied, AgentInputCapture, AgentTaskInput,
               CaptureInputRequest, CaptureGraphRequest, CancelGraphRequest, TaskEvent,
               TaskGraphService, PromptLimits, ObservationError, WaitResult):
    assert symbol is not None
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
