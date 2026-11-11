#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime composition and independent domain public import contracts."""

import subprocess
import sys

import pytest


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
