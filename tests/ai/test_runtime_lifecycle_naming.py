#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime lifecycle naming contract coverage."""

import linktools.ai.runtime as runtime_api
from linktools.ai.runtime import StartEvaluationRequest
from linktools.ai.runtime._agent import Agent
from linktools.ai.runtime._evaluation import DefaultEvaluationService
from linktools.ai.runtime.service_api import EvaluationService


def test_evaluation_submission_uses_start_naming() -> None:
    assert hasattr(Agent, "start_evaluation")
    assert not hasattr(Agent, "run_evaluation")
    assert hasattr(DefaultEvaluationService, "start")
    assert not hasattr(DefaultEvaluationService, "run")
    assert hasattr(EvaluationService, "start")
    assert not hasattr(EvaluationService, "run")
    assert runtime_api.StartEvaluationRequest is StartEvaluationRequest
    assert not hasattr(runtime_api, "RunEvaluationRequest")
