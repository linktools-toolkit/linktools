#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation executors: drive a target against a case and produce an
EvalExecution. DirectEvalExecutor runs inline through the existing Runtime;
TaskEvalExecutor routes a case through TaskRuntime (task mode, gaining
retries + lease/recovery + retry_count capture)."""

from .direct import DirectEvalExecutor, EvalTargetResolver
from .task import TaskEvalExecutor

__all__: "list[str]" = [
    "DirectEvalExecutor",
    "EvalTargetResolver",
    "TaskEvalExecutor",
]
