#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Built-in task handlers."""

from .callable import CallableTaskHandler
from .runtime import (
    MappingRunnableResolver,
    RunnableRef,
    RunnableResolver,
    RuntimeTaskHandler,
    RuntimeTaskInput,
    TaskRunDispatcher,
)

__all__: "list[str]" = [
    "CallableTaskHandler",
    "RuntimeTaskHandler",
    "RunnableRef",
    "RunnableResolver",
    "MappingRunnableResolver",
    "RuntimeTaskInput",
    "TaskRunDispatcher",
]
