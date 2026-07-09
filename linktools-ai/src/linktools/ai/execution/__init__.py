#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai.execution: ExecutionBackend protocol + the local filesystem
backend that supplies builtin file/terminal tools."""

from .local import LocalExecutionBackend
from .protocols import ExecutionBackend

__all__ = ["ExecutionBackend", "LocalExecutionBackend"]
