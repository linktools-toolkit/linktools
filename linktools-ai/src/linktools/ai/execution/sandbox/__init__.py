#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai.execution.sandbox: isolated (non-trusted-local) execution
backend implementations. LocalExecutionBackend stays at the execution/ top
level (it is the trusted-local path, not an isolation boundary)."""

from .container import ContainerExecutionBackend

__all__ = ["ContainerExecutionBackend"]
