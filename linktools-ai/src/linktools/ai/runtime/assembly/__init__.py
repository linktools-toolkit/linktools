#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime assembly internals: the support modules that back an ASSEMBLED
runtime but are not part of the public top-level runtime layout (facade /
builder / dependencies / dispatcher). Holds capability inspection
(:func:`inspect_capabilities`, behind ``Runtime.inspect``) and the shared
run-lifecycle helpers (:func:`resolve_session` / :func:`create_run_context`)
used by the run entry points."""

from .inspection import inspect_capabilities
from .lifecycle import create_run_context, resolve_session

__all__ = [
    "inspect_capabilities",
    "resolve_session",
    "create_run_context",
]
