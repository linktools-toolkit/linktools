#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime assembly internals: the support modules that back an ASSEMBLED
runtime but are not part of the public top-level runtime layout (facade /
builder / dependencies / dispatcher). Holds capability inspection
(:func:`inspect_capabilities`, behind ``Runtime.inspect``)."""

from .inspection import inspect_capabilities

__all__ = ["inspect_capabilities"]
