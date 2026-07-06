#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RunnableSpec: the common Protocol both AgentSpec and SwarmSpec satisfy,
so Runtime.run() can type-hint a single union/Protocol instead of isinstance dispatch."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class RunnableSpec(Protocol):
    id: str
    name: str
