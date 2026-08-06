#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task gateway adapter boundary."""

from typing import Protocol

from ..core import Principal
from ..task.model import TaskGraphView


class TaskGateway(Protocol):
    async def inspect_graph(self, graph_id: str, *, principal: Principal) -> TaskGraphView: ...


__all__ = ["TaskGateway"]
