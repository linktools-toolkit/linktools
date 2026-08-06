#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DAG inspection helpers."""

from .model import TaskGraph, TaskNode


def ready_nodes(graph: TaskGraph, completed: 'frozenset[str]') -> 'tuple[TaskNode, ...]':
    return tuple(
        node for node in graph.nodes if node.task_id not in completed and all(dependency in completed for dependency in node.dependencies)
    )


__all__ = ["ready_nodes"]
