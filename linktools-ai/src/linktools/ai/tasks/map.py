#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Canonical mapping for the former public task symbols."""

COMPATIBILITY_MAP: "dict[str, str]" = {
    "linktools.ai.tasks.TaskPlan": "linktools.ai.domain.task.TaskPlan",
    "linktools.ai.tasks.TaskExecution": "linktools.ai.domain.task.TaskExecution",
    "linktools.ai.tasks.Job": "linktools.ai.domain.task.Job",
    "linktools.ai.tasks.Swarm": "linktools.ai.domain.task.Swarm",
}

__all__ = ["COMPATIBILITY_MAP"]
