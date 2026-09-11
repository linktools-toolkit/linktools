#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone local TaskGraph service composition root."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from ..core import AuthorizationPolicy
from ._local import LocalTaskGraphLauncher, TaskNodeRunner
from ._service import TaskGraphService
from ._service_impl import DefaultTaskService, TaskPersistence


@asynccontextmanager
async def open_local_task_graph_service(
    persistence: TaskPersistence,
    authorization: AuthorizationPolicy,
    *,
    runner: TaskNodeRunner,
    owner: str,
) -> "AsyncIterator[TaskGraphService]":
    """Open the durable local TaskGraph service."""
    launcher = LocalTaskGraphLauncher(persistence.tasks, runner, owner=owner)
    service = DefaultTaskService(persistence, authorization, launcher)
    try:
        await service.recover_pending()
        yield service
    finally:
        await service.drain_owned_finalizers()
        await service.preflight_close()
        await launcher.shutdown()


__all__ = ["open_local_task_graph_service"]
