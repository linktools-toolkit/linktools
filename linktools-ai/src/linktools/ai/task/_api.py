"""Standalone local Task API composition root."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from ..core import AuthorizationPolicy
from ._local import LocalTaskGraphLauncher, TaskNodeRunner
from ._service import TaskApi
from ._service_impl import DefaultTaskService, TaskPersistence


@asynccontextmanager
async def open_local_task_api(
    persistence: TaskPersistence,
    authorization: AuthorizationPolicy,
    *,
    runner: TaskNodeRunner,
    owner: str,
) -> "AsyncIterator[TaskApi]":
    """Open only the durable local Task launcher and service façade."""
    launcher = LocalTaskGraphLauncher(persistence.tasks, runner, owner=owner)
    service = DefaultTaskService(persistence, authorization, launcher)
    try:
        yield service
    finally:
        await launcher.shutdown()


__all__ = ["open_local_task_api"]
