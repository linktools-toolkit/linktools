#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""State command entry points."""

from typing import TYPE_CHECKING

from ._execution_commands import ExecutionStateCommands
from ._runtime_commands import StateCommands

from ._execution_management import StoredExecutionManagementAdapter
from ._recovery_commands import RuntimeRecoveryCommands

if TYPE_CHECKING:
    from ._task_repository import TaskRepositoryImpl


def graph_task_repository(repository: object) -> "TaskRepositoryImpl | None":
    """Select the native grouped Task adapter, not an arbitrary implementation."""
    from ._task_repository import TaskRepositoryImpl

    return repository if type(repository) is TaskRepositoryImpl else None


__all__ = ['ExecutionStateCommands', 'RuntimeRecoveryCommands', 'StateCommands', 'StoredExecutionManagementAdapter', 'graph_task_repository']
