#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared local composition for AI CLI commands."""

import asyncio
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TypeVar

from filelock import FileLock

from linktools.cli import CommandError

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_metrics_sqlite, validate_metrics_sqlite
from linktools.ai.model import ModelRegistry
from linktools.ai.observe import Metrics
from linktools.ai.runtime import Runtime, RuntimeState
from linktools.ai.workspace import Workspace

ResultT = TypeVar("ResultT")


def _load_workspace(root: "Path | None" = None) -> Workspace:
    start = Path.cwd()
    try:
        return Workspace.discover(start) if root is None else Workspace.discover(start, root=root)
    except AIError as error:
        if error.code is not ErrorCode.WORKSPACE_CONFIG_INVALID:
            raise
        return Workspace.initialize(start if root is None else root)


def _local_runtime_root(workspace: Workspace) -> Path:
    return workspace.storage_root / "runtime"


def _local_runtime_state(workspace: Workspace) -> RuntimeState:
    return RuntimeState.from_root(_local_runtime_root(workspace))


async def _local_metrics(workspace: Workspace) -> Metrics:
    runtime_root = _local_runtime_root(workspace)
    runtime_root.mkdir(parents=True, exist_ok=True)
    path = runtime_root / "metrics.db"
    with FileLock(str(path) + ".lock"):
        if path.exists():
            await validate_metrics_sqlite(path)
        else:
            await provision_metrics_sqlite(path)
    return Metrics.sqlite(path, namespace=workspace.workspace_id)


@asynccontextmanager
async def _open_local_runtime(
    workspace: Workspace,
    *,
    models: "ModelRegistry | None" = None,
) -> AsyncIterator[Runtime]:
    async with Runtime.open(
        workspace,
        state=_local_runtime_state(workspace),
        metrics=await _local_metrics(workspace),
        models=models,
    ) as runtime:
        yield runtime


def _run_async(coroutine: "Coroutine[object, object, ResultT]") -> ResultT:
    try:
        return asyncio.run(coroutine)
    except (AIError, TypeError, ValueError) as error:
        raise CommandError(str(error)) from error


__all__ = []
