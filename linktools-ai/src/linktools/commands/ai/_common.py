#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared local composition for AI CLI commands."""

import asyncio
import os
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TypeVar

from filelock import FileLock

from linktools.cli import CommandError
from linktools.core import environ

from linktools.ai.capability import CapabilityGroup
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
def _local_models(workspace: Workspace) -> ModelRegistry:
    configured = workspace.config.get("model")
    model = (
        configured.strip()
        if isinstance(configured, str) and configured.strip()
        else os.getenv("OPENAI_MODEL", "").strip()
    )
    if not model:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "model is required")
    raw_vision = os.getenv("OPENAI_VISION")
    try:
        vision = (
            False
            if raw_vision is None or not raw_vision.strip()
            else environ.config.cast(raw_vision, bool)
        )
    except (TypeError, ValueError) as error:
        raise AIError(
            ErrorCode.MODEL_CONFIG_INVALID,
            retryable=False,
            safe_details={"provider": "openai", "field": "vision"},
        ) from error
    return ModelRegistry.openai(
        model=model,
        vision=vision,
        base_url=os.getenv("OPENAI_BASE_URL", "").strip() or None,
        api_key=os.getenv("OPENAI_API_KEY", "").strip() or None,
    )


@asynccontextmanager
async def _open_local_runtime(
    workspace: Workspace,
    *,
    models: "ModelRegistry | None" = None,
) -> AsyncIterator[Runtime]:
    async with Runtime.open(
        workspace.workspace_id,
        state=_local_runtime_state(workspace),
        metrics=await _local_metrics(workspace),
        models=_local_models(workspace) if models is None else models,
        capabilities=(CapabilityGroup.from_workspace(workspace),),
    ) as runtime:
        yield runtime


def _run_async(coroutine: "Coroutine[object, object, ResultT]") -> ResultT:
    try:
        return asyncio.run(coroutine)
    except (AIError, TypeError, ValueError) as error:
        raise CommandError(str(error)) from error


__all__ = []
