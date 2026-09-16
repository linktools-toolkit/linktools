#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared local composition for AI CLI commands."""

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_metrics_sqlite
from linktools.ai.observe import Metrics
from linktools.ai.runtime import RuntimeState
from linktools.ai.workspace import Workspace


def _load_workspace(root: "Path | None" = None) -> Workspace:
    workspace_root = Path.cwd() if root is None else root
    try:
        return Workspace.discover(Path.cwd(), root=workspace_root)
    except AIError as error:
        if error.code is not ErrorCode.WORKSPACE_CONFIG_INVALID:
            raise
        return Workspace.initialize(workspace_root)


def _local_runtime_root(workspace: Workspace) -> Path:
    return workspace.storage_root / "runtime"


def _local_runtime_state(workspace: Workspace) -> RuntimeState:
    return RuntimeState.from_root(_local_runtime_root(workspace))


async def _local_metrics(workspace: Workspace) -> Metrics:
    runtime_root = _local_runtime_root(workspace)
    runtime_root.mkdir(parents=True, exist_ok=True)
    path = runtime_root / "metrics.db"
    if not path.exists():
        await provision_metrics_sqlite(path)
    return Metrics.sqlite(path, namespace=workspace.workspace_id)


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _json_dumps(value: object, *, indent: "int | None" = None) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
    )


__all__ = []
