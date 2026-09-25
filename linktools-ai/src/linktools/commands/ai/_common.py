#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared local composition for AI CLI commands."""

import asyncio
import os
from argparse import Namespace
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from linktools.cli import CommandError
from linktools.cli.argparse import ConfigAction
from linktools.core import ConfigField, environ

from linktools.ai.asset import (
    AssetStore,
    DirectoryAssetBackend,
    PrefixAssetPathAdapter,
)
from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import DEFAULT_DISCOVERY_POLICY
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_metrics_sqlite, validate_metrics_sqlite
from linktools.ai.model import ModelRegistry
from linktools.ai.observe import Metrics
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.storage import FilesystemMutationLock, StorageOverlay
from linktools.ai.workspace import Workspace

ResultT = TypeVar("ResultT")

if TYPE_CHECKING:
    from linktools.cli import CommandParser

OPENAI_BASE_URL = ConfigField(name="OPENAI_BASE_URL", cast=str, default=None)
OPENAI_MODEL = ConfigField(name="OPENAI_MODEL", cast=str, default=None)
OPENAI_API_KEY = ConfigField(name="OPENAI_API_KEY", cast=str, default=None, secret=True)
OPENAI_VISION = ConfigField(name="OPENAI_VISION", cast=bool, default=False)


def _add_local_runtime_arguments(parser: "CommandParser") -> None:
    parser.add_argument("--project", type=Path, default=None, help="working directory")
    parser.add_argument("--base-url", action=ConfigAction, config=OPENAI_BASE_URL)
    parser.add_argument("--model", action=ConfigAction, config=OPENAI_MODEL)
    parser.add_argument("--api-key", action=ConfigAction, config=OPENAI_API_KEY)
    parser.add_argument("--vision", action=ConfigAction, config=OPENAI_VISION)
    parser.add_argument(
        "--memory",
        default=None,
        help="caller-owned memory scope (default: default)",
    )


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


def _local_runtime_assets(workspace: Workspace) -> AssetStore:
    return AssetStore(
        StorageOverlay(
            DirectoryAssetBackend(
                str(workspace.storage_root),
                path_adapter=PrefixAssetPathAdapter(
                    {"agent": "agents", "skill": "skills", "mcp": "mcp"}
                ),
                kinds=("agent", "skill", "mcp"),
                follow_external_symlinks=True,
                ignore_paths=DEFAULT_DISCOVERY_POLICY.ignores,
            )
        )
    )


def _local_runtime_storage(workspace: Workspace) -> RuntimeStorage:
    return RuntimeStorage.from_root(_local_runtime_root(workspace))


async def _local_metrics(workspace: Workspace) -> Metrics:
    runtime_root = _local_runtime_root(workspace)
    runtime_root.mkdir(parents=True, exist_ok=True)
    path = runtime_root / "metrics.db"
    async with FilesystemMutationLock(str(path) + ".lock"):
        if path.exists():
            await validate_metrics_sqlite(path)
        else:
            await provision_metrics_sqlite(path)
    return Metrics.sqlite(path, namespace="default")


def _local_models(
    workspace: Workspace,
    *,
    model: "str | None" = None,
    vision: "bool | None" = None,
    base_url: "str | None" = None,
    api_key: "str | None" = None,
) -> ModelRegistry:
    configured = workspace.config.get("model")
    selected_model = (
        model.strip()
        if isinstance(model, str) and model.strip()
        else configured.strip()
        if isinstance(configured, str) and configured.strip()
        else os.getenv("OPENAI_MODEL", "").strip()
    )
    if not selected_model:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "model is required")

    if vision is None:
        raw_vision = os.getenv("OPENAI_VISION")
        try:
            selected_vision = (
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
    else:
        selected_vision = vision

    return ModelRegistry.openai(
        model=selected_model,
        vision=selected_vision,
        base_url=(
            base_url
            if base_url is not None
            else os.getenv("OPENAI_BASE_URL", "").strip() or None
        ),
        api_key=(
            api_key
            if api_key is not None
            else os.getenv("OPENAI_API_KEY", "").strip() or None
        ),
    )


def _local_runtime_models(workspace: Workspace, args: Namespace) -> ModelRegistry:
    return _local_models(
        workspace,
        model=args.model,
        vision=args.vision,
        base_url=args.base_url,
        api_key=args.api_key,
    )


@asynccontextmanager
async def _open_local_runtime(
    workspace: Workspace,
    *,
    models: "ModelRegistry | None" = None,
) -> AsyncIterator[Runtime]:
    assets = _local_runtime_assets(workspace)
    await assets.initialize()
    try:
        async with Runtime.open(
            "default",
            storage=_local_runtime_storage(workspace),
            metrics=await _local_metrics(workspace),
            models=_local_models(workspace) if models is None else models,
            capabilities=(
                CapabilityGroup(
                    "workspace",
                    workspace=workspace,
                    assets=assets,
                ),
            ),
        ) as runtime:
            yield runtime
    finally:
        await assets.close()


def _run_async(coroutine: "Coroutine[object, object, ResultT]") -> ResultT:
    try:
        return asyncio.run(coroutine)
    except (AIError, TypeError, ValueError) as error:
        raise CommandError(str(error)) from error


__all__ = []
