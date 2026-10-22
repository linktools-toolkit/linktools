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
from linktools.ai.core import DEFAULT_DISCOVERY_POLICY
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_metrics_sqlite
from linktools.ai.observe import Metrics
from linktools.ai.storage import FilesystemMutationLock, StorageOverlay
from linktools.ai.workspace import Workspace

ResultT = TypeVar("ResultT")

if TYPE_CHECKING:
    from linktools.cli import CommandParser
    from linktools.ai.model import ModelBinding, create_model
    from linktools.ai.runtime import Runtime, RuntimeStorage

MODEL_APIS = ("openai", "openai-responses", "anthropic")
MODEL_API = ConfigField(name="MODEL_API", cast=str, default=None)


def _add_local_runtime_arguments(parser: "CommandParser") -> None:
    parser.add_argument("--project", type=Path, default=None, help="working directory")
    parser.add_argument("--model-api", action=ConfigAction, config=MODEL_API, choices=MODEL_APIS,
                        help="model API protocol (default: openai)")
    parser.add_argument("--base-url", default=None, help="selected protocol endpoint")
    parser.add_argument("--model", default=None, help="selected protocol model name")
    parser.add_argument("--api-key", default=None, help="one-shot selected protocol credential")
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


def _local_runtime_storage(workspace: Workspace) -> "RuntimeStorage":
    from linktools.ai.runtime import RuntimeStorage

    return RuntimeStorage.from_root(_local_runtime_root(workspace))


async def _local_metrics(workspace: Workspace) -> Metrics:
    runtime_root = _local_runtime_root(workspace)
    runtime_root.mkdir(parents=True, exist_ok=True)
    path = runtime_root / "metrics.db"
    async with FilesystemMutationLock(str(path) + ".lock"):
        if not path.exists():
            await provision_metrics_sqlite(path)
    return Metrics.sqlite(path, namespace="default")


def _local_input_config(
    workspace: Workspace,
) -> tuple[str, ...]:

    try:
        values = workspace.config.get("input_types", [])
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError("input_types must be a list of input type strings")
        return tuple(values)
    except (TypeError, ValueError) as error:
        raise AIError(
            ErrorCode.MODEL_CONFIG_INVALID,
            "Invalid workspace input_types configuration",
            retryable=False,
            safe_details={"reason": "model_input_configuration_invalid"},
        ) from error


def _local_model_options(workspace: Workspace, args: Namespace) -> dict[str, str | None]:
    api = args.model_api or workspace.config.get("model_api") or os.getenv("MODEL_API") or "openai"
    if api not in MODEL_APIS:
        raise AIError(ErrorCode.MODEL_CONFIG_INVALID, "model_api must be one of: " + ", ".join(MODEL_APIS))
    prefix = "ANTHROPIC" if api == "anthropic" else "OPENAI"
    options = {"api": api}
    for name, explicit in (("model", args.model), ("base_url", args.base_url), ("api_key", args.api_key)):
        field = ConfigField(name=f"{prefix}_{name.upper()}", cast=str, default=None, secret=name == "api_key")
        value = explicit if explicit is not None else environ.config.get(field)
        if name == "model" and (not isinstance(value, str) or not value.strip()):
            value = workspace.config.get("model")
        if value is None:
            value = os.getenv(field.name, "").strip() or None
        if name != "model" and (value is None or not value.strip()):
            value = os.getenv(field.name) or None
        options[name] = value.strip() if name == "model" and isinstance(value, str) else value
    return options


def _local_model(
    workspace: Workspace,
    *,
    model: "str | None" = None,
    api: "str | None" = None,
    base_url: "str | None" = None,
    api_key: "str | None" = None,
) -> "ModelBinding":
    from linktools.ai.model import ModelBinding, create_model

    api = api or workspace.config.get("model_api") or os.getenv("MODEL_API") or "openai"
    if not isinstance(api, str) or api not in MODEL_APIS:
        raise AIError(ErrorCode.MODEL_CONFIG_INVALID, "unsupported model_api")
    prefix = "ANTHROPIC" if api == "anthropic" else "OPENAI"
    input_types = _local_input_config(workspace)
    configured = workspace.config.get("model")
    selected_model = (
        model.strip()
        if isinstance(model, str) and model.strip()
        else configured.strip()
        if isinstance(configured, str) and configured.strip()
        else os.getenv(f"{prefix}_MODEL", "").strip()
    )
    if not selected_model:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "model is required")

    try:
        return create_model(
            protocol=api,
            model=selected_model,
            input_types=input_types,
            base_url=(
                base_url
                if base_url is not None
                else os.getenv(f"{prefix}_BASE_URL", "").strip() or None
            ),
            api_key=(
                api_key
                if api_key is not None
                else os.getenv(f"{prefix}_API_KEY", "").strip() or None
            ),
        )
    except (TypeError, ValueError) as error:
        raise AIError(
            ErrorCode.MODEL_CONFIG_INVALID,
            "Invalid local model configuration; check input_types and connection settings",
            retryable=False,
            safe_details={"reason": "model_configuration_invalid"},
        ) from error


def _local_runtime_model(workspace: Workspace, args: Namespace) -> "ModelBinding":
    return _local_model(workspace, **_local_model_options(workspace, args))


@asynccontextmanager
async def _open_local_runtime(
    workspace: Workspace,
    *,
    model: "ModelBinding | None" = None,
) -> "AsyncIterator[Runtime]":
    from linktools.ai.capability import AssetCapability, WorkspaceCapability
    from linktools.ai.runtime import RuntimeFeatures
    from linktools.ai.capability import CheckpointCapability, ExecutionCapability, TraceCapability
    from linktools.ai.runtime import Runtime

    assets = _local_runtime_assets(workspace)
    await assets.initialize()
    try:
        features = RuntimeFeatures("workspace")
        features.model("default", _local_model(workspace) if model is None else model)
        features.capability(AssetCapability(assets))
        features.capability(WorkspaceCapability(workspace))
        features.capability(ExecutionCapability(retain_history=True))
        features.capability(CheckpointCapability())
        features.capability(TraceCapability())
        async with Runtime.open(
            "default",
            storage=_local_runtime_storage(workspace),
            features=features,
            metrics=await _local_metrics(workspace),
        ) as runtime:
            yield runtime
    finally:
        await assets.close()


def _run_async(coroutine: "Coroutine[object, object, ResultT]") -> ResultT:
    try:
        return asyncio.run(coroutine)
    except (AIError, ImportError, TypeError, ValueError) as error:
        raise CommandError(str(error)) from error


__all__ = []
