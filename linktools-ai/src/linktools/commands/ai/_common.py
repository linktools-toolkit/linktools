#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared local composition for AI CLI commands."""

import asyncio
import os
from argparse import Namespace
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, cast

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
    from linktools.ai.model import ModelInputCapability, ModelRegistry, UnknownModelInputs
    from linktools.ai.runtime import Runtime, RuntimeStorage

OPENAI_BASE_URL = ConfigField(name="OPENAI_BASE_URL", cast=str, default=None)
OPENAI_MODEL = ConfigField(name="OPENAI_MODEL", cast=str, default=None)
OPENAI_API_KEY = ConfigField(name="OPENAI_API_KEY", cast=str, default=None, secret=True)


def _add_local_runtime_arguments(parser: "CommandParser") -> None:
    parser.add_argument("--project", type=Path, default=None, help="working directory")
    parser.add_argument("--base-url", action=ConfigAction, config=OPENAI_BASE_URL)
    parser.add_argument("--model", action=ConfigAction, config=OPENAI_MODEL)
    parser.add_argument("--api-key", action=ConfigAction, config=OPENAI_API_KEY)
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
) -> "tuple[tuple[ModelInputCapability, ...], UnknownModelInputs]":
    from linktools.ai.model import ModelInputCapability

    missing = object()
    if (
        any(key in workspace.config for key in ("vision", "OPENAI_VISION", "LINKTOOLS_OPENAI_VISION"))
        or "OPENAI_VISION" in os.environ
        or "LINKTOOLS_OPENAI_VISION" in os.environ
        or environ.config.get("OPENAI_VISION", default=missing) is not missing
    ):
        raise AIError(
            ErrorCode.MODEL_CONFIG_INVALID,
            "Legacy vision configuration was removed; remove vision/OPENAI_VISION "
            "and configure input_capabilities and unknown_inputs in .linktools/config.yaml",
            retryable=False,
            safe_details={"reason": "legacy_vision_configuration"},
        )
    try:
        values = workspace.config.get("input_capabilities", [])
        unknown = workspace.config.get("unknown_inputs", "reject")
        if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
            raise ValueError("input_capabilities must be a list of declarations")
        if unknown not in ("reject", "passthrough"):
            raise ValueError("unknown_inputs must be reject or passthrough")
        return tuple(ModelInputCapability(**item) for item in values), cast("UnknownModelInputs", unknown)
    except (TypeError, ValueError) as error:
        raise AIError(
            ErrorCode.MODEL_CONFIG_INVALID,
            "Invalid workspace input_capabilities or unknown_inputs configuration",
            retryable=False,
            safe_details={"reason": "model_input_configuration_invalid"},
        ) from error


def _local_models(
    workspace: Workspace,
    *,
    model: "str | None" = None,
    base_url: "str | None" = None,
    api_key: "str | None" = None,
) -> "ModelRegistry":
    from linktools.ai.model import ModelRegistry

    input_capabilities, unknown_inputs = _local_input_config(workspace)
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

    try:
        return ModelRegistry.openai(
            model=selected_model,
            input_capabilities=input_capabilities,
            unknown_inputs=unknown_inputs,
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
    except (TypeError, ValueError) as error:
        raise AIError(
            ErrorCode.MODEL_CONFIG_INVALID,
            "Invalid local model configuration; check input_capabilities, unknown_inputs, and connection settings",
            retryable=False,
            safe_details={"reason": "model_configuration_invalid"},
        ) from error


def _local_runtime_models(workspace: Workspace, args: Namespace) -> "ModelRegistry":
    return _local_models(
        workspace,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )


@asynccontextmanager
async def _open_local_runtime(
    workspace: Workspace,
    *,
    models: "ModelRegistry | None" = None,
) -> "AsyncIterator[Runtime]":
    from linktools.ai.capability import CapabilityGroup
    from linktools.ai.runtime import Runtime

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
